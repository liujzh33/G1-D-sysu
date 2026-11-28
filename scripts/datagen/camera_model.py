"""Real-camera image model applied to each rendered frame before it is encoded (collect.py --camera-model), so the
recorded images carry what the real cameras put on theirs: motion blur from the camera's own motion over the
exposure, the lens / ISP softness or sharpening, white balance off by a little and signal-dependent sensor noise; the
frame is then JPEG-encoded once, as on the robot.

Profiles are calibrated against real episodes (runs/school_v2/image_stats.py). g1d_sysu: the Unitree G1-D of the SYSU
lab (data/real/g1-d-sysu, episodes 34 / 36, 640x480 JPEG q95). There the head camera shows a fine grain (high-pass std
2.2-2.7 levels on the green screen against 1.0-1.6 rendered) and is no softer than the render; the wrist cameras are
much softer (Laplacian variance 45-180 against 200-240 rendered), carry less grain (about 1 level against 0.3) and
smear whenever the hand moves. Each episode draws its own parameters per camera from the profile's ranges (rng from
the layout), every frame of the episode uses them.

Per frame: lens distortion (cameras with `lens`; the recorded intrinsics stay the pinhole K) -> linear light -> white balance gains -> motion blur (a line kernel along the image motion, since the last
frame, of points depth_m ahead, over exposure_ms) -> defocus (Gaussian) -> unsharp mask -> sRGB -> noise (sigma at mid
grey, scaled by the square root of the signal; luminance plus a chroma share; slightly correlated as after
demosaicing) -> uint8. Done frame by frame inside the episode: one pass over a whole episode at its end blocks Kit's
main loop for minutes and its hang detector fires.
"""

import numpy as np

PROFILES = {
    "g1d_sysu": {
        # lens: OpenCV k1 k2 p1 p2 k3 of the checkerboard-calibrated head_left (scripts/g1d/calib_toolkit/examples/
        # intrinsics/camera_intrinsics_head_left.json), for the g1d_calib_head_left render (same K); up to 9 px.
        "cam_head": dict(exposure_ms=(3.0, 12.0), defocus_px=(0.0, 0.3), sharpen=(0.0, 0.25), noise=(1.0, 2.4),
                         chroma=(0.2, 0.5), wb=0.04, depth_m=1.0,
                         lens=(0.093616, -0.105871, 0.007352, -0.002534, 0.031005)),
        "cam_left_wrist": dict(exposure_ms=(8.0, 25.0), defocus_px=(0.3, 0.9), sharpen=(0.0, 0.0), noise=(0.4, 1.2),
                               chroma=(0.2, 0.5), wb=0.04, depth_m=0.35),
        "cam_right_wrist": dict(exposure_ms=(8.0, 25.0), defocus_px=(0.3, 0.9), sharpen=(0.0, 0.0), noise=(0.4, 1.2),
                                chroma=(0.2, 0.5), wb=0.04, depth_m=0.35),
    },
}  # fmt: skip
NOISE_CORRELATION_PX = 0.5  # Gaussian sigma of the noise field (demosaicing spreads it), std kept
MIN_BLUR_PX = 0.5
SHARPEN_PX = 1.0  # unsharp mask radius (the head camera's ISP)
LIN_BITS = 12  # linear -> sRGB through a table of this many bits

TO_LIN = np.where(np.arange(256) / 255 <= 0.04045, np.arange(256) / 255 / 12.92,
                  ((np.arange(256) / 255 + 0.055) / 1.055) ** 2.4).astype(np.float32)  # fmt: skip
_l = np.arange(2**LIN_BITS) / (2**LIN_BITS - 1)
TO_SRGB = (np.where(_l <= 0.0031308, _l * 12.92, 1.055 * _l ** (1 / 2.4) - 0.055) * 255).astype(np.float32)


def draw(rng, profile):
    """{camera: parameters} of one episode."""
    out = {}
    for cam, r in PROFILES[profile].items():
        u = lambda k: float(rng.uniform(*r[k]))  # noqa: E731
        gains = 1 + rng.uniform(-r["wb"], r["wb"], 3)
        out[cam] = dict(exposure_ms=u("exposure_ms"), defocus_px=u("defocus_px"), sharpen=u("sharpen"), noise=u("noise"),
                        chroma=u("chroma"), wb_gains=(gains / gains[1]).round(4).tolist(), depth_m=r["depth_m"],
                        seed=int(rng.integers(2**31)))  # fmt: skip
        if r.get("lens"):
            out[cam]["lens"] = list(r["lens"])
    return out


def lens_maps(K, dist, size):
    """cv2.remap maps that turn a pinhole render with intrinsics K into the distorted image of the same K."""
    import cv2

    w, h = size
    g = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).reshape(-1, 1, 2).astype(np.float64)
    u = cv2.undistortPoints(g, K, np.asarray(dist, np.float64), P=K).reshape(h, w, 2).astype(np.float32)
    return u[..., 0], u[..., 1]


def image_motion(T0, T1, K, size, depth):
    """Pixel motion (2,) of the image centre region (points `depth` ahead of the camera at T1) from camera pose T0
    to T1 (camera-to-world, USD camera: looks along -z, +y up)."""
    w, h = size
    g = np.stack(np.meshgrid([w * 0.3, w * 0.5, w * 0.7], [h * 0.3, h * 0.5, h * 0.7]), -1).reshape(-1, 2)
    ray = np.c_[(g - K[:2, 2]) / [K[0, 0], K[1, 1]], np.ones(len(g))] * depth  # x right, y down, z ahead
    pts = np.c_[ray[:, 0], -ray[:, 1], -ray[:, 2]] @ T1[:3, :3].T + T1[:3, 3]

    def uv(Tc):
        c = (pts - Tc[:3, 3]) @ Tc[:3, :3]
        c = np.c_[c[:, 0], -c[:, 1], -c[:, 2]]
        return c[:, :2] / np.maximum(c[:, 2:3], 1e-6) * [K[0, 0], K[1, 1]] + K[:2, 2]

    return np.median(uv(T1) - uv(T0), 0)


def line_kernel(vec):
    L = float(np.hypot(*vec))
    n = (int(np.ceil(L)) + 2) | 1
    k = np.zeros((n, n), np.float32)
    c = n // 2
    for s in np.linspace(-0.5, 0.5, 4 * n):
        x, y = c + s * vec[0], c + s * vec[1]
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            if 0 <= x0 + dx < n and 0 <= y0 + dy < n:
                k[y0 + dy, x0 + dx] += wt
    return k / k.sum()


def apply_frame(img, p, blur_vec, rng, lens=None):
    """One uint8 RGB frame through the model with parameters p and motion blur vector blur_vec (px); lens: lens_maps."""
    import cv2

    if lens is not None:
        img = cv2.remap(img, lens[0], lens[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    x = TO_LIN[img] * np.asarray(p["wb_gains"], np.float32)
    if np.hypot(*blur_vec) >= MIN_BLUR_PX:
        x = cv2.filter2D(x, -1, line_kernel(blur_vec), borderType=cv2.BORDER_REFLECT)
    if p["defocus_px"] > 0.05:
        x = cv2.GaussianBlur(x, (0, 0), p["defocus_px"])
    if p.get("sharpen", 0) > 0:
        x = x + p["sharpen"] * (x - cv2.GaussianBlur(x, (0, 0), SHARPEN_PX))
    top = 2**LIN_BITS - 1
    y = TO_SRGB[np.clip(x * top + 0.5, 0, top).astype(np.int32)]
    h, w = y.shape[:2]
    lum = rng.standard_normal((h, w), dtype=np.float32)
    chroma = rng.standard_normal((h, w, 3), dtype=np.float32)
    if NOISE_CORRELATION_PX:
        keep = np.float32(np.sqrt(4 * np.pi) * NOISE_CORRELATION_PX)  # white noise std after that blur: 1 / keep
        lum = cv2.GaussianBlur(lum, (0, 0), NOISE_CORRELATION_PX) * keep
        chroma = cv2.GaussianBlur(chroma, (0, 0), NOISE_CORRELATION_PX) * keep
    c = np.float32(p["chroma"])
    field = np.sqrt(1 - c * c) * lum[..., None] + c * chroma
    sigma = np.float32(p["noise"]) * np.sqrt((y + 16) / np.float32(128 + 16))
    return np.clip(y + sigma * field + 0.5, 0, 255).astype(np.uint8)


class CameraModel:
    """Per-episode model over the env's cameras: call it on each camera's frame once per step, in step order."""

    def __init__(self, env, profile, rng):
        self.params = draw(rng, profile)
        cm = env.camera_manager
        names = list(cm.camera_names[0])
        self.cm, self.index = cm, {c: names.index(c) for c in self.params if c in names}
        self.K = {c: np.asarray(cm.get_camera_intrinsics(k, 0)) for c, k in self.index.items()}
        self.rng = {c: np.random.default_rng(p["seed"]) for c, p in self.params.items()}
        self.last, self.blur, self.lens = {}, {c: [] for c in self.params}, {}

    def __call__(self, camera, img, fps):
        p = self.params.get(camera)
        if p is None:
            return img
        if p.get("lens") and camera in self.K and camera not in self.lens:
            self.lens[camera] = lens_maps(self.K[camera], p["lens"], img.shape[1::-1])
        vec = np.zeros(2)
        if camera in self.index:
            T = np.asarray(self.cm.get_camera_extrinsics(self.index[camera], 0), dtype=np.float64)
            if camera in self.last:
                h, w = img.shape[:2]
                vec = image_motion(self.last[camera], T, self.K[camera], (w, h), p["depth_m"]) * p["exposure_ms"] * 1e-3 * fps
            self.last[camera] = T
        self.blur[camera].append(float(np.hypot(*vec)))
        return apply_frame(img, p, vec, self.rng[camera], self.lens.get(camera))

    def summary(self):
        out = {}
        for c, p in self.params.items():
            b = self.blur[c]
            out[c] = dict(p, blur_px_p50_p95=np.percentile(b, [50, 95]).round(2).tolist() if b else None)
        return out


def apply_episode(images, poses, Ks, params, fps):
    """Offline: images {camera: [uint8 RGB frames]}, poses {camera: (T, 4, 4)}, Ks {camera: K} -> processed frames."""
    out = dict(images)
    for cam, p in params.items():
        if cam not in images:
            continue
        rng = np.random.default_rng(p["seed"])
        frames = []
        lens = None
        for t, img in enumerate(images[cam]):
            if lens is None and p.get("lens") and cam in Ks:
                lens = lens_maps(Ks[cam], p["lens"], img.shape[1::-1])
            vec = np.zeros(2)
            if cam in poses and t > 0:
                h, w = img.shape[:2]
                vec = image_motion(poses[cam][t - 1], poses[cam][t], Ks[cam], (w, h), p["depth_m"]) * p["exposure_ms"] * 1e-3 * fps
            frames.append(apply_frame(img, p, vec, rng, lens))
        out[cam] = frames
    return out
