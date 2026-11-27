"""Pre-evaluation real/sim alignment check of the G1-D on the school_v2 table (real_scene4_school_v2), from one real
head-left frame (Unitree color_0).

The sim head camera is the robot root (env_cfg/robot/g1d_school_v2.yml default_root_pos / default_root_rot) x FK of
head_link (waist from the frame's joints, lift 0) x the head mount (camera_override.cam_head), with the calibrated
head_left intrinsics and distortion. The table edges, the black tape rectangle (tape centreline) and the grommet frame
(usd/build_params.json of the scene) are projected with it. Along each projected point's normal the nearest real image
edge of the same orientation is found (so wood grain and cloth folds rarely match; the arms are masked out when the
frame's joints are given): the signed offset, real minus sim, in px. Each of the 11 segments gets the median offset of
its points.

Verdict: the largest |segment offset| under the current config. Under --threshold (5 px) the config is used as is.
Either way the stand (root x, y; the yaw too with --fit-yaw) is refitted to the edges (ICP-style, each segment
weighted equally) and printed;
--apply writes it into the yml when over threshold. With --data, the head mount is checked against the arm and gripper
outlines (FK at the recorded joints); --fit-mount refits it (several minutes).

    align_check.py --image <color_0.jpg | side-by-side head jpg> [--data <data.json> --frame N] [--out DIR]
                   [--threshold 5] [--apply] [--fit-yaw] [--fit-mount] [--start DX_CM DY_CM DYAW_DEG]
Exit code 0 = pass, 2 = recalibrate.
"""
import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from urdf_fk import FK, URDF, arm_q, rpy_m  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ROBOT_YML = ROOT / "env_cfg/robot/g1d_school_v2.yml"
SCENE = ROOT / "real2sim/scenes/real_scene4_school_v2/usd/build_params.json"
INTR = ROOT / "scripts/g1d/calib_toolkit/examples/intrinsics/camera_intrinsics_head_left.json"
ROOM_POS = np.array([-0.015, -0.30, 0.0])  # scripts/g1d/build_house4_layouts.py SCENES["g1d_school_v2"]
USD_TO_CV = np.diag([1.0, -1.0, -1.0])
SIZE = (640, 480)
STEP = 0.003  # model edge sampling (m)
BINS = 12  # edge orientation bins (15 deg); a model point matches its bin and both neighbours
MIN_SEG_POINTS = 15
ICP_RADII = (25, 15, 10, 6, 4, 3, 3)  # px searched along the normal, per iteration
ARM_LINKS = [f"{s}_{n}" for s in ("left", "right") for n in
             ("shoulder_yaw_link", "elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link",
              "dex1_model_base_link", "dex1_finger_link_1", "dex1_finger_link_2", "dex1_finger_link_1_mid",
              "dex1_finger_link_2_mid", "dex1_finger_link_1_tip", "dex1_finger_link_2_tip")]


# ---------------- configuration ----------------

def load_config(path=ROBOT_YML):
    r = yaml.safe_load(Path(path).read_text())["robots"][0]
    w, x, y, z = r["default_root_rot"]
    if abs(x) + abs(y) > 1e-6:
        raise SystemExit("default_root_rot is not a pure yaw")
    head = (r.get("camera_override") or {}).get("cam_head")
    if head is None:
        raise SystemExit(f"{path}: no camera_override.cam_head (the asset default mount is not this robot's)")
    return dict(root=np.array(r["default_root_pos"], float), yaw=float(np.degrees(2 * np.arctan2(z, w))),
                mount_pos=np.array(head["pos"], float), mount_wxyz=np.array(head["ori"], float), cam_type=head.get("type"))


def mount_matrix(pos, wxyz, delta=np.zeros(6)):
    """head_link -> OpenCV camera; delta = rotvec (deg, camera frame) + shift (mm, head_link)."""
    R = Rotation.from_quat(np.asarray(wxyz)[[1, 2, 3, 0]]).as_matrix() @ USD_TO_CV
    R = R @ Rotation.from_rotvec(np.radians(delta[:3])).as_matrix()
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(pos) + np.asarray(delta[3:]) / 1000
    return T


def root_matrix(root, yaw_deg):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rpy_m(0, 0, np.radians(yaw_deg)), root
    return T


def scene_model(path=SCENE):
    """[(segment name, (2, 3) world endpoints)] and the 10 key corners (10, 3)."""
    p = json.loads(Path(path).read_text())
    t, g, b = p["table"], p["grommet"], p["target_box"]
    z, hl, hw = t["top_z"], t["length"] / 2, t["width"] / 2

    def seg(name, a, bb):
        return name, np.array([[*a, z], [*bb, z]]) + ROOM_POS

    gy1 = hw - g["far_inset"]
    gx0, gx1, gy0 = g["x"] - g["length"] / 2, g["x"] + g["length"] / 2, gy1 - g["width"]
    (bx0, bx1), (by0, by1) = b["x"], b["y"]
    segs = [seg("table far edge", (-hl, hw), (hl, hw)),  # the near edge is under the robot
            seg("table left end", (-hl, -hw), (-hl, hw)), seg("table right end", (hl, -hw), (hl, hw)),
            seg("tape far", (bx0, by1), (bx1, by1)), seg("tape near", (bx0, by0), (bx1, by0)),
            seg("tape left", (bx0, by0), (bx0, by1)), seg("tape right", (bx1, by0), (bx1, by1)),
            seg("grommet far", (gx0, gy1), (gx1, gy1)), seg("grommet near", (gx0, gy0), (gx1, gy0)),
            seg("grommet left", (gx0, gy0), (gx0, gy1)), seg("grommet right", (gx1, gy0), (gx1, gy1))]
    corners = np.array([[-hl, hw], [hl, hw], [bx0, by1], [bx1, by1], [bx0, by0], [bx1, by0],
                        [gx0, gy1], [gx1, gy1], [gx0, gy0], [gx1, gy0]])
    return segs, np.c_[corners, np.full(len(corners), z)] + ROOM_POS


def sample(segs):
    """Points (N, 3), unit segment directions (N, 3), segment index (N,)."""
    pts, dirs, idx = [], [], []
    for k, (_, (a, b)) in enumerate(segs):
        n = max(4, int(np.linalg.norm(b - a) / STEP))
        s = np.linspace(0.03, 0.97, n)[:, None]
        pts.append(a + s * (b - a))
        dirs.append(np.repeat(((b - a) / np.linalg.norm(b - a))[None], n, 0))
        idx.append(np.full(n, k))
    return np.concatenate(pts), np.concatenate(dirs), np.concatenate(idx)


# ---------------- image ----------------

def load_head(path):
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"cannot read {path}")
    if img.shape[1] == 2 * SIZE[0]:  # side-by-side stereo: left half is head_left
        img = img[:, : SIZE[0]]
    if img.shape[1::-1] != SIZE:
        raise SystemExit(f"{path}: expected 640x480 head_left (or 1280x480 side-by-side), got {img.shape[1::-1]}")
    return img


def oriented_edges(img, mask=None):
    """(BINS, H, W) bool: Canny edges whose gradient orientation is in the bin or a neighbouring one."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    e = cv2.Canny(g, 30, 90) > 0
    if mask is not None:
        e &= mask == 0
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    ang = ((np.degrees(np.arctan2(gy, gx)) % 180) // (180 / BINS)).astype(int)
    per = np.stack([e & (ang == b) for b in range(BINS)])
    return per | np.roll(per, 1, 0) | np.roll(per, -1, 0)


class Projector:
    def __init__(self):
        intr = json.loads(INTR.read_text())
        self.K, self.D = np.array(intr["K"]), np.array(intr["dist"])

    def __call__(self, cam_world, P):
        C = np.linalg.inv(cam_world)
        X = (C[:3, :3] @ P.T + C[:3, 3:4]).T
        uv = np.full((len(P), 2), np.nan)
        ok = X[:, 2] > 0.05
        if ok.any():
            uv[ok] = cv2.projectPoints(X[ok], np.zeros(3), np.zeros(3), self.K, self.D)[0].reshape(-1, 2)
        return uv


def offsets(proj, cam, pts, dirs, edges, mask, radius):
    """Per point: projected uv, image unit normal and the signed offset (px) to the nearest real edge of the same
    orientation along the normal within radius (nan: none, masked or out of view)."""
    uv = proj(cam, pts)
    d = proj(cam, pts + dirs * 0.002) - uv
    n = np.c_[-d[:, 1], d[:, 0]]
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    off = np.full(len(pts), np.nan)
    ok = np.isfinite(uv).all(1) & (uv[:, 0] > 2) & (uv[:, 0] < SIZE[0] - 3) & (uv[:, 1] > 2) & (uv[:, 1] < SIZE[1] - 3)
    if mask is not None:
        ui, vi = np.clip(uv[:, 0], 0, SIZE[0] - 1), np.clip(uv[:, 1], 0, SIZE[1] - 1)
        ok &= mask[np.nan_to_num(vi).astype(int), np.nan_to_num(ui).astype(int)] == 0
    if not ok.any():
        return uv, n, off
    i = np.flatnonzero(ok)
    b = ((np.degrees(np.arctan2(n[i, 1], n[i, 0])) % 180) // (180 / BINS)).astype(int)  # gradient ~ normal
    ts = np.arange(-radius, radius + 0.25, 0.5)
    P = uv[i, None, :] + ts[None, :, None] * n[i, None, :]
    u, v = P[..., 0].round().astype(int), P[..., 1].round().astype(int)
    inb = (u >= 0) & (u < SIZE[0]) & (v >= 0) & (v < SIZE[1])
    hit = np.zeros(u.shape, bool)
    hit[inb] = edges[np.broadcast_to(b[:, None], u.shape)[inb], v[inb], u[inb]]
    cost = np.where(hit, np.abs(ts)[None, :], np.inf)
    j = cost.argmin(1)
    found = np.isfinite(cost[np.arange(len(i)), j])
    off[i[found]] = ts[j[found]]
    return uv, n, off


def segment_offsets(off, idx, nseg):
    out = []
    for k in range(nseg):
        o = off[idx == k]
        good = np.isfinite(o)
        out.append(float(np.median(o[good])) if good.sum() >= max(MIN_SEG_POINTS, 0.25 * len(o)) else None)
    return out


# ---------------- robot ----------------

def frame_joints(data, i):
    s = json.loads(Path(data).read_text())["data"][i]["states"]
    body = s["body"]["qpos"]
    q = {**arm_q("left", s["left_arm"]["qpos"]), **arm_q("right", s["right_arm"]["qpos"]),
         "torso_Joint": body[12], "Yaw_Joint": body[14]}
    for side in ("left", "right"):
        g = float(np.clip(s[f"{side}_ee"]["qpos"][0] / 5.4, 0, 1))
        q[f"{side}_dex1_finger_joint_1"] = q[f"{side}_dex1_finger_joint_2"] = -0.02 + 0.0445 * (1 - g)  # open .. closed
    return q


def arm_meshes():
    import trimesh

    root, out = ET.parse(URDF).getroot(), {}
    for link in root.findall("link"):
        if link.get("name") not in ARM_LINKS:
            continue
        ms = []
        for vis in link.findall("visual"):
            geo = vis.find("geometry/mesh")
            if geo is None:
                continue
            m = trimesh.load(f"{URDF.rsplit('/', 1)[0]}/{geo.get('filename')}", force="mesh")
            o = vis.find("origin")
            if o is not None:
                T = np.eye(4)
                T[:3, :3] = rpy_m(*[float(x) for x in o.get("rpy", "0 0 0").split()])
                T[:3, 3] = [float(x) for x in o.get("xyz", "0 0 0").split()]
                m.apply_transform(T)
            ms.append(m)
        if ms:
            out[link.get("name")] = trimesh.util.concatenate(ms)
    return out


def arm_mask(fk, meshes, q, head_cam, proj, skip=()):
    """Filled silhouette of the arms and grippers in the head camera (uint8)."""
    mask = np.zeros(SIZE[::-1], np.uint8)
    to_cam = np.linalg.inv(fk.pose("head_link", q) @ head_cam)
    for name, m in meshes.items():
        if any(s in name for s in skip):
            continue
        T = to_cam @ fk.pose(name, q)
        P = (T[:3, :3] @ m.vertices.T + T[:3, 3:4]).T
        if (P[:, 2] < 0.05).any():
            continue
        uv = cv2.projectPoints(P, np.zeros(3), np.zeros(3), proj.K, proj.D)[0].reshape(-1, 2)
        cv2.fillPoly(mask, list(uv[m.faces].round().astype(np.int32)), 255)
    return mask


def outline_cost(mask, dt):
    c = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(c)
    keep = (xs > 2) & (xs < SIZE[0] - 3) & (ys > 2) & (ys < SIZE[1] - 3)
    return np.minimum(dt[ys[keep], xs[keep]], 15.0) if keep.sum() > 50 else None


def plain_dt(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    return cv2.distanceTransform(255 - cv2.Canny(g, 40, 110), cv2.DIST_L2, 3)


# ---------------- checks ----------------

def check_mount(args, cfg, fk, proj, meshes, report):
    """Arm / gripper outline error of the configured mount; with --fit-mount the refitted mount matrix, else None."""
    frames = json.loads(args.data.read_text())["data"]
    mf = []
    for i in np.linspace(0, len(frames) - 1, args.mount_frames).round().astype(int):
        p = args.data.parent / "colors" / f"{i:06d}_color_0.jpg"
        if p.exists():
            mf.append((frame_joints(args.data, int(i)), plain_dt(load_head(p))))
    if not mf:
        print("[mount] no colors/<frame>_color_0.jpg next to --data: mount check skipped")
        return None

    def cost(delta):
        # the shoulders sit half under the head shell and the chest edge: only the elbows down are scored
        T = mount_matrix(cfg["mount_pos"], cfg["mount_wxyz"], delta)
        vals = [outline_cost(arm_mask(fk, meshes, q, T, proj, skip=("shoulder",)), dt) for q, dt in mf]
        vals = [v for v in vals if v is not None]
        return float(np.mean(np.concatenate(vals))) if vals else np.nan

    m0 = cost(np.zeros(6))
    report["mount_outline_px"] = round(m0, 2)
    print(f"[mount] arm / gripper outline to real edges: mean {m0:.2f} px over {len(mf)} frames "
          f"({'ok' if m0 <= args.mount_threshold else 'high: refit with --fit-mount'}; 2.6-2.8 px right after the 2026-11-20 fit)")
    if not args.fit_mount:
        return None
    res = minimize(cost, np.zeros(6), method="Powell", options=dict(xtol=0.05, ftol=1e-3, maxfev=500))
    T = mount_matrix(cfg["mount_pos"], cfg["mount_wxyz"], res.x)
    qu = Rotation.from_matrix(T[:3, :3] @ USD_TO_CV).as_quat()[[3, 0, 1, 2]]
    qu = qu if qu[0] >= 0 else -qu
    report["mount_fit"] = dict(delta_rotvec_deg=res.x[:3].round(2).tolist(), delta_shift_mm=res.x[3:].round(1).tolist(),
                               outline_px=round(res.fun, 2), pos=T[:3, 3].round(4).tolist(), ori=qu.round(4).tolist())
    print(f"[mount] refit: rot {res.x[:3].round(2)} deg (camera frame), shift {res.x[3:].round(1)} mm -> {res.fun:.2f} px\n"
          f"        camera_override.cam_head: pos {T[:3, 3].round(4).tolist()}, ori {qu.round(4).tolist()} "
          "(cam_head_right: the same ori, pos shifted by the same mm); the stand below already uses this mount")
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, type=Path, help="real head_left frame (color_0)")
    ap.add_argument("--data", type=Path, help="the episode's data.json: waist angles, arm mask and the mount check")
    ap.add_argument("--frame", type=int, default=0, help="frame index of --image in --data")
    ap.add_argument("--robot-yml", type=Path, default=ROBOT_YML)
    ap.add_argument("--out", type=Path, default=Path("align_check_out"))
    ap.add_argument("--threshold", type=float, default=5.0, help="largest segment offset (px) that passes")
    ap.add_argument("--mount-threshold", type=float, default=3.5, help="arm outline error (px) above which to refit")
    ap.add_argument("--start", type=float, nargs=3, default=(0, 0, 0), metavar=("DX_CM", "DY_CM", "DYAW_DEG"),
                    help="offset the configured stand before checking (to test the check)")
    ap.add_argument("--apply", action="store_true", help="write the refitted stand into --robot-yml when over threshold")
    ap.add_argument("--fit-yaw", action="store_true",
                    help="refit the yaw too (it trades against x on these features: +-2 deg spread; default keeps it)")
    ap.add_argument("--fit-mount", action="store_true", help="also refit the head mount on the arms (needs --data)")
    ap.add_argument("--mount-frames", type=int, default=12, help="frames of --data used by the mount check / fit")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.robot_yml)
    fk, proj = FK(), Projector()
    q = frame_joints(args.data, args.frame) if args.data else {}
    head = fk.pose("head_link", q)  # root frame
    img = load_head(args.image)
    mount = mount_matrix(cfg["mount_pos"], cfg["mount_wxyz"])
    report = dict(image=str(args.image), robot_yml=str(args.robot_yml), camera_type=cfg["cam_type"],
                  root=cfg["root"].tolist(), yaw_deg=round(cfg["yaw"], 3),
                  waist_yaw_pitch=[q.get("torso_Joint", 0.0), q.get("Yaw_Joint", 0.0)])

    meshes, mask = (arm_meshes(), None) if args.data else (None, None)
    if args.data:
        refit = check_mount(args, cfg, fk, proj, meshes, report)
        mount = refit if refit is not None else mount
        mask = cv2.dilate(arm_mask(fk, meshes, q, mount, proj), np.ones((9, 9), np.uint8))

    segs, corners = scene_model()
    names = [s[0] for s in segs]
    pts, dirs, idx = sample(segs)
    edges = oriented_edges(img, mask)
    base = np.array([args.start[0] / 100, args.start[1] / 100, args.start[2]])

    def cam_of(x):
        return root_matrix(cfg["root"] + np.array([x[0], x[1], 0.0]), cfg["yaw"] + x[2]) @ head @ mount

    # ICP: correspondences along the normals, then a robust least-squares step on the stand
    x = base.copy()
    for radius in ICP_RADII:
        uv, n, off = offsets(proj, cam_of(x), pts, dirs, edges, mask, radius)
        good = np.isfinite(off)
        if good.sum() < 50:
            break
        target = uv[good] + off[good, None] * n[good]
        cnt = np.bincount(idx[good], minlength=len(segs))
        w = 1 / np.sqrt(cnt[idx[good]])
        P, N = pts[good], n[good]

        def resid(xx):
            xx = np.r_[xx, base[2]] if not args.fit_yaw else xx
            return w * np.einsum("ij,ij->i", proj(cam_of(xx), P) - target, N)

        if not args.fit_yaw:
            x = np.r_[least_squares(resid, x[:2], loss="soft_l1", f_scale=1.5, x_scale=[0.01, 0.01]).x, base[2]]
        else:
            x = least_squares(resid, x, loss="soft_l1", f_scale=1.5, x_scale=[0.01, 0.01, 0.5]).x

    _, _, off0 = offsets(proj, cam_of(base), pts, dirs, edges, mask, 25)
    _, _, off1 = offsets(proj, cam_of(x), pts, dirs, edges, mask, 25)
    seg0, seg1 = segment_offsets(off0, idx, len(segs)), segment_offsets(off1, idx, len(segs))
    vis0 = [abs(v) for v in seg0 if v is not None]
    worst = max(vis0) if vis0 else float("inf")
    uv0, uv1 = proj(cam_of(base), corners), proj(cam_of(x), corners)
    ok = np.isfinite(uv0).all(1) & np.isfinite(uv1).all(1)
    move = np.linalg.norm(uv1 - uv0, axis=1)[ok]
    new_root = [round(float(v), 4) for v in cfg["root"] + [x[0], x[1], 0.0]]
    new_yaw = cfg["yaw"] + x[2]
    new_rot = [round(float(np.cos(np.radians(new_yaw) / 2)), 5), 0, 0, round(float(np.sin(np.radians(new_yaw) / 2)), 5)]
    passed = worst <= args.threshold and len(vis0) >= 6
    c = dict(dx_cm=round((x[0] - base[0]) * 100, 2), dy_cm=round((x[1] - base[1]) * 100, 2), dyaw_deg=round(x[2] - base[2], 2))
    r2 = lambda v: None if v is None else round(v, 2)  # noqa: E731
    report.update(segment_offset_px_config={k: r2(v) for k, v in zip(names, seg0)},
                  segment_offset_px_refit={k: r2(v) for k, v in zip(names, seg1)},
                  worst_offset_px=r2(worst), corner_move_px=dict(max=r2(float(move.max())), mean=r2(float(move.mean()))),
                  correction=c, refit_stand=dict(default_root_pos=new_root, default_root_rot=new_rot, yaw_deg=round(new_yaw, 3)),
                  passed=bool(passed), threshold_px=args.threshold)

    print("[stand] segment offset, real minus sim along the normal (px):")
    for k, a, b in zip(names, seg0, seg1):
        print(f"        {k:16s} config {'   -' if a is None else f'{a:+5.1f}'}   refit {'   -' if b is None else f'{b:+5.1f}'}")
    print(f"[stand] worst segment under the config {worst:.1f} px; key corners move {move.max():.1f} px max "
          f"({move.mean():.1f} mean) to the refit")
    print(f"[stand] refit = config + dx {c['dx_cm']:+.1f} cm (world x, + = robot's right), dy {c['dy_cm']:+.1f} cm "
          f"(+ = toward the far edge), yaw {c['dyaw_deg']:+.2f} deg (+ = turned to the robot's left)")
    move_real, move_table = real_moves(c)
    report.update(move_real_robot=move_real, move_table=move_table)
    print(f"[move] 把真机挪到仿真的站位: {move_real}")
    print(f"[move] 或者机器人不动、挪桌子: {move_table}")
    if passed:
        print(f"PASS: every segment within {args.threshold:g} px; the configured stand matches, no change needed.")
    else:
        why = "fewer than 6 segments visible" if len(vis0) < 6 else f"a segment is off by more than {args.threshold:g} px"
        print(f"RECALIBRATE ({why}).\n"
              f"  Evaluating a trained policy: move the real robot as [move] says, record a new frame, rerun until PASS.\n"
              f"  Making new sim data for this stand instead: in {args.robot_yml}\n"
              f"    default_root_pos: {new_root},\n    default_root_rot: {new_rot},")
        if args.apply and len(vis0) >= 6:
            text = args.robot_yml.read_text()
            text = re.sub(r"default_root_pos: \[[^\]]*\]", f"default_root_pos: {new_root}", text, count=1)
            text = re.sub(r"default_root_rot: \[[^\]]*\]", f"default_root_rot: {new_rot}", text, count=1)
            args.robot_yml.write_text(text)
            print(f"    written to {args.robot_yml}; rebuild the layouts (build_house4_layouts.py reads it), rerun this check")

    vis = img.copy()
    if mask is not None:
        vis[mask > 0] = (0.6 * vis[mask > 0]).astype(np.uint8)
    for xx, col in ((base, (0, 0, 255)), (x, (0, 255, 0))):
        for _, (a, b) in segs:
            uv = proj(cam_of(xx), a + np.linspace(0, 1, 40)[:, None] * (b - a))
            uv = uv[np.isfinite(uv).all(1)].round().astype(np.int32)
            if len(uv) > 1:
                cv2.polylines(vis, [uv], False, col, 1, cv2.LINE_AA)
    cv2.putText(vis, f"red: config  green: refit  worst {worst:.1f}px  {'PASS' if passed else 'RECALIBRATE'}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(args.out / "align_overlay.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (args.out / "align_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {args.out / 'align_overlay.jpg'} and {args.out / 'align_report.json'}")
    return 0 if passed else 2


def real_moves(c, min_cm=0.3, min_deg=0.3):
    """Operator instructions from the correction (real stand = config + c): move the real robot by -c, or the table
    by +c (translation only; the table cannot take the yaw). The robot faces +y, so world +x is its right."""
    def lin(v, pos, neg):
        return f"{pos if v > 0 else neg} {abs(v):.1f} cm" if abs(v) >= min_cm else None

    robot = [lin(-c["dx_cm"], "向机器人右侧", "向机器人左侧"), lin(-c["dy_cm"], "向前（靠近桌子）", "向后（远离桌子）"),
             f"{'向左' if -c['dyaw_deg'] > 0 else '向右'}转 {abs(c['dyaw_deg']):.1f} 度" if abs(c["dyaw_deg"]) >= min_deg else None]
    table = [lin(c["dx_cm"], "向机器人右侧", "向机器人左侧"), lin(c["dy_cm"], "远离机器人", "靠近机器人")]
    robot, table = [s for s in robot if s], [s for s in table if s]
    if abs(c["dyaw_deg"]) >= min_deg:
        table.append("（转角只能靠转机器人）")
    return "，".join(robot) or "不用挪", "，".join(table) or "不用挪"


if __name__ == "__main__":
    sys.exit(main())
