"""CPU preview of real_scene4_school_v2 with the G1-D at its stand (no Isaac, no GPU): point-splat renders of
usd/scene.usda plus the g1d_v4 meshes posed by forward kinematics at env_cfg/robot/g1d_school_v2.yml's joints, from
the side (a section through the robot, with the tape checks drawn on), from above (stand), from behind the robot
(like reference/video_behind_robot.jpg) and an overview. Writes preview/*.jpg.

    python real2sim/scenes/real_scene4_school_v2/preview.py
"""

import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from pxr import Usd, UsdGeom, UsdShade

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
# The robot the sim loads (robot g1d): g1d_v4 with the Dex1-1 grippers, their wrist camera modules and the head D435
# frame, written by scripts/g1d/build_asset.py; same joint chain as g1d_v4. Falls back to the bare g1d_v4 URDF.
URDF = ROOT / "Assets/Robots/g1d/g1d_dex1.urdf"
if not URDF.exists():
    URDF = ROOT / "real2sim/robot/g1d_v4/g1_d_full_dex1_v4.urdf"
# g1d_v4/meshes are LFS pointers in some checkouts; the same files (sha256 equal) are real in these
MESHES = [ROOT / "Assets/Robots/g1d/meshes", ROOT / "real2sim/robot/g1d_v4/meshes",
          ROOT / "real2sim/robot/g1d_aurora_lite_ws/meshes/g1d"]  # fmt: skip
P = json.loads((HERE / "usd/build_params.json").read_text())
ARMS = dict(  # env_cfg/robot/g1d_school_v2.yml (real teleop first frame)
    left_shoulder_pitch_joint=0.194, left_shoulder_roll_joint=-0.046, left_shoulder_yaw_joint=-0.214,
    left_elbow_joint=-0.315, left_wrist_roll_joint=-0.029, left_wrist_pitch_joint=-0.249, left_wrist_yaw_joint=0.119,
    right_shoulder_pitch_joint=0.201, right_shoulder_roll_joint=0.026, right_shoulder_yaw_joint=0.158,
    right_elbow_joint=-0.308, right_wrist_roll_joint=0.026, right_wrist_pitch_joint=-0.280, right_wrist_yaw_joint=0.055,
)  # fmt: skip
ROBOT_RGB = (175, 178, 180)


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])  # fmt: skip


def axis_angle(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k


def load_stl(name):
    for d in MESHES:
        p = d / name
        if p.exists() and p.stat().st_size > 1000:
            b = p.read_bytes()
            n = struct.unpack("<I", b[80:84])[0]
            a = np.frombuffer(b[84 : 84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
            return a["v"].reshape(-1, 3, 3).astype(np.float64)
    return None


def robot_triangles():
    r = P["robot"]
    q = dict(ARMS, **r["init_joint_pos"])
    root = ET.parse(URDF).getroot()
    yaw = math.radians(r["heading_deg"])
    world = {"AGV_link": (axis_angle([0, 0, 1], yaw), np.array([*r["root_xy"], r["root_z"]]))}
    joints = root.findall("joint")
    while len(world) < len(root.findall("link")):
        grew = False
        for j in joints:
            par, ch = j.find("parent").get("link"), j.find("child").get("link")
            if par not in world or ch in world:
                continue
            o = j.find("origin")
            xyz = np.array([float(v) for v in o.get("xyz").split()])
            rot = rpy_matrix(*[float(v) for v in o.get("rpy").split()])
            ax = j.find("axis")
            v = q.get(j.get("name"), 0.0)
            if j.get("type") in ("revolute", "continuous"):
                rot = rot @ axis_angle([float(t) for t in ax.get("xyz").split()], v)
            elif j.get("type") == "prismatic":
                xyz = xyz + rot @ (np.array([float(t) for t in ax.get("xyz").split()]) * v)
            R, t = world[par]
            world[ch] = (R @ rot, t + R @ xyz)
            grew = True
        if not grew:
            break
    tris = []
    for link in root.findall("link"):
        for vis in link.findall("visual"):
            m = vis.find("geometry/mesh")
            if m is None or link.get("name") not in world:
                continue
            v = load_stl(m.get("filename").replace("meshes/", ""))
            if v is None:
                continue
            o = vis.find("origin")
            R0 = rpy_matrix(*[float(x) for x in o.get("rpy").split()]) if o is not None else np.eye(3)
            t0 = np.array([float(x) for x in o.get("xyz").split()]) if o is not None else np.zeros(3)
            R, t = world[link.get("name")]
            tris.append((link.get("name"), (v @ R0.T + t0) @ R.T + t))
    return tris, world


BLACK_LINKS = ("head_link", "Pitching_Link", "Yaw_Link", "Left_Wheel_Link", "RIght_Wheel_Link", "dex1")  # as in sim


def write_robot_layer(parts):
    """usd/robot_preview.usdc (the posed robot as static meshes, display only) and usd/preview.usda = scene + robot,
    which scripts/scenes/export_glb.py prefers over scene.usda for preview/scene.glb."""
    from pxr import Gf, Sdf, UsdPhysics, Vt

    st = Usd.Stage.CreateNew(str(HERE / "usd/robot_preview.usdc"))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/Robot").GetPrim())
    mats = {}
    for key, rgb, rough, metal in (("body", (0.62, 0.64, 0.66), 0.4, 0.6), ("black", (0.03, 0.03, 0.03), 0.5, 0.2)):
        mat = UsdShade.Material.Define(st, f"/Robot/Looks/{key}")
        sh = UsdShade.Shader.Define(st, f"/Robot/Looks/{key}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        mats[key] = mat
    for i, (link, tris) in enumerate(parts):
        m = UsdGeom.Mesh.Define(st, f"/Robot/{link}_{i}")  # a link can carry several visuals
        pts = tris.reshape(-1, 3).astype(np.float32)
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
        m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray(list(range(len(pts)))))
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mats["black" if any(b in link for b in BLACK_LINKS) else "body"])
    st.GetRootLayer().Save()
    pv = Usd.Stage.CreateNew(str(HERE / "usd/preview.usda"))
    UsdGeom.SetStageUpAxis(pv, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(pv, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(pv, 1.0)
    pv.SetDefaultPrim(UsdGeom.Xform.Define(pv, "/World").GetPrim())
    for nm in ("Background", "Table", "Lights"):
        pv.DefinePrim(f"/World/{nm}").GetReferences().AddReference(f"./{nm.lower()}.usda")
    pv.DefinePrim("/World/Robot").GetReferences().AddReference("./robot_preview.usdc")
    pv.GetRootLayer().Save()


def scene_triangles():
    stage = Usd.Stage.Open(str(HERE / "usd/scene.usda"))
    cache = UsdGeom.XformCache()
    tris, cols = [], []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        if mesh.GetPurposeAttr().Get() == UsdGeom.Tokens.guide:
            continue
        pts = np.array(mesh.GetPointsAttr().Get(), np.float64)
        idx = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        m = np.array(cache.GetLocalToWorldTransform(prim)).T
        pts = pts @ m[:3, :3].T + m[:3, 3]
        mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        rgb, tex = (0.5, 0.5, 0.5), None
        if mat:
            sh = UsdShade.Shader(stage.GetPrimAtPath(mat.GetPath().AppendChild("Shader")))
            src = sh.GetInput("diffuseColor")
            if src.HasConnectedSource():
                tex = mat.GetPath().name
            else:
                rgb = tuple(src.Get())
        c = np.array([v ** (1 / 2.2) * 255 for v in rgb])
        if tex == "board":
            c = np.array([195, 178, 155.0])
        elif tex == "floor":
            c = np.array([200, 202, 202.0])
        tris.append(pts[idx])
        cols.append(np.repeat(c[None], len(idx), 0))
    return np.concatenate(tris), np.concatenate(cols)


def sample(tris, cols, density):
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    cr = np.cross(e1, e2)
    area = np.linalg.norm(cr, axis=1) / 2
    nrm = cr / np.maximum(2 * area[:, None], 1e-12)
    n = np.random.default_rng(0).poisson(area * density) + 1
    ti = np.repeat(np.arange(len(tris)), n)
    u, v = np.random.default_rng(1).random((2, len(ti)))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    p = tris[ti, 0] + u[:, None] * e1[ti] + v[:, None] * e2[ti]
    return p, nrm[ti], cols[ti]


def render(p, n, c, eye, target, f_px, size=(960, 720), ortho=None, keep=None):
    if keep is not None:
        m = keep(p)
        p, n, c = p[m], n[m], c[m]
    w, h = size
    fwd = np.asarray(target, float) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1])
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    d = p - eye
    x, y, z = d @ right, d @ up, d @ fwd
    if ortho:
        u, v, ok = w / 2 + x * ortho, h / 2 - y * ortho, z > 0
    else:
        ok = z > 0.05
        u, v = w / 2 + f_px * x / np.maximum(z, 1e-6), h / 2 - f_px * y / np.maximum(z, 1e-6)
    ui, vi = u.astype(int), v.astype(int)
    ok &= (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    light = np.array([0.3, -0.5, 0.8])
    light /= np.linalg.norm(light)
    view = -(d / np.linalg.norm(d, axis=1, keepdims=True))
    nn = np.where((np.sum(n * view, 1) < 0)[:, None], -n, n)
    shade = 0.55 + 0.45 * np.abs(nn @ light)
    col = np.clip(c * shade[:, None], 0, 255)
    order = np.argsort(-z[ok])  # far first, near overwrite
    img = np.full((h, w, 3), 30, np.uint8)
    pix = vi[ok][order] * w + ui[ok][order]
    flat = img.reshape(-1, 3)
    flat[pix] = col[ok][order][:, ::-1].astype(np.uint8)
    return img


def main():
    out = HERE / "preview"
    out.mkdir(exist_ok=True)
    st, sc = scene_triangles()
    parts, world = robot_triangles()
    write_robot_layer(parts)
    if "d435_link" in world:  # head camera (G1 URDF D435 frame, X along the optical axis), for the web viewer
        R, t = world["d435_link"]
        cam = dict(pos=np.round(t, 4).tolist(), forward=np.round(R[:, 0], 4).tolist(), up=np.round(R[:, 2], 4).tolist())
        (out / "head_camera.json").write_text(json.dumps(cam))
        print("head camera", cam)
    rt = np.concatenate([t for _, t in parts])
    rc = np.repeat(np.array([ROBOT_RGB], float), len(rt), 0)
    p, n, c = sample(np.concatenate([st, rt]), np.concatenate([sc, rc]), 1.5e5)
    r, t = P["robot"], P["table"]
    rx, ry = r["root_xy"]
    views = {
        # section through the robot's centre plane, seen from its right: the table end and screen in front are cut
        "side": dict(eye=np.array([3.0, -0.2, 0.75]), target=np.array([0.0, -0.2, 0.75]), ortho=420,
                     keep=lambda q: (q[:, 0] < rx + 0.3) & (q[:, 0] > rx - 0.6)),  # fmt: skip
        "top": dict(eye=np.array([0.0, -0.3, 6.0]), target=np.array([0.0, -0.2999, 0.0]), ortho=330),
        "behind": dict(eye=np.array([rx - 0.05, ry - 1.4, 1.75]), target=np.array([rx, 0.2, 0.7]), f_px=700),
        "overview": dict(eye=np.array([1.6, -1.9, 1.9]), target=np.array([0.0, 0.0, 0.7]), f_px=700),
    }
    for name, v in views.items():
        img = render(p, n, c, v["eye"], v["target"], v.get("f_px", 600), ortho=v.get("ortho"), keep=v.get("keep"))
        cv2.imwrite(str(out / f"{name}.jpg"), img)
    # side view annotations: tabletop and waist-top heights
    img = cv2.imread(str(out / "side.jpg"))
    for z, label in ((t["top_z"], f"table top {t['top_z']:.3f}"), (r["checks"]["waist_top_z"], "waist top "
                     f"{r['checks']['waist_top_z']:.3f}"), (r["checks"]["plate_top_z"], f"plate top {r['checks']['plate_top_z']:.3f}")):  # fmt: skip
        yy = int(360 - (z - 0.75) * 420)
        cv2.line(img, (0, yy), (960, yy), (0, 0, 255), 1)
        cv2.putText(img, label, (10, yy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    ck = r["checks"]
    for y, label in ((ry + ck["waist_back_x"], "waist back"), (ry + ck["waist_front_x"], "waist front"),
                     (t["front_y"], "near edge"), (-t["front_y"], "far edge")):  # fmt: skip
        xx = int(480 + (y + 0.2) * 420)
        cv2.line(img, (xx, 120), (xx, 700), (255, 120, 0), 1)
        cv2.putText(img, label, (xx + 3, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 1)
    yb, yf = int(480 + (ry + ck["waist_back_x"] + 0.2) * 420), int(480 + (-t["front_y"] + 0.2) * 420)
    cv2.putText(img, f"{-t['front_y'] - ry - ck['waist_back_x']:.3f} (tape 0.81)", (yb, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 1)  # fmt: skip
    cv2.putText(img, f"{-t['front_y'] - ry - ck['waist_front_x']:.3f} (tape 0.66)", (yb + 60, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 1)  # fmt: skip
    cv2.imwrite(str(out / "side.jpg"), img)
    print("wrote", out)


if __name__ == "__main__":
    main()
