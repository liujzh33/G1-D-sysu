"""Rebuild the school lab green-screen table (real_scene4, v2) from tape measurements, the robot-height and stand
photos and the walk-around video (data/sim_scene/image/: 环境.mp4, 机器人高度/, 机器人距离桌面距离/; copies and
frames in reference/). Replaces the fitted geometry of real_scene4_school (colleague's MuJoCo twin) with tape values.

Scene frame: Z up, meters. Origin on the floor under the tabletop centre; +X along the long side (the robot's right);
+Y away from the robot, towards the green screen (same convention as real_scene1-5).

Tape (2026-11-19):
  * tabletop 110 x 60 cm, 75.8 cm above the floor;
  * robot G1-D (real2sim/robot/g1d_v4), lift raised from the lowest position (column reading 7 -> 14.5 cm): waist
    top 48.5 cm above the top of the black column base plate on the chassis;
  * far table edge (screen side) to the back of the waist 81 cm, robot front (waist front) to the far edge 66 cm.
The robot geometry the tape refers to comes from the g1d_v4 URDF chain (parsed here) and its meshes (LFS in
g1d_v4/meshes, the same files sit in g1d_aurora_lite_ws/meshes/g1d, sha256 equal); see robot_stand() and PLAN.md.

Usage (numpy, usd-core, trimesh, opencv-python-headless, shapely):
    python real2sim/scenes/real_scene4_school_v2/build_scene.py
"""

import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt
from shapely.geometry import box as sbox
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "usd"
TEX = OUT / "textures"
ROOM_NAME = "real_scene4_v2"
URDF = ROOT / "real2sim/robot/g1d_v4/g1_d_full_dex1_v4.urdf"

TAPE = dict(
    table_length=1.10,
    table_width=0.60,
    table_top_z=0.758,
    waist_top_above_plate=0.485,  # reference/height_485_waist_top.jpg, tape zero on the plate: height_tape_zero_at_plate
    lift_reading=(0.07, 0.145),  # waist top to the outer column top, lowest / now (height_lift_lowest, _raised)
    far_edge_to_waist_back=0.81,  # dist_far_edge_to_waist_back.jpg, dist_81_waist_back.jpg
    far_edge_to_waist_front=0.66,  # dist_waist_front.jpg, dist_66_far_edge.jpg, top_view_tape.jpg
    # robot's left table end to the waist centre line (stand_565_from_left*.jpg: the tape's hook on the end on the
    # robot's left, crossing the tape run from the waist to the far edge at 56.5)
    left_end_to_waist_centre=0.565,
    # cable grommet frame: 40.4-70.0 cm from the robot's right end (grommet_along_length.jpg, grommet_close.jpg: the
    # screen side is on the photo's left, so the hook end at the top is the robot's right), 43.0-54.8 cm from the near
    # edge (grommet_across_width.jpg)
    grommet_from_right_end=(0.404, 0.700),
    grommet_from_near_edge=(0.430, 0.548),
    # black tape rectangle (placement target) outer edges: 40.7-61.5 cm from the robot's right end
    # (target_box_along_length.jpg), 20.0-26.3 cm from the near edge (target_box_across_width.jpg, grommet_across_width)
    target_box_from_right_end=(0.407, 0.615),
    target_box_from_near_edge=(0.200, 0.263),
    date="2026-11-19",
)
# Robot surfaces the tape touched, in their link frames (g1d_v4 meshes, measured from the STL files):
#   black column base plate: LZ_ot_Link up-facing square face z = -0.438 (15.8 cm square, 4 mm above the AGV top);
#   waist box: Pitching_Link up-facing face z = +0.017, the box spans x -0.0555..+0.0555 about the column axis;
#   waist front: Yaw_Link (the waist pitch brackets) x -0.044..+0.044 about its joint.
MESH = dict(plate_top_in_ot=-0.438, waist_top_in_pitching=0.017, waist_back_in_pitching=-0.0555, yaw_front_in_yaw=0.044)
ROOT_Z_NOMINAL = 0.1115  # AGV_link over the floor with the wheels on it (env_cfg/robot/g1d*.yml)
# Waist centre (the column axis, over the root within 0.2 mm) 56.5 cm from the robot's left end: 1.5 cm right of the
# table centre. Chassis square to the table edge (not measured; the tapes run square to it in top_view_tape.jpg).
ROBOT_X = round(TAPE["left_end_to_waist_centre"] - TAPE["table_length"] / 2, 4)
ROBOT_YAW_DEG = 0.0

TABLE = dict(
    length=TAPE["table_length"],
    width=TAPE["table_width"],
    top_z=TAPE["table_top_z"],
    thick=0.025,  # by eye (video_t02s), common 25 mm board
    corner_r=0.006,
)
# Dark grey steel frame (video_leg_frame, video_t02s): one closed rectangle per end (front and back posts, floor foot
# bar, rail under the top) and a back beam along the table. Sizes by eye.
LEGS = dict(post=(0.05, 0.05), end_inset=0.06, edge_inset=0.04, foot_h=0.03, rail_h=0.05, beam=(0.03, 0.05))
# Cable grommet (hidden power strip) near the far edge, tape: aluminium frame 29.6 x 11.8 cm, black brush slot along
# the robot side.
_gx, _gy = TAPE["grommet_from_right_end"], TAPE["grommet_from_near_edge"]
GROMMET = dict(
    length=round(_gx[1] - _gx[0], 4),
    width=round(_gy[1] - _gy[0], 4),
    far_inset=round(TAPE["table_width"] - _gy[1], 4),
    x=round(TAPE["table_length"] / 2 - (_gx[0] + _gx[1]) / 2, 4),
    slot_w=0.012,
    rim=0.008,
)
_bx, _by = TAPE["target_box_from_right_end"], TAPE["target_box_from_near_edge"]
TARGET_BOX = dict(
    x=(round(TAPE["table_length"] / 2 - _bx[1], 4), round(TAPE["table_length"] / 2 - _bx[0], 4)),
    y=(round(_by[0] - TAPE["table_width"] / 2, 4), round(_by[1] - TAPE["table_width"] / 2, 4)),
    line_w=0.005,  # by eye
)
# Green screen on a black tube frame, U-shaped round the table (video_t01s/t02s: corner beyond the right end;
# video_t09s/t10s: left panel ends at a pole near the robot; video_t19s: blue curtain beyond it). Positions by eye:
# a person stands between the far edge and the screen (dist_66_far_edge, top_view_tape), back screen 2 m wide.
SCREEN = dict(back_gap=0.45, half_width=1.0, front_y=-0.45, height=2.3, sweep=0.12, bend_r=0.06, tube=0.025)
ROOM = dict(x0=-1.8, x1=2.4, y0=-2.6, y1=1.9, height=2.8)
CURTAIN = dict(x=ROOM["x0"] + 0.12, y0=-2.2, y1=1.6, z0=0.55, z1=2.7)  # blue window curtain on the left wall
FLOOR_TILE = 0.6  # light ceramic tiles (video_t13s)
LIGHTS = dict(
    panels=[(-0.9, -0.9), (0.9, -0.9), (-0.9, 0.7), (0.9, 0.7)],
    size=(1.2, 0.3),
    intensity=26000.0,
    color_temperature=6000.0,
    dome=150.0,
)


def lin(srgb):
    c = np.asarray(srgb, float) / 255
    return tuple(round(float(v), 4) for v in np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))


# name: (diffuse linear RGB, roughness, metallic, texture). sRGB sampled in the video frames (exposure varies: the
# green screen reads (27,120,68) in shade to (55,187,144) lit; the board (191,175,155)-(198,181,155)).
MATERIALS = {
    "board": ((1.0, 1.0, 1.0), 0.5, 0.0, "board.png"),
    "board_edge": (lin((200, 186, 164)), 0.5, 0.0, None),
    "frame_steel": (lin((62, 64, 66)), 0.45, 0.4, None),
    "foot_pad": (lin((20, 20, 20)), 0.8, 0.0, None),
    "alu": (lin((212, 214, 216)), 0.22, 0.9, None),  # silver-white brushed aluminium, visibly reflective
    "tape_black": (lin((18, 18, 18)), 0.6, 0.0, None),
    "slot": (lin((15, 15, 15)), 0.9, 0.0, None),
    "screen_green": (lin((40, 160, 100)), 0.95, 0.0, None),
    "tube_black": (lin((25, 25, 25)), 0.5, 0.3, None),
    "floor": ((1.0, 1.0, 1.0), 0.35, 0.0, "floor_tile.png"),
    "wall": (lin((225, 225, 220)), 0.9, 0.0, None),
    "ceiling": (lin((230, 230, 230)), 0.9, 0.0, None),
    "curtain": (lin((95, 135, 180)), 0.9, 0.0, None),
}
WRAP = {"board.png": "mirror", "floor_tile.png": "repeat"}
PHYS = {"default": (0.5, 0.45, 0.0), "board": (0.5, 0.4, 0.0)}


# ----------------------------------------------------------------------------- the robot stand from the tape
def urdf_joints():
    joints = {}
    for j in ET.parse(URDF).getroot().findall("joint"):
        o = j.find("origin")
        joints[j.get("name")] = np.array([float(v) for v in o.get("xyz").split()]) if o is not None else np.zeros(3)
    return joints


def robot_stand():
    """Root pose and lift that put the real waist where the tape says. In AGV_link coordinates (X forward, Z up)."""
    j = urdf_joints()
    ot = j["LZ_ot_Joint"]
    pitching = ot + j["LZ_mt_Joint"] + j["LZ_it_Joint"] + j["Pitching_Joint"]  # at zero lift
    yaw = pitching + j["Yaw_Joint"]
    plate_z = ot[2] + MESH["plate_top_in_ot"]
    waist_top_z0 = pitching[2] + MESH["waist_top_in_pitching"]
    waist_back_x = pitching[0] + MESH["waist_back_in_pitching"]
    waist_front_x = yaw[0] + MESH["yaw_front_in_yaw"]
    # The URDF column is taller than the real one: at zero lift its waist top already sits 0.491 m over the plate, the
    # real one 0.485 m with the lift raised 7.5 cm. Lift stays at its lowest (0) and the root drops by the rest, so the
    # waist, torso, arms and cameras are at the measured height over the plate; the wheels sink that much into the floor.
    lift_total = max(0.0, TAPE["waist_top_above_plate"] - (waist_top_z0 - plate_z))
    root_drop = (waist_top_z0 - plate_z) + lift_total - TAPE["waist_top_above_plate"]
    far_from_back = waist_back_x + TAPE["far_edge_to_waist_back"]
    far_from_front = waist_front_x + TAPE["far_edge_to_waist_front"]
    far_x = (far_from_back + far_from_front) / 2
    root_z = ROOT_Z_NOMINAL - root_drop
    return dict(
        far_edge_x=far_x,
        near_edge_x=far_x - TABLE["width"],
        far_edge_x_from_back=far_from_back,
        far_edge_x_from_front=far_from_front,
        root_z=root_z,
        root_drop=root_drop,
        lift_total=lift_total,
        plate_top_z=root_z + plate_z,
        waist_top_z=root_z + waist_top_z0 + lift_total,
        waist_back_x=waist_back_x,
        waist_front_x=waist_front_x,
        urdf_waist_over_plate_at_zero_lift=waist_top_z0 - plate_z,
        real_lift_from_reading=TAPE["lift_reading"][1] - TAPE["lift_reading"][0],
    )


# ----------------------------------------------------------------------------- geometry helpers
def box_xyz(x0, x1, y0, y1, z0, z1):
    m = np.eye(4)
    m[:3, 3] = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]
    return trimesh.creation.box(extents=[x1 - x0, y1 - y0, z1 - z0], transform=m)


def box_c(x, y, sx, sy, z0, z1):
    return box_xyz(x - sx / 2, x + sx / 2, y - sy / 2, y + sy / 2, z0, z1)


def rounded_rect(cx, cy, sx, sy, r):
    return sbox(cx - sx / 2 + r, cy - sy / 2 + r, cx + sx / 2 - r, cy + sy / 2 - r).buffer(r, quad_segs=4)


def grid_mesh(pts):
    """Triangulated sheet from an (n, m, 3) grid of points."""
    n, m = pts.shape[:2]
    idx = np.arange(n * m).reshape(n, m)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    faces = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    return trimesh.Trimesh(vertices=pts.reshape(-1, 3), faces=faces, process=False)


def fabric(p0, p1, inward, top, sweep, bend_r, seed, amp=0.012, n_per_m=60, nz=60):
    """Hanging cloth from p0 to p1 (floor points), clipped to a bar at `top`, falling into a sweep `sweep` m long on
    the floor towards `inward` (unit 2D). Vertical folds (random wavelengths 0.07-0.45 m) grow towards the bottom,
    plus a few shallow diagonal creases; they fade out over the bend so the sweep lies flat."""
    rng = np.random.default_rng(seed)
    p0, p1, inward = np.asarray(p0, float), np.asarray(p1, float), np.asarray(inward, float)
    length = float(np.linalg.norm(p1 - p0))
    along = (p1 - p0) / length
    s = np.linspace(0, length, max(8, int(length * n_per_m)))
    # path down the cloth: straight from top to bend_r, quarter circle, flat sweep
    arc = math.pi / 2 * bend_r
    total = (top - bend_r) + arc + (sweep - bend_r)
    u = np.concatenate([np.linspace(0, top - bend_r, nz), np.linspace(0, arc, 8)[1:] + top - bend_r,
                        np.linspace(0, sweep - bend_r, 4)[1:] + top - bend_r + arc])  # fmt: skip
    z, out, fade = np.empty_like(u), np.empty_like(u), np.empty_like(u)
    for i, ui in enumerate(u):
        if ui <= top - bend_r:
            z[i], out[i], fade[i] = top - ui, 0.0, 1.0
        elif ui <= top - bend_r + arc:
            a = (ui - (top - bend_r)) / bend_r
            z[i], out[i], fade[i] = bend_r * (1 - math.sin(a)), bend_r * (1 - math.cos(a)), math.cos(a)
        else:
            z[i], out[i], fade[i] = 0.002, bend_r + ui - (top - bend_r + arc), 0.0
    k = 9
    lam = rng.uniform(0.07, 0.45, k)
    ph = rng.uniform(0, 2 * math.pi, k)
    a_k = amp * rng.uniform(0.3, 1.0, k) * (lam / 0.45) ** 0.5
    S, Z = np.meshgrid(s, z, indexing="ij")
    hang = 0.35 + 0.65 * (1 - Z / top)  # clipped flat at the bar, loose at the bottom
    d = sum(a * np.sin(2 * math.pi * S / L + p) for a, L, p in zip(a_k, lam, ph)) * hang
    for _ in range(4):  # diagonal creases
        c0, c1, w = rng.uniform(0, length), rng.uniform(0.3, top - 0.2), rng.uniform(0.25, 0.6)
        ang = rng.uniform(-0.8, 0.8)
        r = (S - c0) * math.cos(ang) + (Z - c1) * math.sin(ang)
        q = -(S - c0) * math.sin(ang) + (Z - c1) * math.cos(ang)
        d += 0.004 * np.exp(-(q / 0.02) ** 2) * np.exp(-(r / w) ** 2)
    d *= fade[None, :]
    base = p0[None, None, :] + S[..., None] * along[None, None, :]
    xy = base + (out[None, :, None] - d[..., None]) * inward[None, None, :]
    return grid_mesh(np.concatenate([xy, Z[..., None]], -1))


def pleated_curtain(x, y0, y1, z0, z1, pleat=0.13, amp=0.035):
    """Curtain hanging in the plane x = const (pleats along Y), facing +X."""
    y = np.linspace(y0, y1, int((y1 - y0) / pleat * 8))
    z = np.linspace(z1, z0, 12)
    Y, Z = np.meshgrid(y, z, indexing="ij")
    X = x + amp * (0.5 + 0.5 * np.sin(2 * math.pi * Y / pleat))
    return grid_mesh(np.stack([X, Y, Z], -1))


# ----------------------------------------------------------------------------- USD helpers
class Layer:
    def __init__(self, path, root):
        self.stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(self.stage, 1.0)
        self.rootpath = f"/{root}"
        self.stage.SetDefaultPrim(UsdGeom.Xform.Define(self.stage, self.rootpath).GetPrim())
        self.mats, self.pmats = {}, {}

    def material(self, key):
        if key in self.mats:
            return self.mats[key]
        color, rough, metal, tex = MATERIALS[key]
        mat = UsdShade.Material.Define(self.stage, f"{self.rootpath}/Looks/{key}")
        sh = UsdShade.Shader.Define(self.stage, f"{self.rootpath}/Looks/{key}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
        if tex:
            st = UsdShade.Shader.Define(self.stage, f"{self.rootpath}/Looks/{key}/stReader")
            st.CreateIdAttr("UsdPrimvarReader_float2")
            st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            st.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            tx = UsdShade.Shader.Define(self.stage, f"{self.rootpath}/Looks/{key}/Texture")
            tx.CreateIdAttr("UsdUVTexture")
            tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./textures/{tex}")
            tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            for w in ("wrapS", "wrapT"):
                tx.CreateInput(w, Sdf.ValueTypeNames.Token).Set(WRAP.get(tex, "clamp"))
            tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
            tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
        else:
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        self.mats[key] = mat
        return mat

    def phys_material(self, key):
        if key in self.pmats:
            return self.pmats[key]
        sf, df, rs = PHYS.get(key, PHYS["default"])
        pm = UsdShade.Material.Define(self.stage, f"{self.rootpath}/PhysicsMaterials/{key}")
        api = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
        api.CreateStaticFrictionAttr(sf)
        api.CreateDynamicFrictionAttr(df)
        api.CreateRestitutionAttr(rs)
        self.pmats[key] = pm
        return pm

    def mesh(self, path, m, mat=None, uv=None, guide=False, smooth=False, double=False):
        um = UsdGeom.Mesh.Define(self.stage, f"{self.rootpath}/{path}")
        pts = m.vertices.astype(np.float32)
        um.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
        um.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(m.faces)))
        um.CreateFaceVertexIndicesAttr(Vt.IntArray(m.faces.reshape(-1).tolist()))
        um.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        um.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, pts.min(0))), Gf.Vec3f(*map(float, pts.max(0)))]))
        if smooth:
            um.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(m.vertex_normals.astype(np.float32)))
            um.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        else:
            um.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.repeat(m.face_normals, 3, axis=0).astype(np.float32)))
            um.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        if double:
            um.CreateDoubleSidedAttr(True)
        if uv is not None:
            fv = m.vertices[m.faces.reshape(-1)]
            pv = UsdGeom.PrimvarsAPI(um).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
            )
            pv.Set(Vt.Vec2fArray.FromNumpy(uv(fv).astype(np.float32)))
        if guide:
            um.CreatePurposeAttr(UsdGeom.Tokens.guide)
        elif mat:
            UsdShade.MaterialBindingAPI.Apply(um.GetPrim()).Bind(self.material(mat))
        return um

    def collider(self, path, m, pkey="default"):
        """Static convex-hull collider (guide purpose); all boxes (8 vertices)."""
        prim = self.mesh(path, m, guide=True).GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            self.phys_material(pkey), UsdShade.Tokens.weakerThanDescendants, "physics"
        )

    def part(self, path, m, mat, pkey="default"):
        self.mesh(path, m, mat)
        self.collider(f"Colliders/{path.replace('/', '_')}", m, pkey)

    def save(self):
        self.stage.GetRootLayer().Save()


# ----------------------------------------------------------------------------- textures
BOARD_TEX_M = (1.10, 0.60)  # one texture over the whole board: no visible repeat


def board_texture(px_per_m=1400, seed=4):
    """Light oak laminate: long grain streaks along the board (X), a few darker cathedral bands, sRGB mean
    (195, 178, 155) as in the video frames."""
    rng = np.random.default_rng(seed)
    w, h = int(BOARD_TEX_M[0] * px_per_m), int(BOARD_TEX_M[1] * px_per_m)
    fine = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=60, sigmaY=1.2)
    mid = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=180, sigmaY=6)
    fine /= fine.std()
    mid /= mid.std()
    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]
    bands = np.zeros((h, w), np.float32)
    for _ in range(5):  # stretched arches of the flat-sawn veneer
        cy, cx, sx = rng.uniform(0, h), rng.uniform(0, w), rng.uniform(0.25, 0.6) * w
        ring = (y - cy) + ((x - cx) / sx) ** 2 * h * 0.25 + 4 * mid
        bands += 0.5 * (np.sin(ring / rng.uniform(9, 16)) > 0.85) * np.exp(-(((x - cx) / sx) ** 2))
    lum = 1.0 + 0.035 * fine + 0.03 * mid - 0.05 * cv2.GaussianBlur(bands, (0, 0), 1.5)
    base = np.array([155, 178, 195], np.float32)  # BGR
    img = np.clip(base[None, None, :] * lum[..., None], 0, 255).astype(np.uint8)
    return img


def floor_texture(n=512, grout=3):
    """One 0.6 m light ceramic tile with a grey grout line on two edges and a faint speckle."""
    rng = np.random.default_rng(7)
    img = np.full((n, n, 3), (200, 203, 204), np.float32)
    img += cv2.GaussianBlur(rng.normal(0, 6, (n, n)).astype(np.float32), (0, 0), 2)[..., None]
    img[:grout], img[:, :grout] = (150, 150, 148), (150, 150, 148)
    return np.clip(img, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- the table
def add_table(L):
    t, g, gr = TABLE, LEGS, GROMMET
    hl, hw, top = t["length"] / 2, t["width"] / 2, t["top_z"]
    z_under = top - t["thick"]
    slab = trimesh.creation.extrude_polygon(rounded_rect(0, 0, t["length"], t["width"], t["corner_r"]), t["thick"])
    slab.apply_translation([0, 0, z_under])
    up = slab.face_normals[:, 2] > 0.99
    top_face = trimesh.Trimesh(slab.vertices, slab.faces[up], process=False)
    sides = trimesh.Trimesh(slab.vertices, slab.faces[~up], process=False)
    L.mesh("Top/board", top_face, "board", uv=lambda fv: (fv[:, :2] + [hl, hw]) / BOARD_TEX_M)
    L.mesh("Top/edges", sides, "board_edge")
    L.collider("Colliders/top", box_xyz(-hl, hl, -hw, hw, z_under, top), pkey="board")

    # grommet: frame 2 mm proud of the board, flush lid, black brush slot on the robot side
    y1 = hw - gr["far_inset"]
    y0, x0, x1 = y1 - gr["width"], gr["x"] - gr["length"] / 2, gr["x"] + gr["length"] / 2
    ring = sbox(x0, y0, x1, y1).difference(sbox(x0 + gr["rim"], y0 + gr["rim"], x1 - gr["rim"], y1 - gr["rim"]))
    L.mesh("Grommet/frame", trimesh.creation.extrude_polygon(ring, 0.002).apply_translation([0, 0, top]), "alu")
    r = gr["rim"]
    L.mesh("Grommet/slot", box_xyz(x0 + r, x1 - r, y0 + r, y0 + r + gr["slot_w"], top - 0.0005, top + 0.0003), "slot")
    L.mesh("Grommet/lid", box_xyz(x0 + r, x1 - r, y0 + r + gr["slot_w"], y1 - r, top - 0.0005, top + 0.0008), "alu")

    bx0, bx1 = TARGET_BOX["x"]
    by0, by1 = TARGET_BOX["y"]
    w = TARGET_BOX["line_w"]
    outline = sbox(bx0, by0, bx1, by1).difference(sbox(bx0 + w, by0 + w, bx1 - w, by1 - w))
    L.mesh("TargetBox/tape", trimesh.creation.extrude_polygon(outline, 0.0003).apply_translation([0, 0, top]), "tape_black")

    pw, pd = g["post"]
    rail_z0 = z_under - g["rail_h"]
    for sx, side in ((1, "pos"), (-1, "neg")):
        x = sx * (hl - g["end_inset"])
        yf, yb = -hw + g["edge_inset"] + pd / 2, hw - g["edge_inset"] - pd / 2
        for end, y in (("front", yf), ("back", yb)):
            L.part(f"Legs/{side}/post_{end}", box_c(x, y, pw, pd, g["foot_h"], rail_z0), "frame_steel")
            L.mesh(f"Legs/{side}/pad_{end}", box_c(x, y, pw - 0.006, pd - 0.006, 0.0, 0.008), "foot_pad")
        L.part(f"Legs/{side}/foot", box_xyz(x - pw / 2, x + pw / 2, yf - pd / 2, yb + pd / 2, 0.008, g["foot_h"]),
               "frame_steel")  # fmt: skip
        L.part(f"Legs/{side}/rail", box_xyz(x - pw / 2, x + pw / 2, yf - pd / 2, yb + pd / 2, rail_z0, z_under),
               "frame_steel")  # fmt: skip
    bw, bh = g["beam"]
    xb = hl - g["end_inset"] - pw / 2
    yb = hw - g["edge_inset"] - bw / 2
    L.part("Frame/beam", box_xyz(-xb, xb, yb - bw / 2, yb + bw / 2, z_under - bh, z_under), "frame_steel")


# ----------------------------------------------------------------------------- green screen, room
def screen_panels():
    """(name, p0, p1, inward) of the three cloth panels, floor points at the cloth plane."""
    s, hw = SCREEN, TABLE["width"] / 2
    yb, xw = hw + s["back_gap"], s["half_width"]
    return [
        ("left", (-xw, s["front_y"]), (-xw, yb), (1.0, 0.0)),
        ("back", (-xw, yb), (xw, yb), (0.0, -1.0)),
        ("right", (xw, yb), (xw, s["front_y"]), (-1.0, 0.0)),
    ]


def add_screen(L):
    s = SCREEN
    for i, (name, p0, p1, inward) in enumerate(screen_panels()):
        cloth = fabric(p0, p1, inward, s["height"] - 0.03, s["sweep"], s["bend_r"], seed=11 + i)
        L.mesh(f"Furniture/GreenScreen/{name}", cloth, "screen_green", smooth=True, double=True)
        # thin box behind the cloth so objects do not pass through it
        a, b, n = np.array(p0), np.array(p1), np.array(inward)
        q0, q1 = a - 0.03 * n, b - 0.02 * n
        xs, ys = sorted([q0[0], q1[0]]), sorted([q0[1], q1[1]])
        L.collider(f"Colliders/screen_{name}", box_xyz(xs[0], xs[1] + 1e-3, ys[0], ys[1] + 1e-3, 0.0, s["height"]))
    # frame: posts at the two back corners and the front ends of the side panels, top bars over all three panels
    t, h = s["tube"], s["height"]
    behind = 0.04  # the tubes stand just behind the cloth
    yb = TABLE["width"] / 2 + s["back_gap"] + behind
    xw = s["half_width"] + behind
    corners = [(-xw, s["front_y"]), (-xw, yb), (xw, yb), (xw, s["front_y"])]
    for i, (x, y) in enumerate(corners):
        L.mesh(f"Furniture/GreenScreen/frame/post_{i}", box_c(x, y, t, t, 0.0, h), "tube_black")
        L.mesh(f"Furniture/GreenScreen/frame/foot_{i}", box_c(x, y, 0.08, 0.08, 0.0, 0.01), "tube_black")
    for i, (a, b) in enumerate(zip(corners[:-1], corners[1:])):
        lo, hi = np.minimum(a, b) - t / 2, np.maximum(a, b) + t / 2
        L.mesh(f"Furniture/GreenScreen/frame/top_{i}", box_xyz(lo[0], hi[0], lo[1], hi[1], h - t, h), "tube_black")
    # the left panel's front pole carries clips (reference/video_t10s.jpg)
    x, y = corners[0]
    for k, z in enumerate(np.arange(0.35, h - 0.1, 0.35)):
        L.mesh(f"Furniture/GreenScreen/frame/clip_{k}", box_c(x, y + 0.02, 0.03, 0.02, z, z + 0.03), "tube_black")


def add_background(L):
    r = ROOM
    L.mesh(
        "Floor",
        trimesh.Trimesh(
            [[r["x0"], r["y0"], 0], [r["x1"], r["y0"], 0], [r["x1"], r["y1"], 0], [r["x0"], r["y1"], 0]],
            [[0, 1, 2], [0, 2, 3]],
            process=False,
        ),
        "floor",
        uv=lambda fv: fv[:, :2] / FLOOR_TILE,
    )
    L.collider("Colliders/floor", box_xyz(r["x0"] - 1, r["x1"] + 1, r["y0"] - 1, r["y1"] + 1, -0.1, 0.0))
    walls = {
        "left": [(r["x0"], r["y1"]), (r["x0"], r["y0"])],
        "front": [(r["x0"], r["y0"]), (r["x1"], r["y0"])],
        "right": [(r["x1"], r["y0"]), (r["x1"], r["y1"])],
        "back": [(r["x1"], r["y1"]), (r["x0"], r["y1"])],
    }
    for name, (a, b) in walls.items():  # single quads facing into the room
        v = [[*a, 0], [*b, 0], [*b, r["height"]], [*a, r["height"]]]
        L.mesh(f"Walls/{name}", trimesh.Trimesh(v, [[0, 1, 2], [0, 2, 3]], process=False), "wall",
               uv=lambda fv, a=a: np.c_[np.linalg.norm(fv[:, :2] - a, axis=1), fv[:, 2]])  # fmt: skip
    ceil = [[r["x0"], r["y0"], r["height"]], [r["x0"], r["y1"], r["height"]], [r["x1"], r["y1"], r["height"]],
            [r["x1"], r["y0"], r["height"]]]  # fmt: skip
    L.mesh("Ceiling", trimesh.Trimesh(ceil, [[0, 1, 2], [0, 2, 3]], process=False), "ceiling")
    c = CURTAIN
    L.mesh("Furniture/Curtain/cloth", pleated_curtain(c["x"], c["y0"], c["y1"], c["z0"], c["z1"]), "curtain",
           smooth=True, double=True)  # fmt: skip
    L.mesh("Furniture/Curtain/rail", box_xyz(c["x"] - 0.02, c["x"] + 0.06, c["y0"], c["y1"], c["z1"], c["z1"] + 0.03),
           "wall")  # fmt: skip
    add_screen(L)


def add_lights(L):
    lt = LIGHTS
    for i, (x, y) in enumerate(lt["panels"]):
        rl = UsdLux.RectLight.Define(L.stage, f"{L.rootpath}/CeilingPanel_{i}")
        rl.CreateWidthAttr(lt["size"][0])
        rl.CreateHeightAttr(lt["size"][1])
        rl.CreateIntensityAttr(lt["intensity"])
        rl.CreateEnableColorTemperatureAttr(True)
        rl.CreateColorTemperatureAttr(lt["color_temperature"])
        UsdGeom.Xformable(rl).AddTranslateOp().Set(Gf.Vec3d(x, y, ROOM["height"] - 0.02))
    dome = UsdLux.DomeLight.Define(L.stage, f"{L.rootpath}/AmbientDome")
    dome.CreateIntensityAttr(lt["dome"])
    dome.CreateColorAttr(Gf.Vec3f(0.95, 0.95, 1.0))


def r4(v):
    return round(float(v), 4)


def main():
    stand = robot_stand()
    # scene frame: the far edge is at y = +width/2, the robot root `far_edge_x` behind it
    root_y = TABLE["width"] / 2 - stand["far_edge_x"]
    if OUT.exists():
        shutil.rmtree(OUT)
    TEX.mkdir(parents=True)
    cv2.imwrite(str(TEX / "board.png"), board_texture())
    cv2.imwrite(str(TEX / "floor_tile.png"), floor_texture())
    layers = [("background", add_background), ("table", add_table), ("lights", add_lights)]
    for name, fn in layers:
        layer = Layer(OUT / f"{name}.usda", name.capitalize())
        fn(layer)
        layer.save()
    names = [n.capitalize() for n, _ in layers]

    st = Usd.Stage.CreateNew(str(OUT / "scene.usda"))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(st, 1.0)
    st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/World").GetPrim())
    UsdPhysics.Scene.Define(st, "/World/PhysicsScene").CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    for nm in names:
        st.DefinePrim(f"/World/{nm}").GetReferences().AddReference(f"./{nm.lower()}.usda")
    st.GetRootLayer().Save()
    # RoboDojo Room entry (layout_manager.select_room loads the only .usd in Assets/Room/<name>/): no PhysicsScene
    room = Usd.Stage.CreateNew(str(HERE / f"{ROOM_NAME}.usd"))
    UsdGeom.SetStageUpAxis(room, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(room, 1.0)
    room.SetDefaultPrim(UsdGeom.Xform.Define(room, "/Scene").GetPrim())
    for nm in names:
        room.DefinePrim(f"/Scene/{nm}").GetReferences().AddReference(f"./usd/{nm.lower()}.usda")
    room.GetRootLayer().Save()

    t = TABLE
    yaw = math.radians(90 + ROBOT_YAW_DEG)
    params = dict(
        frame="Z-up, meters; origin on the floor under the tabletop centre; +X along the table; +Y away from the robot",
        tape=TAPE,
        table=dict(top_z=t["top_z"], length=t["length"], width=t["width"], front_y=-t["width"] / 2, thick=t["thick"]),
        robot=dict(
            root_xy=[ROBOT_X, r4(root_y)],
            root_z=r4(stand["root_z"]),
            heading_deg=90 + ROBOT_YAW_DEG,
            quat_wxyz=[r4(math.cos(yaw / 2)), 0.0, 0.0, r4(math.sin(yaw / 2))],
            lift_total=r4(stand["lift_total"]),
            init_joint_pos=dict(LZ_mt_Joint=r4(stand["lift_total"] / 2), LZ_it_Joint=r4(stand["lift_total"] / 2)),
            table_edge_from_root=r4(stand["near_edge_x"]),
            table_centre_x_from_root=r4(-ROBOT_X),
            checks=dict(
                far_edge_from_root_via_waist_back=r4(stand["far_edge_x_from_back"]),
                far_edge_from_root_via_waist_front=r4(stand["far_edge_x_from_front"]),
                waist_back_x=r4(stand["waist_back_x"]),
                waist_front_x=r4(stand["waist_front_x"]),
                plate_top_z=r4(stand["plate_top_z"]),
                waist_top_z=r4(stand["waist_top_z"]),
                waist_top_below_table_top=r4(t["top_z"] - stand["waist_top_z"]),
                urdf_waist_over_plate_at_zero_lift=r4(stand["urdf_waist_over_plate_at_zero_lift"]),
                root_drop_below_nominal=r4(stand["root_drop"]),
                real_lift_from_reading=r4(stand["real_lift_from_reading"]),
            ),
        ),
        legs=LEGS,
        grommet=GROMMET,
        target_box=TARGET_BOX,
        screen=dict(SCREEN, panels=[[n, list(a), list(b)] for n, a, b, _ in screen_panels()]),
        room=ROOM,
        curtain=CURTAIN,
        lights=LIGHTS,
        layers=dict(scene="scene.usda", **{n: f"{n}.usda" for n, _ in layers}),
    )
    (OUT / "build_params.json").write_text(json.dumps(params, indent=1), encoding="utf-8")
    for f in OUT.glob("*.usda"):  # USD ends text layers with a blank line, which `git diff --check` rejects
        f.write_text(f.read_text().rstrip("\n") + "\n")
    print(json.dumps(params["table"]), json.dumps(params["robot"], indent=1))


if __name__ == "__main__":
    main()
