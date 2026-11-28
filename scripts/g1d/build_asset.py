"""Build the Unitree G1-D + Dex1-1 robot asset for RoboDojo.

Takes the team's assembled G1-D + Dex1-1 (camera version) URDF, v4: official g1_d.urdf with the standalone
dex1_1.urdf mounted on each wrist (real2sim/robot/g1d_v4/: g1_d_full_dex1_v4.urdf, meshes/, physics_contract.json),
rebuilds the Dex1-1 wrist camera module at its real mount (scripts/g1d/dex1_camera), adds the G1 head D435 frame from
unitree_ros g1_29dof_mode_15_with_dex1_1.urdf and locks the wheels.
Writes:
    Assets/Robots/g1d/                     g1d_dex1.urdf (tabletop, robot g1d), g1d_mobile_dex1.urdf (mobile chassis,
                                           robot g1d_mobile), meshes/, physics_contract.json (self-collision filters),
                                           {left,right}_robot_config.yml
    Assets/Room/real_scene{1,...,6}/      the scenes in SCENES (office, counter, lab, school, ShenZhen lab, HIMA store):
                                           <room>.usd + usd/
    Assets/Eval_Layout/RoboDojo/<config>/  per scene config in SCENES, layouts for the G1-D tasks in G1D_TASKS (bottle
                                           swapped for our pitcher of red beans or two cups, set in the real scene);
                                           g1d/ also gets the arx_x5 layouts minus the camera stand (it would sit
                                           inside the robot)
    Assets/Material/Fluid/RedBean.mdl

Usage:
    python scripts/g1d/build_asset.py              # fetch the unitree_ros G1 URDF from GitHub into --src
    python scripts/g1d/build_asset.py --src DIR    # DIR mirrors unitree_ros (robots/...); reused if present
"""

import argparse
import colorsys
import copy
import json
import math
from pathlib import Path
import random
import shutil
import sys
import urllib.request
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from env.global_configs import object_dir  # noqa: E402
from scripts.objects import block_atomic, pick_anything  # noqa: E402  (stdlib only at import)

UNITREE_RAW = "https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master/"
V4_URDF = "g1_d_full_dex1_v4.urdf"
G1_DEX1_URDF = "robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf"
# Real scenes reconstructed from phone video / photos (real2sim/scenes/<dir>/: usd/ and the Room entry <room>.usd), one
# env config each, replacing the stock room, table cube, ground cube and HDR dome of the G1-D task layouts. Each room
# sits with its robot-side table edge on the stock one (y=-0.60); the tabletop keeps its real height
# (usd/build_params.json) and the objects move onto it. A taller table raises the robot's lift column by as much
# (env_cfg/robot/<config>.yml) and the objects move with the robot by `shift`, so its reach is the same everywhere.
SCENES_DIR = ROOT / "real2sim" / "scenes"
SCENES = {  # env config: (scene dir, Room category, Room position, object (x, y) shift)
    "g1d": ("real_scene1_office", "real_scene1", [0.0, -0.305, 0.0], (0.0, 0.0)),
    "g1d_aurora_lite_ws": ("real_scene1_office", "real_scene1", [0.0, -0.305, 0.0], (0.0, 0.0)),
    "g1d_mobile": ("real_scene1_office", "real_scene1", [0.0, -0.305, 0.0], (0.0, 0.0)),  # same start pose as g1d
    # floor cabinet: the chassis cannot go under it, so the robot stands 7 cm further back (root y=-0.80)
    "g1d_counter": ("real_scene2_counter", "real_scene2", [-0.35, -0.32, 0.0], (0.0, -0.07)),
    "g1d_counter_mobile": ("real_scene2_counter", "real_scene2", [-0.35, -0.32, 0.0], (0.0, -0.07)),  # + its lounge
    "g1d_lab": ("real_scene3_lab", "real_scene3", [0.0, -0.257, 0.0], (0.0, 0.0)),
    # school green-screen table: robot 0.19 m behind the edge (root y=-0.79), 0.083 m left of the table centre, lift at
    # zero (the torso keeps its fitted height over this table, see env_cfg/robot/g1d_school.yml)
    "g1d_school": ("real_scene4_school", "real_scene4", [0.0828, -0.302, 0.0], (0.0, -0.06)),
    "g1d_pick_bucket": ("real_scene4_school", "real_scene4", [0.0835832937, -0.3648223366, 0.0], (0.0, 0.0)),
    # the same school table rebuilt from tape (real2sim/scenes/real_scene4_school_v2/PLAN.md): root 0.157 m behind the
    # edge (y=-0.757), 1.5 cm right of the table centre, lift at zero (env_cfg/robot/g1d_school_v2.yml)
    "g1d_school_v2": ("real_scene4_school_v2", "real_scene4_v2", [-0.015, -0.30, 0.0], (0.0, 0.0)),
    # ShenZhen lab desk (tape-measured 1.2 x 0.6 x 0.75 m): front edge at y=-0.60 like the office, stand not measured
    "g1d_shenzhen_lab": ("real_scene5_shenzhen_lab", "real_scene5", [0.0, -0.30, 0.0], (0.0, 0.0)),
    # HIMA user centre (not a twin, built from public descriptions): 1.4 x 0.7 x 0.75 m negotiation table, front edge
    # at y=-0.60; doors, drawers and props move (real2sim/scenes/real_scene6_hima_store/PLAN.md)
    "g1d_hima_store": ("real_scene6_hima_store", "real_scene6", [0.0, -0.25, 0.0], (0.0, 0.0)),
}
# ponytail: lights live in the per-env Room, so N parallel envs get N suns; move them to /World if num_envs > 1 matters.
# Red (adzuki) beans: Water_Opaque.mdl (glossy) with a dark maroon base colour (linear RGB).
RED_BEAN_COLOR = "color(0.15, 0.008, 0.01)"
RED_BEAN_MDL = "$Robodojo_ASSETS/Material/Fluid/RedBean.mdl"
# Beans are PBD solid (granular) particles: 5.6 mm spheres of 0.1 g, the volume and mass of a real 7 x 5 x 5 mm bean,
# no cohesion. Friction 0.3 is fitted to the real G1D-PourBeansEps380 pours (scripts/g1d/bean_tilt_probe.py, 2026-11-01:
# the tall cup filled to the real 131 mm, rolled at the real ~2 deg/s, sheds 5% at 88 deg and 25% at 98 deg; the real
# cups first catch beans at 86-90 deg and are about half full, a quarter of the tall cup, by the 93-99 deg peak).
# At 0.2 the same cup sheds at 82 / 92 deg and a poured pile stands at 13 deg (atan 0.2), at 0.3 at 18 deg; real
# beans stand at ~25-30 deg. Bean size hardly matters (4.5 mm: within 1 deg). The 13 mm pitcher pour opening jams
# beans above ~5 mm, so the pitcher task no longer pours (4 mm spheres did; check_pitcher_fluid.py --red-beans).
RED_BEAN_PHYSICS = {
    "granular": True,
    "particle_mass": 1e-4,
    "particle_system": {
        "particle_contact_offset": 0.0034,
        "contact_offset": 0.0034,
        "rest_offset": 0.0028,
        "solid_rest_offset": 0.0028,
        "fluid_rest_offset": 0.0017,
        "max_velocity": 2.5,
        "smoothing": False,
        "anisotropy": False,
        "isosurface": False,  # render the particles as beans, not a liquid surface
    },
    "particle_material": {
        "friction": 0.3,
        "particle_friction_scale": 1.0,
        "damping": 0.0,
        "adhesion": 0.0,
        "particle_adhesion_scale": 0.0,
        "cohesion": 0.0,
        "viscosity": 0.0,
        "surface_tension": 0.0,
        "drag": 0.0,
        "lift": 0.0,
    },
}
# Half-full pitcher: FluidObject fills the fluid mesh (a 60 x 60 x 70 mm ellipsoid, 130 ml) scaled by visual.scale with
# particles, then scales the particle set again by scale * visual.scale; scale = 1 / visual.scale keeps the spacing.
# 1.2 x 1.2 x 2.7 -> 72 x 72 x 189 mm (513 ml, about half of the 1.1 l jug), standing on the inside bottom.
PITCHER_FILL = (1.2, 1.2, 2.7)
PITCHER_FLUID_Z = -0.017  # fluid centre relative to the pitcher centre
PITCHER_AXIS_X = -0.0137  # body axis in the pitcher frame (metadata contact_circle): the bbox centre is off-axis
# The second arm picks the cup and holds it under the spout: it reaches the cup's rim only ~0.40 m in front of the
# robot root (scripts/g1d/scripted_demo.py), so the cup comes this much closer than the jug; layouts where it then
# overlaps the jug are dropped.
POUR_CUP_PULL = -0.09
POUR_MIN_GAP = 0.03  # m between the jug's and the cup's footprints
# Cup-to-cup pour (two plastic cups): each cup within the side grasp's reach of the arm on its side, |x| and y ranges
# (from |x| = 0.178 the right arm cannot lift its cup up to the pour, 0.162 it can); the source cup a sixth full: the
# 60 x 60 x 70 mm fluid ellipsoid scaled to 46 x 46 x 50 mm (~55 ml), its centre this far above the cup centre (inside
# bottom + half its height + 2.8 mm)
CUP_PAIR_X, CUP_PAIR_Y = (0.10, 0.16), (-0.39, -0.31)
CUP_FILL = (0.77, 0.77, 0.71)
CUP_FLUID_Z = -0.0167
# Measuring cups (scripts/objects/make_plastic_cups.py): 500 ml source, 400 ml target, as in the desk photos. Turned
# 45 deg about z so the pads, closing across the 45-deg-inward wrist, meet the square body's flats (63-68 mm) and not
# its corners (75-88 mm, nearly the 95 mm jaw). Fluid 54 x 54 x 63 mm (~95 ml) 0.8 mm above the 2 mm inside bottom of
# the 170 mm cup.
MEASURING_CUPS = ((1, "source_cup"), (0, "target_cup"))  # measuring_cup index, label
MEASURING_ORI = [0.92388, 0.0, 0.0, 0.38268]
MEASURING_FILL = (0.9, 0.9, 0.9)
MEASURING_FLUID_Z = -0.0487
OBJECT_SECTIONS = ("Rigid", "Geometry", "Fluid", "Articulation", "Garment", "Clutter", "Dynamic")
D435_INERTIAL = (
    '<inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.01"/>'
    '<inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/></inertial>'
)


def object_meta(assets, rel):
    return json.loads((Path(object_dir(rel, assets)) / "metadata.json").read_text())


def mesh_files(root):
    return {m.get("filename") for m in root.iter("mesh")}


def fetch(src, rel):
    dst = src / rel
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetch {rel}")
        urllib.request.urlretrieve(UNITREE_RAW + rel, dst)
    return dst


def build_urdf(v4, src, out):
    g1d = ET.parse(v4 / V4_URDF).getroot()
    g1 = ET.parse(fetch(src, G1_DEX1_URDF)).getroot()
    g1d.extend(el for el in g1 if el.tag in ("link", "joint") and el.get("name") in ("d435_joint", "d435_link"))

    for joint in g1d.findall("joint"):
        if joint.get("name").endswith("Wheel_Joint"):
            joint.set("type", "fixed")
    for link in g1d.findall("link"):
        if link.find("inertial") is None:  # d435_link is massless in the G1 URDF; PhysX wants mass
            link.append(ET.fromstring(D435_INERTIAL))

    movable = [j.get("name") for j in g1d.findall("joint") if j.get("type") != "fixed"]
    for side in ("left", "right"):
        assert sum(n.startswith(side) and "dex1" not in n for n in movable) == 7, f"{side} arm must have 7 joints"
        assert sum(n.startswith(f"{side}_dex1_finger") for n in movable) == 2, f"{side} Dex1 must have 2 fingers"

    for mesh in mesh_files(g1d):
        (out / mesh).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(v4 / mesh, out / mesh)
    seat_wrist_camera(g1d, v4, out)
    recess_carrier_hulls(g1d, out)
    shutil.copy(v4 / "physics_contract.json", out)
    for usd in ("usd", "usd_mobile"):  # UrdfFileCfg hashes only the URDF path, not mesh files
        shutil.rmtree(out / usd, ignore_errors=True)
    ET.ElementTree(g1d).write(out / "g1d_dex1.urdf", encoding="utf-8", xml_declaration=True)
    print(f"wrote {out / 'g1d_dex1.urdf'} ({len(movable)} movable joints)")
    mobile = mobile_urdf(g1d)
    ET.ElementTree(mobile).write(out / "g1d_mobile_dex1.urdf", encoding="utf-8", xml_declaration=True)
    print(f"wrote {out / 'g1d_mobile_dex1.urdf'}")


def mobile_urdf(g1d):
    """G1-D Flagship (differential chassis): the tabletop URDF on three virtual planar joints (x, y, yaw in the frame of
    the start pose), driven from AgvClient-style (vx, vyaw) commands by env/robot_manager/robot_class/g1d.py. The
    official URDF has only the two drive wheels (no casters), so a wheel-driven chassis would tip over: the wheels stay
    locked and lose their colliders, which would drag on the floor; the chassis body (3.4 cm ground clearance) keeps
    its collider."""
    mobile = copy.deepcopy(g1d)
    for link in mobile.findall("link"):
        if link.get("name").endswith("Wheel_Link"):
            for collision in link.findall("collision"):
                link.remove(collision)
    chain = [("base_world", None), ("base_x_link", ("base_x_joint", "prismatic", "1 0 0"))]
    chain += [
        ("base_y_link", ("base_y_joint", "prismatic", "0 1 0")),
        ("AGV_link", ("base_yaw_joint", "revolute", "0 0 1")),
    ]
    for i, (child, joint) in enumerate(chain):
        if child != "AGV_link":
            mobile.insert(i, ET.fromstring(f'<link name="{child}">{D435_INERTIAL}</link>'))
        if joint:
            name, kind, axis = joint
            limit = 1e3 if kind == "prismatic" else 1e4  # unbounded in practice; continuous joints import unreliably
            mobile.append(
                ET.fromstring(
                    f'<joint name="{name}" type="{kind}"><parent link="{chain[i - 1][0]}"/><child link="{child}"/>'
                    f'<origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="{axis}"/>'
                    f'<limit lower="{-limit}" upper="{limit}" effort="1e4" velocity="10"/></joint>'
                )
            )
    return mobile


def seat_wrist_camera(g1d, v4, out):
    """Dex1-1 camera version: the v4 base mesh (= official dex1_1_service base_link.STL) holds the camera module up on
    a ~6 cm post; scripts/g1d/dex1_camera rebuilds the module on its upright at the real height, smooths the tail by
    the wrist and adds the mount lip (approximate geometry, *profile.json), as body / support / housing / lip visual
    meshes and convex colliders. Its lens lies on the cam_wrist optical axis, 11 mm ahead of the robot-config sensor,
    inside the camera's 3 cm near plane. The three steps run as delivered (each checks its inputs and the finger
    clearance) in a scratch dir; only the two dex1 base links and their meshes come back."""
    import tempfile

    import dex1_camera
    from dex1_camera.add_mount_lip import add_mount_lip
    from dex1_camera.build import build
    from dex1_camera.geometry import BASE_MESH
    from dex1_camera.smooth_tail import smooth_tail

    profiles = Path(dex1_camera.__file__).parent
    links = ("left_dex1_model_base_link", "right_dex1_model_base_link")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "base").mkdir()
        (tmp / "base/meshes").symlink_to((out / "meshes").resolve())  # also when --assets is relative
        ET.ElementTree(g1d).write(tmp / "base/g1d_dex1.urdf", encoding="utf-8", xml_declaration=True)
        build(tmp / "base/g1d_dex1.urdf", v4 / BASE_MESH, profiles / "profile.json", tmp / "camera")
        smooth_tail(tmp / "camera", tmp / "tail", profiles / "tail_profile.json")
        add_mount_lip(tmp / "tail", tmp / "lip", profiles / "mount_lip_profile.json")
        built = ET.parse(tmp / "lip/g1d_dex1.urdf").getroot()
        for name in links:
            link, new = (r.find(f"link[@name='{name}']") for r in (g1d, built))
            link[:] = list(new)
        shutil.rmtree(out / "meshes/camera_geometry", ignore_errors=True)
        shutil.copytree(tmp / "lip/meshes/camera_geometry", out / "meshes/camera_geometry")
    (out / BASE_MESH).unlink()


def recess_carrier_hulls(g1d, out):
    """Link{1,2}_2 is L-shaped (slider lug + finger beam); its single convex hull bridges the L and sticks 3.4 mm
    past the frictionless carrier into the grip zone ahead of the Link{1,2}_3 pad. Trim the lug on the collision copy
    so the carrier hull stays 0.5 mm behind the pad face."""
    import numpy as np
    import trimesh

    for idx, clip in ((1, lambda x: np.maximum(x, -0.002)), (2, lambda x: np.minimum(x, 0.002))):
        rel = f"meshes/dex1_1/Link{idx}_2_col.STL"
        mesh = trimesh.load(out / f"meshes/dex1_1/Link{idx}_2.STL", process=False)
        mesh.vertices[:, 0] = clip(mesh.vertices[:, 0])
        mesh.export(out / rel)
        for el in g1d.findall(f".//link/collision/geometry/mesh[@filename='meshes/dex1_1/Link{idx}_2.STL']"):
            el.set("filename", rel)


def shift_objects(layout, offset):
    """Translate every task object (not the room/table fixtures) by `offset`, keeping their relative layout."""
    for section in OBJECT_SECTIONS:
        for entries in layout.get(section, {}).values():
            for entry in entries:
                entry["default_pos"] = [p + d for p, d in zip(entry["default_pos"], offset)]
    return layout


def squeeze(layout, scale):
    """Scale every task object's (x, y) about the stock table centre: X5 layouts span 0.9 x 0.3 m in front of its
    arms, G1-D standing at a 0.68 m table reaches about 0.5 x 0.15 m with its hands down."""
    for section in OBJECT_SECTIONS:
        for entries in layout.get(section, {}).values():
            for entry in entries:
                entry["default_pos"][0] *= scale[0]
                entry["default_pos"][1] *= scale[1]
    return layout


def move_into_real_scene(layout, xy, room, room_pos, table_top):
    """Swap the stock room/table/ground/dome for the real scene; move the objects by `xy` and down onto its tabletop."""
    table = layout.pop("Table")
    shift_objects(layout, (*xy, table_top - (table["default_pos"][2] + table["scale"][2] / 2)))
    layout.pop("Ground", None)
    layout.pop("Background", None)
    layout["Room"] = room_entry(room, room_pos, table_top)
    return layout


def room_entry(room, room_pos, table_top):
    return {
        "default": room,
        "default_pos": room_pos,
        "default_ori": [1.0, 0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "collision": False,  # the USD carries its own static colliders (floor box included)
        "table_height": table_top,  # for the layout stability check
    }


def swap_bottle_for_pitcher(layout, assets, layout_index, table_top):
    """Replace the wuliangye bottle (and the fluid inside it) by our fridge jug, spout turned toward the cup, and the
    cup by our water cup, POUR_CUP_PULL closer to the robot; None if the cup then sits too close to the jug."""
    index = layout_index % 3  # jug colour variant
    meta = object_meta(assets, f"Rigid/pitcher/{index:05d}")
    bottle = layout["Rigid"].pop("wuliangye")[0]
    pitcher = dict(bottle, category="pitcher", category_idx=index, physics=dict(meta["physics"], type="rigid"))
    height = meta["physics"]["size"][2]
    x, y, _ = bottle["default_pos"]  # the bottle's origin is not its bbox centre, the pitcher's is
    z = table_top + height / 2 + 1e-4
    pitcher["default_pos"] = [x, y, z]
    flip = x < 0
    pitcher["default_ori"] = [0, 0, 0, 1] if flip else [1, 0, 0, 0]  # spout is -X, handle +X
    layout["Rigid"]["pitcher"] = [pitcher]
    fluid = layout["Fluid"]["wuliangye"][0]
    fluid["zlim"] = [PITCHER_FLUID_Z] * 2
    fluid["default_pos"] = [x - PITCHER_AXIS_X if flip else x + PITCHER_AXIS_X, y, z + PITCHER_FLUID_Z]
    fluid["physics"] = dict(fluid["physics"], **RED_BEAN_PHYSICS)
    fluid["visual"]["visual_mdl_path"] = RED_BEAN_MDL
    fluid["visual"]["scale"] = list(PITCHER_FILL)
    fluid["scale"] = [1 / s for s in PITCHER_FILL]
    key, mug = next((k, e) for k, es in layout["Rigid"].items() for e in es if e.get("label") == "cup")
    layout["Rigid"][key].remove(mug)
    if not layout["Rigid"][key]:
        del layout["Rigid"][key]
    meta = object_meta(assets, "Rigid/water_cup/00000")
    x, y, _ = mug["default_pos"]
    cup = dict(mug, category="water_cup", category_idx=0, physics=dict(meta["physics"], type="rigid"))
    cup["default_pos"] = [x, y + POUR_CUP_PULL, table_top + meta["physics"]["size"][2] / 2 + 1e-4]
    layout["Rigid"]["water_cup"] = [cup]
    if _gap_xy(_obb(assets, "Rigid", "pitcher", pitcher), _obb(assets, "Rigid", cup["category"], cup)) < POUR_MIN_GAP:
        return None
    return layout


def two_cups(layout, assets, layout_index, table_top):
    """Replace the bottle and the mug by two plastic cups: the source (red beans) on the robot's left, the target on its
    right; positions drawn from the bottle's original one, so every seed folder and layout differs (indices repeat).
    The right hand does not pour: the cup turns between the pads (one Dex1 finger sits on its open stop, on the other
    side of the cup than in the left hand) and the beans stay in at 85 deg."""
    meta = object_meta(assets, "Rigid/plastic_cup/00000")
    z = table_top + meta["physics"]["size"][2] / 2 + 1e-4
    template = layout["Rigid"].pop("wuliangye")[0]
    rng = random.Random(json.dumps(template["default_pos"]))
    side = -1
    layout["Rigid"] = {}
    cups = []
    for label, sign in (("source_cup", side), ("target_cup", -side)):
        xy = [sign * rng.uniform(*CUP_PAIR_X), rng.uniform(*CUP_PAIR_Y)]
        cups.append(
            dict(
                template,
                category="plastic_cup",
                category_idx=0,
                label=label,
                default_pos=[*xy, z],
                default_ori=[1.0, 0.0, 0.0, 0.0],
                physics=dict(meta["physics"], type="rigid"),
            )
        )
    layout["Rigid"]["plastic_cup"] = cups
    beans = layout["Fluid"]["wuliangye"][0]
    beans["label"], beans["relative_plane"] = "beans", "source_cup"
    beans["zlim"] = [CUP_FLUID_Z] * 2
    beans["default_pos"] = [*cups[0]["default_pos"][:2], z + CUP_FLUID_Z]
    beans["physics"] = dict(beans["physics"], **RED_BEAN_PHYSICS)
    beans["visual"]["visual_mdl_path"] = RED_BEAN_MDL
    beans["visual"]["scale"] = list(CUP_FILL)
    beans["scale"] = [1 / s for s in CUP_FILL]
    return layout


def two_measuring_cups(layout, assets, layout_index, table_top):
    """two_cups with the measuring cups (MEASURING_CUPS): same positions as two_cups draws for this layout."""
    layout = two_cups(layout, assets, layout_index, table_top)
    cups = layout["Rigid"].pop("plastic_cup")
    for cup, (idx, label) in zip(cups, MEASURING_CUPS):
        assert cup["label"] == label
        meta = object_meta(assets, f"Rigid/measuring_cup/{idx:05d}")
        cup.update(category="measuring_cup", category_idx=idx, default_ori=MEASURING_ORI)
        cup["physics"] = dict(meta["physics"], type="rigid")
        cup["default_pos"][2] = table_top + meta["physics"]["size"][2] / 2 + 1e-4
    layout["Rigid"]["measuring_cup"] = cups
    beans = layout["Fluid"]["wuliangye"][0]
    beans["zlim"] = [MEASURING_FLUID_Z] * 2
    beans["default_pos"][2] = cups[0]["default_pos"][2] + MEASURING_FLUID_Z
    beans["visual"]["scale"] = list(MEASURING_FILL)
    beans["scale"] = [1 / s for s in MEASURING_FILL]
    return layout


def build_red_bean_material(assets):
    fluid = assets / "Material" / "Fluid"
    mdl = (fluid / "Water_Opaque.mdl").read_text()
    bean = mdl.replace("Water_Opaque(", "RedBean(").replace("color(0.008, 0.02, 0.03)", RED_BEAN_COLOR)
    assert bean.count("RedBean(") == 1 and RED_BEAN_COLOR in bean, "Water_Opaque.mdl changed"
    (fluid / "RedBean.mdl").write_text(bean)


def install_rooms(assets):
    """Install every real scene as its Room category (select_room loads the one .usd at the top of the folder)."""
    for scene, room, _, _ in SCENES.values():
        src, dst = SCENES_DIR / scene, assets / "Room" / room
        shutil.copytree(src / "usd", dst / "usd", dirs_exist_ok=True)
        shutil.copy(src / f"{room}.usd", dst)
        print(f"wrote {dst}")


def _obb(assets, kind, cat, entry):
    from scipy.spatial.transform import Rotation

    idx = entry.get("category_idx", 0)
    sub = "Clutter" if entry.get("type") == "cluttered" else kind
    meta = object_meta(assets, f"{sub}/{cat}/{idx:05d}")
    verts = meta["geometry"]["oriented_bbox"]["vertices"]
    scale, pos, q = entry.get("scale", [1, 1, 1]), entry["default_pos"], entry["default_ori"]
    rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
    axes = rot.as_matrix()
    up = max(range(3), key=lambda a: abs(axes[2, a]))
    ext = [(max(v[a] for v in verts) - min(v[a] for v in verts)) * scale[a] for a in range(3)]
    wverts = [rot.apply([v[a] * scale[a] for a in range(3)]) + pos for v in verts]
    lo = [min(v[a] for v in wverts) for a in range(2)]
    hi = [max(v[a] for v in wverts) for a in range(2)]
    return min(ext[a] for a in range(3) if a != up), ext[up], lo, hi


def _gap_xy(a, b):
    return max(a[2][0] - b[3][0], b[2][0] - a[3][0], a[2][1] - b[3][1], b[2][1] - a[3][1])


def filter_pickup(layout, assets, layout_index, table_top):
    """Drop layouts whose target cannot be grasped by the 90 mm Dex1-1 jaw (narrow horizontal side > 80 mm, or flat
    < 15 mm), and drop clutter items that the 0.6x squeeze pushed into the target or into each other."""
    rigid = layout["Rigid"]
    target_key, target = next((k, e) for k, es in rigid.items() for e in es if e.get("label") == "target")
    info = _obb(assets, "Rigid", target.get("category") or target_key, target)
    if info[0] > 0.08 or info[1] < 0.015:
        return None
    kept = {target_key: [target]}
    boxes = [(True, info)]
    for k, entries in rigid.items():
        if k == target_key:
            continue
        out = []
        for e in entries:
            box = _obb(assets, "Rigid", e.get("category") or k, e)
            if all(_gap_xy(box, b) >= (0.02 if is_target else 0.005) for is_target, b in boxes):
                out.append(e)
                boxes.append((False, box))
        if out:
            kept[k] = out
    layout["Rigid"] = kept
    return layout


# G1-D tasks reuse a benchmark task's eval layouts with the objects moved into G1-D reach (the X5 arms sit on the
# table; G1-D stands behind it, its arms reach about 0.42 m in front of the shoulders at table height):
# task: (benchmark task, (x, y) offset, (x, y) squeeze, post-filter/swap or None)
G1D_TASKS = {
    "g1d_pour_water_into_cup": ("pour_liquid_into_cup", (0.0, -0.20), (1.0, 1.0), swap_bottle_for_pitcher),
    "g1d_pour_beans_cup_to_cup": ("pour_liquid_into_cup", (0.0, -0.20), (1.0, 1.0), two_cups),
    "g1d_pour_beans_measuring_cups": ("pour_liquid_into_cup", (0.0, -0.20), (1.0, 1.0), two_measuring_cups),
    "g1d_general_pickup": ("general_pickup", (0.0, -0.36), (0.6, 0.6), filter_pickup),
    "g1d_stack_blocks": ("stack_blocks", (0.0, -0.35), (0.6, 0.6), None),
}


# Size and colour generalization without new assets, per task and object label, drawn per layout (so every episode is
# reproducible from its layout file): {"scale": (lo, hi)} uniform factor, "color": True for a random plain colour
# (RigidObject binds it over the asset's looks; on the meshes in metadata visual.color_parts if listed). Off for every
# current task: their eval sets and datasets were made at scale 1 in the asset colours. A task that turns it on must
# read sizes from the layout manager's bboxes (scaled), not fixed geometry; the beans tasks count beans in fixed
# cavities.
RANDOMIZE = {}  # e.g. "g1d_general_pickup": {"target": {"scale": (0.85, 1.15), "color": True}}


def randomize_objects(layout, rules, table_top, assets, seed):
    """Apply RANDOMIZE `rules` to the labelled rigid objects of `layout`: scale about the tabletop (the object keeps
    standing on it), mass by the volume, a plain colour in linear RGB."""
    rng = random.Random(seed)
    for cat, entries in layout.get("Rigid", {}).items():
        for entry in entries:
            rule = rules.get(entry.get("label"))
            if not rule:
                continue
            if "scale" in rule:
                s = rng.uniform(*rule["scale"])
                entry["scale"] = [v * s for v in entry.get("scale", [1.0, 1.0, 1.0])]
                entry["default_pos"][2] = table_top + (entry["default_pos"][2] - table_top) * s
                if "mass" in entry.get("physics", {}):
                    entry["physics"] = dict(entry["physics"], mass=entry["physics"]["mass"] * s**3)
            if rule.get("color"):
                rgb = colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.45, 0.9), rng.uniform(0.35, 0.9))
                sub, index = "Clutter" if entry.get("type") == "cluttered" else "Rigid", entry.get("category_idx", 0)
                name = entry.get("category") or cat
                meta = object_meta(assets, f"{sub}/{name}/{index:05d}")
                visual = dict(entry.get("visual") or {}, color=[round(c**2.2, 4) for c in rgb])  # sRGB -> linear
                if meta.get("visual", {}).get("color_parts"):
                    visual["color_parts"] = meta["visual"]["color_parts"]
                entry["visual"] = visual
    return layout


# Mobile tasks in a scene's lounge (Room usd/build_params.json "lounge"), drawn per layout from a seeded RNG rather than
# from a benchmark task's layouts: env config: {task: generator}; LOUNGE_LAYOUTS layouts in each seed folder.
LOUNGE_SEEDS, LOUNGE_LAYOUTS = (0, 1, 2), 10
# cup on the free countertop in front of g1d_counter's root (0, -0.80), |x| on either side: the arm on that side grasps
# it from the outside (the cup pours' side grasp, scripted_demo.cup_grasp)
CUP_X, CUP_Y = (0.06, 0.16), (-0.46, -0.38)
CHAIR_JITTER, CHAIR_YAW_JITTER = 0.05, 10.0  # m, deg about each chair's photo pose
# g1d_counter_mobile's chassis stands this much further back than g1d_counter's root (its -0.80 overlaps the cabinet)
MOBILE_BACK = -0.053
# jug within JUG_MAX_X of the midline and at least JUG_MAX_Y out (world y): its handle sits 6 cm further out, and the
# jug hand fell short at x 0.26 and knocked the jug over at (0.14, -0.39); probed only at (0.09, -0.49)
JUG_MAX_X, JUG_MAX_Y = 0.12, -0.44


def _yaw_quat(deg):
    a = math.radians(deg) / 2
    return [math.cos(a), 0.0, 0.0, math.sin(a)]


def deliver_cup_layout(assets, room, room_pos, params, rng):
    """The plastic cup somewhere on the countertop in front of the robot, any yaw; the lounge chairs (rigid, at their
    real mass so the chassis pushes rather than flings them) at their photo poses, jittered."""

    def entry(cat, idx, label, xy, yaw, plane, z_base):
        meta = object_meta(assets, f"Rigid/{cat}/{idx:05d}")
        physics = dict(meta["physics"], type="rigid")
        if cat == "lounge_chair":
            physics["mass_cap"] = physics["mass"]
        return {
            "category": cat,
            "category_idx": idx,
            "label": label,
            "default_pos": [*map(float, xy), z_base + meta["physics"]["size"][2] / 2 + 1e-3],
            "default_ori": _yaw_quat(yaw),
            "scale": [1.0, 1.0, 1.0],
            "physics": physics,
            "visual": {},
            "relative_plane": plane,
            # the layout check also fails anything resting > 5 cm below the tabletop: floor furniture is left out
            "need_check_stable": plane != "ground",
        }

    top = params["table"]["top_z"]
    cup_xy = (rng.choice((-1, 1)) * rng.uniform(*CUP_X), rng.uniform(*CUP_Y))
    chairs = []
    for i, chair in enumerate(params["lounge"]["chairs"]):
        xy = [c + o + rng.uniform(-CHAIR_JITTER, CHAIR_JITTER) for c, o in zip(chair["xy"], room_pos)]
        yaw = chair["yaw_deg"] + rng.uniform(-CHAIR_YAW_JITTER, CHAIR_YAW_JITTER)
        chairs.append(entry("lounge_chair", chair["index"], f"chair_{i}", xy, yaw, "ground", 0.0))
    return {
        "Rigid": {
            "plastic_cup": [entry("plastic_cup", 0, "cup", cup_xy, rng.uniform(0, 360), "Table", top)],
            "lounge_chair": chairs,
        },
        "Room": room_entry(room, room_pos, top),
    }


# Water: the stock pour_liquid_into_cup liquid (PBD fluid, isosurface) made to flow at real mass
WATER_MDL = "$Robodojo_ASSETS/Material/Fluid/Linen_Blue.mdl"  # the stock liquid's look: opaque blue
# ~300 ml in the jug (the beans' PITCHER_FILL is ~510 ml, too heavy as water for the handle grip), its bottom kept
WATER_FILL = (1.2, 1.2, 1.6)
WATER_Z = PITCHER_FLUID_Z - 0.035 * (PITCHER_FILL[2] - WATER_FILL[2])
WATER_PHYSICS = {
    "type": "fluid",
    "particle_system": {
        "particle_contact_offset": 0.0025,
        "contact_offset": 0.0025,
        "rest_offset": 0.002,
        "fluid_rest_offset": 0.0015,
        "solid_rest_offset": 0.0015,
        "max_velocity": 2,
        "smoothing": False,
        "anisotropy": False,
        "isosurface": True,
    },
    # the stock cohesion 0.005 / surface tension 0.0074 hold the light (1 g/ml) particles together as one ball that
    # leaves the vessel in a lump, and its damping 0.99 keeps them in up past 70 deg of tilt: off, the water pours
    "particle_material": {"cohesion": 0.0, "friction": 0.1, "damping": 0.0, "gravity_scale": 1.0, "viscosity": 0.0091},
    "interaction_with_object": True,
    # 3 mm particles (2 x fluid_rest_offset) at 1 g/ml; FluidObject's 0.5 g default weighs ~20x that
    "particle_mass": 2.7e-5,
}


def pour_deliver_layout(assets, room, room_pos, params, rng):
    """g1d_counter's jug-and-cup layout (pour_liquid_into_cup through swap_bottle_for_pitcher), drawn at random, moved
    MOBILE_BACK with the chassis, without the clutter, the jug's beans made ~300 ml of water; the lounge chairs as in
    deliver_cup_layout."""
    _, _, xy, scale, swap = ("g1d_pour_water_into_cup", *G1D_TASKS["g1d_pour_water_into_cup"])
    top = params["table"]["top_z"]
    shift = next(v[3] for v in SCENES.values() if v[1] == room and v[2] == room_pos)
    files = sorted((assets / "Eval_Layout/RoboDojo/arx_x5").rglob("pour_liquid_into_cup_*.json"), key=_layout_sort_key)
    layout = None
    while layout is None:
        index = rng.randrange(len(files))
        layout = squeeze(json.loads(files[index].read_text()), scale)
        layout.pop("Geometry", None)  # the X5 camera stand, where the robot stands (build_layouts drops it too)
        layout = swap(move_into_real_scene(layout, xy, room, room_pos, top), assets, index, top)
        jug = layout and layout["Rigid"]["pitcher"][0]["default_pos"]
        if jug and (abs(jug[0] + shift[0]) > JUG_MAX_X or jug[1] + shift[1] + MOBILE_BACK > JUG_MAX_Y):
            layout = None
    shift_objects(layout, (shift[0], shift[1] + MOBILE_BACK, 0.0))
    # only the jug and the cup: the stock clutter on the counter keeps failing the layout stability check
    layout["Rigid"] = {k: es for k, es in layout["Rigid"].items() if k in ("pitcher", "water_cup")}
    fluid = layout["Fluid"]["wuliangye"][0]
    fluid["label"], fluid["physics"] = "water", WATER_PHYSICS
    fluid["visual"] = {"visible": True, "visual_mdl_path": WATER_MDL, "scale": list(WATER_FILL)}
    fluid["scale"] = [1 / s for s in WATER_FILL]
    fluid["zlim"] = [WATER_Z] * 2
    fluid["default_pos"][2] += WATER_Z - PITCHER_FLUID_Z
    layout["Rigid"]["lounge_chair"] = deliver_cup_layout(assets, room, room_pos, params, rng)["Rigid"]["lounge_chair"]
    return layout


LOUNGE_TASKS = {
    "g1d_counter_mobile": {
        "g1d_deliver_cup_to_lounge": deliver_cup_layout,
        "g1d_pour_water_deliver_to_lounge": pour_deliver_layout,
    }
}


def build_lounge_layouts(assets, config, room, room_pos, dst):
    params = json.loads((assets / "Room" / room / "usd" / "build_params.json").read_text())
    for task, make in LOUNGE_TASKS.get(config, {}).items():
        for seed in LOUNGE_SEEDS:
            (dst / str(seed)).mkdir(parents=True, exist_ok=True)
            for i in range(LOUNGE_LAYOUTS):
                layout = make(assets, room, room_pos, params, random.Random(f"{task}/{seed}/{i}"))
                (dst / str(seed) / f"{task}_{i}.json").write_text(json.dumps(layout, indent=4))
        print(f"wrote {len(LOUNGE_SEEDS) * LOUNGE_LAYOUTS} {task} layouts to {dst}")


# Block patterns (scripts/objects/block_patterns.py), drawn per layout from a seeded RNG: env config: {task: generator};
# PATTERN_LAYOUTS layouts in each seed folder, the patterns in turn. The card lies flat beyond the build space (the
# robot builds about block_patterns.BUILD_XY), its picture upright to the robot; the pattern's blocks
# (unpainted, index 0) lie to the side of their place in it, each footprint clear of the others and of the build
# space, turned so the hand on their side can grip them across.
PATTERN_SEEDS, PATTERN_LAYOUTS = (0, 1, 2), 12
# |x|, y of a scattered block's centre: further out than |x| 0.19 or nearer than y -0.37 the hands grip at the edge of
# their reach (blocks at y -0.34 .. -0.35 and a cube at x -0.20, y -0.45 were missed)
SCATTER_X, SCATTER_Y = (0.10, 0.19), (-0.53, -0.37)
# blocks under 2 cm high (planks, tile, half-cylinder), gripped near the fingertips, only in this band: a grasp probe
# held all three kinds at y -0.40 .. -0.43 (60 deg fingers, tips 5-6 mm up); nearer the robot is not probed
FLAT_Y = (-0.45, -0.37)
BUILD_CLEAR, SCATTER_GAP = 0.09, 0.015  # m: footprints this far out from the midline, this far apart
GRASP_YAW = 20.0  # deg: a block's grip yaw (its local Y, mod its symmetry) this close to block_patterns.grip_yaw
SIDE_X = 0.005  # m: a piece whose target is further off the midline is bound to that side
# blocks that are no piece of the pattern, scattered with its pieces: up to DISTRACTORS of these shapes and paints (a
# distractor never looks like a piece, which would make it the piece the check expects)
DISTRACTORS = 2
DISTRACTOR_SHAPES = tuple(f"wood_block_{s}" for s in ("cube", "short_cylinder", "half_cylinder", "small_prism", "tile"))
DISTRACTOR_PAINTS = (0, 1, 2, 4, 5, 7)  # natural, blue, red, green, yellow, pink


def table_entry(assets, top, cat, idx, label, xy, yaw):
    """A labelled rigid object standing on the tabletop (z `top`) at `xy`, turned `yaw` deg."""
    meta = object_meta(assets, f"Rigid/{cat}/{idx:05d}")
    return {
        "category": cat,
        "category_idx": idx,
        "label": label,
        "default_pos": [*map(float, xy), top + meta["physics"]["size"][2] / 2 + 1e-3],
        "default_ori": _yaw_quat(yaw),
        "scale": [1.0, 1.0, 1.0],
        "physics": dict(meta["physics"], type="rigid"),
        "visual": {},
        "relative_plane": "Table",
        "need_check_stable": True,
    }


def block_pattern_layout(assets, room, room_pos, params, rng, index):
    """Pattern `index` (mod the patterns): its card, its blocks in their paints and 0 to DISTRACTORS other blocks
    (labelled distractor_<j>) scattered as PATTERN_TASKS says."""
    from scripts.objects import block_patterns  # numpy: not at import, the platform (stdlib only) imports this

    top = params["table"]["top_z"]
    pattern_idx = index % len(block_patterns.PATTERNS)
    pattern = block_patterns.PATTERNS[pattern_idx]

    def entry(cat, idx, label, xy, yaw):
        return table_entry(assets, top, cat, idx, label, xy, yaw)

    rigid = {"block_pattern_card": [entry("block_pattern_card", pattern_idx, "card", block_patterns.CARD_XY, 0.0)]}

    def placing_order(piece):
        """Flat blocks first (their band is narrow), then those bound to a side, then the larger ones."""
        cat, x, _ = piece[1]
        w, d, h = block_patterns.SIZES[cat]
        return h >= 0.02, abs(x) <= SIDE_X, -math.hypot(w, d)

    keys = set(zip((c for c, _, _ in pattern["pieces"]), block_patterns.paints(pattern)))
    others = [(c, p) for c in DISTRACTOR_SHAPES for p in DISTRACTOR_PAINTS if (c, p) not in keys]
    extra = [rng.choice(others) for _ in range(rng.randint(0, DISTRACTORS))]
    n = len(pattern["pieces"])

    def scatter(k):
        """{block index: xy} or None, the pieces then `k` distractors (index n + j, at x 0: either side). A piece off
        the midline lies on its target's side, so the hand that picks it does not reach across (a cube carried over to
        x 0.07 landed 1-2 cm off and the bridge's top fell); a centred piece goes to the side with less taken so far.
        A side-bound piece with no room on its side: None."""
        placed, taken, out = [], {-1: 0.0, 1: 0.0}, {}
        first = rng.choice((-1, 1))
        blocks = sorted(enumerate(pattern["pieces"]), key=placing_order) + [
            (n + j, (cat, 0.0, 0.0)) for j, (cat, _) in enumerate(extra[:k])
        ]
        for i, (cat, x, _) in blocks:
            w, d, h = block_patterns.SIZES[cat]
            radius = math.hypot(w, d) / 2
            ys = FLAT_Y if h < 0.02 else SCATTER_Y
            bound = abs(x) > SIDE_X
            side = (1 if x > 0 else -1) if bound else min((first, -first), key=lambda s: taken[s])
            xy = None
            for s in (side,) if bound else (side, -side):
                for _ in range(1000):
                    xy = (s * rng.uniform(*SCATTER_X), rng.uniform(*ys))
                    if abs(xy[0]) - radius >= BUILD_CLEAR and all(
                        math.dist(xy, q) >= radius + r + SCATTER_GAP for q, r in placed
                    ):
                        break
                    xy = None
                if xy:
                    side = s
                    break
            if xy is None:
                return None
            placed.append((xy, radius))
            taken[side] += w * d
            out[i] = xy
        return out

    spots = next((s for k in range(len(extra), -1, -1) for _ in range(50) if (s := scatter(k))), None)
    if spots is None:
        raise RuntimeError(f"no room for the blocks of {pattern['name']}")
    paints = block_patterns.paints(pattern)
    pieces = {}  # labels keep the pattern's piece order
    for i, xy in spots.items():
        cat, paint = (pattern["pieces"][i][0], paints[i]) if i < n else extra[i - n]
        label = f"block_{i}" if i < n else f"distractor_{i - n}"
        # turned so the top grasp across the block's depth (local Y, at its yaw + 90, or every 90 deg for a square
        # block) is one the hand on its side reaches well; round blocks are gripped so at any yaw
        period = block_patterns.YAW_PERIOD[cat]
        if period:
            yaw = (
                block_patterns.grip_yaw(*xy)
                - 90.0
                + rng.uniform(-GRASP_YAW, GRASP_YAW)
                + period * rng.randrange(360 // period)
            )
        else:
            yaw = rng.uniform(0.0, 360.0)
        pieces[i] = (cat, entry(cat, paint, label, xy, yaw))
    for i in sorted(pieces):
        rigid.setdefault(pieces[i][0], []).append(pieces[i][1])
    return {"Rigid": rigid, "Room": room_entry(room, room_pos, top)}


def block_sort_layout(by):
    """Layout generator of the sorting task by `by` ("color" / "shape"), block_sorting.py: two classes drawn at random
    (two paints, or two shapes), one per tray; the left tray (box_0) and the right one (box_1) with the card of their
    class behind them (card_0, card_1); PER_CLASS blocks of each class (by colour: any sorting shape in that paint;
    by shape: that shape in any sorting paint) out to both sides, block_sorting.ACROSS of each class or none on the
    side away from its tray, labels shuffled, each turned so the hand on its side grips it best."""

    def make(assets, room, room_pos, params, rng, index):
        # numpy: not at import (the platform imports this)
        from scripts.objects import block_patterns, block_sorting as bs

        top = params["table"]["top_z"]
        classes = rng.sample(bs.COLORS if by == "color" else bs.SHAPES, 2)
        rigid = {bs.BOX[0]: [], "block_sort_card": []}
        for box, key in enumerate(classes):
            rigid[bs.BOX[0]].append(table_entry(assets, top, bs.BOX[0], bs.BOX[1], f"box_{box}", bs.box_xy(box), 0.0))
            card = table_entry(
                assets, top, "block_sort_card", bs.card_index(by, key), f"card_{box}", bs.card_xy(box), 0.0
            )
            rigid["block_sort_card"].append(card)
        # the same number of each class across from its box (ACROSS or none), so each side holds PER_CLASS blocks
        across = rng.choice([0] + [bs.ACROSS] * 3)
        blocks = []
        for box, key in enumerate(classes):
            for n in range(bs.PER_CLASS):
                if by == "color":
                    cat, paint = rng.choice(bs.SHAPES), key
                else:
                    cat, paint = key, rng.choice(bs.COLORS)
                lies = -bs.side(box) if n < across else bs.side(box)  # the side of the midline it lies on
                blocks.append((lies, cat, bs.PAINTS.index(paint)))
        rng.shuffle(blocks)
        for _ in range(1000):
            placed = []
            for lies, cat, _ in blocks:
                w, d, h = block_patterns.SIZES[cat]
                radius = math.hypot(w, d) / 2
                for _ in range(200):
                    xy = (lies * rng.uniform(*bs.SCATTER_X), rng.uniform(*(bs.FLAT_Y if h < 0.02 else bs.SCATTER_Y)))
                    if all(math.dist(xy, q) >= radius + r + bs.SCATTER_GAP for q, r in placed):
                        placed.append((xy, radius))
                        break
                else:
                    break
            if len(placed) == len(blocks):
                break
        else:
            raise RuntimeError(f"no room for the blocks of sort layout {index}")
        for i, ((lies, cat, paint), (xy, _)) in enumerate(zip(blocks, placed)):
            period = block_patterns.YAW_PERIOD[cat]
            if period:  # turned for the hand on its side, which picks it
                grip = block_patterns.grip_yaw(lies, xy[1])
                yaw = grip - 90.0 + rng.uniform(-GRASP_YAW, GRASP_YAW) + period * rng.randrange(360 // period)
            else:
                yaw = rng.uniform(0.0, 360.0)
            rigid.setdefault(cat, []).append(table_entry(assets, top, cat, paint, f"block_{i}", xy, yaw))
        return {"Rigid": rigid, "Room": room_entry(room, room_pos, top)}

    return make


def block_atomic_layout(task):
    """Layout generator of an atomic block task (scripts/objects/block_atomic.py LAYOUTS)."""

    def make(assets, room, room_pos, params, rng, index):
        top = params["table"]["top_z"]

        def entry(cat, idx, label, xy, yaw):
            return table_entry(assets, top, cat, idx, label, xy, yaw)

        return {"Rigid": block_atomic.LAYOUTS[task](rng, entry), "Room": room_entry(room, room_pos, top)}

    return make


def pick_anything_layout(assets, room, room_pos, params, rng, index):
    """g1d_pick_anything: a pool object (target) among 3-5 others (scripts/objects/pick_anything.py)."""
    top = params["table"]["top_z"]

    def entry(cat, idx, label, xy, yaw):
        return table_entry(assets, top, cat, idx, label, xy, yaw)

    return {"Rigid": pick_anything.layout(rng, index, entry), "Room": room_entry(room, room_pos, top)}


def pick_bucket_layout(assets, room, room_pos, params, rng, index):
    from scripts.g1d.pick_bucket_alignment import layout

    return layout(assets, table_entry, room_entry)


# the layout generators above, per env config; build_pattern_layouts writes PATTERN_LAYOUTS of each per seed
PATTERN_TASKS = {
    "g1d_school": {"pick_bucket": pick_bucket_layout},
    "g1d_pick_bucket": {"pick_bucket": pick_bucket_layout},
    "g1d": {
        "g1d_build_block_pattern": block_pattern_layout,
        "g1d_sort_blocks_by_color": block_sort_layout("color"),
        "g1d_sort_blocks_by_shape": block_sort_layout("shape"),
        **{task: block_atomic_layout(task) for task in block_atomic.TASKS},
        "g1d_pick_anything": pick_anything_layout,
    },
}


def write_pattern_layouts(assets, config, task, folder, seed, count):
    """`count` layouts of PATTERN_TASKS[config][task] into `folder`, layout i drawn from its own RNG keyed by task,
    seed and i (so no two seeds share a layout)."""
    _, room, room_pos, _ = SCENES[config]
    params = json.loads((assets / "Room" / room / "usd" / "build_params.json").read_text())
    make = PATTERN_TASKS[config][task]
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        layout = make(assets, room, room_pos, params, random.Random(f"{task}/{seed}/{i}"), i)
        (folder / f"{task}_{i}.json").write_text(json.dumps(layout, indent=4))


def build_pattern_layouts(assets, config, room, room_pos, dst):
    for task in PATTERN_TASKS.get(config, {}):
        seeds, count = ([0], 1) if task == "pick_bucket" else (PATTERN_SEEDS, PATTERN_LAYOUTS)
        for seed in seeds:
            write_pattern_layouts(assets, config, task, dst / str(seed), seed, count)
        print(f"wrote {len(seeds) * count} {task} layouts to {dst}")


def _layout_sort_key(path):
    tail = path.stem.rpartition("_")[2]
    return str(path.parent), int(tail) if tail.isdigit() else -1, path.stem


def build_layouts(assets):
    """Per scene config: the G1-D task layouts; the office one (g1d) also gets every arx_x5 layout."""
    src = assets / "Eval_Layout" / "RoboDojo" / "arx_x5"
    files = sorted(src.rglob("*.json"), key=_layout_sort_key)
    sys.path.insert(0, str(ROOT / "scripts" / "scenes"))
    import appearance

    for config, (scene, room, room_pos, shift) in SCENES.items():
        table_top = json.loads((assets / "Room" / room / "usd" / "build_params.json").read_text())["table"]["top_z"]
        # the platform's fixed materials / desk height of the scene (appearance.json), if set; random pools are for
        # datagen layouts (scripts/scenes/appearance.py --randomize)
        looks = appearance.load(SCENES_DIR / scene) if (SCENES_DIR / scene / "appearance.json").is_file() else None
        dst = assets / "Eval_Layout" / "RoboDojo" / config
        for old in dst.rglob("g1d_*_*.json"):
            old.unlink()
        counts = {}
        for f in files:
            layout = json.loads(f.read_text())
            geometry = layout.get("Geometry", {})
            geometry.pop("camera_stand", None)
            if "Geometry" in layout and not geometry:
                del layout["Geometry"]
            out = dst / f.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            if config == "g1d":
                out.write_text(json.dumps(layout, indent=4))
            base, _, index = f.stem.rpartition("_")
            for task, (base_task, xy, scale, swap) in G1D_TASKS.items():
                if base == base_task and index.isdigit():
                    shifted = squeeze(copy.deepcopy(layout), scale)
                    shifted = move_into_real_scene(shifted, xy, room, room_pos, table_top)
                    key = (out.parent, task)
                    seq = counts.get(key, 0)
                    if swap:
                        shifted = swap(shifted, assets, seq, table_top)
                    if shifted is not None and task in RANDOMIZE:
                        seed = f"{task}/{out.parent.name}/{seq}"  # the same draws in every scene config
                        randomize_objects(shifted, RANDOMIZE[task], table_top, assets, seed)
                    if shifted is not None:
                        shift_objects(shifted, (*shift, 0.0))
                        if looks:
                            appearance.apply(shifted, SCENES_DIR / scene, loaded=looks)
                        (out.parent / f"{task}_{seq}.json").write_text(json.dumps(shifted, indent=4))
                        counts[key] = seq + 1
        print(f"wrote {sum(counts.values())} G1-D task layouts to {dst}")
        build_lounge_layouts(assets, config, room, room_pos, dst)
        build_pattern_layouts(assets, config, room, room_pos, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--v4", type=Path, default=ROOT / "real2sim" / "robot" / "g1d_v4", help="assembled v4 URDF directory"
    )
    parser.add_argument("--src", type=Path, default=ROOT / ".cache" / "unitree_ros")
    parser.add_argument("--assets", type=Path, default=ROOT / "Assets")
    args = parser.parse_args()

    install_rooms(args.assets)
    out = args.assets / "Robots" / "g1d"
    build_urdf(args.v4, args.src, out)
    for side in ("left", "right"):
        shutil.copy(HERE / f"{side}_robot_config.yml", out)
    build_red_bean_material(args.assets)
    build_layouts(args.assets)


if __name__ == "__main__":
    main()
