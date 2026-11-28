"""Shapes the G1-D builds from the wooden blocks (Assets/Object/Blocks, scripts/objects/make_wood_blocks.py), shared by
the layouts (build_house_layouts.py), the task check (task/RoboDojo/tasks/g1d_build_house.py), the scripted expert
(scripted_demo.build_actions) and the datagen plugin. Plain numpy: no Isaac imports.

A shape is a list of parts in the shape frame (x' along the row, z up from the tabletop), each with the parts it rests
on. Each house block's layout entry carries its part of the drawn shape in "build" (shape frame, part, world target
pose); shape_entry() gathers them.

Every part lies with its block's local y across the row: all profiles are extruded along local y (30 mm), so the pads
close on those two faces and never between neighbours in the row. Each hand comes in from its own side, fingers
pointing inward (wrist yaw near -90 deg left, +90 deg right): scripts/g1d/reach_map.py finds both arms over the whole
row at every layer this way, and the row faces the head camera (the arch opening and the roof's gable are seen).
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from scripts.g1d.reach_map import HAND

ROOT = Path(__file__).resolve().parents[2]
BLOCKS = ROOT / "Assets/Object/Blocks/Rigid"
# category_idx = colour, as make_wood_blocks.PALETTE
COLORS = ("natural wood", "blue", "red", "orange-red", "green", "yellow", "orange", "pink")
COLORS_ZH = ("原木色", "蓝色", "红色", "橙红色", "绿色", "黄色", "橙色", "粉色")
DRAW_COLORS = (0, 1, 2, 4, 5, 6, 7)  # orange-red reads as red or orange in the head camera: not drawn
# en, zh, symmetry about z (deg): a turn by it looks the same and the pads grip the same faces
KINDS = {
    "wood_block_cube": ("cube", "正方体", 90),
    "wood_block_arch": ("arch", "拱形积木", 180),
    "wood_block_cuboid": ("long block", "长方体", 180),
    "wood_block_prism": ("triangular roof block", "三角形积木", 180),
    "wood_block_cylinder": ("cylinder", "圆柱", 360),
    "wood_block_short_cylinder": ("short cylinder", "短圆柱", 360),
    "wood_block_tile": ("square tile", "方片", 90),
    "wood_block_half_cylinder": ("half cylinder", "半圆柱", 180),
    "wood_block_small_prism": ("small triangular block", "小三角积木", 180),
    "wood_block_short_plank": ("short plank", "短木板", 180),
}
BLOCK_KINDS = tuple(KINDS)
DISTRACTORS = ("wood_block_cube", "wood_block_cylinder", "wood_block_short_cylinder", "wood_block_tile",
               "wood_block_half_cylinder", "wood_block_small_prism", "wood_block_short_plank")  # fmt: skip
GAP = 0.003  # m between neighbours in a row
# success: every part this close to its target, its top face up
POS_TOL, Z_TOL, YAW_TOL, TILT_TOL = 0.015, 0.010, 15.0, 10.0  # m, m, deg, deg
# grasp
FINGER_TIPS, TABLE_CLEAR = HAND["finger_tips"], 0.006  # scripted_demo: grip point -> fingertips; fingertips above the surface
TOP_GRIP = 0.004  # pads this far below a block's top
LOW_GRIP = 0.004  # fingertips this far above a low-gripped block's bottom (the prism: its faces narrow to the ridge)
VERTICAL, VERTICAL_TIPS = 85.0, 0.5  # deg: fingers this steep grip with the fingertips at this share of the height
ARM_YAW = {"left": -90.0, "right": 90.0}  # wrist yaw (world z, deg) with the fingers pointing inward


@dataclass(frozen=True)
class Part:
    name: str
    category: str  # "" in the template: the layout's kind (shape_entry) fills it in, parts_of()
    x: float  # along the row (shape frame), m
    supports: tuple = ()
    low_grip: bool = False
    en: str = ""  # where it goes, for the captions; {0}, {1} are the supports, {into} the container
    zh: str = ""
    # a part set into a container (a fixed object of the layout or another part: a tray, the box, a cup) is released
    # with its bottom release_dz above the container's measured top and counts as placed anywhere inside it
    # (CONTAINERS); upright: its tilt is checked too (not for a round fruit)
    into: str = ""
    release_dz: float = 0.0
    upright: bool = True


HOUSE = (
    Part("base_left", "wood_block_cube", -0.048, en="at the left end of the bottom row of the house", zh="房子底层的最左边"),
    Part("base_mid", "wood_block_arch", 0.0, en="in the middle of the bottom row of the house", zh="房子底层的中间"),
    Part("base_right", "wood_block_cube", 0.048, en="at the right end of the bottom row of the house",
         zh="房子底层的最右边"),
    Part("mid_left", "wood_block_cube", -0.0315, ("base_left", "base_mid"),
         en="on top of the {0} and the {1}, on the left of the second layer", zh="{0}和{1}上面，第二层的左边"),
    Part("mid_right", "wood_block_cube", 0.0315, ("base_mid", "base_right"),
         en="on top of the {0} and the {1}, on the right of the second layer", zh="{0}和{1}上面，第二层的右边"),
    Part("beam", "wood_block_cuboid", 0.0, ("mid_left", "mid_right"), en="across the {0} and the {1}",
         zh="{0}和{1}上面，横跨两块"),
    Part("roof", "wood_block_prism", 0.0, ("beam",), low_grip=True, en="on top of the {0} as the roof",
         zh="{0}上面作为屋顶"),
)  # fmt: skip
# the small house: the same bottom row, the roof straight on the arch (it is narrower than the arch's crown); the roof
# rests on the whole row so it always goes last (a cube set down beside a roofed arch turns the roof), its caption
# names the arch ({1})
HOUSE4 = (
    *HOUSE[:3],
    Part("roof", "wood_block_prism", 0.0, ("base_left", "base_mid", "base_right"), low_grip=True,
         en="on top of the {1} as the roof", zh="{1}上面作为屋顶"),
)  # fmt: skip
# the Aurora Lite tasks (scripts/g1d/aurora_task_layouts.py): no row, every part goes into a container
# sort: a cube and a short cylinder of each of two colours into the tray of their colour (card behind it), dropped in
# from just above the 2.5 cm walls
SORT = tuple(
    Part(f"{side}_{what}", cat, 0.0, en=f"into the {side} tray", zh=f"放进{zh}边的托盘", into=f"tray_{side}",
         release_dz=0.005)
    for side, zh in (("left", "左"), ("right", "右"))
    for what, cat in (("cube", "wood_block_cube"), ("cylinder", "wood_block_short_cylinder"))
)  # fmt: skip
# nest: the tall slim purple cup into the big base cup (blue or white, a fixed object), the small clear cup into the
# purple one; each released with its bottom 3.5 cm down inside the cup below, so the fingers stay above that cup's
# rim, and slides down. Only the slim cups are picked: the big cups' 80 mm rim passes the Aurora's 85 mm jaw with
# 2.4 mm a side on the way down to the grasp
NEST = (
    Part("mid", "plastic_cup/00007", 0.0, en="into the {into}", zh="放进{into}里", into="base", release_dz=-0.035),
    Part("top", "plastic_cup/00003", 0.0, ("mid",), en="into the {into}", zh="放进{into}里", into="mid",
         release_dz=-0.035),
)  # fmt: skip
# box: the named fruit or vegetable among others into the storage box, let go 1.5 cm above its rim
BOX = (Part("item", "", 0.0, en="into the {into}", zh="放进{into}", into="box", release_dz=0.015, upright=False),)
SHAPES = {"house": HOUSE, "house4": HOUSE4, "sort": SORT, "nest": NEST, "box": BOX}


LIBRARIES = ("Blocks", "Real2Sim", "RoboDojo")  # Assets/Object/<library>/Rigid, as env.global_configs.object_dir
# other objects of the Aurora tasks, kind "<category>/<index>" (one asset each, unlike a block kind, whose index is
# its colour): en, zh, symmetry about z (deg)
OBJECTS = {
    "plastic_cup/00001": ("blue plastic cup", "蓝色塑料杯", 360),
    "plastic_cup/00002": ("white plastic cup", "白色塑料杯", 360),
    "plastic_cup/00003": ("clear plastic cup", "透明塑料杯", 360),
    "plastic_cup/00007": ("purple plastic cup", "紫色塑料杯", 360),
    "sand3d_food_and_drink/00105": ("apple", "苹果", 360),
    "sand3d_food_and_drink/00183": ("kiwi", "猕猴桃", 360),
    "sand3d_food_and_drink/00202": ("lime", "青柠", 360),
    "sand3d_food_and_drink/00223": ("mushroom", "蘑菇", 360),
    "sand3d_food_and_drink/00293": ("tangerine", "橘子", 360),
    "sand3d_food_and_drink/00298": ("tomato", "番茄", 360),
    "storage_box/00000": ("storage box", "收纳盒", 180),
    "block_tray/00000": ("tray", "托盘", 180),
}
FRUITS = tuple(k for k in OBJECTS if k.startswith("sand3d_food_and_drink/"))
# a container's success region: its centre in the container frame within the half size less `margin` ("rect") or
# within `radius` of its axis ("circle"), its bottom at least `inside` below the container's top
CONTAINERS = {
    "block_tray/00000": dict(region="rect", margin=0.012, inside=0.01),
    "storage_box/00000": dict(region="rect", margin=0.02, inside=0.03),
    **{f"plastic_cup/0000{i}": dict(region="circle", radius=0.015, inside=0.05) for i in (1, 2, 3, 7)},
}
# vertical grasps: the fingertips at this share of the height (else VERTICAL_TIPS): a fruit below its middle (pads
# above the equator squeeze it out); a cup by its rim, the fingertips CUP_TIPS_UNDER_RIM below it (gripped lower, the
# cup widening to its rim wedges against the fingers above the pads and turns in the hand)
GRIP_TIPS = {**{k: 0.3 for k in FRUITS}}
CUP_TIPS_UNDER_RIM = 0.040  # m: the Aurora pads' upper edge 8 mm under the rim
# m a nested cup's rim stands above the rim of the cup below it (layouts' reach check only): the purple cup (150 mm)
# stands on the big cup's bottom (133 mm), 2 cm above its rim
NEST_RISE = 0.02
# kinds the expert's v4 perturbation does not move under the open hand: a tall light cup set down against a finger is
# shoved aside by the descending hand and knocked over
UNMOVED = {k for k in OBJECTS if k.startswith("plastic_cup/")}


def kind(category, index):
    """Shape kind of a layout object: an OBJECTS key, else the block category."""
    key = f"{category}/{int(index):05d}"
    return key if key in OBJECTS else category


def asset_dir(category, index):
    for lib in LIBRARIES:
        d = ROOT / "Assets/Object" / lib / "Rigid" / category / f"{int(index):05d}"
        if d.is_dir():
            return d
    raise FileNotFoundError(f"no asset {category}/{int(index):05d} under {ROOT / 'Assets/Object'}")


def _meta(category):
    return json.loads((BLOCKS / category / "00000" / "metadata.json").read_text())


SIZES = {c: np.array(_meta(c)["physics"]["size"]) for c in KINDS}
for _k in OBJECTS:
    try:
        SIZES[_k] = np.array(json.loads((asset_dir(*_k.split("/")) / "metadata.json").read_text())["physics"]["size"])
    except FileNotFoundError:  # a library without the Aurora tasks' objects: the house still works
        pass
KINDS.update(OBJECTS)
GRIP_TIPS.update({k: 1 - CUP_TIPS_UNDER_RIM / SIZES[k][2] for k in UNMOVED if k in SIZES})


def block_name(category, color, zh=False):
    en, cn, _ = KINDS[category]
    if category in OBJECTS:
        return cn if zh else en
    return f"{COLORS_ZH[color]}{cn}" if zh else f"{COLORS[color]} {en}"


def color_name(category, color):
    """The block's paint, "" for another object."""
    return COLORS[color] if category in BLOCK_KINDS else ""


def yaw_quat(deg):
    """wxyz"""
    q = Rotation.from_euler("z", deg, degrees=True).as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def yaw_diff(a, b, sym):
    """Smallest |a - b| in deg modulo the block's symmetry."""
    return abs((a - b + sym / 2) % sym - sym / 2)


def topo_orders(parts):
    """Every order that puts each part after its supports (the house has 30)."""
    out = []

    def rec(done, left):
        if not left:
            out.append(done)
        for p in left:
            if set(p.supports) <= {d.name for d in done}:
                rec(done + [p], [q for q in left if q is not p])

    rec([], list(parts))
    return [[p.name for p in o] for o in out]


def targets(parts, origin, yaw_deg, table_z):
    """{part: (centre xyz, yaw deg)}: rows on their supports' tops, GAP between neighbours."""
    rot = Rotation.from_euler("z", yaw_deg, degrees=True)
    bottom, out = {}, {}
    for p in parts:
        size = SIZES[p.category]
        z0 = max((bottom[s] + SIZES[by_name(parts, s).category][2] for s in p.supports), default=0.0)
        bottom[p.name] = z0
        xy = np.asarray(origin, float) + rot.apply([p.x, 0.0, 0.0])[:2]
        out[p.name] = (np.r_[xy, table_z + z0 + size[2] / 2], float(yaw_deg))
    return out


def by_name(parts, name):
    return next(p for p in parts if p.name == name)


def grip_offset(category, low_grip, tilt):
    """Grip point height above the block's bottom: near the top (the arch's crown is only 9.6 mm thick), or low enough
    that the pads cover the prism's broad lower faces; the fingertips (FINGER_TIPS further along the fingers, `tilt`
    deg down) stay TABLE_CLEAR above the bottom either way."""
    h = SIZES[category][2]
    tips = FINGER_TIPS * np.sin(np.deg2rad(tilt)) + TABLE_CLEAR
    if tilt >= VERTICAL and not low_grip:
        # straight-down pads pinching from the fingertips up to about the top hold the block below its centre of mass
        # and it rolls over in the hand on the lift: fingertips at mid-height
        return float(max(tips, h * GRIP_TIPS.get(category, VERTICAL_TIPS) + FINGER_TIPS))
    return float(max(tips, FINGER_TIPS * np.sin(np.deg2rad(tilt)) + LOW_GRIP if low_grip else h - TOP_GRIP))


def wrist_yaw(block_yaw, category, side):
    """Wrist yaw (deg) closing the pads across the block's local y, nearest the arm's inward ARM_YAW."""
    sym = KINDS[category][2]
    across = block_yaw + 90.0  # world angle of the block's local y
    want = ARM_YAW[side]
    step = min(sym, 180.0)  # the jaw is symmetric: turning it by 180 grips the same faces
    return float(want + (across - want + step / 2) % step - step / 2)


def hand_rot(yaw, tilt):
    """scripted_demo.TOP_ROT tilted `tilt` deg below horizontal and turned `yaw` deg (reach_map.hand_rot)."""
    return (
        Rotation.from_euler("z", yaw, degrees=True)
        * Rotation.from_euler("x", -tilt, degrees=True)
        * Rotation.from_euler("z", 90, degrees=True)
    )


TILTS = (60.0, 45.0, 30.0)  # fingers below horizontal; the prism wants the flattest (its broad faces are low)
HOVER = (0.04, 0.07)  # m above the grasp / release point, drawn per subtask
# per-frame phase labels (privileged/phase indexes this); "idle" before the first reach and between subtasks
PHASES = ("idle", "approach", "descend", "close", "lift", "transport", "lower", "release", "withdraw", "retreat")
# per-frame events (privileged/event, 0 = none): the failures a "recover" subtask recovers from, and the v4
# perturbations' bounds
EVENTS = ("none", "grasp_fail", "drop", "perturbed", "noise_on", "noise_off")
FAILURES = ("grasp_fail", "drop", "perturbed")
HOVER_MIN = 0.025
LIFT = 0.05  # least lift off the supply spot
OVER = 0.02  # held block's bottom above the house's top while carried
# grip point at most this far above the table while carrying: higher over the build site the arms run out of reach and
# lag a held roof through the house (seen at 0.95 m, tabletop 0.763 m)
CEIL = 0.16
PLACE_GAP = 0.004  # scripted_demo.PLACE_GAP


def place_hover(place_z, hover, table_z):
    """`hover` lowered so the hover point stays below CEIL, but at least HOVER_MIN."""
    return float(max(HOVER_MIN, min(hover, table_z + CEIL - place_z)))


def waypoints(part, src, tgt, side, tilt, hover, house_top, table_z, lift_up=0.0):
    """Nominal (grip point, hand rotation) of one pick-and-place: above the block, on it, lifted to the carry height,
    `lift_up` higher halfway over (the carry detour), above the target, on it. `src` / `tgt` are (centre xyz, yaw deg)
    of the block on the table / in the shape."""
    size = SIZES[part.category]
    up = grip_offset(part.category, part.low_grip, tilt)
    pick = np.r_[src[0][:2], src[0][2] - size[2] / 2 + up]
    place = np.r_[tgt[0][:2], tgt[0][2] - size[2] / 2 + up + PLACE_GAP]
    drop = place_hover(place[2], hover, table_z)
    carry = max(pick[2] + LIFT, place[2] + drop, house_top + OVER + up)
    a = hand_rot(wrist_yaw(src[1], part.category, side), tilt)
    # a part set into a container keeps the hand's turn (check() ignores its yaw; turning a fruit or a cylinder to
    # the target's yaw took the wrist roll up to 180 deg, into its limit, and the hand off the path into the torso)
    b = a if part.into else hand_rot(wrist_yaw(tgt[1], part.category, side), tilt)
    via = np.r_[(pick[:2] + place[:2]) / 2, max(carry, min(carry + lift_up, table_z + CEIL))]
    return [
        (pick + [0, 0, hover], a),
        (pick, a),
        (np.r_[pick[:2], carry], a),
        (via, Slerp([0, 1], Rotation.concatenate([a, b]))(0.5)),
        (np.r_[place[:2], carry], b),
        (place + [0, 0, drop], b),
        (place, b),
    ]


def feasible(arm, part, src, tgt, side, house_top, table_z, tilts=TILTS, hovers=HOVER, lift_up=0.03):
    """Finger tilts at which `arm` (reach_map.Arm, URDF kinematics) reaches the pick-and-place at every hover, the
    hand clear of the torso all the way (clearance.BODY_CLEAR: the big Aurora Lite over a block near the table's
    front edge)."""
    from scripts.g1d.clearance import path_clear

    def ok(t, h):
        qs = arm.follow(waypoints(part, src, tgt, side, t, h, house_top, table_z, lift_up))
        return qs is not None and path_clear(arm, side, qs)

    return [t for t in tilts if all(ok(t, h) for h in hovers)]


def shape_entry(layout):
    """The drawn shape {shape, origin, yaw_deg, table_z, parts: {part: target}} from the "build" field of its blocks'
    Rigid entries (the scene manager spawns every other top-level layout key as objects); None without one."""
    frame, parts, fixed = None, {}, {}
    for entries in layout.get("Rigid", {}).values():
        for e in entries:
            if e.get("build"):
                frame = e["build"]["frame"]
                info = dict(label=e["label"], category=kind(e["category"], e["category_idx"]), color=e["category_idx"])
                if "fixed" in e["build"]:  # a container that stays where it is
                    fixed[e["build"]["fixed"]] = dict(info, pos=list(e["default_pos"]))
                else:
                    parts[e["build"]["part"]] = dict(e["build"]["target"], **info)
    if not parts:
        return None
    return dict(frame, parts=parts, **({"fixed": fixed} if fixed else {}))


def parts_of(entry):
    """The entry's shape template with each part's kind from the layout (a template part's may be open)."""
    from dataclasses import replace

    return tuple(replace(p, category=entry["parts"][p.name]["category"]) for p in SHAPES[entry["shape"]])


def thing(entry, name):
    """Layout info of a part or a fixed object by name."""
    return entry["parts"].get(name) or entry.get("fixed", {})[name]


def release_targets(parts, fixed, offsets, yaw_deg=0.0):
    """{part: (centre xyz, yaw deg)} where the parts set into containers are let go (layouts' reach check; the expert
    measures the containers instead): `fixed` {name: (centre xyz, kind)}, `offsets` {part: xy off its container's
    centre}. A container part stands NEST_RISE above the top of its own container."""

    def where(name):
        if name in fixed:
            pos, k = fixed[name]
            return np.asarray(pos[:2], float), float(pos[2] + SIZES[k][2] / 2)
        p = by_name(parts, name)
        xy, top = where(p.into)
        return xy + offsets.get(name, 0.0), top + NEST_RISE

    out = {}
    for p in parts:
        xy, top = where(p.into)
        out[p.name] = (np.r_[xy + offsets.get(p.name, 0.0), top + p.release_dz + SIZES[p.category][2] / 2], float(yaw_deg))
    return out


def _inside(part, pos, rot, cpos, crot, csize, ckind, cbottom_top):
    """{ok, xy, z, tilt} of a part in its container: `cbottom_top` the container's (bottom, top) z."""
    spec = CONTAINERS[ckind]
    local = crot.inv().apply(np.asarray(pos, float) - cpos)
    if spec["region"] == "circle":
        xy_err = float(np.linalg.norm(local[:2]))
        in_xy = xy_err < spec["radius"]
    else:
        over = np.abs(local[:2]) - (np.asarray(csize[:2]) / 2 - spec["margin"])
        xy_err = float(max(0.0, over.max()))
        in_xy = bool((over < 0).all())
    bottom = float(pos[2] - SIZES[part["category"]][2] / 2)
    z_err = float(bottom - (cbottom_top[1] - spec["inside"]))
    tilt = float(np.degrees(np.arccos(np.clip(rot.apply([0, 0, 1])[2], -1, 1))))
    ok = in_xy and z_err < 0 and (tilt < TILT_TOL or not part.get("upright", True))
    return dict(xy=xy_err, z=max(0.0, z_err), yaw=0.0, tilt=tilt, ok=bool(ok))


def check(entry, poses):
    """Per part {ok, xy, z, yaw, tilt} against the layout's BuildShape `entry`; `poses` {label: (xyz, wxyz)} (with the
    containers' for parts set into one)."""
    out = {}
    for name, part in entry["parts"].items():
        pos, quat = poses[part["label"]]
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        if part.get("into"):
            c = thing(entry, part["into"])
            cpos, cquat = poses[c["label"]]
            crot = Rotation.from_quat([cquat[1], cquat[2], cquat[3], cquat[0]])
            h = SIZES[c["category"]][2]
            # the container's centre: the bbox centre of every asset here is its origin
            cz = float(cpos[2])
            out[name] = _inside(part, pos, rot, np.asarray(cpos, float), crot, SIZES[c["category"]], c["category"],
                                (cz - h / 2, cz + h / 2))  # fmt: skip
            continue
        up = rot.apply([0, 0, 1])
        yaw = float(np.degrees(np.arctan2(*rot.apply([1, 0, 0])[1::-1])))
        err = dict(
            xy=float(np.linalg.norm(np.asarray(pos[:2]) - part["pos"][:2])),
            z=float(abs(pos[2] - part["pos"][2])),
            yaw=yaw_diff(yaw, part["yaw_deg"], KINDS[part["category"]][2]),
            tilt=float(np.degrees(np.arccos(np.clip(up[2], -1, 1)))),
        )
        err["ok"] = err["xy"] < POS_TOL and err["z"] < Z_TOL and err["yaw"] < YAW_TOL and err["tilt"] < TILT_TOL
        out[name] = err
    return out


def built(entry, poses):
    return all(e["ok"] for e in check(entry, poses).values())


def instruction(entry, zh=False):
    """The whole build, naming every block by colour (unique per kind on the table)."""
    p = {name: block_name(part["category"], part["color"], zh) for name, part in entry["parts"].items()}
    fixed = {name: block_name(f["category"], f["color"], zh) for name, f in entry.get("fixed", {}).items()}
    if entry.get("shape") == "sort":
        colour = {s: (COLORS_ZH if zh else COLORS)[entry["parts"][f"{s}_cube"]["color"]] for s in ("left", "right")}
        if zh:
            return f"按颜色把积木分拣到托盘里：{colour['left']}的积木放进左边的托盘，{colour['right']}的积木放进右边的托盘。"
        return (
            f"Sort the blocks by colour: put the {colour['left']} blocks into the left tray and the "
            f"{colour['right']} blocks into the right tray."
        )
    if entry.get("shape") == "nest":
        if zh:
            return f"把杯子叠起来：先把{p['mid']}放进{fixed['base']}里，再把{p['top']}放进{p['mid']}里。"
        return f"Stack the cups: put the {p['mid']} into the {fixed['base']}, then the {p['top']} into the {p['mid']}."
    if entry.get("shape") == "box":
        return f"把{p['item']}放进{fixed['box']}。" if zh else f"Put the {p['item']} into the {fixed['box']}."
    if entry.get("shape") == "house4":
        if zh:
            return (
                f"用积木搭一座房子：底层从左到右依次放{p['base_left']}、{p['base_mid']}和{p['base_right']}，"
                f"再把{p['roof']}放在{p['base_mid']}上面作为屋顶。"
            )
        return (
            f"Build a house with the wooden blocks: put the {p['base_left']}, the {p['base_mid']} and the "
            f"{p['base_right']} in a row from left to right and the {p['roof']} on top of the {p['base_mid']} as the roof."
        )
    if zh:
        return (
            f"用积木搭一座房子：底层从左到右依次放{p['base_left']}、{p['base_mid']}和{p['base_right']}，"
            f"第二层放{p['mid_left']}和{p['mid_right']}，再把{p['beam']}横放在上面，最上面放{p['roof']}作为屋顶。"
        )
    return (
        f"Build a house with the wooden blocks: put the {p['base_left']}, the {p['base_mid']} and the "
        f"{p['base_right']} in a row from left to right, the {p['mid_left']} and the {p['mid_right']} on top of them, "
        f"the {p['beam']} across the second layer and the {p['roof']} on top as the roof."
    )


# sort / box, the detailed texts: how the objects look (colour, shape: en, zh); a block's colour is its paint
LOOKS = {
    "sand3d_food_and_drink/00105": ("red", "红色", "round", "圆形"),
    "sand3d_food_and_drink/00183": ("brown", "棕色", "oval", "椭圆形"),
    "sand3d_food_and_drink/00202": ("green", "绿色", "round", "圆形"),
    "sand3d_food_and_drink/00223": ("beige", "米色", "mushroom-shaped", "蘑菇形"),
    "sand3d_food_and_drink/00293": ("orange", "橙色", "round", "圆形"),
    "sand3d_food_and_drink/00298": ("red", "红色", "round", "圆形"),
    "storage_box/00000": ("light grey", "浅灰色", "open rectangular", "敞口长方形"),
}
DETAILED = ("sort", "box")
NEAR_Y = -0.495  # env-local y: nearer the robot than this is "near the robot" (the supply spans y -0.59..-0.40)


def spot_text(pos, zh=False):
    """Where on the table an object lies, from the robot's view: left / right of the middle, near / far."""
    left, near = pos[0] < 0, pos[1] < NEAR_Y
    if zh:
        return f"桌面{'左' if left else '右'}侧{'靠近机器人' if near else '离机器人较远'}处"
    return f"on the {'left' if left else 'right'} side of the table {'near the robot' if near else 'farther from the robot'}"


def size_text(kind, zh=False):
    cm = round(float(max(SIZES[kind][:2])) * 100)
    return f"约{cm}厘米大小" if zh else f"about {cm} cm across"


def describe(entry, name, pos, zh=False):
    """A sort / box part named in full: colour, shape, size and where it lies (`pos`, env-local)."""
    part = entry["parts"][name]
    k = kind(*part["category"].split("/")) if "/" in part["category"] else part["category"]
    noun = block_name(k, part["color"], zh)
    if k in LOOKS:
        c_en, c_zh, s_en, s_zh = LOOKS[k]
        look = f"{c_zh}、{s_zh}，{size_text(k, True)}" if zh else f"{c_en}, {s_en}, {size_text(k)}"
    else:  # a painted block: its colour and shape are in its name
        look = size_text(k, zh)
    if zh:
        return f"{spot_text(pos, True)}的{noun}（{look}）"
    return f"the {noun} ({look}) {spot_text(pos)}"


def container_text(entry, name, zh=False):
    """A part's container in full: a tray by its side and the colour card behind it, the box by its look."""
    into = by_name(SHAPES[entry["shape"]], name).into
    if entry["shape"] == "sort":
        side = into.split("_")[1]
        card = (COLORS_ZH if zh else COLORS)[entry["parts"][f"{side}_cube"]["color"]]
        if zh:
            return f"{'左' if side == 'left' else '右'}边的木托盘（后面是{card}卡片）"
        return f"the {side} wooden tray (the one with the {card} card behind it)"
    c_en, c_zh, s_en, s_zh = LOOKS["storage_box/00000"]
    if zh:
        return f"桌子中间{c_zh}的{s_zh}收纳盒"
    return f"the {c_en} {s_en} storage box in the middle of the table"


DETAILED_HEADS = {
    "sort": ("Sort the blocks by colour into the two wooden trays in the middle of the table: ",
             "按颜色把积木分拣到桌子中间的两个木托盘里："),
    "box": ("", ""),
}  # fmt: skip


def detailed_instruction(entry, order, where, zh=False):
    """ordered_instruction of a sort / box episode, each object in full (describe, from `where`: part -> its start
    position) and its container in full."""
    steps = []
    for i, name in enumerate(order):
        obj, into = describe(entry, name, where[name], zh), container_text(entry, name, zh)
        if zh:
            word = "" if len(order) == 1 else "先" if i == 0 else "最后" if i == len(order) - 1 else ("再", "然后")[(i - 1) % 2]
            steps.append(f"{word}拿起{obj}，放进{into}")
        else:
            word = "" if len(order) == 1 else "first " if i == 0 else "and finally " if i == len(order) - 1 else "then "
            steps.append(f"{word}pick up {obj} and put it into {into}")
    head = DETAILED_HEADS[entry["shape"]][int(zh)]
    if zh:
        return head + "；".join(steps) + "。"
    text = head + ", ".join(steps) + "."
    return text[0].upper() + text[1:]


def ordered_instruction(entry, order, zh=False, where=None):
    """The whole build step by step in `order` (part names, e.g. the order the episode was built in), each block
    named by colour and put where the subtask captions put it. `entry["parts"]` needs category and colour per part.
    Sort / box with `where` (part -> start position): detailed_instruction."""
    if where is not None and entry.get("shape") in DETAILED:
        return detailed_instruction(entry, order, where, zh)
    p = {name: block_name(part["category"], part["color"], zh) for name, part in entry["parts"].items()}
    parts = SHAPES[entry["shape"]]
    steps = []
    for i, name in enumerate(order):
        tmpl = by_name(parts, name)
        where = place_text(entry, tmpl, zh)
        if zh:
            word = "先" if i == 0 else "最后" if i == len(order) - 1 else ("再", "然后")[(i - 1) % 2]
            verb = "" if tmpl.into else "放到"
            steps.append(f"{word}把{p[name]}{verb}{where}")
        else:
            word = "first" if i == 0 else "and finally" if i == len(order) - 1 else "then"
            steps.append(f"{word} put the {p[name]} {where}")
    if len(order) == 1:
        steps = [s.removeprefix("先").removeprefix("first ") for s in steps]
    head = TASK_HEADS[entry["shape"]][int(zh)]
    if zh:
        return head + "，".join(steps) + "。"
    text = head + ", ".join(steps) + "."
    return text[0].upper() + text[1:]


# ordered_instruction's lead-in per shape: en, zh
TASK_HEADS = {
    "house": ("Build a house with the wooden blocks: ", "用积木搭一座房子："),
    "house4": ("Build a house with the wooden blocks: ", "用积木搭一座房子："),
    "sort": ("Sort the blocks by colour into the trays: ", "按颜色把积木分拣到托盘里："),
    "nest": ("Stack the cups: ", "把杯子叠起来："),
    "box": ("", ""),
}


def place_text(entry, tmpl, zh=False):
    """Where a part goes in words: its template's en / zh with the supports and the container named."""
    names = {n: block_name(i["category"], i["color"], zh) for n, i in [*entry["parts"].items(),
                                                                       *entry.get("fixed", {}).items()]}  # fmt: skip
    sup = [names[s] for s in tmpl.supports]
    text = tmpl.zh if zh else tmpl.en
    return text.format(*sup, into=names.get(tmpl.into, ""))


def table_side(x, house_x, zh=False):
    left = x < house_x
    if zh:
        return "左侧" if left else "右侧"
    return "on the left side of the table" if left else "on the right side of the table"


VIEW_EN = {
    "composed": "This is a composed robot video with two wrist views in the right side.",
    "head": "This is a video from the robot's head camera.",
}


def caption(entry, part_name, arm, other_visible, src_x, view="composed", zh=False):
    """VPP2-style caption of one pick-and-place: the gripper, the block (colour, kind, where it lies at the start),
    where it goes (named by the blocks it rests on), the other gripper, the view. Sort / box with `src_x` the whole
    start position: the object and its container in full (describe, container_text)."""
    part = entry["parts"][part_name]
    tmpl = by_name(SHAPES[entry["shape"]], part_name)
    house_x = entry["origin"][0]
    other = "right" if arm == "left" else "left"
    if entry.get("shape") in DETAILED and np.ndim(src_x) == 1:
        obj, into = describe(entry, part_name, src_x, zh), container_text(entry, part_name, zh)
        if zh:
            side = {"left": "左", "right": "右"}
            rest = "保持不动" if other_visible else "不在画面中"
            return f"机器人{side[arm]}夹爪拿起{obj}，放进{into}，{side[other]}夹爪{rest}。"
        rest = "stays still" if other_visible else "is not visible in the video"
        return f"The {arm} robot gripper picks up {obj} and puts it into {into}, the {other} gripper {rest}. {VIEW_EN[view]}"
    src_x = src_x[0] if np.ndim(src_x) == 1 else src_x
    if zh:
        where = place_text(entry, tmpl, zh=True)
        side = {"left": "左", "right": "右"}
        rest = "保持不动" if other_visible else "不在画面中"
        verb = "" if tmpl.into else "放到"
        return (
            f"机器人{side[arm]}夹爪拿起桌面{table_side(src_x, house_x, True)}的"
            f"{block_name(part['category'], part['color'], True)}，{verb}{where}，{side[other]}夹爪{rest}。"
        )
    where = place_text(entry, tmpl)
    rest = "stays still" if other_visible else "is not visible in the video"
    return (
        f"The {arm} robot gripper picks up the {block_name(part['category'], part['color'])} "
        f"{table_side(src_x, house_x)} and places it {where}, the {other} gripper {rest}. {VIEW_EN[view]}"
    )


def retreat_caption(arm, other_visible, view="composed", zh=False):
    """Caption of the arm going back out to its side (scripted_demo's park pose, beside the start pose) after its last
    block before an arm swap."""
    other = "right" if arm == "left" else "left"
    if zh:
        side = {"left": "左", "right": "右"}
        rest = "保持不动" if other_visible else "不在画面中"
        return f"机器人{side[arm]}夹爪收回到{side[arm]}侧让开，{side[other]}夹爪{rest}。"
    rest = "stays still" if other_visible else "is not visible in the video"
    return f"The {arm} robot gripper moves back out of the way to the {arm} side, the {other} gripper {rest}. {VIEW_EN[view]}"


RECOVER_EN = {
    "grasp_fail": "fails to hold the {} and grasps it again",
    "drop": "drops the {} on the way and picks it up again",
    "perturbed": "finds the {} moved and reaches for it again",
}
RECOVER_ZH = {"grasp_fail": "没抓稳{}，重新抓取", "drop": "搬运途中{}掉落，重新捡起", "perturbed": "发现{}被挪动，重新去抓"}


def recover_caption(failure, category, color, arm, view="composed", zh=False):
    """Caption of a recovery: from a failure (FAILURES) until the block is lifted and carried again."""
    if zh:
        side = {"left": "左", "right": "右"}
        return f"机器人{side[arm]}夹爪{RECOVER_ZH[failure].format(block_name(category, color, True))}。"
    return f"The {arm} robot gripper {RECOVER_EN[failure].format(block_name(category, color))}. {VIEW_EN[view]}"
