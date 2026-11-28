#!/usr/bin/env python3
"""Layouts for the small house (build_shapes.HOUSE4: a cube, the arch and a cube in a row, the roof on the arch) with
fixed colours: red and blue cubes, a natural wood arch, an orange roof; no distractors. Same task as the seven-block
house (g1d_build_house: expert, plugin and check read the shape from the layout).

Against the shortcuts of the seven-block layouts (build_house_layouts.py), where the row's left half always lay on the
left (left arm) and its right half on the right, and every block lay in a narrow band near the table's front edge:
- which cube goes to which end is drawn per layout (red-left or blue-left, half each);
- the cube at an end of the row lies on the other side of the table ("across": the arm on that side carries it to the
  far end of the row) in ACROSS of the layouts, at most one of the two cubes (the arm setting a cube down at the far
  end sweeps its forearm over the rest of the row, so that cube goes first, before the arch and the other cube: a
  "before" rule scripted_demo.build_actions follows; with both cubes across neither could go first); the arch and the
  roof lie on either side, half each;
- a block lies anywhere on its side its arm reaches (SUPPLY_X / SUPPLY_Y), in front of and behind the house too, but
  not in the build area: the house's footprint grown by BUILD_MARGIN (room for the hands).
The arm is the one on the block's side (reaching across the table for a block, the simulated arm pushes it away
instead of lifting it). A draw is kept only when each block's arm reaches its whole pick-and-place
(build_shapes.feasible, URDF kinematics); the side is drawn first and only the spot is redrawn.

The room is one of the school lab's appearance variants (scripts/g1d/school_variants.py, ROOMS): lights, shadows,
tabletop and green-screen colour drawn around the real robot's head camera; a layout adds at most one dim extra
light.

Usage (any python with numpy/scipy/yaml, repo root):
    python scripts/g1d/build_house4_layouts.py --seed 4000 --count 2000 --split train
    python scripts/g1d/build_house4_layouts.py --seed 4000 --summary      # side balance of a written folder
    python scripts/g1d/build_house4_layouts.py --seed 4012 --shape house --fixed-house --tilts 90 --reach-slack 0.03 --v4a  # task2
    # the tape-measured school table, house4 in the black tape rectangle (real_scene4_school_v2: x -0.080..0.128,
    # y -0.400..-0.337 in the env frame; the row 1.2 cm toward the robot of its middle, where both arms reach their
    # end), its base room only
    python scripts/g1d/build_house4_layouts.py --seed 4021 --config g1d_school_v2 --house-at 0.024 -0.38 0 \
        --rooms real_scene4_v2 --tilts 90 60 --no-across --v4
"""

import argparse
import collections
from functools import partial
import json
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.g1d import build_shapes as bs  # noqa: E402
from scripts.g1d.build_asset import SCENES, room_entry  # noqa: E402
from scripts.g1d.build_house_layouts import entry, footprint, house_top_before  # noqa: E402
from scripts.g1d.reach_map import HAND_NAME, Arm  # noqa: E402

TASK = "g1d_build_house"
CONFIG = "g1d_school"
ROOMS = [f"school_v3_{k}" for k in range(32)]  # --v4: school_v4_<k> (the same lights and colours, wood / weave maps)
HOUSE_X, HOUSE_Y, HOUSE_YAW = (-0.06, 0.06), (-0.55, -0.48), 15.0  # m, m, +-deg
# x off the house's centre (outward) and y of a supply block's centre; the table's front edge is at y -0.60, +y away
# from the robot (v2: x 0.14-0.32, y -0.59 to -0.45)
# (URDF reach: no arm gets a block further than about y -0.42 down to the table and back over the house)
SUPPLY_X, SUPPLY_Y = (0.0, 0.40), (-0.59, -0.40)
# m a block's footprint keeps from the table's front edge (TABLE_FRONT): the Aurora Lite's open jaws reach 7.5 cm to
# the side of the grip point, over a block at the edge into the torso (its front at y -0.653); feasible() also keeps
# the hand clearance.BODY_CLEAR off the torso all along the pick-and-place
TABLE_FRONT = -0.60
EDGE_GAP = {"aurora_lite": 0.03}.get(HAND_NAME, 0.0)
# m kept free around the house's footprint (house frame: along the row / across it), plus the block's own radius
BUILD_MARGIN = (0.05, 0.05)
ACROSS = {"none": 0.2, "base_left": 0.4, "base_right": 0.4}  # which end cube (if any) lies across the table
SUPPLY_YAW = 35.0
CLEAR = 0.045  # the pads approach fully open (95 mm apart)
TRIES = 2000  # spots per block: a cube reached across the table has a small region
# finger tilts (deg below horizontal) a draw may use, written to the frame for scripted_demo; --tilts 90: every block
# picked and placed with the fingers straight down
GRASP_TILTS = bs.TILTS
# m: the place must also be reached this much further from the arm (x): straight-down fingers reach less, and at the
# URDF's edge the simulated arm stops ~6 cm short of a far-end place (v3e pilot); 0 with the default tilts (v3d)
REACH_SLACK = 0.0
# --order cube-arch-cube: one end cube, the arch next to it, then the other end cube, the roof last (v3f: on the real
# robot an arch set down between two standing cubes rarely fits); the first cube is the one across, if any, else
# either end half each
ORDER = "any"
CUBE, ARCH, ROOF = "wood_block_cube", "wood_block_arch", "wood_block_prism"
NATURAL, BLUE, RED, GREEN, ORANGE = 0, 1, 2, 4, 6  # build_shapes.COLORS
# --shape house (v4a task2): the seven-block house (build_shapes.HOUSE) with fixed colours and places: red cube, natural
# wood arch, blue cube left to right below, the green cube and a natural wood cube on them, the natural wood long block
# across, the orange roof on top
SHAPE = "house4"
FIXED = False
HOUSE7_COLORS = dict(base_left=RED, base_mid=NATURAL, base_right=BLUE, mid_left=GREEN, mid_right=NATURAL, beam=NATURAL,
                     roof=ORANGE)  # fmt: skip
FIXED_HOUSE = (0.0, -0.515, 0.0)  # --fixed-house (task2): centre x, y (m) and yaw (deg) of every layout's house, the
# middle of HOUSE_X / HOUSE_Y with the row square to the robot: the build site is the same spot on every layout
SECOND = ("mid_left", "mid_right")  # second layer: the cube set down by the far arm goes first, never both
LIGHTS = dict(count=(0, 1), types=("Sphere", "Rect", "Disk"), intensity=(0.1, 0.4), temperature=(4500, 6500))
LIGHT_REF = {"Sphere": 3e4, "Disk": 3e4, "Rect": 3e4}
LIGHT_BOX = ((-0.8, 0.8), (-1.0, 0.4), (1.8, 2.6))
LIGHT_TILT = 20
LIGHT_SHAPE = {"Sphere": {"radius_range": (0.05, 0.2)}, "Disk": {"radius_range": (0.05, 0.2)},
               "Rect": {"width_range": (0.3, 1.0), "height_range": (0.3, 1.0)}}  # fmt: skip
# --v4 (docs/superpowers/specs/2026-11-11-g1d-house4-v4-design.md), drawn from their own stream so the blocks and the
# house come out as without it: up to two extra lights of a wider colour and power; per block mass (x the asset's)
# and friction (the asset's 0.5 / 0.4 inside the range); the expert's perturbations (scripted_demo, frame "v4")
V4 = False
# layout "Robot" entry ({"init_joint_pos": {joint: rad}}): the env places those joints at reset instead of driving
# them there from the spawn pose (an arm rising from hanging straight down catches under the tabletop)
ROBOT = None
V4_LIGHTS = dict(count=(0, 2), types=("Sphere", "Rect", "Disk"), intensity=(0.05, 0.5), temperature=(4000, 6500))
V4_MASS, V4_STATIC, V4_DYNAMIC = (0.7, 1.3), (0.4, 0.8), (0.75, 0.9)  # dynamic: x the static friction
V4_INIT_JITTER = (0.03, 0.1)  # m, rad: each arm's start pose (uniform per axis)
V4_PERTURB = dict(p=0.3, most=2, shift=(0.02, 0.05), yaw=20.0)  # per pick: the block moved under the hand
V4_NOISE = dict(p=0.5, sigma_m=0.005, sigma_deg=1.5, tau_s=0.3, length_s=(1.0, 3.0))  # per episode, OU on the target
# --v4a (implies --v4): the rooms school_v4a_<k> (school_variants.py --v4a: cloth and tabletop colours moved to the real
# head camera's, low-frequency cloth weave, twice the variants) and a smooth drift instead of a jitter for the
# execution noise (tau 0.3 s read as high-frequency shaking in the v4 demo; 0.8 s looks like real tracking error)
V4A_ROOMS, V4A_NOISE_TAU = 64, 0.8
# --fixed-colors: red cube at the right end, wood arch, blue cube at the left end, orange roof, built in that order
FIXED_COLORS = False
# --keepout X0 X1 Y0 Y1 (env frame): no supply block's footprint within KEEPOUT_GAP of it (school_v2: the black
# rectangle taped on the table)
KEEPOUT, KEEPOUT_GAP = None, 0.02
# --supply-near X0 X1 Y0 Y1 P: each block is first looked for in this |x - house x| / y region with probability P (the
# real school_v2 episodes' blocks: 8-15 cm beside the house centre, in front of or beside the tape rectangle), else
# (or when no spot there passes) in SUPPLY_X / SUPPLY_Y
SUPPLY_NEAR = None
# m: the room (table, tape) and the fixed house shifted together along y by up to this per layout (frame["stance_dy"]),
# i.e. the robot standing that much closer to / farther from the table; a shift the arms cannot build at is drawn again
STANCE = 0.0


def draw(rng, arms, table_z):
    parts = bs.SHAPES[SHAPE]
    origin = [rng.uniform(*HOUSE_X), rng.uniform(*HOUSE_Y)]
    yaw = rng.uniform(-HOUSE_YAW, HOUSE_YAW)
    dy = float(rng.uniform(-STANCE, STANCE)) if STANCE else 0.0
    keepout = KEEPOUT and (KEEPOUT[0], (KEEPOUT[1][0] + dy, KEEPOUT[1][1] + dy))
    if FIXED:
        origin, yaw = [FIXED_HOUSE[0], FIXED_HOUSE[1] + dy], FIXED_HOUSE[2]
    goal = bs.targets(parts, origin, yaw, table_z)
    if SHAPE == "house":
        colors = dict(HOUSE7_COLORS)
    elif FIXED_COLORS:
        colors = dict(base_left=BLUE, base_mid=NATURAL, base_right=RED, roof=ORANGE)
    else:
        left_cube, right_cube = (RED, BLUE) if rng.random() < 0.5 else (BLUE, RED)
        colors = dict(base_left=left_cube, base_mid=NATURAL, base_right=right_cube, roof=ORANGE)
    across = str(rng.choice(list(ACROSS), p=list(ACROSS.values())))
    sides = {p.name: str(rng.choice(["left", "right"])) for p in parts}
    sides["base_left"] = "right" if across == "base_left" else "left"
    sides["base_right"] = "left" if across == "base_right" else "right"
    before = [[across, "base_mid"], [across, "base_right" if across == "base_left" else "base_left"]] if across != "none" else []
    if ORDER == "cube-arch-cube":
        first = across if across != "none" else str(rng.choice(["base_left", "base_right"]))
        other = "base_right" if first == "base_left" else "base_left"
        before = [[first, "base_mid"], [first, other], ["base_mid", other]]
    if FIXED_COLORS:  # red cube, wood arch, blue cube, orange roof
        before = [["base_right", "base_mid"], ["base_right", "base_left"], ["base_mid", "base_left"], ["base_left", "roof"]]
    if SHAPE == "house":
        far = [n for n, home in zip(SECOND, ("left", "right")) if sides[n] != home]
        if len(far) == 2:
            return None
        before += [[far[0], next(n for n in SECOND if n != far[0])]] if far else []
    placed = []
    c, s = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
    half = (max(abs(p.x) + bs.SIZES[p.category][0] / 2 for p in parts), max(bs.SIZES[p.category][1] for p in parts) / 2)

    def in_build_area(xy, r):
        d = xy - origin
        along, across = d[0] * c + d[1] * s, -d[0] * s + d[1] * c
        return abs(along) < half[0] + BUILD_MARGIN[0] + r and abs(across) < half[1] + BUILD_MARGIN[1] + r

    def spot(category, side, check):
        regions = [(SUPPLY_X, SUPPLY_Y, TRIES)]
        if SUPPLY_NEAR and rng.random() < SUPPLY_NEAR[4]:
            regions.insert(0, (SUPPLY_NEAR[:2], SUPPLY_NEAR[2:4], TRIES // 2))
        for sx, sy, tries in regions:
            got = spot_in(category, side, check, sx, sy, tries)
            if got is not None:
                return got
        return None

    def spot_in(category, side, check, sx, sy, tries):
        sign = -1 if side == "left" else 1
        r = footprint(category)
        for _ in range(tries):
            xy = np.array([origin[0] + sign * rng.uniform(*sx), rng.uniform(*sy)])
            if abs(xy[0] - origin[0]) < 0.01 or in_build_area(xy, r) or xy[1] - r < TABLE_FRONT + EDGE_GAP:
                continue
            if keepout and all(lo - r - KEEPOUT_GAP < v < hi + r + KEEPOUT_GAP for v, (lo, hi) in zip(xy, keepout)):
                continue
            if any(np.linalg.norm(xy - q) < r + rq + CLEAR for q, rq in placed):
                continue
            sym = bs.KINDS[category][2]
            turn = rng.uniform(-SUPPLY_YAW, SUPPLY_YAW) if sym < 360 else rng.uniform(0, 360)
            turn += float(rng.choice([0.0, 180.0])) if sym == 360 or sym <= 90 else 0.0
            pos = np.r_[xy, table_z + bs.SIZES[category][2] / 2 + 0.001]
            if check((pos, turn)):
                placed.append((xy, r))
                return pos, turn
        return None

    frame = dict(shape=SHAPE, origin=[float(v) for v in origin], yaw_deg=float(yaw), table_z=table_z, before=before,
                 grasp_tilts=list(GRASP_TILTS))  # fmt: skip
    if HAND_NAME != "dex1":  # reach checked with that gripper (G1D_HAND); the expert refuses another one
        frame["hand"] = HAND_NAME
    if STANCE:
        frame["stance_dy"] = round(dy, 4)
    rigid, tilts_by_part = {}, {}
    for p in sorted(parts, key=lambda p: -footprint(p.category)):
        side = arm = sides[p.name]
        top = house_top_before(parts, goal, p.name, table_z)
        tilts = {}

        def ok(src, p=p, arm=arm, top=top, tilts=tilts):
            tilts["ok"] = bs.feasible(arms[arm], p, src, goal[p.name], arm, top, table_z, tilts=GRASP_TILTS)
            if tilts["ok"] and REACH_SLACK:
                far = (goal[p.name][0] + [REACH_SLACK if arm == "left" else -REACH_SLACK, 0, 0], goal[p.name][1])
                tilts["ok"] = [t for t in tilts["ok"] if bs.feasible(arms[arm], p, src, far, arm, top, table_z, tilts=(t,))]
            return bool(tilts["ok"])

        got = spot(p.category, side, ok)
        if got is None:
            return None
        pos, turn = got
        target = dict(pos=[float(v) for v in goal[p.name][0]], yaw_deg=goal[p.name][1], supports=list(p.supports),
                      arm=arm, side=side, tilts=tilts["ok"])  # fmt: skip
        e = entry(p.category, colors[p.name], p.name, pos, turn)
        rigid.setdefault(p.category, []).append(dict(e, build=dict(frame=frame, part=p.name, target=target)))
        tilts_by_part[p.name] = tilts["ok"]
    return rigid, frame, tilts_by_part


def lights(rng, spec=LIGHTS):
    out = {}
    for n in range(int(rng.integers(spec["count"][0], spec["count"][1] + 1))):
        kind = str(rng.choice(spec["types"]))
        pos = [round(float(rng.uniform(*r)), 3) for r in LIGHT_BOX]
        ori = [round(float(rng.uniform(-LIGHT_TILT, LIGHT_TILT)), 1) for _ in range(2)] + [0.0]
        temp = round(float(rng.uniform(*spec["temperature"])))
        power = round(LIGHT_REF[kind] * float(rng.uniform(*spec["intensity"])), 1)
        shape = {k: [v := round(float(rng.uniform(*r)), 3), v] for k, r in LIGHT_SHAPE[kind].items()}
        out[f"light_{n}"] = {
            "types": kind,
            "common": {"initial_pos_range": pos + pos, "initial_ori_range": ori + ori,
                       "color_temperature_range": [temp, temp]},
            kind: {"intensity_range": [power, power], **shape},
        }  # fmt: skip
    return out


def v4_draws(rng, rigid, frame):
    """--v4: block physics into the Rigid entries, the expert's perturbations into the frame (frame["v4"]); the
    extra lights, returned."""
    for entries in rigid.values():
        for e in entries:
            base = json.loads((bs.BLOCKS / e["category"] / f"{e['category_idx']:05d}" / "runtime_config.json").read_text())
            static = float(rng.uniform(*V4_STATIC))
            e["physics"] = dict(type="rigid", mass=round(base["physics"]["mass"] * float(rng.uniform(*V4_MASS)), 5),
                                static_friction=round(static, 3),
                                dynamic_friction=round(static * float(rng.uniform(*V4_DYNAMIC)), 3), restitution=0.0)  # fmt: skip
    m, r = V4_INIT_JITTER
    jitter = {s: dict(shift=[round(float(v), 4) for v in rng.uniform(-m, m, 3)],
                      turn=[round(float(v), 4) for v in rng.uniform(-r, r, 3)]) for s in ("left", "right")}  # fmt: skip
    parts = [p.name for p in bs.SHAPES[frame["shape"]]]
    moved = [p for p in parts if rng.random() < V4_PERTURB["p"]][: V4_PERTURB["most"]]
    perturb = {p: dict(shift=round(float(rng.uniform(*V4_PERTURB["shift"])), 4),
                       heading_deg=round(float(rng.uniform(0, 360)), 1),
                       yaw_deg=round(float(rng.uniform(-V4_PERTURB["yaw"], V4_PERTURB["yaw"])), 1)) for p in moved}  # fmt: skip
    noise = dict(V4_NOISE, on=bool(rng.random() < V4_NOISE["p"]), seed=int(rng.integers(2**31)))
    frame["v4"] = dict(init_jitter=jitter, perturb=perturb, noise=noise)
    return lights(rng, V4_LIGHTS)


def write_layout(seed, out, arms, table_z, room_pos, i):
    attempt = 0
    while True:
        rng = np.random.default_rng([seed, i, attempt])
        attempt += 1
        if (drawn := draw(rng, arms, table_z)) is not None:
            break
    rigid, frame, tilts = drawn
    room_pos = [room_pos[0], round(room_pos[1] + frame.get("stance_dy", 0.0), 4), room_pos[2]]
    layout = dict(Rigid=rigid, Room=room_entry(str(rng.choice(ROOMS)), room_pos, table_z))
    light = lights(rng)
    if V4:
        light = v4_draws(np.random.default_rng([seed, i, 4]), rigid, frame)
    if light:
        layout["Light"] = light
    if ROBOT:
        layout["Robot"] = ROBOT
    (out / f"{TASK}_{i}.json").write_text(json.dumps(layout, indent=4))
    print(f"{TASK}_{i}: draw {attempt - 1}, origin {np.round(frame['origin'], 4).tolist()}, "
          f"yaw {frame['yaw_deg']:+.1f}, {len(light)} lights, tilts {tilts}", flush=True)  # fmt: skip
    return attempt


def summary(folder):
    """Side balance of a folder: per house place and per block, how often it lies left / right; supply spread."""
    by_part, by_block, xy = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter), []
    by_arm = collections.defaultdict(collections.Counter)
    files = sorted(folder.glob(f"{TASK}_*.json"))
    for f in files:
        layout = json.loads(f.read_text())
        for entries in layout["Rigid"].values():
            for e in entries:
                b = e["build"]
                side = b["target"].get("side", b["target"]["arm"])
                by_part[b["part"]][side] += 1
                by_block[bs.block_name(e["category"], e["category_idx"])][side] += 1
                by_arm[b["part"]][b["target"]["arm"]] += 1
                xy.append(np.array(e["default_pos"][:2]) - b["frame"]["origin"])
    print(f"{len(files)} layouts in {folder}")
    for title, table in (("house place", by_part), ("block", by_block), ("arm", by_arm)):
        for k in sorted(table):
            n = sum(table[k].values())
            print(f"  {title:11s} {k:28s} left {table[k]['left']:5d}  right {table[k]['right']:5d}  "
                  f"left share {table[k]['left'] / n:.2f}")  # fmt: skip
    xy = np.array(xy)
    if len(xy):
        print(f"  supply |x - house x| p5/p50/p95 {np.percentile(np.abs(xy[:, 0]), [5, 50, 95]).round(3)}, "
              f"y - house y p5/p50/p95 {np.percentile(xy[:, 1], [5, 50, 95]).round(3)}, "
              f"in front of / behind the house (|x - house x| < 0.1) {np.mean(np.abs(xy[:, 0]) < 0.1):.2f}")  # fmt: skip


def complete(path):
    try:
        json.loads(path.read_text())
        return True
    except (OSError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True, help="layout seed folder")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--split", choices=("eval", "train"), default="eval")
    ap.add_argument("--assets", type=Path, default=ROOT / "Assets")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--summary", action="store_true", help="only print the side balance of the folder")
    ap.add_argument("--tilts", type=float, nargs="+", help=f"finger tilts allowed (default {bs.TILTS}; 90: straight down)")
    ap.add_argument("--reach-slack", type=float, default=0.0, help="m further from the arm each place must reach")
    ap.add_argument("--order", choices=("any", "cube-arch-cube"), default="any",
                    help="build order the expert keeps to (layout 'before' rules; any: supports and reach only)")
    ap.add_argument("--resume", action="store_true", help="keep the folder's layouts, write the missing ones (a layout's "
                    "draws depend on seed and index only, so they come out as a full run would write them)")
    ap.add_argument("--v4", action="store_true", help="v4 block physics, lights and expert perturbations (V4_*)")
    ap.add_argument("--v4a", action="store_true", help="--v4 in the school_v4a rooms, execution noise tau V4A_NOISE_TAU")
    ap.add_argument("--shape", choices=("house4", "house"), default="house4",
                    help="house: the seven-block house in HOUSE7_COLORS (v4a task2)")
    ap.add_argument("--fixed-house", action="store_true", help="every house at FIXED_HOUSE (position and yaw)")
    ap.add_argument("--house-at", type=float, nargs=3, metavar=("X", "Y", "YAW"),
                    help="every house at this centre (m, env frame) and yaw (deg); implies --fixed-house")
    ap.add_argument("--config", help="env config (default g1d_school; build_asset.SCENES: room, its position, the robot)")
    ap.add_argument("--no-across", action="store_true", help="each end cube on its own end's side (ACROSS none only): "
                    "the far arm cannot reach the far end of a house set far from the robot")
    ap.add_argument("--rooms", nargs="+", help="Room folders to draw from (default: the school variants)")
    ap.add_argument("--fixed-colors", action="store_true", help="red cube right, arch, blue cube left, roof; that order")
    ap.add_argument("--keepout", type=float, nargs=4, metavar=("X0", "X1", "Y0", "Y1"),
                    help="env-frame rectangle no supply block may lie on (KEEPOUT_GAP around it)")
    ap.add_argument("--supply-x", type=float, nargs=2, help="supply |x - house x| range (default SUPPLY_X)")
    ap.add_argument("--supply-y", type=float, nargs=2, help="supply y range (default SUPPLY_Y)")
    ap.add_argument("--supply-near", type=float, nargs=5, metavar=("X0", "X1", "Y0", "Y1", "P"),
                    help="easy-reach |x - house x| / y region each block is first looked for in, with probability P")
    ap.add_argument("--extra-lights", type=int, help="most extra lights a --v4 layout gets (default V4_LIGHTS: 2)")
    ap.add_argument("--shard", default="0/1", help="k/n: write only the indices i with i %% n == k (one job per shard, "
                    "with --resume; collection takes the layouts in index order)")
    args = ap.parse_args()
    global GRASP_TILTS, REACH_SLACK, ORDER, V4, ROOMS, SHAPE, FIXED, FIXED_HOUSE, CONFIG, ACROSS
    global FIXED_COLORS, KEEPOUT, SUPPLY_X, SUPPLY_Y, SUPPLY_NEAR
    FIXED_COLORS = args.fixed_colors
    SUPPLY_NEAR = tuple(args.supply_near) if args.supply_near else None
    KEEPOUT = (tuple(args.keepout[:2]), tuple(args.keepout[2:])) if args.keepout else None
    SUPPLY_X, SUPPLY_Y = tuple(args.supply_x or SUPPLY_X), tuple(args.supply_y or SUPPLY_Y)
    if args.extra_lights is not None:
        V4_LIGHTS["count"] = (0, args.extra_lights)
    if args.no_across:
        ACROSS = {"none": 1.0}
    SHAPE, FIXED, CONFIG = args.shape, args.fixed_house or bool(args.house_at), args.config or CONFIG
    if args.house_at:
        FIXED_HOUSE = tuple(args.house_at)
    V4 = args.v4 or args.v4a
    ROOMS = [r.replace("school_v3_", "school_v4_") for r in ROOMS] if V4 else ROOMS
    if args.v4a:
        ROOMS = [f"school_v4a_{k}" for k in range(V4A_ROOMS)]
        V4_NOISE["tau_s"] = V4A_NOISE_TAU
    ROOMS = args.rooms or ROOMS
    GRASP_TILTS = tuple(args.tilts) if args.tilts else GRASP_TILTS
    REACH_SLACK = args.reach_slack
    ORDER = args.order
    out = args.assets / "Eval_Layout" / "RoboDojo" / CONFIG / str(args.seed)
    if args.summary:
        summary(out)
        return
    if missing := [r for r in ROOMS if not (args.assets / "Room" / r).is_dir()]:
        raise SystemExit(f"no room {missing[0]}: make the variants with scripts/g1d/school_variants.py")
    _, room, room_pos, _ = SCENES[CONFIG]
    table_z = json.loads((args.assets / "Room" / room / "usd" / "build_params.json").read_text())["table"]["top_z"]
    cfg = yaml.safe_load((ROOT / "env_cfg/robot" / f"{CONFIG}.yml").read_text())["robots"][0]
    arms = {side: Arm(side, cfg) for side in ("left", "right")}
    out.mkdir(parents=True, exist_ok=True)
    k, n = map(int, args.shard.split("/"))
    todo = list(range(k, args.count, n))
    if args.resume:
        todo = [i for i in todo if not complete(out / f"{TASK}_{i}.json")]
    else:
        for old in out.glob(f"{TASK}_*.json"):
            old.unlink()
    if args.split == "train":
        (out / "split.json").write_text(json.dumps({"split": "train"}) + "\n")
    job = partial(write_layout, args.seed, out, arms, table_z, room_pos)
    with Pool(max(1, min(args.workers, len(todo)))) as pool:
        draws = sum(pool.imap_unordered(job, todo))
    print(f"wrote {len(todo)} layouts ({draws} draws) to {out}")
    summary(out)


if __name__ == "__main__":
    main()
