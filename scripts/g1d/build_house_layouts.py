#!/usr/bin/env python3
"""Layouts for g1d_build_house: the seven house blocks (build_shapes.HOUSE) and a few distractor blocks scattered over
the two supply areas either side of the build site, in the green-screen school lab (g1d_school).

Per layout: the house's centre and yaw, a colour per block (each kind/colour pair once on the table), which side each
house block lies on (that arm picks it: the row's left half on the left, its right half on the right, the middle
column either) and its spot and yaw. A block is only laid where its arm reaches the whole
pick-and-place at some finger tilt (build_shapes.waypoints, URDF kinematics of reach_map.py). Physics come from the
blocks' runtime_config.json (real mass, wood friction), so the entries carry none.

Usage (any python with numpy/scipy/yaml):
    python scripts/g1d/build_house_layouts.py --seed 0 --count 20            # eval layouts
    python scripts/g1d/build_house_layouts.py --seed 1000 --count 200 --split train
"""

import argparse
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
from scripts.g1d.reach_map import Arm  # noqa: E402

TASK = "g1d_build_house"
CONFIG = "g1d_school"
HOUSE_X, HOUSE_Y, HOUSE_YAW = (-0.05, 0.05), (-0.55, -0.49), 15.0  # m, m, +-deg
# x off the house's centre (outward), y of a supply block's centre; nearer the house a carried block or the hand
# sweeps a supply block into it; nearer the robot than SUPPLY_Y a block nudged on a regrasp is out of the arm's reach
SUPPLY_X, SUPPLY_Y = (0.14, 0.32), (-0.59, -0.45)
SUPPLY_YAW = 35.0  # +-deg off the row's direction, so the wrist turns at most this much from ARM_YAW
CLEAR = 0.035  # m between two blocks' footprint circles on the table: room for the open pads and the fingers
DISTRACTORS = (0, 4)  # inclusive
TRIES = 400
# extra lights over the room's own: count (inclusive), types, intensity share of the type's reference, colour temp
LIGHTS = dict(count=(0, 2), types=("Sphere", "Rect", "Disk"), intensity=(0.2, 0.8), temperature=(3500, 7500))
LIGHT_REF = {"Sphere": 3e4, "Disk": 3e4, "Rect": 3e4}  # as scripts/datagen/rsi_profile.py
LIGHT_BOX = ((-0.8, 0.8), (-1.0, 0.4), (1.8, 2.6))  # m, x / y / z
LIGHT_TILT = 20  # deg about x and y; 0 points the light down
LIGHT_SHAPE = {"Sphere": {"radius_range": (0.05, 0.2)}, "Disk": {"radius_range": (0.05, 0.2)},
               "Rect": {"width_range": (0.3, 1.0), "height_range": (0.3, 1.0)}}  # fmt: skip


def footprint(category):
    return float(np.hypot(*bs.SIZES[category][:2]) / 2)


def dependents(parts, name):
    out, todo = set(), [name]
    while todo:
        n = todo.pop()
        for p in parts:
            if n in p.supports and p.name not in out:
                out.add(p.name)
                todo.append(p.name)
    return out


def house_top_before(parts, goal, name, table_z):
    """Highest top among the parts that may already stand when `name` is placed (all but it and those on it)."""
    skip = dependents(parts, name) | {name}
    return max(
        (goal[p.name][0][2] + bs.SIZES[p.category][2] / 2 for p in parts if p.name not in skip), default=table_z
    )


def entry(category, color, label, pos, yaw):
    return dict(
        category=category,
        category_idx=int(color),
        label=label,
        default_pos=[float(v) for v in pos],
        default_ori=bs.yaw_quat(yaw),
        scale=[1.0, 1.0, 1.0],
        relative_plane="Table",
        need_check_stable=True,
        check_mode="bbox",
        margin=0.01,
        physics={"type": "rigid"},
        visual={},
    )


def draw(rng, arms, table_z):
    parts = bs.HOUSE
    origin = [rng.uniform(*HOUSE_X), rng.uniform(*HOUSE_Y)]
    yaw = rng.uniform(-HOUSE_YAW, HOUSE_YAW)
    goal = bs.targets(parts, origin, yaw, table_z)
    used = set()

    def color_for(category):
        free = [c for c in bs.DRAW_COLORS if (category, c) not in used]
        c = int(rng.choice(free))
        used.add((category, c))
        return c

    colors = {p.name: color_for(p.category) for p in parts}
    # each half of the row comes from its own side: an arm placing at the far end of the row reaches past what the
    # simulated arm tracks (it stalls and sets the block down short); the middle column comes from either side
    sides = {p.name: "left" if p.x < 0 else "right" if p.x > 0 else str(rng.choice(["left", "right"])) for p in parts}
    placed = []  # (xy, radius)

    def spot(category, side, check=None):
        sign = -1 if side == "left" else 1
        r = footprint(category)
        for _ in range(TRIES):
            xy = np.array([origin[0] + sign * rng.uniform(*SUPPLY_X), rng.uniform(*SUPPLY_Y)])
            if any(np.linalg.norm(xy - q) < r + rq + CLEAR for q, rq in placed):
                continue
            sym = bs.KINDS[category][2]
            turn = rng.uniform(-SUPPLY_YAW, SUPPLY_YAW) if sym < 360 else rng.uniform(0, 360)
            turn += float(rng.choice([0.0, 180.0])) if sym == 360 or sym <= 90 else 0.0
            pos = np.r_[xy, table_z + bs.SIZES[category][2] / 2 + 0.001]
            if check is None or check((pos, turn)):
                placed.append((xy, r))
                return pos, turn
        return None

    frame = dict(shape="house", origin=[float(v) for v in origin], yaw_deg=float(yaw), table_z=table_z)
    rigid, tilts_by_part = {}, {}
    for p in sorted(parts, key=lambda p: -footprint(p.category)):
        side = sides[p.name]
        top = house_top_before(parts, goal, p.name, table_z)
        tilts = {}

        def ok(src, p=p, side=side, top=top, tilts=tilts):
            tilts["ok"] = bs.feasible(arms[side], p, src, goal[p.name], side, top, table_z)
            return bool(tilts["ok"])

        got = spot(p.category, side, ok)
        if got is None:
            return None
        pos, turn = got
        target = dict(
            pos=[float(v) for v in goal[p.name][0]],
            yaw_deg=goal[p.name][1],
            supports=list(p.supports),
            arm=side,
            tilts=tilts["ok"],
        )
        e = entry(p.category, colors[p.name], p.name, pos, turn)
        rigid.setdefault(p.category, []).append(dict(e, build=dict(frame=frame, part=p.name, target=target)))
        tilts_by_part[p.name] = tilts["ok"]
    for i in range(int(rng.integers(DISTRACTORS[0], DISTRACTORS[1] + 1))):
        category = str(rng.choice(bs.DISTRACTORS))
        if len({c for c in bs.DRAW_COLORS if (category, c) not in used}) == 0:
            continue
        got = spot(category, str(rng.choice(["left", "right"])))
        if got is None:
            continue
        rigid.setdefault(category, []).append(entry(category, color_for(category), f"distractor_{i}", *got))
    return rigid, frame, tilts_by_part


def lights(rng):
    """Layout "Light" section (scene_manager's Light, every range collapsed to the drawn value); {} for none."""
    out = {}
    for n in range(int(rng.integers(LIGHTS["count"][0], LIGHTS["count"][1] + 1))):
        kind = str(rng.choice(LIGHTS["types"]))
        pos = [round(float(rng.uniform(*r)), 3) for r in LIGHT_BOX]
        ori = [round(float(rng.uniform(-LIGHT_TILT, LIGHT_TILT)), 1) for _ in range(2)] + [0.0]
        temp = round(float(rng.uniform(*LIGHTS["temperature"])))
        power = round(LIGHT_REF[kind] * float(rng.uniform(*LIGHTS["intensity"])), 1)
        shape = {k: [v := round(float(rng.uniform(*r)), 3), v] for k, r in LIGHT_SHAPE[kind].items()}
        out[f"light_{n}"] = {
            "types": kind,
            "common": {"initial_pos_range": pos + pos, "initial_ori_range": ori + ori,
                       "color_temperature_range": [temp, temp]},
            kind: {"intensity_range": [power, power], **shape},
        }  # fmt: skip
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True, help="layout seed folder")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--split", choices=("eval", "train"), default="eval")
    ap.add_argument("--assets", type=Path, default=ROOT / "Assets")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    _, room, room_pos, _ = SCENES[CONFIG]
    table_z = json.loads((args.assets / "Room" / room / "usd" / "build_params.json").read_text())["table"]["top_z"]
    cfg = yaml.safe_load((ROOT / "env_cfg/robot" / f"{CONFIG}.yml").read_text())["robots"][0]
    arms = {side: Arm(side, cfg) for side in ("left", "right")}
    out = args.assets / "Eval_Layout" / "RoboDojo" / CONFIG / str(args.seed)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob(f"{TASK}_*.json"):
        old.unlink()
    if args.split == "train":
        (out / "split.json").write_text(json.dumps({"split": "train"}) + "\n")
    job = partial(write_layout, args.seed, out, arms, table_z, room, room_pos)
    with Pool(min(args.workers, args.count)) as pool:
        draws = sum(pool.imap_unordered(job, range(args.count)))
    print(f"wrote {args.count} layouts ({draws} draws) to {out}")


def write_layout(seed, out, arms, table_z, room, room_pos, i):
    """Layout i of a seed folder, drawn from its own random streams (so the folder's layouts build in parallel)."""
    attempt = 0
    while True:
        rng = np.random.default_rng([seed, i, attempt])
        attempt += 1
        if (drawn := draw(rng, arms, table_z)) is not None:
            break
    rigid, frame, tilts = drawn
    layout = dict(Rigid=rigid, Room=room_entry(room, room_pos, table_z))
    if light := lights(rng):
        layout["Light"] = light
    (out / f"{TASK}_{i}.json").write_text(json.dumps(layout, indent=4))
    n_blocks = sum(len(v) for v in rigid.values())
    print(f"{TASK}_{i}: draw {attempt - 1}, origin {np.round(frame['origin'], 3).tolist()}, "
          f"yaw {frame['yaw_deg']:+.1f}, {n_blocks} blocks, {len(light)} lights, tilts {tilts}", flush=True)  # fmt: skip
    return attempt


if __name__ == "__main__":
    main()
