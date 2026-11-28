#!/usr/bin/env python3
"""How close a G1-D gripper comes to the robot's torso and to the other gripper, from the URDF's collision meshes
alone (no Isaac). The simulated arm does not collide with its own body, so a hand driven through the torso or into
the other hand is not stopped: the layouts, the expert and the plugins keep these gaps open instead.

A hand is the collision surface of its wrist yaw link and everything mounted on it (the Aurora Lite's base, carriers,
sensors and gel pads, the jaws fully open), as points SPACING apart in the frame the experts track: the grip point,
the mount's rotation (reach_map.Arm.fk). The torso is torso_link's collision surface at the stand's lift and waist.

Usage (any python with numpy/scipy, repo root): replay the gaps of recorded episodes (privileged/ee_pose)
    python scripts/g1d/clearance.py logs/aurora_final/v4a/episode_*.hdf5
"""

import argparse
import functools
import os
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.g1d.reach_map import HAND, HAND_NAME, PAD_REACH, URDF, chain  # noqa: E402

SPACING = 0.005  # m between surface points
# m of gap planned (surface to surface): a hand from the torso (layouts and the expert's tilts, build_shapes.feasible),
# the two hands from each other (scripted_demo.park_offsets adds PARK_CLEAR); None: not checked (the Dex1-1 house data
# up to v4a / task2 was collected without these checks: G1D_CLEARANCE=0)
BODY_CLEAR = {"aurora_lite": 0.025, "dex1": 0.025}.get(HAND_NAME)
HANDS_CLEAR = {"aurora_lite": 0.03, "dex1": 0.03}.get(HAND_NAME)
if os.environ.get("G1D_CLEARANCE") == "0":
    BODY_CLEAR = HANDS_CLEAR = None
# m: gaps the simulated hands (tracking error, the v4 noise, a block moved under the hand) may close to before the
# episode is rejected (Watch); under the planned ones so the planning margins, not the reject, keep them open
BODY_TOUCH, HANDS_TOUCH = 0.01, 0.01
WRIST = "{side}_wrist_yaw_link"
BODY = ("torso_link",)


def read_stl(path):
    data = Path(path).read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    if len(data) == 84 + 50 * n:
        rec = np.frombuffer(data[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
        return rec["v"].astype(float)
    tris = [[float(t) for t in line.split()[1:4]] for line in data.decode().splitlines() if line.strip().startswith("vertex")]
    return np.asarray(tris, float).reshape(-1, 3, 3)


def surface(tris, spacing=SPACING, seed=0):
    """Points on the triangles about `spacing` apart (vertices, plus area-weighted samples), thinned to one per cell."""
    rng = np.random.default_rng(seed)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    k = np.ceil(area / spacing**2).astype(int)
    idx = np.repeat(np.arange(len(tris)), k)
    u, v = rng.random(len(idx)), rng.random(len(idx))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = np.r_[tris.reshape(-1, 3), a[idx] + u[:, None] * (b - a)[idx] + v[:, None] * (c - a)[idx]]
    _, keep = np.unique(np.floor(pts / spacing).astype(np.int64), axis=0, return_index=True)
    return pts[np.sort(keep)]


@functools.lru_cache(maxsize=None)
def _urdf(urdf):
    root = ET.parse(urdf).getroot()
    meshes = {}
    for link in root.findall("link"):
        for col in link.findall("collision"):
            mesh = col.find("geometry/mesh")
            if mesh is None:
                continue
            o = col.find("origin")
            xyz = np.array([float(v) for v in ((o.get("xyz") if o is not None else None) or "0 0 0").split()])
            rpy = [float(v) for v in ((o.get("rpy") if o is not None else None) or "0 0 0").split()]
            scale = np.array([float(v) for v in (mesh.get("scale") or "1 1 1").split()])
            meshes.setdefault(link.get("name"), []).append((Path(urdf).parent / mesh.get("filename"), xyz, rpy, scale))
    children = {}
    for j in root.findall("joint"):
        children.setdefault(j.find("parent").get("link"), []).append(j.find("child").get("link"))
    return meshes, children


def link_cloud(urdf, link):
    """Collision surface points of one link in its own frame."""
    out = [np.zeros((0, 3))]
    for path, xyz, rpy, scale in _urdf(str(urdf))[0].get(link, []):
        pts = surface(read_stl(path) * scale)
        out.append(xyz + Rotation.from_euler("xyz", rpy).apply(pts))
    return np.concatenate(out)


def walk(joints, values, pos=None, rot=None):
    """Pose of the chain's last link: `joints` from reach_map.chain, `values` by joint name (missing ones 0)."""
    pos = np.zeros(3) if pos is None else pos
    rot = Rotation.identity() if rot is None else rot
    for j in joints:
        pos, rot = pos + rot.apply(j["xyz"]), rot * j["rot"]
        v = values.get(j["name"], 0.0)
        if j["type"] == "revolute":
            rot = rot * Rotation.from_rotvec(j["axis"] * v)
        elif j["type"] == "prismatic":
            pos = pos + rot.apply(j["axis"] * v)
    return pos, rot


def subtree(children, link):
    out = [link]
    for c in children.get(link, []):
        out += subtree(children, c)
    return out


@functools.lru_cache(maxsize=None)
def hand_cloud(side="left", urdf=str(URDF)):
    """(N, 3) points of the hand in its grip frame (x along the fingers, PAD_REACH ahead of the mount), jaws open."""
    _, children = _urdf(urdf)
    wrist = WRIST.format(side=side)
    base = {j["name"]: j for j in chain(urdf, wrist)}
    open_ = {}
    pts = []
    for link in subtree(children, wrist):
        joints = [j for j in chain(urdf, link) if j["name"] not in base]
        for j in joints:
            if j["type"] == "prismatic" and j["lim"]:
                open_[j["name"]] = j["lim"][1]
        p, r = walk(joints, open_)
        cloud = link_cloud(urdf, link)
        if len(cloud):
            pts.append(p + r.apply(cloud))
    tip = walk([j for j in chain(urdf, HAND["tip"].format(side=side)) if j["name"] not in base], {})
    # wrist frame -> grip frame: the tip (mount) frame's rotation, PAD_REACH ahead along its x
    pts = tip[1].inv().apply(np.concatenate(pts) - tip[0]) - [PAD_REACH, 0, 0]
    return pts


class Clearance:
    """Gaps of the hands (grip point, rotation: reach_map.Arm.fk, scripted_demo Arm.grasp_point / pose, the
    plugins' ee_pose) from the torso and from each other, in the robot config's frame (env-local)."""

    def __init__(self, cfg=None, urdf=URDF, base=None, values=None):
        """From a robot config entry (env_cfg/robot/*.yml robots[0]), or a stand pose `base` (pos, Rotation) and
        joint `values` (a reach_map.Arm's base and fixed)."""
        urdf = str(urdf)
        if cfg is not None:
            values = cfg.get("init_joint_pos", {})
            q = cfg["default_root_rot"]
            base = (np.array(cfg["default_root_pos"], float), Rotation.from_quat([q[1], q[2], q[3], q[0]]))
        body = []
        for link in BODY:
            p, r = walk(chain(urdf, link), values, *base)
            body.append(p + r.apply(link_cloud(urdf, link)))
        self.body_points = np.concatenate(body)
        self.body = cKDTree(self.body_points)
        self.hand = {s: hand_cloud(s, urdf) for s in ("left", "right")}

    def points(self, side, grip, rot):
        return np.asarray(grip, float) + rot.apply(self.hand[side])

    def body_gap(self, side, grip, rot):
        pts = self.points(side, grip, rot)
        lo, hi = self.body_points.min(0), self.body_points.max(0)
        box = float(np.linalg.norm(np.maximum(0, np.maximum(lo - pts, pts - hi)), axis=1).min())
        if box > 0.1:  # far outside the torso's bounding box: its distance is enough
            return box
        return float(self.body.query(pts, distance_upper_bound=1.0)[0].min())

    def hands_gap(self, left, right):
        """`left` / `right`: (grip, rot) of each hand."""
        a, b = self.points("left", *left), self.points("right", *right)
        box = float(np.linalg.norm(np.maximum(0, np.maximum(a.min(0) - b.max(0), b.min(0) - a.max(0)))))
        if box > 0.1:  # the hands' bounding boxes this far apart
            return box
        return float(cKDTree(a).query(b, distance_upper_bound=1.0)[0].min())


@functools.lru_cache(maxsize=None)
def for_config(name="g1d_aurora_school"):
    cfg = yaml.safe_load((ROOT / "env_cfg/robot" / f"{name}.yml").read_text())["robots"][0]
    return Clearance(cfg)


_ARMS = {}


def for_arm(arm):
    """The Clearance of a reach_map.Arm's robot (cached per stand pose and joint values)."""
    key = (tuple(arm.base[0]), tuple(arm.base[1].as_quat()), tuple(sorted(arm.fixed.items())))
    if key not in _ARMS:
        _ARMS[key] = Clearance(base=arm.base, values=arm.fixed)
    return _ARMS[key]


def path_clear(arm, side, qs, steps=3):
    """Whether the hand keeps BODY_CLEAR from the torso at the joint solutions `qs` (reach_map.Arm.follow) and
    `steps` - 1 points between neighbours (joint space)."""
    if BODY_CLEAR is None:
        return True
    c = for_arm(arm)
    qs = [np.asarray(q) for q in qs]
    for a, b in zip(qs, qs[1:] + qs[-1:]):
        for s in np.linspace(0, 1, steps, endpoint=False) if b is not a else [0.0]:
            if c.body_gap(side, *arm.fk(a + s * (b - a))) < BODY_CLEAR:
                return False
    return True


def ee_rot(ee):
    """Rotation of an ee_pose row (xyz wxyz)."""
    return Rotation.from_quat(np.r_[ee[4:7], ee[3]])


class Watch:
    """Per frame (both hands' grip point and rotation): the first violation as a reject reason, kept once set."""

    def __init__(self, clearance):
        self.c, self.reason, self.min = clearance, None, dict(body_left=1.0, body_right=1.0, hands=1.0)

    def __call__(self, frame, left, right):
        if BODY_CLEAR is None or self.reason:
            return self.reason
        gaps = dict(body_left=self.c.body_gap("left", *left), body_right=self.c.body_gap("right", *right),
                    hands=self.c.hands_gap(left, right))  # fmt: skip
        for k, v in gaps.items():
            self.min[k] = min(self.min[k], v)
        for side in ("left", "right"):
            if gaps[f"body_{side}"] < BODY_TOUCH:
                self.reason = f"{side}_hand_near_body ({gaps[f'body_{side}'] * 100:.1f} cm, frame {frame})"
        if gaps["hands"] < HANDS_TOUCH:
            self.reason = self.reason or f"hands_too_close ({gaps['hands'] * 100:.1f} cm, frame {frame})"
        return self.reason


def main():
    import h5py

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episodes", nargs="+")
    parser.add_argument("--config", default="g1d_aurora_school")
    parser.add_argument("--every", type=int, default=3)
    parser.add_argument("--verbose", action="store_true", help="both grip points at each worst frame")
    args = parser.parse_args()
    c = for_config(args.config)
    print(f"hand points {len(c.hand['left'])}, torso points {len(c.body_points)}, torso y {c.body_points[:, 1].min():.3f}"
          f"..{c.body_points[:, 1].max():.3f} z {c.body_points[:, 2].min():.3f}..{c.body_points[:, 2].max():.3f}")
    for path in args.episodes:
        with h5py.File(path, "r") as f:
            ee = f["privileged/ee_pose"][:]
            phases = [n.decode() for n in f["privileged/phase_names"][:]] if "privileged/phase_names" in f else None
            phase = f["privileged/phase"][:] if phases else None
        rows = []
        for t in range(0, len(ee), args.every):
            left, right = (ee[t, 0, :3], ee_rot(ee[t, 0])), (ee[t, 1, :3], ee_rot(ee[t, 1]))
            rows.append((c.body_gap("left", *left), c.body_gap("right", *right), c.hands_gap(left, right)))
        rows = np.array(rows)
        worst = rows.argmin(0) * args.every
        at = [f"@{t}" + (f" {phases[phase[t]]}" if phases and 0 <= phase[t] < len(phases) else "") for t in worst]
        print(f"{path}: body L {rows[:, 0].min() * 100:.1f} cm {at[0]}  body R {rows[:, 1].min() * 100:.1f} cm "
              f"{at[1]}  hands {rows[:, 2].min() * 100:.1f} cm {at[2]}")
        if args.verbose:
            for k, t in zip(("body L", "body R", "hands"), worst):
                print(f"   {k} @{t}: L {ee[t, 0, :3].round(3).tolist()} R {ee[t, 1, :3].round(3).tolist()}")


if __name__ == "__main__":
    main()
