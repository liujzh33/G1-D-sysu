#!/usr/bin/env python3
"""Where each G1-D arm reaches the scripted top grasp (scripted_demo.TOP_ROT turned by a wrist yaw), from the URDF alone.

Kinematics of Assets/Robots/g1d/g1d_dex1.urdf (AGV_link -> <side>_dex1_base_link), the stand and arm start of an env
robot config, joint limits REAL_JOINT_MARGIN inside the URDF ones (what Limiter allows). Each target is solved by damped
least squares from the start pose and counts as reached within POS_TOL / ROT_TOL. No Isaac, no collision check.

Usage:
    python scripts/g1d/reach_map.py [--robot-cfg env_cfg/robot/g1d_school.yml] [--z 0.79 0.83 0.87 0.91]
        [--yaw -60 -30 0 30 60 90] [--json out.json]
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[2]
# The G1-D's grippers, picked by G1D_HAND (default dex1); the layouts and the expert must use the same one. `tip`: the
# hand frame (x along the fingers, y closing, z camera up; both mount at the wrist yaw link + 0.0415 m in x), `reach`:
# m from it along x to the grip point, `finger_tips`: m from the grip point to the fingertips, `pad_gap`: pad gap (m)
# at gripper command 0 and m more at 1, `grip_rate`: gripper command per action at the real jaw speed, `squeeze`: m
# per finger hold_cmd presses past the object, `asset`: the robot class asset_name.
HANDS = {
    "dex1": dict(
        urdf="Assets/Robots/g1d/g1d_dex1.urdf", tip="{side}_dex1_base_link", reach=0.12, finger_tips=0.023,
        pad_gap=(0.006, 0.089), grip_rate=0.2 / 0.0445 / 30, squeeze=0.0015, asset="g1d",
    ),
    # Xense Aurora Lite (real2sim/robot/g1d_aurora_lite_ws/model_contract.json): the TCP is the pads' axial midpoint
    # 0.1463 m from the mount (gel 0.1316-0.1609, sensor housing to 0.1638: the fingertips), the pads close to 0 and
    # open 85 mm, 0.07 m/s per jaw; 40 N per jaw at 1e4 N/m (3 mm)
    "aurora_lite": dict(
        urdf="Assets/Robots/g1d_aurora_lite_ws/g1d_aurora_lite_ws.urdf", tip="{side}_aurora_mount_link",
        reach=0.1462769233970408, finger_tips=0.0175, pad_gap=(0.0, 0.085), grip_rate=0.07 / 0.0425 / 30,
        squeeze=0.003, asset="g1d_aurora_lite_ws",
    ),
}
HAND_NAME = os.environ.get("G1D_HAND", "dex1")
HAND = HANDS[HAND_NAME]
URDF = ROOT / HAND["urdf"]
PAD_REACH = HAND["reach"]  # scripted_demo.PAD_REACH
GRASP_ROT = Rotation.from_euler("z", 90, degrees=True)  # scripted_demo.GRASP_ROT: fingers forward, camera up


def hand_rot(yaw, tilt):
    """Fingers `tilt` deg below horizontal (scripted_demo.TOP_ROT is 60), turned `yaw` deg about world z."""
    return Rotation.from_euler("z", yaw, degrees=True) * Rotation.from_euler("x", -tilt, degrees=True) * GRASP_ROT
REAL_JOINT_MARGIN = 0.05
POS_TOL, ROT_TOL = 0.003, np.deg2rad(3)


def chain(urdf, tip):
    joints = {j.find("child").get("link"): j for j in ET.parse(urdf).getroot().findall("joint")}
    out, link = [], tip
    while link in joints:
        j = joints[link]
        o, a, lim = j.find("origin"), j.find("axis"), j.find("limit")
        out.append(
            dict(
                name=j.get("name"),
                type=j.get("type"),
                xyz=np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]),
                rot=Rotation.from_euler("xyz", [float(v) for v in (o.get("rpy") or "0 0 0").split()]),
                axis=np.array([float(v) for v in a.get("xyz").split()]) if a is not None else None,
                lim=(float(lim.get("lower")), float(lim.get("upper"))) if lim is not None else None,
            )
        )
        link = j.find("parent").get("link")
    return out[::-1]


class Arm:
    def __init__(self, side, cfg, urdf=URDF, tip=HAND["tip"]):
        """`tip`: the hand frame (x along the fingers, y closing, z camera up), PAD_REACH behind the grip point."""
        self.joints = chain(urdf, tip.format(side=side))
        init = cfg.get("init_joint_pos", {})
        self.fixed = {j["name"]: float(init.get(j["name"], 0.0)) for j in self.joints if j["type"] != "fixed"}
        self.arm = [j["name"] for j in self.joints if j["name"].startswith(f"{side}_") and j["type"] == "revolute"]
        lims = {j["name"]: j["lim"] for j in self.joints if j["lim"]}
        self.lo = np.array([lims[n][0] + REAL_JOINT_MARGIN for n in self.arm])
        self.hi = np.array([lims[n][1] - REAL_JOINT_MARGIN for n in self.arm])
        self.q0 = np.array([self.fixed[n] for n in self.arm])
        q = cfg["default_root_rot"]
        self.base = (np.array(cfg["default_root_pos"], float), Rotation.from_quat([q[1], q[2], q[3], q[0]]))

    def fk(self, q):
        vals = dict(self.fixed, **dict(zip(self.arm, q)))
        pos, rot = self.base
        for j in self.joints:
            pos, rot = pos + rot.apply(j["xyz"]), rot * j["rot"]
            if j["type"] == "revolute":
                rot = rot * Rotation.from_rotvec(j["axis"] * vals[j["name"]])
            elif j["type"] == "prismatic":
                pos = pos + rot.apply(j["axis"] * vals[j["name"]])
        return pos + rot.apply([PAD_REACH, 0, 0]), rot

    def solve(self, point, rot_t, iters=150, q=None, pos_tol=POS_TOL, rot_tol=ROT_TOL):
        q = self.q0.copy() if q is None else q.copy()
        for _ in range(iters):
            p, r = self.fk(q)
            err = np.r_[point - p, (rot_t * r.inv()).as_rotvec()]
            if np.linalg.norm(err[:3]) < pos_tol and np.linalg.norm(err[3:]) < rot_tol:
                return q, err
            jac = np.empty((6, len(q)))
            for i in range(len(q)):
                dq = np.zeros(len(q))
                dq[i] = 1e-5
                p2, r2 = self.fk(q + dq)
                jac[:, i] = np.r_[p2 - p, (r2 * r.inv()).as_rotvec()] / 1e-5
            step = jac.T @ np.linalg.solve(jac @ jac.T + 0.03**2 * np.eye(6), err)
            q = np.clip(q + np.clip(step, -0.2, 0.2), self.lo, self.hi)
        return q, err

    def follow(self, targets, q=None):
        """Solutions along [(point, rot)], each warm-started from the one before; None once a target is missed."""
        out = []
        for point, rot in targets:
            q, err = self.solve(np.asarray(point, float), rot, q=q)
            if not reached(err):
                return None
            out.append(q)
        return out


def reached(err):
    return bool(np.linalg.norm(err[:3]) < POS_TOL and np.linalg.norm(err[3:]) < ROT_TOL)


def reach_row(side, cfg, xs, y, zs, yaw, tilt):
    """[z][x] reached for one y. Up through zs from the start pose's solution at the lowest z, as the arm rises after
    a grasp: a solve from the start pose alone stalls in local minima at the higher targets."""
    arm, rot_t = Arm(side, cfg), hand_rot(yaw, tilt)
    out = np.zeros((len(zs), len(xs)), bool)
    for i, x in enumerate(xs):
        q = None
        for k, z in enumerate(zs):
            sol, err = arm.solve(np.array([x, y, z]), rot_t, q=q)
            out[k, i] = reached(err)
            q = sol if out[k, i] else q
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-cfg", type=Path, default=ROOT / "env_cfg/robot/g1d_school.yml")
    ap.add_argument("--x", type=float, nargs=3, default=(-0.30, 0.40, 0.05), help="min max step (env frame, m)")
    ap.add_argument("--y", type=float, nargs=3, default=(-0.62, -0.20, 0.04))
    ap.add_argument("--z", type=float, nargs="+", default=[0.79, 0.85, 0.90, 0.95], help="grasp point heights")
    ap.add_argument("--yaw", type=float, nargs="+", default=[-30, 0, 30], help="wrist yaw (deg)")
    ap.add_argument("--tilt", type=float, nargs="+", default=[60], help="fingers below horizontal (deg)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.robot_cfg.read_text())["robots"][0]
    xs = np.arange(args.x[0], args.x[1] + 1e-9, args.x[2])
    ys = np.arange(args.y[0], args.y[1] + 1e-9, args.y[2])
    zs = sorted(args.z)
    out = {"robot_cfg": str(args.robot_cfg), "x": xs.tolist(), "y": ys.tolist(), "z": zs, "arms": {}}
    combos = [(yaw, tilt) for yaw in args.yaw for tilt in args.tilt]
    for side in ("left", "right"):
        arm = Arm(side, cfg)
        print(f"{side}: start grasp point {arm.fk(arm.q0)[0].round(3)}", flush=True)
        jobs = [(side, cfg, xs, y, zs, yaw, tilt) for yaw, tilt in combos for y in ys]
        with ProcessPoolExecutor() as pool:
            rows = np.array(list(pool.map(reach_row, *zip(*jobs)))).reshape(len(combos), len(ys), len(zs), len(xs))
        out["arms"][side] = {
            "start": arm.fk(arm.q0)[0].tolist(),
            "grids": {f"{z:.2f}/{yaw:g}/{tilt:g}": rows[c, :, k].tolist()
                      for c, (yaw, tilt) in enumerate(combos) for k, z in enumerate(zs)},
        }  # fmt: skip
        for tilt in args.tilt:
            every_yaw = np.all([rows[c] for c, (_, t) in enumerate(combos) if t == tilt], axis=0)
            print(f"  tilt {tilt:g}: reached at every yaw, per z: {[int(every_yaw[:, k].sum()) for k in range(len(zs))]}")
    if args.json:
        args.json.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
