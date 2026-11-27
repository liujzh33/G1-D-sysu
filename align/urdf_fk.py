"""Minimal URDF forward kinematics (numpy) for the G1-D + Dex1-1 asset."""
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

URDF = str(Path(__file__).resolve().parents[1] / "Assets/Robots/g1d/g1d_dex1.urdf")
ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]


def rpy_m(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_m(a, q):
    a = np.asarray(a, float) / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


class FK:
    def __init__(self, path=URDF):
        root = ET.parse(path).getroot()
        self.joints = {}
        self.parent_of = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            a = j.find("axis")
            axis = [float(v) for v in a.get("xyz").split()] if a is not None else [0, 0, 1]
            T = np.eye(4); T[:3, :3] = rpy_m(*rpy); T[:3, 3] = xyz
            child, parent = j.find("child").get("link"), j.find("parent").get("link")
            self.joints[child] = (j.get("name"), j.get("type"), parent, T, axis)
            self.parent_of[child] = parent

    def chain(self, link):
        out = []
        while link in self.joints:
            out.append(link); link = self.parent_of[link]
        return out[::-1], link

    def pose(self, link, q, base=None):
        """4x4 of `link` in `base` (default: the URDF root); q: dict joint name -> value."""
        links, root = self.chain(link)
        if base is not None:
            bl, _ = self.chain(base)
            links = links[len(bl):]
        T = np.eye(4)
        for l in links:
            name, typ, _, T0, axis = self.joints[l]
            M = np.eye(4)
            v = q.get(name, 0.0)
            if typ in ("revolute", "continuous"):
                M[:3, :3] = axis_m(axis, v)
            elif typ == "prismatic":
                M[:3, 3] = np.asarray(axis) * v
            T = T @ T0 @ M
        return T


def arm_q(side, vals):
    return {f"{side}_{n}_joint": float(v) for n, v in zip(ARM, vals)}
