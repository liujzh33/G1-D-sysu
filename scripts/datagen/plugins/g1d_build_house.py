"""scripts/datagen plugin for g1d_build_house: the scripted one-arm-at-a-time build of the seven-block house
(scripted_demo.build_actions, commands through Limiter). An episode is accepted when the expert has finished and every
house block sits within build_shapes' tolerances of its target for HOLD frames in a row (acceptance house_built_v1).

Annotations come from the simulator's ground truth, VPP2-style (clips cover the episode from the first reach on):
- quality.json "subtasks", in time order, each with its frames [start, end) (frame t is the image before action t):
  - type "pick_place": also grasp, lift and release frames, the arm, the block (label, kind, colour), its start / goal
    / final pose, the blocks it rests on, the head-camera bbox of the block at the clip's first and last frame, its
    visible share there and the share of the clip's frames it is occluded in, and captions (English for the composed
    video and the head view, Chinese);
  - type "retreat": the arm going back out to its side before the other arm's turn (and at the end), its captions.
  "plan" is the build order, "instruction_en" / "instruction_zh" the whole task.
- privileged/: per frame (aligned with the images) every block's pose, both grasp points in the head image, the subtask
  index (into "subtasks"; -1 before the first reach) and, for the block a pick_place moves only, its head-camera bbox
  [u0, v0, u1, v1] in pixels, the bbox of its visible part, the visible share and whether it is occluded (target -1
  and NaN / False in retreats or out of view). Visibility: points on the camera-facing faces of the block's box, each visible unless the head
  depth image has something nearer (the gripper, the other arm, another block) at its pixel.
- v4 labels (any profile): per frame the phase (bs.PHASES, from the expert's rec["events"]) and its arm, the failure
  events (bs.EVENTS) and whether a recovery is under way (from a failure until the block is carried again), both grasp
  points' pose (env frame, wxyz) and pixel in each camera, whether each is visible in the head view (EE_RADIUS sphere
  against the depth), and the wrist cameras' camera-to-world. quality.json "recoveries": the recovery clips, each
  with the index of the pick_place it belongs to ("parent"; they overlap it, so subtask_index never points at them).
- 0..1 progress (scripts/g1d/house4_progress.py, flora_v2a style): privileged/progress, semantic_progress and
  subtask_progress per frame; quality.json "progress": each block's pick and place milestone with its key frames
  (t_start, t_contact, t_effect, t_end), status and weight.
- HOUSE4_PROFILE (env, default v3e): v4 keeps KEEP_REJECTED_SHARE of the rejected trajectories (accepted false and
  the reason in the sidecar), for failure detection; HOUSE4_KEEP_REJECTED (env) overrides the share (1: keep all).
- grippers with clearance checks (scripts/g1d/clearance.py, the Aurora Lite): a hand closer than BODY_TOUCH to the
  torso or HANDS_TOUCH to the other hand (the simulated arm passes through both) rejects the episode and ends it there
  (quality.json "aborted"); "clearance" has the smallest gaps of the episode.
"""

import os

import numpy as np

from scripts.g1d import build_shapes as bs
from scripts.g1d import clearance
from scripts.g1d import house4_progress as hp
from scripts.g1d import scripted_demo as demo

TASK = "g1d_build_house"
ACCEPTANCE = "house_built_v1"
STEP_LIM = 6000  # actions: seven straight-down pick-and-places of ~450 actions (v4a), arm swaps, recoveries, home
TAIL = 30  # actions the last command is held after the script ends
HOLD = 30  # frames (1 s) the acceptance must hold
KEEP_REJECTED = False  # a rejected episode keeps its sidecar (why) but not its ~350 MB trajectory
CAMERA = "cam_head"
ARMS = ("left", "right")
FACE_GRID = 7  # visibility sample points per face side
DEPTH_TOL = 0.006  # m a sample may lie behind the rendered depth and still count as seen (its own surface)
OCCLUDED = 0.5  # visible share under which the target block counts as occluded
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist", "cam_third")  # those the env config has
EE_RADIUS, EE_DEPTH_TOL = 0.01, 0.015  # m: sphere about the grasp point (between the pads) for its head visibility
PROFILE = os.environ.get("HOUSE4_PROFILE", "v3e")
KEEP_REJECTED_SHARE = float(os.environ.get("HOUSE4_KEEP_REJECTED", 0.2 if PROFILE == "v4" else 0.0))
# HOUSE4_INSTRUCTION=ordered: the instruction says the build step by step in the order the expert built
# (build_shapes.ordered_instruction, as runs/relabel_ordered_instructions.py), the whole-house one kept as *_original
ORDERED = os.environ.get("HOUSE4_INSTRUCTION", "") == "ordered"


def camera_axes(T_cw):
    """World point -> (x right, y down, depth) in the camera of a camera-to-world matrix: USD cameras look along -z
    with +y up."""
    R, t = T_cw[:3, :3], T_cw[:3, 3]

    def to_cam(p):
        c = (np.asarray(p) - t) @ R
        return np.stack([c[..., 0], -c[..., 1], -c[..., 2]], axis=-1)

    return to_cam


def project(K, size, to_cam, points):
    """Pixels (N, 2) of world points; NaN behind the camera."""
    c = to_cam(points)
    uv = (c[..., :2] / c[..., 2:3]) * [K[0, 0], K[1, 1]] + K[:2, 2]
    return np.where(c[..., 2:3] > 1e-3, uv, np.nan)


def face_points(lo, hi, eye_local):
    """Points on the faces of the box [lo, hi] (block frame) that face a camera at `eye_local` (block frame)."""
    g = (np.arange(FACE_GRID) + 0.5) / FACE_GRID
    a, b = np.meshgrid(g, g)
    out = []
    for axis in range(3):
        u, v = [k for k in range(3) if k != axis]
        for side, plane in ((-1, lo[axis]), (1, hi[axis])):
            if side * (eye_local[axis] - plane) <= 0:
                continue
            pts = np.empty((a.size, 3))
            pts[:, axis] = plane
            pts[:, u] = lo[u] + a.ravel() * (hi[u] - lo[u])
            pts[:, v] = lo[v] + b.ravel() * (hi[v] - lo[v])
            out.append(pts)
    return np.concatenate(out) if out else np.empty((0, 3))


SPHERE = np.random.default_rng(0).normal(size=(40, 3))
SPHERE /= np.linalg.norm(SPHERE, axis=1, keepdims=True)


def sphere_points(centre, eye, radius=EE_RADIUS):
    """The centre and points on the half of a sphere about it that faces the camera at `eye`."""
    d = np.where((SPHERE @ (eye - centre) < 0)[:, None], -SPHERE, SPHERE)
    return np.r_[centre[None], centre + radius * d]


def visibility(K, size, to_cam, depth, points, tol=DEPTH_TOL):
    """(visible share, [u0, v0, u1, v1] of the visible points) of world points against a depth image; (NaN, NaN box)
    when none of them is in view."""
    c = to_cam(points)
    uv = (c[:, :2] / np.maximum(c[:, 2:3], 1e-6)) * [K[0, 0], K[1, 1]] + K[:2, 2]
    px = np.floor(uv).astype(int)
    w, h = size
    inside = (c[:, 2] > 1e-3) & (px[:, 0] >= 0) & (px[:, 1] >= 0) & (px[:, 0] < w) & (px[:, 1] < h)
    if not inside.any():
        return np.nan, np.full(4, np.nan)
    seen = np.zeros(len(points), bool)
    seen[inside] = c[inside, 2] <= depth[px[inside, 1], px[inside, 0]] + tol
    if not seen.any():
        return 0.0, np.full(4, np.nan)
    return float(seen.sum() / inside.sum()), np.r_[uv[seen].min(0), uv[seen].max(0)]


def bbox(uv, size):
    """[u0, v0, u1, v1] of projected corners clipped to the image; NaN when none of it is in view."""
    if np.isnan(uv).any():
        return np.full(4, np.nan)
    lo, hi = uv.min(0), uv.max(0)
    w, h = size
    if hi[0] < 0 or hi[1] < 0 or lo[0] >= w or lo[1] >= h:
        return np.full(4, np.nan)
    return np.r_[np.clip(lo, 0, [w - 1, h - 1]), np.clip(hi, 0, [w - 1, h - 1])]


class Episode:
    def __init__(self, env):
        self.env, self.lm = env, env.scene_manager.layout_manager
        self.layout_id = int(env.current_env_seed_map[0])
        layout = env.seed_manager.get_seed_scene_info(self.layout_id)
        self.entry = bs.shape_entry(layout)
        self.labels = [e["label"] for entries in layout["Rigid"].values() for e in entries]
        self.kinds = {e["label"]: (e["category"], e["category_idx"]) for es in layout["Rigid"].values() for e in es}
        self.rng = np.random.default_rng([int(env.eval_seed), self.layout_id])
        self.log, self.done = [], False
        cm = env.camera_manager
        self.cam = cm.camera_names[0].index(CAMERA)
        self.K = cm.get_camera_intrinsics(self.cam, 0)
        # calibrated cameras have an off-centre principal point, so the size comes from the camera, not 2 * (cx, cy)
        size = lambda k: tuple(int(v) for v in cm.cameras[0][k].get_resolution())  # noqa: E731
        self.size = size(self.cam)
        self.cams = {}  # name: (index, K, (w, h))
        for name in CAMERAS:
            if name in cm.camera_names[0]:
                k = cm.camera_names[0].index(name)
                self.cams[name] = (k, cm.get_camera_intrinsics(k, 0), size(k))
        self.origin = demo.as_np(env.scene_manager.env_origins[0])
        self.arms = {a.arm_key.split("_")[0]: a for a in demo.Robot(env).arms}
        self.history = dict(
            block_poses=[], head_bboxes=[], visible_bboxes=[], visible_share=[], grasp_points_head=[], camera_to_world=[],
            ee_pose=[], ee_visible_head=[], wrist_camera_to_world=[], **{f"ee_uv_{c}": [] for c in self.cams},
        )  # fmt: skip
        if "cam_third" in self.cams:
            self.history["third_camera_to_world"] = []
        self.final_check, self.last_poses = None, None
        self.reject = None if self.entry else "layout has no BuildShape entry"
        # pad contacts from the PhysX contact report (the pads carry its API for their ContactSensor; the blocks spawn
        # after the sensors are set up, so a filtered ContactSensor cannot list them): pads left 1, 2, right 1, 2
        self.pads = list((getattr(self.arms["left"].robot, "finger_pads", None) or {}).values())
        self.block_paths = [self.lm.get_scene_object(0, self.lm.get_instance_name(0, lb)).prim_path for lb in self.labels]
        self.physics_dt = float(env.dt)
        if self.pads:
            self.history.update(pad_contact_pos=[], pad_contact_force=[], pad_contact_uv_head=[])
        # its own stream: the expert's rng draws stay those of v3e
        keep = np.random.default_rng([int(env.eval_seed), self.layout_id, 1]).random() < KEEP_REJECTED_SHARE
        self.keep_rejected = bool(keep)
        self.watch = None
        if clearance.BODY_CLEAR is not None:
            from omegaconf import OmegaConf

            cfg = OmegaConf.to_container(env.robot_config, resolve=True)["robots"][0]
            self.watch = clearance.Watch(clearance.Clearance(cfg, urdf=self.arms["left"].robot.urdf_path))

    def doomed(self):
        """collect.py's early end (scripts/datagen/early_abort.py): the clearance violation, once there is one."""
        return self.watch and self.watch.reason

    def poses(self):
        return {
            label: [demo.as_np(v) for v in self.lm.get_instance_pose(0, label=label)] for label in self.labels
        }

    def local_box(self, label):
        local = self.lm.get_instance_bbox_vertices(self.lm.get_instance_name(0, label), 0)
        return local, local.min(0), local.max(0)

    def observe(self, obs):
        """Record the frame of an observation (collect.py hands over each one, the first before any action)."""
        poses = self.poses()
        T = self.env.camera_manager.get_camera_extrinsics(self.cam, 0)
        to_cam = camera_axes(T)
        depth = np.asarray(obs["vision"][CAMERA]["depth"], dtype=np.float32).reshape(self.size[1], self.size[0])
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, np.inf)
        eye = T[:3, 3]
        boxes, seen, share = [], [], []
        for lb in self.labels:
            pos, quat = poses[lb]
            rot = demo.from_wxyz(quat)
            local, lo, hi = self.local_box(lb)
            world = pos + self.origin
            boxes.append(bbox(project(self.K, self.size, to_cam, world + rot.apply(local)), self.size))
            pts = world + rot.apply(face_points(lo, hi, rot.inv().apply(eye - world)))
            s, vb = visibility(self.K, self.size, to_cam, depth, pts)
            share.append(s)
            seen.append(bbox(vb.reshape(2, 2), self.size) if np.isfinite(vb).all() else vb)
        grips = [project(self.K, self.size, to_cam, self.arms[s].grasp_point() + self.origin) for s in ARMS]
        ee, visible = [], []
        for s in ARMS:
            point, rot = self.arms[s].grasp_point(), self.arms[s].pose()[1]
            x, y, z, w = rot.as_quat()
            ee.append(np.r_[point, w, x, y, z])
            ee_share, _ = visibility(self.K, self.size, to_cam, depth, sphere_points(point + self.origin, eye), EE_DEPTH_TOL)
            visible.append(bool(ee_share > 0))
        self.history["ee_pose"].append(ee)
        self.history["ee_visible_head"].append(visible)
        if self.watch:
            hands = [(self.arms[s].grasp_point(), self.arms[s].pose()[1]) for s in ARMS]
            self.watch(len(self.history["ee_pose"]) - 1, *hands)
        wrists = []
        for name, (k, K, size) in self.cams.items():
            Tc = T if k == self.cam else self.env.camera_manager.get_camera_extrinsics(k, 0)
            if name == "cam_third":
                self.history["third_camera_to_world"].append(Tc)
            elif k != self.cam:
                wrists.append(Tc)
            pts = np.array([e[:3] + self.origin for e in ee])
            self.history[f"ee_uv_{name}"].append(project(K, size, camera_axes(Tc), pts))
        self.history["wrist_camera_to_world"].append(wrists)
        if self.pads:
            pos, force = self.pad_contacts()
            self.history["pad_contact_pos"].append(pos.reshape(2, 2, len(self.labels), 3))
            self.history["pad_contact_force"].append(force.reshape(2, 2, len(self.labels)))
            self.history["pad_contact_uv_head"].append(
                project(self.K, self.size, to_cam, pos + self.origin).reshape(2, 2, len(self.labels), 2)
            )
        self.history["block_poses"].append([np.r_[poses[lb][0], poses[lb][1]] for lb in self.labels])
        self.history["head_bboxes"].append(boxes)
        self.history["visible_bboxes"].append(seen)
        self.history["visible_share"].append(share)
        self.history["grasp_points_head"].append(grips)
        self.history["camera_to_world"].append(T)
        self.last_poses = poses

    def pad_contacts(self):
        """Per pad and block: contact point (env frame; impulse-weighted mean of the last physics step's points, NaN
        apart) and force (N, |total impulse| / dt), from the PhysX contact report."""
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools

        shape = (len(self.pads), len(self.labels))
        moment, weight, impulse, count = np.zeros((*shape, 3)), np.zeros(shape), np.zeros((*shape, 3)), np.zeros(shape)
        headers, data = get_physx_simulation_interface().get_contact_report()
        for h in headers:
            ends = [str(PhysicsSchemaTools.intToSdfPath(a)) for a in (h.actor0, h.actor1)]
            pad = [k for k, link in enumerate(self.pads) for e in ends if e.endswith("/" + link)]
            block = [i for i, b in enumerate(self.block_paths) for e in ends if e == b or e.startswith(b + "/")]
            if not pad or not block:
                continue
            pts = data[h.contact_data_offset : h.contact_data_offset + h.num_contact_data]
            if not len(pts):
                continue
            xyz = np.array([[c.position[0], c.position[1], c.position[2]] for c in pts])
            imp = np.array([[c.impulse[0], c.impulse[1], c.impulse[2]] for c in pts])
            w = np.linalg.norm(imp, axis=1) + 1e-9
            p, b = pad[0], block[0]
            moment[p, b] += (xyz * w[:, None]).sum(0)
            weight[p, b] += w.sum()
            impulse[p, b] += imp.sum(0)
            count[p, b] += len(pts)
        with np.errstate(invalid="ignore", divide="ignore"):
            pos = np.where((count > 0)[..., None], moment / weight[..., None] - self.origin, np.nan)
        return pos, np.linalg.norm(impulse, axis=-1) / self.physics_dt

    def actions(self):
        limiter = demo.Limiter(self.env)
        for action in demo.build_actions(self.env, self.rng, self.log):
            yield limiter(action)
        self.done = True

    def step(self):
        """After each action (its frame observed): True once the expert is done and the house stands."""
        self.final_check = bs.check(self.entry, self.last_poses)
        return self.done and all(e["ok"] for e in self.final_check.values()) and not self.doomed()

    def frames(self):
        """Per-frame arrays aligned with the images: frame t is before action t (the last record is after the end)."""
        return {k: np.asarray(v[:-1]) for k, v in self.history.items()}

    def in_view(self, uv):
        return bool(np.isfinite(uv).all() and 0 <= uv[0] < self.size[0] and 0 <= uv[1] < self.size[1])

    def occluded_share(self, i, start, end):
        """Share of a subtask's frames in which its block is occluded (or out of view)."""
        share = np.asarray(self.history["visible_share"][start : end + 1], dtype=float)[:, i]
        return float(np.mean(~(share >= OCCLUDED))) if len(share) else float("nan")

    def timeline(self):
        """(type, record, start, end) in time order: each pick-and-place, then a retreat wherever the arm goes home
        (before the other arm's turn and after the last block)."""
        n = len(self.history["block_poses"]) - 1
        out = []
        for j, rec in enumerate(self.log):
            end = rec["frames"].get("end", n)
            out.append(("pick_place", rec, rec["frames"]["start"], end))
            nxt = self.log[j + 1] if j + 1 < len(self.log) else None
            if "end" in rec["frames"] and (nxt is None or nxt["arm"] != rec["arm"]):
                stop = nxt["frames"]["start"] if nxt else n
                if stop > end:
                    out.append(("retreat", rec, end, stop))
        return out

    def subtasks(self):
        out = []
        for kind, rec, start, end in self.timeline():
            out.append(self.pick_place(rec, len(out)) if kind == "pick_place" else self.retreat(rec, start, end, len(out)))
        return out

    def retreat(self, rec, start, end, index):
        other = ARMS[1 - ARMS.index(rec["arm"])]
        other_visible = self.in_view(self.history["grasp_points_head"][start][ARMS.index(other)])
        return dict(
            index=index,
            type="retreat",
            arm=rec["arm"],
            after=rec["part"],
            frames=dict(start=start, end=end),
            other_gripper_visible=other_visible,
            caption=bs.retreat_caption(rec["arm"], other_visible),
            caption_head=bs.retreat_caption(rec["arm"], other_visible, view="head"),
            caption_zh=bs.retreat_caption(rec["arm"], other_visible, zh=True),
        )

    def pick_place(self, rec, index):
        h = self.history
        f = dict(rec["frames"])
        f.setdefault("end", len(h["block_poses"]) - 1)
        i = self.labels.index(rec["label"])
        other = ARMS[1 - ARMS.index(rec["arm"])]
        other_visible = self.in_view(h["grasp_points_head"][f["start"]][ARMS.index(other)])
        final = h["block_poses"][f["end"]][i]
        sub = dict(
            {k: v for k, v in rec.items() if k != "frames"},
            index=index,
            type="pick_place",
            frames=f,
            color_name=bs.color_name(rec["category"], rec["color"]),
            kind=bs.KINDS[rec["category"]][0],
            final_pos=[float(v) for v in final[:3]],
            final_ori_wxyz=[float(v) for v in final[3:]],
            head_bbox_start=[float(v) for v in h["head_bboxes"][f["start"]][i]],
            head_bbox_end=[float(v) for v in h["head_bboxes"][f["end"]][i]],
            visible_share_start=float(h["visible_share"][f["start"]][i]),
            visible_share_end=float(h["visible_share"][f["end"]][i]),
            occluded_share=self.occluded_share(i, f["start"], f["end"]),
            active_gripper_visible=self.in_view(h["grasp_points_head"][f["start"]][ARMS.index(rec["arm"])]),
            other_gripper_visible=other_visible,
        )
        args = (self.entry, rec["part"], rec["arm"], other_visible,
                rec["start_pos"] if self.entry.get("shape") in bs.DETAILED else rec["start_pos"][0])
        sub["caption"] = bs.caption(*args)
        sub["caption_head"] = bs.caption(*args, view="head")
        sub["caption_zh"] = bs.caption(*args, zh=True)
        return sub

    def failures(self, rec):
        """(failure, frame, end frame, recovered) of a pick_place, for the failures the expert tries again after (an
        approach follows; the last failure before giving up has none). A try ends at the next failure or once the
        block is carried again (its next transport), whichever comes first; recovered: the block is carried again in
        the end."""
        events = rec.get("events", [])
        carried = [u for u, m in events if m == "transport"]
        fails = [(t, name) for t, name in events if name in bs.FAILURES]
        out = []
        for k, (t, name) in enumerate(fails):
            if not any(m == "approach" and u >= t for u, m in events):
                continue
            stops = [u for u in carried if u > t] + [u for u, _ in fails[k + 1 :]]
            out.append((name, t, min(stops) if stops else rec["frames"].get("end", t), any(u > t for u in carried)))
        return out

    def recoveries(self):
        parents = {id(rec): k for k, (kind, rec, *_) in enumerate(self.timeline()) if kind == "pick_place"}
        out = []
        for rec in self.log:
            for failure, start, end, recovered in self.failures(rec):
                args = (failure, rec["category"], rec["color"], rec["arm"])
                out.append(dict(
                    type="recover", failure=failure, parent=parents[id(rec)], arm=rec["arm"], part=rec["part"],
                    label=rec["label"], frames=dict(start=int(start), end=int(end)), recovered=recovered,
                    caption=bs.recover_caption(*args), caption_head=bs.recover_caption(*args, view="head"),
                    caption_zh=bs.recover_caption(*args, zh=True),
                ))  # fmt: skip
        return out

    def phases(self, n):
        """Per-frame phase (bs.PHASES), its arm (0 left, 1 right, -1 none), failure event (bs.EVENTS) and whether a
        recovery is under way."""
        phase, arm = np.zeros(n, np.int16), np.full(n, -1, np.int8)
        event, recovering = np.zeros(n, np.int8), np.zeros(n, bool)
        for kind, rec, start, end in self.timeline():
            k = ARMS.index(rec["arm"])
            if kind == "retreat":
                phase[start:end], arm[start:end] = bs.PHASES.index("retreat"), k
                continue
            arm[start:end] = k
            marks = [(t, name) for t, name in rec.get("events", []) if name in bs.PHASES]
            for (t, name), (u, _) in zip(marks, [*marks[1:], (end, None)]):
                phase[t:u] = bs.PHASES.index(name)
            for t, name in rec.get("events", []):
                if name in bs.EVENTS and t < n:
                    event[t] = bs.EVENTS.index(name)
            for _, t, u, _ in self.failures(rec):
                recovering[t:u] = True
        return phase, arm, event, recovering

    def progress(self):
        """(per-frame progress arrays, json info) of house4_progress, computed once the episode is over."""
        if getattr(self, "_progress", None) is None:
            p = self.privileged_core()
            self._progress = hp.compute(p, self.subtasks(), self.final_check, self.entry["parts"])
        return self._progress

    def privileged(self):
        return dict(self.privileged_core(), **self.progress()[0])

    def privileged_core(self):
        arrays = self.frames()
        n = len(arrays["block_poses"])
        phase, phase_arm, event, recovering = self.phases(n)
        index = np.full(n, -1, dtype=np.int16)
        target = np.full(n, -1, dtype=np.int16)
        for k, (kind, rec, start, end) in enumerate(self.timeline()):
            index[start:end] = k
            if kind == "pick_place":
                target[start:end] = self.labels.index(rec["label"])
        rows, on = np.arange(n), target >= 0
        pick = np.where(on, target, 0)
        box = np.where(on[:, None], arrays.pop("head_bboxes")[rows, pick], np.nan)
        seen = np.where(on[:, None], arrays.pop("visible_bboxes")[rows, pick], np.nan)
        share = np.where(on, arrays.pop("visible_share")[rows, pick], np.nan)
        return dict(
            arrays,
            block_labels=np.array(self.labels, dtype="S"),
            head_intrinsics=self.K,
            subtask_index=index,
            target_block_index=target,
            target_bbox_head=box,
            target_visible_bbox_head=seen,
            target_visible_share=share,
            target_occluded=on & ~(share >= OCCLUDED),
            phase=phase,
            phase_names=np.array(bs.PHASES, dtype="S"),
            phase_arm=phase_arm,
            event=event,
            event_names=np.array(bs.EVENTS, dtype="S"),
            recovering=recovering,
            camera_intrinsics=np.array([K for _, K, _ in self.cams.values()]),
            camera_names=np.array(list(self.cams), dtype="S"),
        )

    def summary(self):
        if not self.entry:
            return {}
        progress = self.progress()[1] if len(self.history["block_poses"]) > 1 else None
        plan = [rec["part"] for rec in self.log]
        text = dict(instruction_en=bs.instruction(self.entry), instruction_zh=bs.instruction(self.entry, zh=True))
        if ORDERED and plan:
            order = list(dict.fromkeys(plan))
            where = {}
            for rec in self.log:
                where.setdefault(rec["part"], rec["start_pos"])
            where = where if self.entry.get("shape") in bs.DETAILED else None
            text = dict(
                instruction=bs.ordered_instruction(self.entry, order, where=where),
                instruction_en=bs.ordered_instruction(self.entry, order, where=where),
                instruction_zh=bs.ordered_instruction(self.entry, order, zh=True, where=where),
                instruction_original=text["instruction_en"],
                instruction_en_original=text["instruction_en"],
                instruction_zh_original=text["instruction_zh"],
                instruction_order=order,
                instruction_style="ordered",
            )
        return dict(
            progress=progress,
            shape=self.entry,
            plan=plan,
            **text,
            blocks={lb: dict(category=c, index=k, color=bs.color_name(c, k), name=bs.block_name(bs.kind(c, k), k))
                    if bs.kind(c, k) in bs.KINDS else dict(category=c, index=k) for lb, (c, k) in self.kinds.items()},
            head_camera=dict(name=CAMERA, size=list(self.size), intrinsics=self.K.tolist()),
            subtasks=self.subtasks(),
            recoveries=self.recoveries(),
            profile=PROFILE,
            expert_done=self.done,  # False: the expert gave up (a block not picked up) or ran out of steps
            final_check=self.final_check,
            **({"clearance": dict(reason=self.watch.reason, **{f"min_{k}_m": round(v, 4) for k, v in self.watch.min.items()},
                                  body_touch_m=clearance.BODY_TOUCH, hands_touch_m=clearance.HANDS_TOUCH)}
               if self.watch else {}),  # fmt: skip
        )


def audit(privileged, quality):
    """Recompute the acceptance from the recorded block poses: every house block on target in the last HOLD frames,
    one clip per house block in a support-respecting order. Raises AssertionError on a violation."""
    labels = [v.decode() for v in privileged["block_labels"]]
    poses = privileged["block_poses"]
    entry = quality["shape"]
    assert poses.shape == (quality["frames"], len(labels), 7)
    for frame in poses[-HOLD:]:
        assert bs.built(entry, {lb: (p[:3], p[3:]) for lb, p in zip(labels, frame.astype(float))})
    done = []
    for sub in quality["subtasks"]:
        if sub.get("type", "pick_place") != "pick_place":
            continue
        assert set(sub["supports"]) <= set(done), sub["part"]
        done.append(sub["part"])
    assert sorted(done) == sorted(entry["parts"])
    return dict(subtasks=len(done), frames=quality["frames"])
