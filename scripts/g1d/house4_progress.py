"""0..1 task progress of a house4 episode (flora_v2a style), from the simulator's ground truth.

Each house block is two milestones, pick and place, each with four key frames t_start <= t_contact <= t_effect <= t_end:
- pick: t_start the clip's start; t_effect the first frame of the block's last lift (bottom LIFTED above where it
  lay, held by the arm's pads) before the release; t_contact the start of the pad contact run that holds it there;
  t_end the transport phase after it. Success: the block was lifted.
- place: t_start the pick's t_effect; t_contact the start of the last lower phase (the block going down onto its
  support; the expert lets go a few mm above it, so the touchdown itself comes after the release); t_effect the first
  frame from the release on with the pads off the block (the gripper let go); t_end the clip's end. Success: the
  release happened and final_check of the part is ok.
Retreats are no milestones. N = 2 x the house's parts, so blocks never attempted count as failed too.

- progress (flora s5_derive): (k + alpha) / N; k = successful milestones done (t_effect) by frame t, alpha = linear
  (t - t_start) / (t_effect - t_start) inside a successful milestone. Plateaus between milestones.
- semantic_progress (flora s5_progress, linear fill): sum_m w_m a_m(t), weights normalized over all N milestones
  (every milestone closes or opens the gripper: GRIP_BOOST for all, so equal); a_m ramps linearly from the same arm's
  previous milestone's t_effect (frame 0 for its first) to its t_effect, starting once the arm moves, rising only on
  the frames the arm is the active one (phase_arm: the other arm's turn and the start-up jitter of the idle arm are
  its static frames) and holding still on recovering frames (a recovery is no progress); running max, at most 1.
  One arm works at a time, so the curve rises without plateaus.
- subtask_progress: linear 0..1 over the current subtask's frames [start, end) (retreats too), 0 before the first.
Failed milestones keep their share but never fill it: a rejected episode stays below 1.
"""

import numpy as np

VERSION = "house4_progress_v1"
CONTACT_N = 0.05  # N: the pads touch the block
LIFTED = 0.015  # m: the block's bottom this far up = lifted (the expert's grasp_fail test)
MOVING = 0.005  # m/s: grasp point speed above which the arm moves
GRIP_BOOST = 2.0
ARMS = ("left", "right")


def _runs(mask):
    """(start, stop) of the True runs."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def milestones(p, subtasks, final_check, n):
    """Pick and place key frames of each pick_place subtask (in time order)."""
    phase_names = [v.decode() if isinstance(v, bytes) else v for v in p["phase_names"]]
    labels = [v.decode() if isinstance(v, bytes) else v for v in p["block_labels"]]
    phase, poses = p["phase"], p["block_poses"]
    force = p.get("pad_contact_force")
    out = []
    for s in subtasks:
        if s.get("type", "pick_place") != "pick_place":
            continue
        f, a, i = s["frames"], ARMS.index(s["arm"]), labels.index(s["label"])
        start, end = int(f["start"]), int(min(f.get("end", n), n))
        release = f.get("release")
        z = poses[:, i, 2]
        held = force[:, a, :, i].sum(1) > CONTACT_N if force is not None else np.zeros(n, bool)
        if force is None and "grasp" in f:  # no contact report: the gripper is closed from grasp to release
            held[f["grasp"] : release or end] = True
        lifted = (z - z[start] > LIFTED) & held
        lifted[:start] = False
        lifted[release if release is not None else end :] = False
        runs = _runs(lifted)
        pick = dict(skill="pick", part=s["part"], label=s["label"], arm=s["arm"], subtask=s["index"], t_start=start)
        if runs:
            eff = int(runs[-1][0])
            con = eff
            while con > start and held[con - 1]:
                con -= 1
            tr = [t for t in np.flatnonzero(phase[eff:end] == phase_names.index("transport"))]
            pick.update(t_contact=int(con), t_effect=eff, t_end=int(eff + tr[0]) if tr else eff, status="success")
        else:
            pick.update(t_contact=None, t_effect=None, t_end=end, status="failure")
        out.append(pick)
        place = dict(skill="place", part=s["part"], label=s["label"], arm=s["arm"], subtask=s["index"])
        ok = bool((final_check or {}).get(s["part"], {}).get("ok"))
        if pick["status"] == "success" and release is not None:
            ps = pick["t_effect"]
            off = [int(t) for t in range(int(release), end) if not held[t]]
            eff = off[0] if off else end - 1
            lower = [ps + int(t) for t in np.flatnonzero(phase[ps:eff] == phase_names.index("lower"))]
            con = lower[-1] if lower else eff
            while con > ps and phase[con - 1] == phase_names.index("lower"):
                con -= 1
            place.update(t_start=ps, t_contact=con, t_effect=eff, t_end=end, status="success" if ok else "failure")
        else:
            place.update(t_start=pick["t_effect"] or start, t_contact=None, t_effect=None, t_end=end,
                         status="failure")  # fmt: skip
        out.append(place)
    return out


def compute(p, subtasks, final_check, parts, fps=30):
    """Per-frame progress arrays and the milestones (json-ready).     p: privileged arrays (block_poses, block_labels,
    phase, phase_names, phase_arm, recovering, ee_pose, subtask_index; pad_contact_force if recorded)."""
    n = len(p["phase"])
    ms = milestones(p, subtasks, final_check, n)
    total = max(1, 2 * len(parts))
    done = [m for m in ms if m["status"] == "success"]
    t = np.arange(n)
    # flora s5_derive
    k = np.zeros(n)
    alpha = np.zeros(n)
    for m in done:
        d, s0 = m["t_effect"], m["t_start"]
        k[d:] += 1
        span = max(1, d - s0)
        fr = np.arange(s0, d)
        alpha[fr] = np.maximum(alpha[fr], (fr - s0) / span)
    progress = np.minimum((k + alpha) / total, 1.0)
    # flora s5_progress, linear fill
    sp = np.zeros((n, 2))
    sp[1:] = np.linalg.norm(np.diff(p["ee_pose"][:, :, :3], axis=0), axis=-1) * fps
    recovering = np.asarray(p["recovering"], bool)
    w = GRIP_BOOST / (GRIP_BOOST * total)
    sem = np.zeros(n)
    last = {arm: 0 for arm in ARMS}
    for m in ms:
        a = ARMS.index(m["arm"])
        if m["status"] != "success":
            m["weight"], m["ramp_start"] = round(w, 6), None
            continue
        d = m["t_effect"]
        s0 = last[m["arm"]]
        mine = np.asarray(p["phase_arm"]) == a
        moving = np.flatnonzero((sp[s0 : d + 1, a] > MOVING) & mine[s0 : d + 1])
        s0 = s0 + int(moving[0]) if len(moving) else d
        active = mine[s0 : d + 1] & ~recovering[s0 : d + 1]
        active[0] = False
        cum = np.cumsum(active)
        ramp = cum / cum[-1] if cum[-1] > 0 else (np.arange(d + 1 - s0) >= d - s0).astype(float)
        a_m = np.zeros(n)
        a_m[s0 : d + 1] = ramp
        a_m[d:] = 1.0
        sem += w * a_m
        last[m["arm"]] = d
        m["weight"], m["ramp_start"] = round(w, 6), int(s0)
    semantic = np.minimum(np.maximum.accumulate(sem), 1.0)
    sub = np.zeros(n)
    idx = p["subtask_index"]
    for s in subtasks:
        st, en = int(s["frames"]["start"]), int(min(s["frames"].get("end", n), n))
        if en > st:
            sub[st:en] = (t[st:en] - st) / (en - st)
    sub[idx < 0] = 0.0
    arrays = dict(progress=progress.astype(np.float32), semantic_progress=semantic.astype(np.float32),
                  subtask_progress=sub.astype(np.float32))  # fmt: skip
    info = dict(version=VERSION, milestones_total=total, milestones_done=len(done), milestones=ms,
                final_progress=float(progress[-1]), final_semantic_progress=float(semantic[-1]))  # fmt: skip
    return arrays, info


def check(arrays, info, accepted):
    """Problems with the labels (empty when fine)."""
    bad = []
    for key in ("progress", "semantic_progress"):
        v = arrays[key]
        if not (np.isfinite(v).all() and v.min() >= 0 and v.max() <= 1 + 1e-6):
            bad.append(f"{key} out of [0, 1]")
        if np.any(np.diff(v) < -1e-6):
            bad.append(f"{key} decreases")
        if v[0] > 1e-6:
            bad.append(f"{key} starts at {v[0]:.3f}")
        full = info["milestones_done"] == info["milestones_total"]
        if full != (v[-1] > 1 - 1e-6):
            bad.append(f"{key} ends at {v[-1]:.3f} with {info['milestones_done']}/{info['milestones_total']} done")
    if accepted and info["milestones_done"] != info["milestones_total"]:
        bad.append("accepted but not every milestone done")
    for m in info["milestones"]:
        keys = [m[k] for k in ("t_start", "t_contact", "t_effect", "t_end") if m.get(k) is not None]
        if keys != sorted(keys):
            bad.append(f"{m['skill']} {m['part']} key frames out of order {keys}")
        if m["status"] == "success":
            d = m["t_effect"]
            if arrays["progress"][d] < arrays["progress"][max(0, d - 1)]:
                bad.append(f"{m['skill']} {m['part']} no step at t_effect")
    return bad
