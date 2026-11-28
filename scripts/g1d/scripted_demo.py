"""Scripted demos for the G1-D tasks: no policy server, privileged object poses.

Runs src/eval_client/main.py in-process with the policy swapped for a Cartesian script, tracked by damped
least-squares IK on the simulator's Jacobian. Videos and _result.json land in eval_result/ like a normal eval,
under policy name "scripted_demo".
    g1d_pour_water_into_cup   bimanual: one arm grasps the pitcher handle from the robot side, the other picks the cup
                              from the outside and holds it up, mouth leaning to the jug; the jug rolls about its lip
                              until a quarter of the beans have left it, the actual spout steered over the cup; back
                              upright
    g1d_pour_beans_cup_to_cup bimanual: each arm picks the plastic cup on its side from the outside and lifts it; the
                              target cup is held up in front of the chest, the source cup rolls about its rim lip over
                              the target's mouth until most beans are over; back upright
    g1d_pour_beans_measuring_cups  the same with the 500 ml (source) and 400 ml measuring cups
    g1d_general_pickup        top-down grasp across the target's narrow side, lift 15 cm
    g1d_pick_anything         the same for the named object among others, at the grip yaw of its spot, lift 13 cm
    g1d_stack_blocks          stack the two outer blocks on the middle one, each with the arm on its side; arms back
    g1d_build_block_pattern   the card's block pattern (scripts/objects/block_patterns.py) about its build space,
                              bottom-up: each block picked top-down by the arm on its side, turned to its place with the
                              fingers closing along the pattern's depth, set down, that arm home before the next
    g1d_deliver_cup_to_lounge (g1d_counter_mobile) pick the cup from the counter, back off, drive to a gap between the
                              chairs at the lounge's round table, face it and set the cup down
    g1d_pour_water_deliver_to_lounge (g1d_counter_mobile) the jug pour above with water, the jug set back down, then
                              the cup, left standing on the counter for the pour, carried to the lounge table as above
    g1d_build_house           (g1d_school) seven wooden blocks from both sides of the table built into a house, one
                              arm at a time (build_actions)

Usage (sim env, repo root):
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:XPolicyLab EVAL_NUM=1 python scripts/g1d/scripted_demo.py \
        --env_cfg_type g1d --task_name g1d_stack_blocks [--seed 0]
"""

import itertools
import os
from pathlib import Path
import runpy
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from env.environment.isaaclab_compat import pose_wxyz  # noqa: E402
from env.reward_manager.pattern_match import pattern_error, yaw_of  # noqa: E402
from scripts.objects import (
    block_atomic,  # noqa: E402
    block_patterns,  # noqa: E402
    block_sorting,  # noqa: E402
    cup_profile,  # noqa: E402
    measuring_cup_shape as cup_shape,  # noqa: E402
    pick_anything,  # noqa: E402
    plastic_mug_shape as mug_shape,  # noqa: E402
    round_cup_shape,  # noqa: E402
)
from scripts.g1d.reach_map import HAND, HAND_NAME  # noqa: E402

POLICY = "scripted_demo"

# Calibration knobs: metres, radians, actions (30 Hz)
PAD_REACH = HAND["reach"]  # hand frame -> grip point on the pads (reach_map.HANDS; Dex1 0.12, fingertips 2.3 cm further)
GRASP_IN_PITCHER = np.array([0.0617, 0.0, 0.059])  # handle bar centre (metadata functional.handle)
OPEN = 0.15  # 19 mm between the pads: the inner finger fits in the 26 mm slot between handle and body
BEHIND = np.array([0.0, -0.09, 0.0])  # fingertips clear the pitcher's back face; descend here, then slide in
SPOUT_IN_PITCHER = np.array([-0.0663, 0.0, 0.1145])  # spout lip, the tilt pivot
BBOX_HALF = np.array([0.0663, 0.0445, 0.1225])
# lip short of the cup centre, inside the near rim of the 7.3 cm mouth (the cup hand holds the body below it)
POUR_BACKOFF = 0.025
SPOUT_CLEAR, BODY_CLEAR = 0.03, 0.02  # above the cup rim
TILT = np.deg2rad(125)
TILT_RATE, RETURN_RATE = np.deg2rad(2.0), np.deg2rad(4.0)  # per action
TILT_DRY, DRY_RATE = np.deg2rad(80), np.deg2rad(4.0)  # nothing leaves the jug below this: tilt faster
JUG_SPEED = 0.0035  # m per action carrying the jug: faster swings it about the handle and it slips out
POUR_TARGET = 0.25  # of the half-full jug (~130 ml, 2/3 of a mug); the stream in flight tops it up
FLOW = 0.0033  # of the beans per action (~50 ml/s): hold the tilt above this
STEP_MARGIN = 12  # actions kept free after the jug is back upright, for the rising-edge success check
# Real G1-D motion limits (Limiter): joints this far inside the URDF limits (the real arm server's margin); the
# fastest arm joint speed in the real replays (elbow ~0.8 rad/s, calib toolkit docs/05); the Dex1-1's 0.2 m/s finger
# speed over its 44.5 mm stroke, per 30 Hz action (real2sim/robot/g1d_v4/assemble.json)
REAL_JOINT_MARGIN, REAL_ARM_VMAX, REAL_GRIP_RATE = 0.05, 0.8, HAND["grip_rate"]  # Dex1: 0.2 m/s over 44.5 mm
# Water in the jug (g1d_pour_water_deliver_to_lounge): it runs from ~45 deg; a third of the ~300 ml fill
WATER_DRY, WATER_POUR = np.deg2rad(40), 0.3
# tilt speeds scaled to stay under REAL_ARM_VMAX: the dry tilt's 4 deg/action is ~2 rad/s at the wrist, and clipped
# to 0.8 rad/s the hand fell 12 cm behind the spout target
WATER_RATE = 0.4
# Second arm holding the cup
CUP_LIFT = 0.015  # cup bottom above the table while held under the spout; higher puts the jug out of reach
CUP_MEET = 0.0  # the cup is carried this much toward the jug (the cup arm is near the edge of its reach)
# the cup's mouth leans toward the jug to meet the stream, in step with the jug's tilt. With a top grasp on a mug the
# hand sat over the mouth and the tilted jug hit it; held 13-20 cm up, the jug hand cannot reach its pour pose.
CUP_TILT = np.deg2rad(0)
SPOUT_GAIN, SPOUT_SLACK = 0.2, 0.05  # per action, share of the spout error (actual vs over the actual cup) fed back
# side grasp on the cup body (probed at the cup): fingers horizontal miss it by 6-14 cm at any height and wrist yaw;
# 35 deg down with the wrist turned 30-60 deg inward reaches it within 1-3 cm, 8-10 cm above the bottom. At 9 cm the
# wrist camera module on top of the gripper stands above the rim and the tilted jug hits it; at 6.5 cm it stays below
CUP_GRIP_UP = 0.065  # pads this far above the cup bottom
CUP_PITCH, CUP_YAW = 35, 45  # deg, fingers below horizontal; wrist turned inward from fingers-forward
CUP_SLIDE = 0.08  # m the open pads slide along the fingers onto the cup body
# Cup-to-cup pour (two 9.5 cm plastic cups, as in the real G1D-PourBeansPlus clips, or the 14 / 17 cm measuring
# cups): pads this far below the rim. Measuring cups: 45 mm keeps the pads clear of the 75 mm flange 12.5 mm down;
# at 32 mm the 17 cm cup pivoted ~45 deg in the grip (16/18 strict pours at 45 mm, 2 failures one spilled bean each)
C2C_RIM_DOWN = {"plastic_cup": 0.02, "measuring_cup": 0.045, "plastic_mug": 0.03}  # mug: pads under its handle top
C2C_RAISE = 0.06  # straight up first
# target cup bottom lifted, by its kind; the mug's mouth then sits about as high as the short measuring cup's (at 0.12
# the tall cup's hand could not reach over it)
C2C_LIFT = {"plastic_cup": 0.12, "measuring_cup": 0.06, "plastic_mug": 0.08}
# fingers this far below horizontal. The cup tilts by rolling about the fingers, cos(tilt) = sin^2 p + cos^2 p cos(r):
# at 35 deg the ~100 deg the beans need in the tall measuring cup takes ~140 deg of roll; the arm reaches ~116
C2C_PITCH = {"plastic_cup": CUP_PITCH, "measuring_cup": 20, "plastic_mug": CUP_PITCH}  # per cup, by its kind
# the square measuring cups stand at any yaw: the wrist turns beyond CUP_YAW (deg, outward .. inward) so the pads close
# across a pair of flats (they repeat every 90 deg); past this window the pads sit off the flats by the rest. Probed at
# the table centre: 25 deg further inward knocks the cup over, 44 deg outward misses the pour
C2C_FLAT_TURN = (-35, 15)
C2C_MEET_X, C2C_MEET_Y = 0.03, -0.36  # target cup axis in mid-air: this far on its own side of the midline
CUP_SPEED = 0.005  # m per action carrying a cup
# source lip this far inside the target's near rim, this high above it. Lying flat, the source cup reaches a cup width
# past its lip: further in, its far rim sweeps into the wrist camera module of the hand holding the target
LIP_IN, LIP_CLEAR = 0.010, 0.035
C2C_TILT, C2C_DRY = np.deg2rad(150), np.deg2rad(60)  # most tilt; nothing leaves a sixth-full cup below C2C_DRY
C2C_DRY_RATE, C2C_TILT_RATE = np.deg2rad(2.0), np.deg2rad(1.0)  # per action; faster and the arm lags, the cup slips
C2C_SETTLE, C2C_SETTLE_ERR = 40, 0.01  # most actions held in mid-air until both grasp points are this close
C2C_POUR, C2C_FLOW = 0.85, 0.004  # of the beans out of the source cup; tilt further while less leaves per action
# g1d_pour_beans_half_cup: the tall cup is full as the real one (~3100 beans) and sheds beans from ~88 deg; stop
# rolling once HALF_FILL of the target cup's capacity has left it (more follows while it rolls back), then put both
# cups back where they stood and let go
HALF_FILL = 0.4
# the first, empty roll at the real ~35 deg/s (eps380 FK: 0 to 70 deg in ~2 s); at C2C_DRY_RATE (60 deg/s) a hand
# still catching up with the meet point jerks the full cup and throws beans over the rim
HALF_DRY_RATE = np.deg2rad(1.0)  # per action
# where the real hands meet (eps380_episodes.py FK at the frame the right wrist rolls most): the target cup 40.7 cm
# ahead of the robot base and 3.3 cm left of it. The real cup is lifted 6 cm; the real tall cup rolls to 87-99 deg
# (median 96, eps380 FK) as here. Rolled to ~100 deg over a cup that low, the tall cup's body knocked the target hand
# (0-80% in the cup), so lift it 12 cm as the other cup-to-cup pours and hold it 3.5 cm under the rim: the hand then
# stays under the tall cup's sweep (4/4 episodes, no bean spilled; 2 cm under: the hands met, 2026-11-01). 40.7 cm
# also keeps the tall cup in reach: at 33-37 cm its hand fell 6-9 cm short of the meet and the jerk threw out 10-22%
HALF_MEET, HALF_LIFT, HALF_RIM_DOWN = (0.033, 0.407), 0.12, {"plastic_cup": 0.035}
# g1d_pour_beans_plus: cups of any shape, by their profile (scripts/objects/cup_profile.py) instead of their kind.
# Pads PLUS_RIM_DOWN (share of the height + m, clipped) below the rim; fingers PLUS_PITCH deg down (the tall cups
# need the measuring cups' shallower pitch to roll far enough); the target's rim held PLUS_MOUTH above the desk
# (the twin plastic cup, 9.5 cm: C2C_LIFT). Pads PLUS_RIM_DOWN (share of the height) under the rim, at least
# PLUS_RIM_MIN, not in the lowest share PLUS_RIM_DOWN[1]: 2 cm under the rim, as C2C_RIM_DOWN, put the rolled source
# hand on the target hand at ~95 deg (2026-11-03). Pads PLUS_PAD tall hold a wall within PLUS_SLOPE deg of vertical
# (friction 0.5: a cone steeper than ~27 deg slides out). At most tilt, the pour stops after PLUS_STALL actions
# without a bean leaving (some always stick in the corners); it rolls back while more than PLUS_SURGE of the beans
# leave per action
PLUS_RIM_DOWN, PLUS_RIM_MIN, PLUS_PAD, PLUS_SLOPE = (0.4, 0.25), 0.02, 0.02, 10
PLUS_PITCH, PLUS_TALL = (CUP_PITCH, 20), 0.14
PLUS_MOUTH, PLUS_LIFT, PLUS_STALL, PLUS_SURGE = 0.215, (0.05, 0.12), 60, 0.015
# The source cup's tilt from its cavity: a bean at any height leaves once the wall from there to the rim slopes down,
# tilt > 90 deg - the least wall flare over >= PLUS_CHORD of height + atan(grain friction 0.3, the "min" combine of
# pour_beans_plus_layouts.GRAIN_COMBINE) + PLUS_TILT_MARGIN (heaps in the corners), within PLUS_NEED. The fingers
# take the steepest pitch of PLUS_PITCHES (deg; the grasp's reach) that gets there within PLUS_ROLL_NEED of roll;
# the roll stops PLUS_ROLL_SPARE past it, at most PLUS_ROLL_MAX (B probe 2026-11-04: 170 deg rolled cleanly; at 30
# deg the loaded cups slid in the hand or fell out of it, C probe; the cups tilt 10-25 deg short of the planned roll
# under load, E probe: 147 deg of roll tipped a cup to 94 deg and the pour ran dry at the step limit)
PLUS_CHORD, PLUS_TILT_MARGIN, PLUS_NEED = 0.015, 8.0, (100.0, 135.0)
PLUS_PITCHES, PLUS_ROLL_NEED, PLUS_ROLL_SPARE, PLUS_ROLL_MAX = (25, 20), 160.0, 30.0, 170.0
# The stream the target mouth takes: a surge is PLUS_SURGE_MOUTH of its area in grain sections (volume-equivalent
# diameter) per action (the twin cup, red beans: ~7), a weak stream a PLUS_WEAK share of that; past PLUS_SLOW deg of
# tilt (the last heap slides all at once, B probe: 7/16 spills went over the far rim at 103-112 deg) the roll is
# PLUS_SLOW_RATE of C2C_TILT_RATE
PLUS_SURGE_MOUTH, PLUS_SURGE_GRAINS, PLUS_WEAK = 0.045, (1.5, 12.0), 0.25
PLUS_SLOW, PLUS_SLOW_RATE = np.deg2rad(95), 0.5
# A surge rolls the cup back C2C_TILT_RATE times how many surges leave per action, at most PLUS_SURGE_BACK times, and
# drops the lip to PLUS_CLEAR_MIN over the rim: a heap that slides at once (rigid beans, 32 GPU partitions,
# 2026-11-10: 5 -> 45 grains an action in 4 actions at a steady 99 deg) outran a back-off of one rate and flew over
# the far rim
PLUS_SURGE_BACK = 4.0
# Spills (privileged: a grain out of both cavities, PLUS_SPILL_DROP under the target rim, off its mouth): each new one
# rolls the cup back PLUS_SPILL_BACK and holds the tilt PLUS_SPILL_CALM actions, lowers the lip PLUS_SPILL_LOWER (to
# PLUS_CLEAR_MIN over the rim) and moves the lip target PLUS_AIM_GAIN of the spill's offset along the pour back toward
# the mouth centre (over the far rim: back; short of the near rim: further in), at most PLUS_AIM_STEP at a time,
# within PLUS_AIM. Under PLUS_SPILL_FROM of tilt no bean runs over the rim: beans out then fell in the grasp or the
# lift and neither steer the lip nor end the dry roll (C probe 2026-11-04: such a bean pushed the lip 1.5 cm in, and
# the full cup's slide at 85 deg then went over the far rim)
PLUS_SPILL_DROP, PLUS_SPILL_BACK, PLUS_SPILL_CALM, PLUS_SPILL_FROM = 0.01, np.deg2rad(2.0), 15, np.deg2rad(30)
# The dry roll also ends once the highest bean is PLUS_LIP_NEAR under the lowest rim point: a full cup's heap then
# slides out at once (C probe: 15% at 54 deg, over the far rim)
PLUS_LIP_NEAR = 0.01
PLUS_SPILL_LOWER, PLUS_CLEAR_MIN, PLUS_AIM_GAIN, PLUS_AIM_STEP, PLUS_AIM = 0.003, 0.015, 0.3, 0.005, (-0.02, 0.015)
# At most tilt the arm runs out of roll (cups stop at 105-108 deg) with 1-3% wedged in the corners: a stall first
# jolts the cup PLUS_SHAKE along the pour and back, PLUS_SHAKE_ACTS actions, up to PLUS_SHAKES times (office
# feasibility 2026-11-04: beans left in the source 1.6 -> 0.7 an episode)
PLUS_SHAKES, PLUS_SHAKE, PLUS_SHAKE_ACTS = 3, 0.008, 12
PLACE_CLEAR = 0.006  # m above the cup's start grip height when it is let go
PLACE_OPEN, PLACE_BACK = 20, 0.06  # actions to open the gripper; m to back off (along the slide-in) and rise
# g1d_pour_beans_plus: the poured cup first rises clear of the target's mouth (E probe 2026-11-04: turned 15 deg in the
# hand, it swept the full target cup over on the way back)
PLACE_RISE = 0.04
LIP_GAIN, LIP_SLACK = 0.2, 0.04  # per action, share of the lip error fed back; bound on the correction
DAMPING = 0.05
GRASP_ROT = Rotation.from_euler("z", 90, degrees=True)  # fingers forward (+y), wrist camera up: the G1-D zero pose
# Pick and place
# fingers 60 deg below horizontal, closing along x: reaches ~8 cm further at table height than straight down
TOP_ROT = Rotation.from_euler("x", -60, degrees=True) * GRASP_ROT
PAD_CLOSED, PAD_TRAVEL = HAND["pad_gap"]  # pad gap (m) at grip=0, m more at 1 (Dex1: 6 .. 95 mm, 2 * 44.5 mm stroke)
GRASP_CLEAR = 0.025  # open the pads this much wider than the grasped side so neighbours are not swept in
FINGER_TIPS = HAND["finger_tips"]  # grip point -> fingertips
TABLE_CLEAR = 0.006  # fingertips above the table
HOVER, LIFT = 0.06, 0.15  # 6 cm hover keeps 3-block stacks below the wrist-pitch limit; pickup lifts 15 cm
PLACE_GAP = 0.004  # held object's bottom above the surface it is released on
LABEL = "__expert__"  # action key: {arm_key: noise-free joint command} when the executed one carries v4 noise
V4_CLEAR = 0.045  # m between a moved block's footprint and the others' (build_house4_layouts.CLEAR)
NOISE_EASE = 10  # actions a v4 noise offset decays over at the end of its segment
NOTICE = (8, 20)  # actions the hand pauses over a block that was just moved away
# Mobile (g1d_mobile): chassis cruise speeds, ramp, tolerances
DRIVE_V, DRIVE_W, DRIVE_ACC = 0.35, 0.5, 0.5 / 30  # m/s, rad/s; m/s per action (a held cup slips at hard starts)
DRIVE_TOL, HEADING_TOL, TURN_FIRST = 0.03, 0.03, 0.25  # m, rad; turn on the spot first when off course by more
SETTLE = 15  # actions standing still after a drive
# m straight back from the counter before turning: the held cup clears its front edge, and the chassis stays clear of
# the bar stools on the far side of the aisle (real_scene2; 0.6 m already touches them)
BACK_OFF = 0.5
# Lounge table: root this far outside its rim; the cup set down this far ahead of the root; approach from this far out
TABLE_STAND, PLACE_AHEAD, APPROACH = 0.20, 0.38, 0.5
CHAIR_GAP = np.deg2rad(70)  # narrowest gap between the chairs round the table to stand in
LOUNGE_LIFT = 0.07  # lift column at the 0.75 m table: 0.07 m above 0.68 m, as the counter's 0.153
CARRY_UP = 0.08  # m the side-grasped cup is lifted off the counter for the drive (clears the chair backs, 0.8 m)


def pad_open(width, robot=None):
    """Gripper command opening the pads GRASP_CLEAR wider than `width`; `robot.pad_gap` (m at 0, m more at 1), else
    the Dex1-1's."""
    closed, travel = getattr(robot, "pad_gap", (PAD_CLOSED, PAD_TRAVEL))
    return float(np.clip((width + GRASP_CLEAR - closed) / travel, 0.0, 1.0))


def hold_cmd(arm, squeeze=HAND["squeeze"]):
    """Gripper command `squeeze` (m per finger) past where the fingers stopped on the object: the linear PD presses
    with stiffness x squeeze instead of the effort limit. Dex1-1: (0.023 - fingers) / 0.0445."""
    robot = arm.robot
    q = as_np(arm.key.data.joint_pos[0, list(robot.gripper_joint_indices)])
    mimic, gain, offset = robot.gripper_move["mimic"]
    base = np.mean([(v - offset) / gain if n == mimic else v for n, v in zip(robot.gripper_joints_name, q)])
    lo, hi = robot.gripper_scale
    cmd = (base - squeeze - lo) / (hi - lo) if robot.gripper_move["sign"] == 1 else (hi - base - squeeze) / (hi - lo)
    return float(np.clip(cmd, 0.0, 1.0))


def as_np(x):
    return np.asarray(x.cpu() if hasattr(x, "cpu") else x, dtype=float).reshape(-1)


def link_jacobians(art):
    """(envs, links - 1, 6, dofs) at the link origins in world, PhysX's fixed-base layout, on every sim backend."""
    view = art.root_view if hasattr(type(art), "root_view") else art.root_physx_view  # IsaacLab 3 / 2 and MuJoCo
    if not hasattr(view, "get_jacobians"):
        from env.backends.newton import physics

        return physics.link_jacobians(view)
    jac = view.get_jacobians()
    if not hasattr(jac, "cpu"):
        import warp as wp

        jac = wp.to_torch(jac)
    return jac


def from_wxyz(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def waypoints(p, rot, keys):
    """Smoothstep-interpolated (grasp point, hand rotation, gripper) per action through `keys` [(point, rotation,
    gripper, actions[, lead])]: the rotation is done after the first `lead` share of the actions (default all)."""
    for p1, rot1, grip, n, *lead in keys:
        slerp = Slerp([0, 1], Rotation.concatenate([rot, rot1]))
        lead = lead[0] if lead else 1.0
        for i in range(1, n + 1):
            s, r = i / n, min(i / (n * lead), 1.0)
            yield p + (p1 - p) * s * s * (3 - 2 * s), slerp(r * r * (3 - 2 * r)), grip
        p, rot = p1, rot1


class Arm:
    REACH = np.array([PAD_REACH, 0.0, 0.0])  # ee link -> grasp point, ee link frame

    def __init__(self, env, robot):
        rm = env.robot_manager
        self.robot, self.key = robot, rm.robot_key[rm.robot_list.index(robot)]
        self.body = self.key.body_names.index(robot.ee_link_name)
        self.cols = list(robot.arm_joint_indices)
        self.origin = as_np(env.scene_manager.env_origins[0])
        self.arm_key, self.grip_key = rm.process_name(robot.arm_name), rm.process_name(robot.gripper_name)

    def q(self):
        return as_np(self.key.data.joint_pos[0, self.cols])

    def root(self):
        return as_np(self.key.data.root_pos_w[0])[:3] - self.origin

    def pose(self):
        link = as_np(pose_wxyz(self.key.data.body_link_pose_w[0, self.body]))
        return link[:3] - self.origin, from_wxyz(link[3:])

    def grasp_point(self):
        pos, rot = self.pose()
        return pos + rot.apply(self.REACH)

    def ik(self, point, rot_t):
        pos, rot = self.pose()
        r = rot.apply(self.REACH)
        err = np.r_[point - (pos + r), 0.3 * (rot_t * rot.inv()).as_rotvec()]
        jac = as_np(link_jacobians(self.key)[0, self.body - 1][:, self.cols]).reshape(6, -1)
        jac = np.r_[jac[:3] - np.cross(r, jac[3:].T).T, 0.3 * jac[3:]]
        q = self.q()
        lim = as_np(self.key.data.soft_joint_pos_limits[0, self.cols]).reshape(-1, 2) + [0.025, -0.025]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.02**2 * np.eye(6), err)
        jac[:, (q + dq <= lim[:, 0]) & (dq < 0) | (q + dq >= lim[:, 1]) & (dq > 0)] = 0.0
        return np.clip(q + jac.T @ np.linalg.solve(jac @ jac.T + 0.02**2 * np.eye(6), err), lim[:, 0], lim[:, 1])


def flat_turn(rot, out):
    """Wrist yaw (deg, world z) beyond out * CUP_YAW that closes the pads across a pair of a square cup's flats, within
    C2C_FLAT_TURN: the pads close along world angle out * CUP_YAW (= 45 mod 90), the flats face the cup's yaw + k * 90
    deg, and out * turn > 0 turns the wrist further inward. Past the window: the edge nearest a pair of flats."""
    want = out * (np.degrees(rot.as_euler("zyx")[0]) % 90 - 45)
    edges = (np.clip(w, *C2C_FLAT_TURN) for w in (want - 90, want, want + 90))
    inward = min(edges, key=lambda c: abs((c - want + 45) % 90 - 45))
    return out * float(inward)


def cup_grasp(lm, out, label="cup", grip_up=CUP_GRIP_UP, pitch=CUP_PITCH, turn=0.0):
    """Side grasp around the body of the (handleless, tapered) cup by the arm on side `out` (+-1 in x): fingers
    `pitch` deg down, turned CUP_YAW (+ `turn`) inward, pads closing across the body below the rim; the mouth stays
    free.

    Returns the grasp point (on the body axis), hand rotation, rim height, rim radius and gripper opening.
    """
    pos, quat = lm.get_instance_pose(0, label=label)
    pos, rot = as_np(pos), from_wxyz(as_np(quat))
    meta = lm.get_instance_metadata(0, label=label) or {}
    bbox = lm.get_instance_bbox_vertices(lm.get_instance_name(0, label), 0)
    lo, hi = bbox.min(0), bbox.max(0)
    circle = meta.get("active", {}).get("place", {}).get("up", {}).get("contact_circle", {})
    axis = pos + rot.apply(np.r_[circle.get("center", (lo + hi) / 2)[:2], 0])
    rim, radius = pos[2] + hi[2], (hi[1] - lo[1]) / 2
    foot = circle.get("radius", radius)
    up = min(grip_up, hi[2] - lo[2])
    grip = np.r_[axis[:2], pos[2] + lo[2] + up]
    hand = Rotation.from_euler("xz", [-pitch, out * CUP_YAW + turn], degrees=True) * GRASP_ROT
    return grip, hand, rim, radius, pad_open(2 * (foot + (radius - foot) * up / (hi[2] - lo[2])))


def plus_grip(profile, scale):
    """g1d_pour_beans_plus: (pads above the cup's underside, fingers' pitch in deg) for a cup by its profile: where
    its wall is within PLUS_SLOPE of vertical (the pads squeeze a cone out of the grip), nearest PLUS_RIM_DOWN, at
    least PLUS_RIM_MIN under the rim; the least sloped height when it is a cone throughout."""
    bottom, top = profile["bottom"] * scale, profile["top"] * scale
    height = top - bottom
    share, low = PLUS_RIM_DOWN
    z = np.linspace(bottom + low * height, top - PLUS_RIM_MIN, 40)
    span = np.r_[-1, 1] * PLUS_PAD / 2
    radius = np.array([[cup_profile.body_radius(profile, zi + d, scale) for d in span] for zi in z])
    slope = np.abs(radius[:, 1] - radius[:, 0]) / PLUS_PAD
    want = top - share * height
    ok = slope <= np.tan(np.radians(PLUS_SLOPE))
    pick = z[ok][np.argmin(np.abs(z[ok] - want))] if ok.any() else z[np.argmin(slope)]
    return float(pick - bottom), PLUS_PITCH[height > PLUS_TALL]


def plus_tilt(profile, tall=False):
    """g1d_pour_beans_plus source cup: (tilt it needs, fingers' pitch, most roll), deg, from its cavity (PLUS_CHORD)."""
    z, r = np.asarray(profile["z"])[1:], np.asarray(profile["r"])[1:]  # the floor sample reads the fillet
    dz, dr = z[None, :] - z[:, None], r[None, :] - r[:, None]
    flare = np.degrees(np.arctan2(dr, dz))[dz >= PLUS_CHORD].min() if (dz >= PLUS_CHORD).any() else 0.0
    need = float(np.clip(90.0 - flare + np.degrees(np.arctan(0.3)) + PLUS_TILT_MARGIN, *PLUS_NEED))
    for pitch in [p for p in PLUS_PITCHES if not tall or p <= PLUS_PITCH[1]]:
        p = np.radians(pitch)
        cos_roll = (np.cos(np.radians(need)) - np.sin(p) ** 2) / np.cos(p) ** 2
        if cos_roll >= -1 and (roll := np.degrees(np.arccos(cos_roll))) <= PLUS_ROLL_NEED:
            return need, pitch, min(roll + PLUS_ROLL_SPARE, PLUS_ROLL_MAX)
    return need, PLUS_PITCHES[-1], PLUS_ROLL_MAX


def profile_grasp(lm, out, label, profile, scale, grip_up, pitch, turn=0.0):
    """cup_grasp of a cup by its profile (scripts/objects/cup_profile.py, at `scale`): pads `grip_up` above its
    underside on the body axis, opened to the widest body (handle left out) they pass sliding in along the pitched
    fingers. Same returns."""
    pos, quat = lm.get_instance_pose(0, label=label)
    pos, rot = as_np(pos), from_wxyz(as_np(quat))
    axis = pos + rot.apply(np.r_[np.asarray(profile["axis"]) * scale, 0.0])
    bottom, top = profile["bottom"] * scale, profile["top"] * scale
    grip = np.r_[axis[:2], pos[2] + bottom + grip_up]
    hand = Rotation.from_euler("xz", [-pitch, out * CUP_YAW + turn], degrees=True) * GRASP_ROT
    width = plus_pad_width(profile, scale, grip_up, pitch)
    return grip, hand, pos[2] + top, cup_profile.body_radius(profile, top, scale), pad_open(width)


def plus_pad_width(profile, scale, grip_up, pitch):
    """Widest body (m, handle left out) the pads pass sliding in along the pitched fingers to `grip_up`."""
    bottom, top = profile["bottom"] * scale, profile["top"] * scale
    path = bottom + grip_up + np.linspace(-PLUS_PAD / 2, CUP_SLIDE * np.sin(np.radians(pitch)) + PLUS_PAD / 2, 8)
    return 2 * max(cup_profile.body_radius(profile, min(zi, top), scale) for zi in path)


def pour_actions(env, fluid_label="wine", dry=TILT_DRY, pour_target=POUR_TARGET, hold_cup=True, rate=1.0):
    """Jug and cup as the module docstring says; fast tilt up to `dry`, then pour until `pour_target` of the jug's
    fill has left it. Water runs from ~45 deg, the beans from TILT_DRY. Without `hold_cup` the cup stays on the table
    and the other arm stays where it is. `rate` scales the tilt speeds."""
    lm = env.scene_manager.layout_manager
    pos, quat = lm.get_instance_pose(0, label="bottle")
    pitcher_pos, pitcher_rot = as_np(pos), from_wxyz(as_np(quat))
    fluid = lm.get_scene_object(0, lm.get_instance_name(0, fluid_label))

    arms = [Arm(env, r) for r in env.robot_manager.robot_list if r.type == "target"]
    arm = min(arms, key=lambda a: abs(a.grasp_point()[0] - pitcher_pos[0]))
    cup_arm = next(a for a in arms if a is not arm)
    cup0 = as_np(lm.get_instance_pose(0, label="cup")[0])
    side = np.sign(cup0[0] - pitcher_pos[0])  # the spout faces the cup (build_asset.py)
    cup_grip, cup_hand, rim0, cup_r, cup_open = cup_grasp(lm, side)  # the cup arm is beyond the cup
    held_grip = cup_grip + ([-side * CUP_MEET, 0, CUP_LIFT] if hold_cup else 0.0)
    meet = held_grip[:2]
    rim = rim0 + CUP_LIFT

    grasp = pitcher_pos + pitcher_rot.apply(GRASP_IN_PITCHER)
    corners = np.array(list(itertools.product(*zip(-BBOX_HALF, BBOX_HALF)))) - GRASP_IN_PITCHER

    def pour(theta, centre=meet, top=rim):
        """Grasp point and hand rotation tilting the pitcher about its lip, held over the cup; and the lip target."""
        tilt = Rotation.from_rotvec([0, side * theta, 0])
        lip = (tilt * pitcher_rot).apply(SPOUT_IN_PITCHER - GRASP_IN_PITCHER)
        low = (tilt * pitcher_rot).apply(corners)[:, 2].min() - lip[2]
        target = np.array([centre[0] - side * POUR_BACKOFF, centre[1], top + max(SPOUT_CLEAR, BODY_CLEAR - low)])
        return target - lip, tilt * GRASP_ROT, target

    start = pour(0)[0]
    mid = np.r_[grasp[:2], start[2]]  # straight up first: pulled sideways while standing, the jug tips over

    def carry(a, b):
        return max(int(np.ceil(np.linalg.norm(b - a) / JUG_SPEED)), 12)

    jug_keys = [
        (grasp + BEHIND + [0, 0, 0.1], GRASP_ROT, OPEN, 30),
        (grasp + BEHIND, GRASP_ROT, OPEN, 20),
        (grasp + BEHIND, GRASP_ROT, OPEN, 8),  # settle
        (grasp, GRASP_ROT, OPEN, 48),
        (grasp, GRASP_ROT, 0, 18),  # close
        (mid, GRASP_ROT, 0, carry(grasp, mid)),
        (start, GRASP_ROT, 0, carry(mid, start)),
    ]
    cup_side = cup_grip - cup_hand.apply([CUP_SLIDE, 0, 0])
    cup_keys = [
        (cup_side + [0, 0, HOVER], cup_hand, cup_open, 54),
        (cup_side, cup_hand, cup_open, 30),
        (cup_grip, cup_hand, cup_open, 40),  # slide in
        (cup_grip, cup_hand, 0, 15),  # close
        (cup_grip + [0, 0, CUP_LIFT], cup_hand, 0, 18),
        (held_grip, cup_hand, 0, 20),  # carry toward the jug
    ]
    check = env.pour_check()

    def beans():
        return np.asarray(fluid.get_particle_positions()[0]).reshape(-1, 3) + fluid.init_pos

    def jug():
        jug_pos, jug_quat = lm.get_instance_pose(0, label="bottle")
        return as_np(jug_pos), from_wxyz(as_np(jug_quat))

    def left_jug():
        """Fraction of the beans outside the jug's bounding box (in the cup, in flight or spilled)."""
        jug_pos, jug_rot = jug()
        return np.mean(np.any(np.abs(jug_rot.inv().apply(beans() - jug_pos)) > BBOX_HALF, axis=1))

    axis_off = from_wxyz(as_np(lm.get_instance_pose(0, label="cup")[1])).inv().apply(cup_grip - cup0) * [1, 1, 0]
    cup_box = lm.get_instance_bbox_vertices(lm.get_instance_name(0, "cup"), 0)
    cup_box = cup_box[cup_box[:, 2] > cup_box[:, 2].max() - 1e-6]  # rim corners: the highest rim point of a leaning cup

    def in_cup():
        """Fraction of the beans inside the cup's rim cylinder, for the log."""
        pos, quat = lm.get_instance_pose(0, label="cup")
        pos, rot = as_np(pos), from_wxyz(as_np(quat))
        local = rot.inv().apply(beans() - pos) - axis_off
        return np.mean((np.hypot(local[:, 0], local[:, 1]) < cup_r) & (local[:, 2] < rim0 - cup0[2]))

    def act(jug_target, cup_target):
        nonlocal step
        if step % 10 == 0:
            jug_pos, jug_rot = jug()
            tilt = np.degrees(np.arccos(jug_rot.apply([0, 0, 1])[2]))
            lean = np.degrees(np.arccos(from_wxyz(as_np(lm.get_instance_pose(0, label="cup")[1])).apply([0, 0, 1])[2]))
            print(
                f"[scripted_demo] {step:3d} err jug {np.linalg.norm(arm.grasp_point() - jug_target[0]):.3f}"
                f" cup {np.linalg.norm(cup_arm.grasp_point() - cup_target[0]):.3f} jug {jug_pos.round(3)}"
                f" tilt {tilt:.0f} cup lean {lean:.0f} left jug {left_jug():.0%} in cup {in_cup():.0%}"
                f" check {env.reward_manager.call_func_parser(check, 0):.0f}",
                flush=True,
            )
        step += 1
        return {
            arm.arm_key: arm.ik(*jug_target[:2]),
            arm.grip_key: [float(jug_target[2])],
            cup_arm.arm_key: cup_arm.ik(*cup_target[:2]),
            cup_arm.grip_key: [float(cup_target[2])],
        }

    print(
        f"[scripted_demo] jug {pitcher_pos.round(3)} cup {cup0.round(3)} rim {rim0:.3f} jug arm {arm.arm_key}"
        f" cup arm {cup_arm.arm_key} grip {cup_grip.round(3)} open {cup_open:.2f}"
    )
    step = 0
    jug_path = list(waypoints(arm.grasp_point(), arm.pose()[1], jug_keys))
    idle = (cup_arm.grasp_point(), cup_arm.pose()[1], 1.0)
    cup_path = list(waypoints(cup_arm.grasp_point(), cup_arm.pose()[1], cup_keys)) if hold_cup else [idle]
    for j, c in itertools.zip_longest(jug_path, cup_path):
        yield act(j or jug_path[-1], c or cup_path[-1])
    held = as_np(lm.get_instance_pose(0, label="cup")[0])
    print(f"[scripted_demo] cup held at {held.round(3)} (from {cup0.round(3)})")

    fix = np.zeros(3)  # the jug slips in the handle grip: steer the actual spout over where the cup actually is

    def steer(theta):
        nonlocal fix
        phi = CUP_TILT * min(theta / TILT_DRY, 1.0)
        lean = Rotation.from_rotvec([0, -side * phi, 0])  # mouth toward the jug
        # the hand sits above the cup and would swing under the tilted jug: move the cup away so the hand stays put
        palm = cup_hand.apply([-PAD_REACH, 0, 0])
        shift = palm - lean.apply(palm)
        held = (held_grip + [shift[0], shift[1], 0], lean * cup_hand, 0)
        pos, quat = lm.get_instance_pose(0, label="cup")
        pos, rot = as_np(pos), from_wxyz(as_np(quat))
        top = (pos + rot.apply(cup_box))[:, 2].max()
        mouth = pos + rot.apply(axis_off + [0, 0, rim0 - cup0[2]])  # a leaning mouth moves toward the jug
        hand, hand_rot, lip_target = pour(theta, mouth[:2], top)
        jug_pos, jug_rot = jug()
        fix = np.clip(
            fix + SPOUT_GAIN * (lip_target - jug_pos - jug_rot.apply(SPOUT_IN_PITCHER)), -SPOUT_SLACK, SPOUT_SLACK
        )
        return act((hand + fix, hand_rot, 0), held if hold_cup else idle)

    returning = int(np.ceil(TILT / (rate * RETURN_RATE))) + STEP_MARGIN
    theta, prev = 0.0, 0.0  # tilt until enough beans have left the jug, then back upright
    while theta < dry:
        theta = min(theta + rate * DRY_RATE, dry)
        yield steer(theta)
    while (now := left_jug()) < pour_target and step < env.step_lim - returning:
        if now - prev < FLOW:  # tilt further only while the stream is weak: a surge overshoots the cup
            theta = min(theta + rate * TILT_RATE, TILT)
        prev = now
        yield steer(theta)
    while theta > 0:
        theta = max(theta - rate * RETURN_RATE, 0.0)
        yield steer(theta)
    for _ in range(STEP_MARGIN):
        yield steer(0.0)


def cup_to_cup_actions(
    env,
    pour_share=None,
    place=False,
    meet=None,
    lift=None,
    rim_down=None,
    dry_rate=C2C_DRY_RATE,
    carry_speed=CUP_SPEED,
    arrive_above=0.0,
    profiles=None,
):
    """Each arm side-grasps the cup on its side and lifts it; the target cup is held up in front of the chest and the
    source cup rolls about its rim lip over the target's mouth until `pour_share` (default C2C_POUR) of the beans have
    gone over; back upright. With `place`, both cups then go back down where they stood and the hands open and back
    off. `meet` (x to the target's side, y ahead; m) puts the target cup's mid-air axis relative to the robot base
    instead of at the world C2C_MEET point; `lift` (m) replaces the target kind's C2C_LIFT; `rim_down` overrides
    C2C_RIM_DOWN entries; `dry_rate` (rad per action) is the first roll's rate, up to C2C_DRY; `carry_speed` (m per
    action) the cups' carry speed; `arrive_above` (m): the source cup comes in this far above its pour start and then
    goes straight down to it. `profiles` ({label: (cup profile, scale)}, scripts/objects/cup_profile.py) takes each
    cup's grip, pitch, lift and cavity from its profile instead of its kind (cups of any shape); the cup frame is then
    the cup's root, its centre at axes[label]."""
    pour_share = C2C_POUR if pour_share is None else pour_share
    rim_down = {**C2C_RIM_DOWN, **(rim_down or {})}
    lm = env.scene_manager.layout_manager
    fluid = lm.get_scene_object(0, lm.get_instance_name(0, "beans"))
    src0 = as_np(lm.get_instance_pose(0, label="source_cup")[0])
    side = np.sign(src0[0])  # the source cup's side; the target is on the other one
    arms = [Arm(env, r) for r in env.robot_manager.robot_list if r.type == "target"]
    src_arm = min(arms, key=lambda a: abs(a.grasp_point()[0] - src0[0]))
    tgt_arm = next(a for a in arms if a is not src_arm)
    heights, kinds, turns, axes, pitches = {}, {}, {}, {}, {}
    for label, out in (("source_cup", side), ("target_cup", -side)) if profiles else ():
        cavity, scale = profiles[label]
        heights[label] = (cavity["top"] - cavity["bottom"]) * scale
        heights[label + "_grip"], pitches[label] = plus_grip(cavity, scale)
        kinds[label], turns[label] = None, 0.0
        axes[label] = np.r_[np.asarray(cavity["axis"]) * scale, (cavity["bottom"] + cavity["top"]) / 2 * scale]
    if profiles:
        need, pitches["source_cup"], most = plus_tilt(profiles["source_cup"][0], heights["source_cup"] > PLUS_TALL)
        print(f"[scripted_demo] source needs {need:.0f} deg of tilt: pitch {pitches['source_cup']}, roll to {most:.0f}")
        most = np.radians(most)
    for label, out in (("source_cup", side), ("target_cup", -side)) if not profiles else ():
        name = lm.get_instance_name(0, label)
        bbox = lm.get_instance_bbox_vertices(name, 0)
        heights[label] = bbox[:, 2].max() - bbox[:, 2].min()
        kinds[label] = kind = next(c for c in rim_down if name.startswith(c))
        heights[label + "_grip"] = heights[label] - rim_down[kind]
        rot = from_wxyz(as_np(lm.get_instance_pose(0, label=label)[1]))
        turns[label] = flat_turn(rot, out) if kind == "measuring_cup" else 0.0
        circle = (lm.get_instance_metadata(0, label=label) or {}).get("active", {}).get("place", {}).get("up", {})
        center = circle.get("contact_circle", {}).get("center", (0.0, 0.0))  # cup axis in the cup (bbox centre) frame
        axes[label] = np.r_[center[0], center[1], 0.0]
    if profiles:
        lift = float(np.clip(PLUS_MOUTH - heights["target_cup"], *PLUS_LIFT)) if lift is None else lift
        grasps = [
            profile_grasp(lm, out, label, *profiles[label], heights[label + "_grip"], pitches[label])
            for label, out in (("source_cup", side), ("target_cup", -side))
        ]
    else:
        lift = C2C_LIFT[kinds["target_cup"]] if lift is None else lift
        grasps = [
            cup_grasp(lm, out, label, heights[label + "_grip"], C2C_PITCH[kinds[label]], turns[label])
            for label, out in (("source_cup", side), ("target_cup", -side))
        ]
    (s_grip, s_hand, s_rim, s_r, s_open), (t_grip, t_hand, t_rim, t_r, t_open) = grasps
    height, t_height = heights["source_cup"], heights["target_cup"]
    up = heights["source_cup_grip"]  # pads above the cup bottom
    centre = axes["source_cup"] if profiles else np.zeros(3)  # of the source cup in its frame

    # the cup rolls about the fingers (the wrist roll); about world y the 35-deg-down hand runs the arm into its joint
    # limits within 15 deg, and fingers forward leave too little roll to empty the cup
    roll_axis = s_hand.apply([1.0, 0.0, 0.0])
    pour_dir = -side * np.cross(roll_axis, [0.0, 0.0, 1.0])  # the way the mouth tips: toward the target, slanted
    pour_dir /= np.linalg.norm(pour_dir)
    if meet is None:
        t_air = np.array([-side * C2C_MEET_X, C2C_MEET_Y, t_grip[2] + lift])
    else:
        root = src_arm.root()
        t_air = np.array([root[0] - side * meet[0], root[1] + meet[1], t_grip[2] + lift])
    # source cup frame (origin at the bbox centre): its rim point that tips lowest, its bbox corners; the grasp point
    # and hand rotation in it, as planned until measured in mid-air (the cup sits lower or turned in the soft grip)
    lip_c = centre + np.r_[s_r * pour_dir[:2], height / 2]
    corners_c = centre + np.array(list(itertools.product((-s_r, s_r), (-s_r, s_r), (-height / 2, height / 2))))
    cup_rot, grasp_c, hand_rot = Rotation.identity(), centre + np.array([0.0, 0.0, up - height / 2]), s_hand
    aim, clear = 0.0, LIP_CLEAR  # lip target moved along the pour, height over the rim (the spill feedback)

    def pour(theta, mouth, rim):
        """Source grasp point and hand rotation rolling the cup about its lip over the target mouth; the lip target."""
        tilt = Rotation.from_rotvec(-side * theta * roll_axis)
        rot = tilt * cup_rot
        low = rot.apply(corners_c - lip_c)[:, 2].min()
        lip = np.r_[mouth - (t_r - LIP_IN - aim) * pour_dir[:2], rim + max(clear, BODY_CLEAR - low)]
        return lip + rot.apply(grasp_c - lip_c), tilt * hand_rot, lip

    def carry(a, b):
        return max(int(np.ceil(np.linalg.norm(b - a) / carry_speed)), 12)

    s_start = pour(0.0, t_air[:2], t_rim + lift)[0]
    keys = []
    for grip, hand, opening, air in ((s_grip, s_hand, s_open, s_start), (t_grip, t_hand, t_open, t_air)):
        approach = grip - hand.apply([CUP_SLIDE, 0, 0])
        raised = grip + [0, 0, C2C_RAISE]
        # the source first rises to its pour height over its own place, then moves across: its bottom is then above
        # the target's rim (pour() keeps BODY_CLEAR), where a straight carry brushed the target cup in ~40% of the
        # ShenZhen episodes (2026-10-21)
        over = air + [0, 0, arrive_above if grip is s_grip else 0.0]
        high = np.r_[raised[:2], max(raised[2], over[2])] if grip is s_grip else raised
        keys.append(
            [
                (approach + [0, 0, HOVER], hand, opening, 54),
                (approach, hand, opening, 30),
                (grip, hand, opening, 40),  # slide in
                (grip, hand, 0, 15),  # close
                (raised, hand, 0, carry(grip, raised)),  # straight up first
                (high, hand, 0, carry(raised, high)),
                (over, hand, 0, carry(high, over)),
                *([(air, hand, 0, carry(over, air))] if arrive_above else []),
            ]
        )

    def cup(label):
        pos, quat = lm.get_instance_pose(0, label=label)
        return as_np(pos), from_wxyz(as_np(quat))

    def beans():
        return np.asarray(fluid.get_particle_positions()[0]).reshape(-1, 3) + fluid.init_pos

    def inside(label):
        """Per bean: inside the cup (cup frame, origin at the bbox centre): the measuring cups' and the mug's actual
        cavity (measuring-cup corners reach past the rim cylinder: beans left there read as poured), else the rim
        cylinder."""
        pos, rot = cup(label)
        if profiles:
            return cup_profile.in_cavity(beans(), pos, np.roll(rot.as_quat(), 1), *profiles[label])
        local = rot.inv().apply(beans() - pos)
        r, h = (s_r, height) if label == "source_cup" else (t_r, t_height)
        if kinds[label] == "measuring_cup":
            return cup_shape.in_cavity(local - cup_shape.origin_shift(h), h)
        if kinds[label] == "plastic_mug":
            return mug_shape.in_cavity(local - mug_shape.origin_shift())
        off = local - axes[label]
        return (np.hypot(off[:, 0], off[:, 1]) < r) & (np.abs(local[:, 2]) < h / 2)

    check = env.pour_check()
    step = 0

    def act(src_target, tgt_target):
        nonlocal step
        if step % 10 == 0:
            tilt = np.degrees(np.arccos(cup("source_cup")[1].apply([0, 0, 1])[2]))
            print(
                f"[scripted_demo] {step:3d} err src {np.linalg.norm(src_arm.grasp_point() - src_target[0]):.3f}"
                f" tgt {np.linalg.norm(tgt_arm.grasp_point() - tgt_target[0]):.3f}"
                f" src {cup('source_cup')[0].round(3)} tgt {cup('target_cup')[0].round(3)} tilt {tilt:.0f}"
                f" left src {1 - inside('source_cup').mean():.0%} in tgt {inside('target_cup').mean():.0%}"
                f" check {env.reward_manager.call_func_parser(check, 0):.0f}",
                flush=True,
            )
        step += 1
        return {
            src_arm.arm_key: src_arm.ik(*src_target[:2]),
            src_arm.grip_key: [float(src_target[2])],
            tgt_arm.arm_key: tgt_arm.ik(*tgt_target[:2]),
            tgt_arm.grip_key: [float(tgt_target[2])],
        }

    print(
        f"[scripted_demo] source {src0.round(3)} with {src_arm.arm_key} grip {s_grip.round(3)} open {s_open:.2f}"
        f" turn {turns['source_cup']:.0f};"
        f" target {as_np(lm.get_instance_pose(0, label='target_cup')[0]).round(3)} with {tgt_arm.arm_key}"
        f" grip {t_grip.round(3)} open {t_open:.2f} turn {turns['target_cup']:.0f}"
    )
    s_path = list(waypoints(src_arm.grasp_point(), src_arm.pose()[1], keys[0]))
    t_path = list(waypoints(tgt_arm.grasp_point(), tgt_arm.pose()[1], keys[1]))
    for s, t in itertools.zip_longest(s_path, t_path):
        yield act(s or s_path[-1], t or t_path[-1])

    hold = (t_air, t_hand, 0)
    fix = np.zeros(3)  # the cup keeps slipping a little: steer the actual lip over the actual target mouth

    def steer(theta, jolt=0.0):
        nonlocal fix
        t_pos, t_rot = cup("target_cup")
        mouth = t_pos + t_rot.apply(axes["target_cup"] + [0.0, 0.0, t_height / 2])  # on the cup axis, not the bbox
        hand, hand_rot, lip_target = pour(theta, mouth[:2], mouth[2])
        s_pos, s_rot = cup("source_cup")
        fix = np.clip(fix + LIP_GAIN * (lip_target - s_pos - s_rot.apply(lip_c)), -LIP_SLACK, LIP_SLACK)
        return act((hand + fix + jolt * np.r_[pour_dir[:2], 0.0], hand_rot, 0), hold)

    s_air = (s_path[-1][0], s_hand, 0)
    for _ in range(C2C_SETTLE):
        if max(np.linalg.norm(a.grasp_point() - p[0]) for a, p in ((src_arm, s_air), (tgt_arm, hold))) < C2C_SETTLE_ERR:
            break
        yield act(s_air, hold)
    pos, cup_rot = cup("source_cup")
    grasp_c = cup_rot.inv().apply(src_arm.grasp_point() - pos)
    hand_rot = src_arm.pose()[1]
    toward = cup_rot.inv().apply(np.r_[pour_dir[:2], 0.0])[:2]  # the cup may have turned about its axis
    lip_c = centre + np.r_[s_r * toward / np.linalg.norm(toward), height / 2]
    print(
        f"[scripted_demo] source grasp point in the cup {grasp_c.round(3)}, planned z {centre[2] + up - height / 2:.3f}"
    )
    most = most if profiles else C2C_TILT
    returning = int(np.ceil(most / RETURN_RATE)) + STEP_MARGIN
    total = len(inside("source_cup"))
    if profiles:  # the stream the target mouth takes, in grains per action
        grain = getattr(fluid, "grain", None) or cup_profile.PBD_GRAIN
        t_cavity, t_scale = profiles["target_cup"]
        sections = (2 * t_cavity["r"][-1] * t_scale / (6 * grain["volume"] / np.pi) ** (1 / 3)) ** 2
        surge_grains = float(np.clip(PLUS_SURGE_MOUTH * sections, *PLUS_SURGE_GRAINS))
        flow, surge = PLUS_WEAK * surge_grains / total, surge_grains / total
        print(f"[scripted_demo] target mouth takes {surge_grains:.1f} grains an action")
    else:  # a few large grains: a weak stream is no grain an action, a surge three
        flow, surge = max(C2C_FLOW, 0.5 / total), max(PLUS_SURGE, 2.5 / total)

    def fallen():
        """Per bean: spilled (out of both cups, under the target rim, off its mouth); its offset along the pour from
        the mouth centre."""
        pts = beans()
        t_pos, t_rot = cup("target_cup")
        mouth = t_pos + t_rot.apply(axes["target_cup"] + [0.0, 0.0, t_height / 2])
        off = pts[:, :2] - mouth[:2]
        out = ~inside("source_cup") & ~inside("target_cup") & (pts[:, 2] < mouth[2] - PLUS_SPILL_DROP)
        return out & (np.hypot(off[:, 0], off[:, 1]) > t_r), off @ pour_dir[:2]

    theta, prev, best, stall, shakes = 0.0, 0.0, 0.0, 0, 0
    leaked = 1 - inside("source_cup").mean()
    while theta < C2C_DRY:
        theta = min(theta + dry_rate, C2C_DRY)
        yield steer(theta)
        prev = 1 - inside("source_cup").mean()
        if not profiles:
            continue
        s_pos, s_rot = cup("source_cup")
        if beans()[:, 2].max() > (s_pos + s_rot.apply(lip_c))[2] - PLUS_LIP_NEAR:  # a full cup runs early
            break
        if theta < PLUS_SPILL_FROM:
            leaked = prev
        elif prev - leaked > flow:
            break
    lowest = theta
    spilled, calm = (fallen()[0] if profiles else None), 0
    while (now := 1 - inside("source_cup").mean()) < pour_share and step < env.step_lim - returning:
        if profiles:
            new = (fell := fallen())[0] & ~spilled
            spilled |= new
        if profiles and new.any() and theta >= PLUS_SPILL_FROM:
            theta, calm = max(theta - PLUS_SPILL_BACK, lowest), PLUS_SPILL_CALM
            clear = max(clear - PLUS_SPILL_LOWER, PLUS_CLEAR_MIN)
            step_aim = np.clip(PLUS_AIM_GAIN * fell[1][new].mean(), -PLUS_AIM_STEP, PLUS_AIM_STEP)
            aim = float(np.clip(aim - step_aim, *PLUS_AIM))
            print(
                f"[scripted_demo] {step:3d} {new.sum()} spilled {fell[1][new].mean():+.3f} along the pour:"
                f" tilt back to {np.degrees(theta):.0f}, lip {aim:+.3f} in, {clear:.3f} over the rim"
            )
        rate = C2C_TILT_RATE * (PLUS_SLOW_RATE if profiles and theta > PLUS_SLOW else 1.0)
        if calm:
            calm -= 1
        elif now - prev < flow:  # tilt further only while the stream is weak
            theta = min(theta + rate, most)
        elif profiles and now - prev > surge:  # and back while it surges: a slug splashes off the target mouth
            theta = max(theta - C2C_TILT_RATE * min((now - prev) / surge, PLUS_SURGE_BACK), lowest)
            clear = PLUS_CLEAR_MIN
        prev = now
        best, stall = (now, 0) if now > best or theta < most else (best, stall + 1)
        if profiles and stall > PLUS_STALL:
            if shakes == PLUS_SHAKES:
                break
            shakes, stall = shakes + 1, 0
            for k in range(PLUS_SHAKE_ACTS):
                yield steer(theta, PLUS_SHAKE if k % 4 < 2 else -PLUS_SHAKE)
        yield steer(theta)
    while theta > 0:
        theta = max(theta - RETURN_RATE, 0.0)
        yield steer(theta)
    for _ in range(STEP_MARGIN):
        yield steer(0.0)
    if not place:
        return
    paths = []
    for arm, grip, hand, opening in ((src_arm, s_grip, s_hand, s_open), (tgt_arm, t_grip, t_hand, t_open)):
        now = arm.grasp_point()
        up = now + [0.0, 0.0, PLACE_RISE if arm is src_arm and profiles else 0.0]
        over = np.r_[grip[:2], max(up[2], grip[2] + C2C_RAISE)]  # back over its place at the carry height, then down
        down = grip + [0.0, 0.0, PLACE_CLEAR]
        back = down - hand.apply([PLACE_BACK, 0.0, 0.0]) + [0.0, 0.0, PLACE_BACK]
        keys = [
            (up, hand, 0, carry(now, up)),
            (over, hand, 0, carry(up, over)),
            (down, hand, 0, carry(over, down) + 10),
            (down, hand, opening, PLACE_OPEN),
            (back, hand, opening, carry(down, back)),
        ]
        paths.append(list(waypoints(now, arm.pose()[1], keys)))
    for s, t in itertools.zip_longest(*paths):
        yield act(s or paths[0][-1], t or paths[1][-1])


def plus_actions(env):
    """g1d_pour_beans_plus: the cup-to-cup pour of all the beans between cups of any shape, by their layout "cavity"
    profiles; both cups put back on the desk."""
    lm = env.scene_manager.layout_manager
    profiles = {}
    for label in ("source_cup", "target_cup"):
        cup = lm.get_scene_object(0, lm.get_instance_name(0, label))
        scale = float(np.asarray(getattr(cup, "scale", (1.0, 1.0, 1.0)), float).reshape(-1)[0])
        profiles[label] = (cup.instance_config["cavity"], scale)
    return cup_to_cup_actions(env, pour_share=1.0, place=True, profiles=profiles)


def half_cup_actions(env):
    """g1d_pour_beans_half_cup(_rigid): the cup-to-cup pour of about half a cup, both cups put back on the desk."""
    lm = env.scene_manager.layout_manager
    fluid = lm.get_scene_object(0, lm.get_instance_name(0, "beans"))
    beans = len(fluid.get_particle_positions()[0])
    cavity = lm.get_scene_object(0, lm.get_instance_name(0, "target_cup")).instance_config["cavity"]
    return cup_to_cup_actions(
        env,
        pour_share=HALF_FILL * round_cup_shape.capacity(cavity, getattr(fluid, "grain", None)) / beans,
        place=True,
        meet=HALF_MEET,
        lift=HALF_LIFT,
        rim_down=HALF_RIM_DOWN,
        dry_rate=HALF_DRY_RATE,
    )


class Robot:
    """Both arms: one moves to Cartesian targets; the other holds its last command, or follows its own Cartesian path
    in the background (send_aside), one step per action of the first. The gripper hooks (make_arm, opening, grip_z,
    closing, holding) are the Dex1 jaw's; scripts/dexhands/block_pattern_demo.py overrides them for a pinch."""

    DESCEND = 30  # actions pick() comes down from its hover to the grip

    def __init__(self, env):
        self.env, self.lm = env, env.scene_manager.layout_manager
        self.arms = [self.make_arm(env, r) for r in env.robot_manager.robot_list if r.type == "target"]
        assets = {getattr(a.robot, "asset_name", None) for a in self.arms}
        if HAND["asset"] in ("g1d", "g1d_aurora_lite_ws") and assets & {"g1d", "g1d_aurora_lite_ws"} - {HAND["asset"]}:
            raise RuntimeError(f"G1D_HAND={HAND_NAME} but the robot is {assets}: set G1D_HAND (reach_map.HANDS)")
        self.hold = {a.arm_key: a.q() for a in self.arms}
        self.grip = {a.grip_key: [1.0] for a in self.arms}
        self.home = {a.arm_key: (a.grasp_point(), a.pose()[1]) for a in self.arms}
        self.home_q = {a.arm_key: a.q() for a in self.arms}
        self.background = {}  # arm key -> iterator of joint targets, advanced while another arm moves
        self.settle = (
            None  # (m, actions): after each key hold its target until the arm is this close, at most this long
        )
        self.start_q = {a.arm_key: a.q() for a in self.arms}
        self.step = 0
        self.models, self.planned = {}, {}
        # v4: noise(arm) -> (m offset, rotvec) on the tracked target, or None; the noise-free command then rides
        # along under LABEL (collect.py records it as the action, the noisy one as executed)
        self.noise = None
        self.jitter = {}  # arm_key: (m shift, rotation) of the start pose the park pose is taken from

    def plan_with(self, models):
        """Track the Cartesian targets with each arm's URDF kinematics ({side: reach_map.Arm}), solved warm from the
        last command along the path, instead of Arm.ik's one simulator-Jacobian step per action, which stalls against
        joint limits on the long reaches across the build site. The model's offset from the measured pose is fed back
        each action."""
        for a in self.arms:
            model = models["left" if a.arm_key.startswith("left") else "right"]
            names = [a.key.joint_names[i] for i in a.cols]
            self.models[a.arm_key] = (model, [model.arm.index(n) for n in names])

    def track(self, arm, point, rot):
        if arm.arm_key not in self.models:
            return arm.ik(point, rot)
        model, order = self.models[arm.arm_key]
        now = np.empty(len(order))
        now[order] = arm.q()
        p, r = model.fk(now)
        pos, hand = arm.pose()
        bias, turn = arm.grasp_point() - p, hand * r.inv()
        start = self.planned.get(arm.arm_key, now)
        q, _ = model.solve(point - bias, turn.inv() * rot, iters=20, q=start, pos_tol=5e-4, rot_tol=np.deg2rad(0.5))
        self.planned[arm.arm_key] = q
        return q[order]

    def make_arm(self, env, robot):
        return Arm(env, robot)

    def opening(self, width):
        """Gripper command that opens wide enough to come down over a side `width` wide."""
        return pad_open(width)

    def grip_z(self, bottom, top):
        """Grip point height for an object between `bottom` and `top`."""
        return max(top - 0.03, bottom + FINGER_TIPS + TABLE_CLEAR)

    def closing(self, width):
        """Gripper command closed on a side `width` wide: shut, both fingers stop against it."""
        return 0.0

    def holding(self, arm, width):
        """Gripper command to carry with once closed: where the fingers stopped, so the PD squeezes 1.5 mm."""
        fingers = as_np(arm.key.data.joint_pos[0, list(arm.robot.gripper_joint_indices)])
        return float(np.clip((0.023 - fingers.mean()) / 0.0445, 0.0, 1.0))

    def yaw_off(self, arm, label, period):
        """rad block_pattern_actions turns the hand back at the place for how far the held block `label` sits turned
        in `arm`'s grip (modulo its yaw `period`): 0, the jaw's flat pads square it."""
        return 0.0

    def arm_for(self, x):
        return min(self.arms, key=lambda a: abs(self.home[a.arm_key][0][0] - x))

    def move(self, arm, keys):
        """Follow `keys` [(grip point, hand rotation, gripper, actions[, lead])] with `arm`; with `settle`, wait at
        each."""
        for path in [keys] if self.settle is None else [[key] for key in keys]:
            for point, rot, grip in waypoints(arm.grasp_point(), arm.pose()[1], path):
                yield self._command(arm, point, rot, grip)
            if self.settle is not None:
                point, rot, grip = path[-1][:3]
                tol, most = self.settle
                for _ in range(most):
                    if np.linalg.norm(arm.grasp_point() - point) < tol:
                        break
                    yield self._command(arm, point, rot, grip)

    def _command(self, arm, point, rot, grip):
        q = self.track(arm, point, rot)
        off = self.noise(arm) if self.noise else None
        if off is not None:
            clean, q = q, self.track(arm, point + off[0], Rotation.from_rotvec(off[1]) * rot)
        self.hold[arm.arm_key] = q
        self.grip[arm.grip_key] = [float(grip)]
        self._advance(arm.arm_key)
        if self.step % 20 == 0:
            err = np.linalg.norm(arm.grasp_point() - point)
            print(f"[scripted_demo] {self.step:3d} {arm.arm_key} err {err:.3f}", flush=True)
        self.step += 1
        out = dict(self.hold) | self.grip
        if off is not None:
            out[LABEL] = {arm.arm_key: clean}
        return out

    def joints_for(self, arm, point_offset):
        """Joint command (simulator order) of the start pose's hand moved `point_offset`, rotation kept (URDF model)."""
        model, order = self.models[arm.arm_key]
        start = np.empty(len(order))
        start[order] = self.start_q[arm.arm_key]
        p, r = model.fk(start)
        shift, turn = self.jitter.get(arm.arm_key, (np.zeros(3), Rotation.identity()))
        q, _ = model.solve(p + shift + np.asarray(point_offset), turn * r, q=start)
        return q[order]

    def joints_at(self, arm, point, rot, seed=None):
        """Joint command (simulator order) of the hand at (point, rot), solved from `seed` (simulator order; default
        the start pose) with the URDF model, the measured model offset fed back as in track; None when it misses."""
        from scripts.g1d.reach_map import reached

        model, order = self.models[arm.arm_key]
        now, start = np.empty(len(order)), np.empty(len(order))
        now[order], start[order] = arm.q(), self.start_q[arm.arm_key] if seed is None else seed
        p, r = model.fk(now)
        bias, turn = arm.grasp_point() - p, arm.pose()[1] * r.inv()
        q, err = model.solve(point - bias, turn.inv() * rot, q=start)
        return q[order] if reached(err) else None

    def move_joints(self, arm, q_cmd, rate):
        """Straight in joint space to `q_cmd` (simulator order), eased in and out, the fastest joint at about `rate`
        rad per action on average."""
        start = np.asarray(self.hold[arm.arm_key], float)
        n = max(10, int(np.ceil(np.abs(q_cmd - start).max() / rate)))
        for w in 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n + 1)[1:]):
            self.hold[arm.arm_key] = start + w * (q_cmd - start)
            self.step += 1
            yield dict(self.hold) | self.grip
        if arm.arm_key in self.models:
            model, order = self.models[arm.arm_key]
            self.planned[arm.arm_key] = np.empty(len(order))
            self.planned[arm.arm_key][order] = q_cmd

    def settle_at(self, arm, point, rot, grip, tol, most):
        """Hold the target until the grasp point is within `tol` of it (at most `most` actions)."""
        for _ in range(most):
            if np.linalg.norm(arm.grasp_point() - point) < tol:
                return
            yield from self.move(arm, [(point, rot, grip, 1)])

    def object(self, label):
        """World centre, rotation, local bbox extents and world z bounds of an object."""
        pos, quat = self.lm.get_instance_pose(0, label=label)
        rot = from_wxyz(as_np(quat))
        bbox = self.lm.get_instance_bbox_vertices(self.lm.get_instance_name(0, label), 0)
        lo, hi = bbox.min(0), bbox.max(0)
        z = (as_np(pos) + rot.apply(bbox))[:, 2]
        return as_np(pos) + rot.apply((lo + hi) / 2), rot, hi - lo, z.min(), z.max()

    def pick(self, label, lift=LIFT, close=False, across=None, prefer=None, period=np.pi, arm=None):
        """Top-down grasp across the narrow side (or the local axis `across`), lifted; returns (arm, hand, offset,
        grip_z, open_cmd, close_cmd). With `prefer` (rad, per arm key): of the hand yaws that grasp the object alike
        (every `period`: pi for the symmetric jaw, less for a symmetric object, 0 for a round one), the one nearest
        it. `arm`: this arm, else the one on the object's side."""
        centre, rot, ext, bottom, top = self.object(label)
        axes = rot.as_matrix()  # columns are the object's local axes in world frame
        up = int(np.argmax(np.abs(axes[2])))
        narrow = across if across is not None else min((a for a in range(3) if a != up), key=lambda a: ext[a])
        yaw = np.arctan2(axes[1, narrow], axes[0, narrow])  # close along the narrow horizontal local axis
        yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2  # the jaw is symmetric: keep the wrist camera facing forward
        open_cmd = self.opening(ext[narrow])
        grip = np.r_[centre[:2], self.grip_z(bottom, top)]
        arm = arm or self.arm_for(centre[0])
        if prefer is None:
            if abs(yaw) > np.deg2rad(60):  # sideways grasp: point the wrist toward the shoulder so it stays in reach
                if (arm.arm_key.startswith("right") and yaw > 0) or (arm.arm_key.startswith("left") and yaw < 0):
                    yaw += -np.pi if yaw > 0 else np.pi
        else:
            want = prefer[arm.arm_key]
            yaw = want if not period else yaw + period * round((want - yaw) / period)
        hand = Rotation.from_euler("z", yaw) * TOP_ROT
        print(
            f"[scripted_demo] pick {label} at {centre.round(3)} size {ext.round(3)}"
            f" width {ext[narrow]:.3f} open {open_cmd:.2f} with {arm.arm_key}"
        )
        yield from self.move(
            arm,
            [
                (grip + [0, 0, lift], hand, open_cmd, 54),
                (grip, hand, open_cmd, self.DESCEND),
                (grip, hand, self.closing(ext[narrow]), 18),  # close on the table: both fingers stop against it
            ],
        )
        tilt_err = np.degrees((hand * arm.pose()[1].inv()).magnitude())
        close_cmd = self.closing(ext[narrow]) if close else self.holding(arm, ext[narrow])
        yield from self.move(arm, [(grip + [0, 0, lift], hand, close_cmd, 30)])
        centre, _, _, _, _ = self.object(label)
        fingers = as_np(arm.key.data.joint_pos[0, list(arm.robot.gripper_joint_indices)])
        print(
            f"[scripted_demo] lifted {label} to {centre.round(3)} (grip {grip.round(3)}), fingers {fingers.round(4)},"
            f" hand {tilt_err:.1f} deg off at the grasp"
        )
        return arm, hand, arm.pose()[1].inv().apply(centre - arm.grasp_point()), grip[2] - bottom, open_cmd, close_cmd

    def place(
        self, arm, hand, offset_local, grip_height, open_cmd, close_cmd, xy, surface, carry=48, label=None, hover=HOVER
    ):
        """Lower the held object onto `surface` (z) at `xy`, release to `open_cmd` and back off upward; `carry`
        actions to above the spot. With the held object's `label`, the hand is moved over the spot by where the
        object actually is once there (it may have slipped in the hand on the way)."""
        point = np.r_[np.asarray(xy) - hand.apply(offset_local)[:2], surface + grip_height + PLACE_GAP]
        print(f"[scripted_demo] place at {point.round(3)} on surface {surface:.3f}")
        keys = [
            (point + [0, 0, hover], hand, close_cmd, carry),
            (point, hand, close_cmd, 30),
            (point, hand, open_cmd, 12),  # release
            (point + [0, 0, hover], hand, open_cmd, 24),
        ]
        if label is None:
            yield from self.move(arm, keys)
            return
        yield from self.move(arm, keys[:1])
        fix = np.asarray(xy) - self.object(label)[0][:2]
        print(f"[scripted_demo] {label} {np.linalg.norm(fix) * 1000:.0f} mm off over the spot, corrected")
        yield from self.move(arm, [(p + [*fix, 0.0], r, g, n) for p, r, g, n in keys[1:]])

    def go_home(self):
        for arm in self.arms:
            point, rot = self.home[arm.arm_key]
            yield from self.move(arm, [(point, rot, 1, 54), (point, rot, 1, 12)])

    def send_aside(self, arm, point, n):
        """Move `arm` to `point`, hand turned as it is, in the background: one step per action while another arm moves;
        finish() runs it out."""
        path = waypoints(arm.grasp_point(), arm.pose()[1], [(point, arm.pose()[1], 1.0, n)])
        self.grip[arm.grip_key] = [1.0]
        self.background[arm.arm_key] = (arm.ik(p, rot) for p, rot, _ in path)

    def _advance(self, busy):
        """One step of every background path but `busy`'s."""
        for key, path in list(self.background.items()):
            if key != busy:
                q = next(path, None)
                if q is None:
                    del self.background[key]
                else:
                    self.hold[key] = q

    def finish(self):
        """Run the background paths out."""
        while self.background:
            self._advance(None)
            self.step += 1
            yield dict(self.hold) | self.grip

    def chassis(self):
        """Chassis (AGV_link) position (x, y) and heading of its +X axis, env frame."""
        rm = self.env.robot_manager
        body = next(r for r in rm.robot_list if hasattr(r, "body_dims"))
        pose = as_np(rm.get_link_pose(body, "AGV_link", env_idx_list=[0], is_relative=True)[0])
        fwd = from_wxyz(pose[3:7]).apply([1.0, 0.0, 0.0])
        return pose[:2], float(np.arctan2(fwd[1], fwd[0]))

    def drive(self, target, heading=None, reverse=False, **body):
        """Drive the chassis to `target` (x, y) with base_velocity, arms held: turn on the spot when off course, then
        straight on with a speed ramp; optionally turn to `heading` at the end. `body` keys (the lift) ride along."""

        def act(v, w):
            self.step += 1
            return dict(self.hold) | self.grip | body | {"base_velocity": [float(v), float(w)]}

        v = 0.0
        while True:
            xy, yaw = self.chassis()
            d = np.subtract(target, xy)
            dist = float(np.linalg.norm(d))
            if dist < DRIVE_TOL:
                break
            err = wrap(np.arctan2(d[1], d[0]) + (np.pi if reverse else 0.0) - yaw)
            want = 0.0 if abs(err) > TURN_FIRST else min(DRIVE_V, 1.2 * dist)
            v = float(np.clip(want, v - DRIVE_ACC, v + DRIVE_ACC))
            if self.step % 30 == 0:
                print(f"[scripted_demo] {self.step:4d} chassis {xy.round(3)} yaw {yaw:.2f} to {np.round(target, 3)}")
            yield act(-v if reverse else v, np.clip(2.0 * err, -DRIVE_W, DRIVE_W))
        while heading is not None and abs(err := wrap(heading - self.chassis()[1])) > HEADING_TOL:
            yield act(0.0, np.clip(2.0 * err, -DRIVE_W, DRIVE_W))
        for _ in range(SETTLE):
            yield act(0.0, 0.0)


def wrap(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def table_approach(robot, table, start):
    """Unit vector from the lounge table centre to where the robot stands: the middle of a gap of at least CHAIR_GAP
    between the chairs round it, the one facing `start` most."""
    bearings = []
    for rec in robot.lm.get_layout_records(0, "Rigid"):
        if str(rec.get("label")).startswith("chair_"):
            xy = robot.object(rec["label"])[0][:2]
            if np.linalg.norm(xy - table) < 1.2:
                bearings.append(np.arctan2(*(xy - table)[::-1]))
    toward = np.arctan2(*(np.subtract(start, table))[::-1])
    bearings = sorted(bearings)
    gaps = [(b, (n - b) % (2 * np.pi) or 2 * np.pi) for b, n in zip(bearings, bearings[1:] + bearings[:1])]
    mids = [b + g / 2 for b, g in gaps if g >= CHAIR_GAP] or [toward]
    mid = min(mids, key=lambda m: abs(wrap(m - toward)))
    return np.array([np.cos(mid), np.sin(mid)])


def deliver_cup_actions(env):
    """Side-grasp the cup on the counter (as the cup pours: a top grasp slips off the 30 g plastic cup), lift it, back
    off, drive through the lounge to a gap between the chairs at the round table, lowering the lift column for it on
    the way, face the table, set the cup down, open and slide the hand back out."""
    robot = Robot(env)
    arm, open_cmd = yield from lift_cup(robot)
    yield from carry_to_lounge(env, robot, arm, "cup", open_cmd)


def lift_cup(robot, arm=None):
    """Side-grasp the cup on the counter from the outside of `arm` (default: the arm on its side), just below the rim,
    and lift it CARRY_UP; returns (arm, open command)."""
    lm = robot.lm
    cup0 = robot.object("cup")[0]
    arm = arm or robot.arm_for(cup0[0])
    out = -1.0 if arm.arm_key.startswith("left") else 1.0
    bbox = lm.get_instance_bbox_vertices(lm.get_instance_name(0, "cup"), 0)
    height = bbox[:, 2].max() - bbox[:, 2].min()
    grip, hand, _, _, open_cmd = cup_grasp(lm, out, "cup", height - C2C_RIM_DOWN["plastic_cup"])
    approach = grip - hand.apply([CUP_SLIDE, 0, 0])
    print(f"[scripted_demo] cup at {cup0.round(3)} with {arm.arm_key} grip {grip.round(3)} open {open_cmd:.2f}")
    yield from robot.move(
        arm,
        [
            (approach + [0, 0, HOVER], hand, open_cmd, 54),
            (approach, hand, open_cmd, 30),
            (grip, hand, open_cmd, 40),  # slide in
            (grip, hand, 0, 15),  # close
            (grip + [0, 0, CARRY_UP], hand, 0, 36),
        ],
    )
    print(f"[scripted_demo] lifted cup to {robot.object('cup')[0].round(3)} (from {cup0.round(3)})")
    return arm, open_cmd


def carry_to_lounge(env, robot, arm, label, open_cmd):
    """With the cup `label` lifted in `arm`: back off from the counter, drive through the lounge to a gap between the
    chairs at the round table, lowering the lift column for it on the way, face the table, set the cup down, open to
    `open_cmd` and slide the hand back out."""
    start, yaw = robot.chassis()
    yield from robot.drive(start - BACK_OFF * np.array([np.cos(yaw), np.sin(yaw)]), reverse=True)
    table, r = np.asarray(env.table_xy[0]), env.table_r[0]
    u = table_approach(robot, table, start)
    lift = {"lift_joint_state": [LOUNGE_LIFT]}
    bearing = np.degrees(np.arctan2(u[1], u[0]))
    print(f"[scripted_demo] lounge table {table.round(3)} r {r:.3f}, approach bearing {bearing:.0f}")
    yield from robot.drive(table + (r + TABLE_STAND + APPROACH) * u, **lift)
    yield from robot.drive(table + (r + TABLE_STAND) * u, heading=float(np.arctan2(-u[1], -u[0])), **lift)
    spot = table + (r + TABLE_STAND - PLACE_AHEAD) * u
    # the cup sits in the hand as it does now (it may have slid in the grip on the way): keep that offset
    point, rot = arm.grasp_point(), arm.pose()[1]
    centre, _, _, bottom, _ = robot.object(label)
    down = np.r_[spot + point[:2] - centre[:2], env.table_z[0] + PLACE_GAP + point[2] - bottom]
    back = down - rot.apply([CUP_SLIDE, 0, 0])
    print(f"[scripted_demo] place cup at {spot.round(3)}, hand to {down.round(3)}")
    yield from robot.move(
        arm,
        [
            (down + [0, 0, HOVER], rot, 0, 48),
            (down, rot, 0, 36),
            (down, rot, open_cmd, 15),  # release
            (back, rot, open_cmd, 30),  # slide out along the fingers
            (back + [0, 0, HOVER], rot, open_cmd, 24),
        ],
    )
    checks = {c[0]: env.reward_manager.call_func_parser(c, 0) for c in env.delivered_checks()}
    print(f"[scripted_demo] cup at {robot.object(label)[0].round(3)}, checks {checks}, env success {env.success[0]}")


def pour_deliver_actions(env):
    """Pour water from the jug into the cup standing on the counter as pour_actions does (the other hand waits), set
    the jug back down where it stood (over its spot, down, open, back out of the handle, home), then pick the cup up
    and carry it to the lounge table."""
    robot = Robot(env)  # home poses from the start
    jug0, _, _, counter_z, _ = robot.object("bottle")
    yield from pour_actions(env, "water", WATER_DRY, WATER_POUR, hold_cup=False, rate=WATER_RATE)
    robot.hold = {a.arm_key: a.q() for a in robot.arms}
    jug_arm = robot.arm_for(jug0[0])
    robot.grip[jug_arm.grip_key] = [0.0]
    point, rot = jug_arm.grasp_point(), jug_arm.pose()[1]
    centre, _, _, bottom, _ = robot.object("bottle")
    down = np.r_[jug0[:2] + point[:2] - centre[:2], counter_z + PLACE_GAP + point[2] - bottom]
    home, home_rot = robot.home[jug_arm.arm_key]
    print(f"[scripted_demo] set the jug down at {jug0[:2].round(3)}, hand {down.round(3)}, chassis {robot.chassis()}")
    yield from robot.move(
        jug_arm,
        [
            (np.r_[down[:2], point[2]], rot, 0, max(int(np.linalg.norm(down[:2] - point[:2]) / JUG_SPEED), 30)),
            (down, rot, 0, max(int((point[2] - down[2]) / JUG_SPEED), 30)),
            (down, rot, OPEN, 15),  # release
            (down + BEHIND, rot, OPEN, 30),  # out of the handle the way the fingers came in
            (down + BEHIND + [0, 0, 0.1], rot, OPEN, 24),
            (home, home_rot, 1.0, 54),
        ],
    )
    # the other hand, from the far side of the jug (the cup may stand on the jug arm's side of the midline)
    arm, open_cmd = yield from lift_cup(robot, next(a for a in robot.arms if a is not jug_arm))
    yield from carry_to_lounge(env, robot, arm, "cup", open_cmd)


def pickup_actions(env):
    robot = Robot(env)
    arm, _, _, _, _, close_cmd = yield from robot.pick("target", close=True)
    point, rot = arm.grasp_point(), arm.pose()[1]
    yield from robot.move(arm, [(point, rot, close_cmd, 36)])  # hold it up


def stack_actions(env):
    robot = Robot(env)
    labels = ["block_0", "block_1", "block_2"]
    base, *others = sorted(labels, key=lambda label: abs(robot.object(label)[0][0]))
    stack = [base]
    for label in others:
        arm, hand, offset_local, grip_height, open_cmd, close_cmd = yield from robot.pick(label, lift=HOVER)
        top = max(robot.object(b)[4] for b in stack)
        yield from robot.place(
            arm, hand, offset_local, grip_height, open_cmd, close_cmd, robot.object(stack[-1])[0][:2], top
        )
        stack.append(label)
    yield from robot.go_home()


# Block patterns: the hand yaw every block is set down with, fingers closing along the pattern's depth. Probed from the
# start pose: each hand reaches its side's scattered blocks within 75 deg of fingers forward, and the build space
# with the fingers pointing across at it (left -90, right +90 deg) from 3 cm up at y -0.40 .. -0.44 (2 cm short at
# y -0.36); turned the other way neither reaches the midline. Blocks are gripped at block_patterns.grip_yaw, the
# layouts turn them to within 20 deg of it (build_asset.GRASP_YAW): the hand turns at most ~95 deg on the way, at
# PATTERN_TURN_RATE per action under Limiter.
PATTERN_PLACE_YAW = {"left_arm_joint_state": -np.pi / 2, "right_arm_joint_state": np.pi / 2}
PATTERN_TURN_RATE = 0.4 / 30  # rad per action, half of REAL_ARM_VMAX
# Limiter slows the long first moves from the start pose: the arm fell 9 cm behind on the way to a block and swept it
# away low down. Every key is held until the arm is this close (m), for at most this many actions.
PATTERN_SETTLE = (0.005, 60)
# |x|, y, height above the table (m) a hand waits at while the other one works, and the actions to get there. Its way
# back to the start pose in joint space first swung it in front of the chest, into the other hand reaching for a block
PATTERN_ASIDE, PATTERN_ASIDE_ACTIONS = (0.16, -0.40, 0.12), 45
LANDED_TOL = 0.03  # m: a block set down further than this from plan does not steer the next places


def block_pattern_actions(env, robot=None):
    """Build the card's pattern about block_patterns.BUILD_XY, bottom-up in its piece order. Every block is gripped
    across its local Y (its 30 mm depth; a half-cylinder then holds its dome to the front) and set down with the hand
    at PATTERN_PLACE_YAW, the fingers closing along the pattern's depth: they pass in front of and behind it, not into
    the gap to a neighbour, and the block gets its pattern yaw. Each place is corrected by how far the blocks already
    set down landed from plan, set onto the measured top of the blocks under it. A hand that picks the next block too
    goes straight on to it; one that does not moves aside, above its own side of the table (PATTERN_ASIDE), while the
    other hand starts on the next block, and picks its next block from there. `robot`: a Robot (subclass) to build
    with, else the Dex1 one."""
    robot = robot or Robot(env)
    robot.settle = PATTERN_SETTLE
    pattern = env.pattern[0]
    labels = block_patterns.labels(pattern)
    table = min(robot.object(label)[3] for label in labels)
    site = np.asarray(block_patterns.BUILD_XY, dtype=float)
    planned, landed = {}, {}
    print(f"[scripted_demo] pattern {pattern['name']} ({pattern['zh']}), {len(labels)} blocks, table {table:.3f}")
    for i, (cat, x, z) in enumerate(pattern["pieces"]):
        label = labels[i]
        period = np.radians(min(block_patterns.YAW_PERIOD[cat], 180))  # the jaw repeats every 180 deg anyway
        prefer = np.radians(block_patterns.grip_yaw(*robot.object(label)[0][:2]))
        arm, hand, offset, grip_height, open_cmd, close_cmd = yield from robot.pick(
            label,
            lift=HOVER,
            across=1,
            prefer=dict.fromkeys(robot.home, prefer),
            period=period,
        )
        width = block_patterns.SIZES[cat][0]
        under = [
            labels[j]
            for j, (c, xb, zb) in enumerate(pattern["pieces"][:i])
            if abs(zb + block_patterns.SIZES[c][2] - z) < 0.0015
            and min(x + width / 2, xb + block_patterns.SIZES[c][0] / 2)
            > max(x - width / 2, xb - block_patterns.SIZES[c][0] / 2)
        ]
        surface = max((robot.object(b)[4] for b in under), default=table + z)
        good = [landed[b] - planned[b] for b in landed if np.linalg.norm(landed[b] - planned[b]) < LANDED_TOL]
        shift = np.mean(good, axis=0) if good else np.zeros(2)
        planned[label] = site + [x, 0.0]
        yaw = (hand * TOP_ROT.inv()).as_euler("zyx")[0]  # pick() turned TOP_ROT this much about z
        turn = wrap(PATTERN_PLACE_YAW[arm.arm_key] - robot.yaw_off(arm, label, period) - yaw)
        target = planned[label] + shift
        print(f"[scripted_demo] {label} {cat} to {target.round(3)} on {surface:.3f}, turn {np.degrees(turn):.0f}")
        carry = max(48, int(np.ceil(abs(turn) / PATTERN_TURN_RATE)))
        held = Rotation.from_euler("z", turn) * hand
        yield from robot.place(
            arm, held, offset, grip_height, open_cmd, close_cmd, target, surface, carry=carry, label=label
        )
        landed[label] = robot.object(label)[0][:2]
        print(f"[scripted_demo] {label} landed at {landed[label].round(3)} (planned {planned[label].round(3)})")
        following = labels[i + 1] if i + 1 < len(labels) else None
        if following is None or robot.arm_for(robot.object(following)[0][0]) is not arm:
            x, y, up = PATTERN_ASIDE
            side = np.sign(robot.home[arm.arm_key][0][0])
            robot.send_aside(arm, np.array([side * x, y, table + up]), PATTERN_ASIDE_ACTIONS)
    yield from robot.finish()
    poses = [robot.lm.get_instance_pose(0, label=label) for label in labels]
    xy, z, tilt, yaw = pattern_error(
        [as_np(p) for p, _ in poses],
        [from_wxyz(as_np(q)).as_matrix() for _, q in poses],
        block_patterns.targets(pattern),
        [block_patterns.YAW_PERIOD[cat] for cat, _, _ in pattern["pieces"]],
        pattern["anchor"],
        block_patterns.groups(pattern),
    )
    built = env.reward_manager.call_func_parser(env.built_check(0), 0)
    print(
        f"[scripted_demo] pattern built {built:.0f}: worst xy {xy * 1000:.0f} mm, z {z * 1000:.0f} mm,"
        f" tilt {tilt:.0f} deg, yaw {yaw:.0f} deg"
    )


def layout_entry(env, label):
    """The layout entry (category, category_idx, ...) of the object `label` in env 0."""
    rigid = env.scene_manager.config["env0"]["Rigid"]
    return next(e for es in rigid.values() for e in es if e.get("label") == label)


# m above the drop point (itself above the tray walls): HOVER there, 2.5 cm higher than on the table, left the hands
# 12-60 cm off when reaching across the midline (2 of 13 short cylinders)
SORT_HOVER = 0.03


def block_sort_actions(env):
    """Sort the blocks into the trays by their cards (block_sorting.py). Each block is picked by the hand on its side
    (the hands neither pick nor reach far across the midline), gripped from above across its local Y, carried over its
    tray and dropped into the tray's next free slot from just above the walls (open fingers inside the 6 cm tray
    caught its walls); the trays stand in the middle, in both hands' reach.
    The left hand's blocks first, one after the other, then the right hand's, the left hand moving aside meanwhile
    (PATTERN_ASIDE: above its own side, where its blocks lay)."""
    robot = Robot(env)
    robot.settle = PATTERN_SETTLE
    groups = env.groups[0]
    order = sorted(robot.arms, key=lambda a: robot.home[a.arm_key][0][0])  # left hand first
    plan = [(robot.arm_for(robot.object(b)[0][0]), box, b) for box in env.BOXES for b in groups[box]]
    plan.sort(key=lambda p: order.index(p[0]))
    free = {box: [0, 1] for box in env.BOXES}
    table = min(robot.object(label)[3] for *_, label in plan)
    print(f"[scripted_demo] sort by {env.BY}: {groups}")
    for n, (arm, box, label) in enumerate(plan):
        cat = layout_entry(env, label)["category"]
        centre = robot.object(label)[0]
        period = np.radians(min(block_patterns.YAW_PERIOD[cat], 180))  # the jaw repeats every 180 deg anyway
        prefer = np.radians(block_patterns.grip_yaw(centre[0], centre[1]))
        arm, hand, offset, grip_height, open_cmd, close_cmd = yield from robot.pick(
            label, lift=HOVER, across=1, prefer=dict.fromkeys(robot.home, prefer), period=period, arm=arm
        )
        slot = free[box].pop(0)
        target = np.asarray(block_sorting.slot_xy(env.BOXES.index(box), slot))
        rim = robot.object(box)[4]
        print(f"[scripted_demo] {label} {cat} into {box} slot {slot} at {target.round(3)}")
        yield from robot.place(
            arm, hand, offset, grip_height, open_cmd, close_cmd, target, rim, label=label, hover=SORT_HOVER
        )
        print(f"[scripted_demo] {label} landed at {robot.object(label)[0].round(3)}")
        if n + 1 == len(plan) or plan[n + 1][0] is not arm:
            x, y, up = PATTERN_ASIDE
            out = np.sign(robot.home[arm.arm_key][0][0])
            robot.send_aside(arm, np.array([out * x, y, table + up]), PATTERN_ASIDE_ACTIONS)
    yield from robot.finish()
    checks = [env.reward_manager.call_func_parser(c, 0) for c in env.sorted_checks(0)]
    print(f"[scripted_demo] sorted checks {checks}")


def atomic_pick(robot, label, lift=HOVER):
    """Top grasp of block `label` across its local Y by the hand on its side, at that hand's best grip yaw."""
    cat = layout_entry(robot.env, label)["category"]
    centre = robot.object(label)[0]
    period = np.radians(min(block_patterns.YAW_PERIOD[cat], 180))  # the jaw repeats every 180 deg anyway
    prefer = np.radians(block_patterns.grip_yaw(centre[0], centre[1]))
    return (yield from robot.pick(label, lift=lift, across=1, prefer=dict.fromkeys(robot.home, prefer), period=period))


def place_turned(robot, grasp, target, surface, label):
    """Set the held block down at `target` on `surface` with the hand at about PATTERN_PLACE_YAW: the block's long side
    ends along world X (its pattern yaw 0) and the fingers close along Y, not into a gap beside it. The hand's final
    yaw takes up how far the block sits turned in it as lifted (a grip a few degrees off, or slipped on the way up)."""
    arm, hand, offset, grip_height, open_cmd, close_cmd = grasp
    yaw = (hand * TOP_ROT.inv()).as_euler("zyx")[0]  # pick() turned TOP_ROT this much about z
    period = np.radians(min(block_patterns.YAW_PERIOD[layout_entry(robot.env, label)["category"]], 180))
    in_hand = yaw_of(robot.object(label)[1].as_matrix()) - (arm.pose()[1] * TOP_ROT.inv()).as_euler("zyx")[0]
    final, off = PATTERN_PLACE_YAW[arm.arm_key], 0.0
    if period:
        off = (final + in_hand + period / 2) % period - period / 2
        final -= off
    turn = wrap(final - yaw)
    print(f"[scripted_demo] {label} sits {np.degrees(off):.1f} deg off in the hand, hand to {np.degrees(final):.1f}")
    carry = max(48, int(np.ceil(abs(turn) / PATTERN_TURN_RATE)))
    held = Rotation.from_euler("z", turn) * hand
    yield from robot.place(
        arm, held, offset, grip_height, TURNED_RELEASE, close_cmd, target, surface, carry=carry, label=label
    )


# Gripper command a block set down by place_turned() is let go with: opened only to its grip width, the fingertips
# caught a 90 mm plank as the hand backed off and turned it 6-17 deg, dragging it 1 cm toward the robot.
TURNED_RELEASE = 1.0


def atomic_done(robot, arm, table):
    """The hand moves aside (PATTERN_ASIDE, above the table top `table`) and both arms settle."""
    x, y, up = PATTERN_ASIDE
    robot.send_aside(arm, np.array([np.sign(robot.home[arm.arm_key][0][0]) * x, y, table + up]), PATTERN_ASIDE_ACTIONS)
    yield from robot.finish()


def place_on_mark_actions(env):
    """g1d_place_block_on_mark: the block onto the mark's centre."""
    robot = Robot(env)
    robot.settle = PATTERN_SETTLE
    table = robot.object("block")[3]
    grasp = yield from atomic_pick(robot, "block")
    mark = robot.object("mark")
    yield from robot.place(*grasp, mark[0][:2], mark[4], label="block")
    yield from atomic_done(robot, grasp[0], table)


def stack_on_block_actions(env):
    """g1d_stack_block_on_block: the top block onto the base block's top (both on the same side)."""
    robot = Robot(env)
    robot.settle = PATTERN_SETTLE
    base = robot.object("base_block")
    grasp = yield from atomic_pick(robot, "top_block")
    yield from robot.place(*grasp, base[0][:2], base[4], label="top_block")
    yield from atomic_done(robot, grasp[0], base[3])


def pick_named_actions(env):
    """g1d_pick_named_block: the named block up PICK_LIFT and held there."""
    robot = Robot(env)
    robot.settle = PATTERN_SETTLE
    arm, hand, _, _, _, close_cmd = yield from atomic_pick(robot, "target_block", lift=PICK_LIFT)
    yield from robot.move(arm, [(arm.grasp_point(), hand, close_cmd, PICK_HOLD)])


PICK_ANYTHING_LIFT = 0.13  # m above the grasp
# m of a tall object above the grip point: at Robot's 3 cm the knuckles behind the 60 deg pads pressed 10 cm pegs and
# bottles over on the way down (5090-2, 11-11)
PICK_ANYTHING_DEPTH = 0.012


class TopGripRobot(Robot):
    # 13 cm down in 30 actions outran the 0.8 rad/s joint limit: the clipped joints bowed the path 4 cm sideways
    # into narrow targets
    DESCEND = 75

    def grip_z(self, bottom, top):
        return max(top - PICK_ANYTHING_DEPTH, bottom + FINGER_TIPS + TABLE_CLEAR)


def pick_anything_actions(env):
    """g1d_pick_anything: top grasp across the target's narrow side, PICK_ANYTHING_DEPTH below its top, at the grip
    yaw of its spot (the layouts turn it so, scripts/objects/pick_anything.py), waiting at each key for the
    rate-limited arm; lifted and held there."""
    robot = TopGripRobot(env)
    robot.settle = PATTERN_SETTLE
    centre = robot.object("target")[0]
    prefer = dict.fromkeys(robot.home, np.radians(pick_anything.grip_yaw(centre[0], centre[1])))
    arm, hand, _, _, _, close_cmd = yield from robot.pick("target", lift=PICK_ANYTHING_LIFT, close=True, prefer=prefer)
    yield from robot.move(arm, [(arm.grasp_point(), hand, close_cmd, PICK_HOLD)])


def put_in_tray_actions(env):
    """g1d_put_block_in_tray: the named block dropped into the tray from just above its walls (block_sort_actions)."""
    robot = Robot(env)
    robot.settle = PATTERN_SETTLE
    table = robot.object("target_block")[3]
    grasp = yield from atomic_pick(robot, "target_block")
    tray = robot.object("tray")
    yield from robot.place(*grasp, tray[0][:2], tray[4], label="target_block", hover=SORT_HOVER)
    yield from atomic_done(robot, grasp[0], table)


def rotate_parallel_actions(env):
    """g1d_rotate_block_parallel: the block set down where it lay, long side along X (place_turned)."""
    robot = Robot(env)
    robot.settle = PATTERN_SETTLE
    start, _, _, bottom, _ = robot.object("block")
    grasp = yield from atomic_pick(robot, "block")
    yield from place_turned(robot, grasp, start[:2], bottom, "block")
    yield from atomic_done(robot, grasp[0], bottom)
    end, rot, _, _, _ = robot.object("block")
    off, moved = (np.degrees(yaw_of(rot.as_matrix())) + 90) % 180 - 90, np.linalg.norm(end[:2] - start[:2])
    print(f"[scripted_demo] block long side {off:.1f} deg off X, moved {moved * 100:.1f} cm")


def line_up_actions(env):
    """g1d_line_up_blocks: block_<i> to block_atomic.row_slots()[i], the left hand's blocks first (place_turned: the
    fingers pass in front of and behind a block, not into the 15 mm gap to its neighbour)."""
    robot = Robot(env)
    robot.settle = PATTERN_SETTLE
    labels = ["block_0", "block_1", "block_2"]
    table = min(robot.object(label)[3] for label in labels)
    order = sorted(robot.arms, key=lambda a: robot.home[a.arm_key][0][0])
    plan = sorted(enumerate(labels), key=lambda p: order.index(robot.arm_for(robot.object(p[1])[0][0])))
    slots = block_atomic.row_slots()
    for n, (i, label) in enumerate(plan):
        grasp = yield from atomic_pick(robot, label)
        yield from place_turned(robot, grasp, np.asarray(slots[i]), table, label)
        following = plan[n + 1][1] if n + 1 < len(plan) else None
        if following is None or robot.arm_for(robot.object(following)[0][0]) is not grasp[0]:
            x, y, up = PATTERN_ASIDE
            side = np.sign(robot.home[grasp[0].arm_key][0][0])
            robot.send_aside(grasp[0], np.array([side * x, y, table + up]), PATTERN_ASIDE_ACTIONS)
    yield from robot.finish()


PICK_LIFT, PICK_HOLD = 0.10, 60  # m above the grasp, actions held there


# Build a shape (g1d_build_house): per action in free space, on the vertical approach / descent, of hand turn
FREE_STEP, SLOW_STEP, TURN_STEP = 0.005, 0.002, np.deg2rad(2.0)
SETTLE_TOL, SETTLE_MAX = 0.003, 40  # the pads close / open once the arm (Limiter caps its speed) is this close
SPEED = (0.8, 1.25)  # duration scale of every move, drawn per subtask
CUP_CARRY = 0.4  # share of FREE_STEP a cup is carried and lowered at
DETOUR = (0.03, 0.03)  # carry via point off the straight line: up to +- m sideways (y), up to m higher
GRASP_SLIDE = 0.3  # grasp point up to this share of (length - 3 cm) / 2 off a long block's centre
PAUSE = (0, 9)  # actions held still before a grasp and before a release
LIFTED = 0.015  # the block's bottom this far up after the lift, or the grasp is retried
HELD_TILT = 30.0  # deg; a block hanging more crooked than this after the lift is set down and grasped again (set
# down, a block hanging 10-20 deg tips onto its side or is pushed off its spot and the next grasps miss; placed, it
# settles flat)
RETREAT_CLEAR = 0.015  # fingertips above the house's top before the hand leaves it
GRASP_TRIES = 3
CARRY_TRIES = 2  # pick-ups of a block that fell out of the hand on the way over
DROPPED = 0.03  # m the release pose moved over the target: the block is no longer in the hand
LEAVE_NEAR, LEAVE_OUT, LEAVE_UP = 0.12, 0.10, 0.02  # m: a hand this near the house moves this far out to its side
# (and this much higher) before turning back to the start pose
# m from the start pose out to the arm's side and back toward the body: an arm waits here while the other works (at
# the start pose, ~10 cm from the house, it is hit by the other arm's forearm placing at the far end or the roof)
PARK_OUT, PARK_BACK = 0.12, 0.03
# with clearance checks (the Aurora Lite: its open jaws and wrist reach 7.5 cm sideways and 20 cm back from the grip
# point), the waiting arm's offset instead is the nearest of this grid out / back / up it reaches whose hand keeps
# HANDS_CLEAR + PARK_CLEAR from the working hand over that arm's planned pick-and-places (the working arm reaches past
# the middle for a block from across or a far place, into a hand waiting at PARK_OUT)
PARK_OUTS, PARK_BACKS, PARK_UPS = (0.12, 0.16, 0.20, 0.24, 0.28), (0.03, 0.08, 0.13), (0.0, 0.05)
PARK_CLEAR = 0.02
# straight-down approach (~90 deg of turn): done turning within this share of the move, at most this per action, so
# the hand is lined up well before it is over the block and only translates there
VERTICAL_TURN = (0.6, np.deg2rad(1.2))
JOINT_RATE = 0.5 / 30  # rad per action, average of the fastest joint in a joint-space move (eased: peak ~0.8 rad/s)
OFF_TARGET = 0.01  # m: a hand this far from the point over the block after the approach is moved there in joint space
CENTRE_GRIP = ("arch",)  # the pads close on the faces with the arch's opening: off centre they hold one leg only


def park_offsets(robot, arms, entry, parts, clear, table_z):
    """{side: offset from its start pose} each arm waits at while the other works (PARK_OUTS grid), with `clear` a
    clearance.Clearance; the default offset without one. The other arm's planned path: each of its parts' pick-and-
    place from where the part lies now to its layout target, at every tilt the layout allows, and halfway points."""
    from scripts.g1d import build_shapes as bs
    from scripts.g1d import clearance
    from scripts.g1d.reach_map import reached

    out = {s: -1 if s == "left" else 1 for s in arms}
    if clear is None or clearance.HANDS_CLEAR is None:
        return {s: np.array([out[s] * PARK_OUT, -PARK_BACK, 0.0]) for s in arms}
    top = max(np.asarray(p["pos"])[2] + bs.SIZES[p["category"]][2] / 2 for p in entry["parts"].values())
    hover = float(np.mean(bs.HOVER))
    paths = {s: [] for s in arms}
    for p in parts:
        info = entry["parts"][p.name]
        centre, rot, _, bottom, _ = robot.object(info["label"])
        side = info.get("arm") or ("left" if centre[0] < entry["origin"][0] else "right")
        src = (np.r_[centre[:2], bottom + bs.SIZES[p.category][2] / 2], heading(rot))
        tgt = (np.asarray(info["pos"], float), float(info["yaw_deg"]))
        for tilt in info.get("tilts") or entry.get("grasp_tilts", bs.TILTS):
            way = bs.waypoints(p, src, tgt, side, tilt, hover, top, table_z)
            paths[side] += way + [((a[0] + b[0]) / 2, a[1]) for a, b in zip(way, way[1:])]
    offsets = {}
    for side, arm in arms.items():
        other = paths["right" if side == "left" else "left"]
        model, order = robot.models[arm.arm_key]
        start = np.empty(len(order))
        start[order] = robot.start_q[arm.arm_key]
        p0, r0 = model.fk(start)
        shift, turn = robot.jitter.get(arm.arm_key, (np.zeros(3), Rotation.identity()))
        grid = sorted(([out[side] * o, -b, u] for o in PARK_OUTS for b in PARK_BACKS for u in PARK_UPS),
                      key=lambda v: np.linalg.norm(v))  # fmt: skip
        best = None
        for off in grid:
            q, err = model.solve(p0 + shift + off, turn * r0, q=start)
            if not reached(err):
                continue
            here = model.fk(q)
            gap = min((clear.hands_gap(here, w) if side == "left" else clear.hands_gap(w, here) for w in other), default=1.0)
            if best is None or gap > best[1]:
                best = (np.array(off, float), gap)
            if gap >= clearance.HANDS_CLEAR + PARK_CLEAR:
                best = (np.array(off, float), gap)
                break
        offsets[side] = best[0] if best else np.array([out[side] * PARK_OUT, -PARK_BACK, 0.0])
        print(f"[scripted_demo] {side} waits at {np.round(offsets[side], 2).tolist()} from its start pose "
              f"({(best[1] if best else float('nan')) * 100:.1f} cm from the other hand's path)", flush=True)
    return offsets


def heading(rot):
    """World yaw (deg) of a rotation's local x."""
    m = rot.as_matrix()
    return float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))


def build_actions(env, rng=None, log=None):
    """Build the layout's BuildShape (scripts/g1d/build_shapes.py), one arm at a time: the parts in a random order
    that keeps each after its supports (and the layout's "before" rules), each picked by the layout's arm for it (by
    default the arm on its supply side), whose fingers point inward. A
    finger tilt is drawn among those the URDF kinematics reach through the whole pick-and-place (the prism takes the
    flattest). The block's measured pose in the hand after the lift gives the release pose that sets it on its target
    (yaw modulo its symmetry), on the measured tops of its supports. Hover, speed, carry detour, grasp point along the
    block and pauses are drawn per subtask. `log` gets one record per subtask (frames, arm, block, poses)."""
    from omegaconf import OmegaConf

    from scripts.g1d import build_shapes as bs
    from scripts.g1d.reach_map import Arm as UrdfArm

    rng = np.random.default_rng(int(env.current_env_seed_map[0])) if rng is None else rng
    log = [] if log is None else log
    robot = Robot(env)
    entry = env.build_shape(0)
    if entry.get("hand", "dex1") != HAND_NAME:
        raise RuntimeError(f"layout made for the {entry.get('hand', 'dex1')} gripper, G1D_HAND={HAND_NAME}")
    parts = bs.parts_of(entry)
    fixed = entry.get("fixed", {})
    cfg = OmegaConf.to_container(env.robot_config, resolve=True)["robots"][0]
    arms = {("left" if a.arm_key.startswith("left") else "right"): a for a in robot.arms}
    model = {side: UrdfArm(side, cfg, urdf=a.robot.urdf_path, tip=a.robot.ee_link_name) for side, a in arms.items()}
    robot.plan_with(model)
    orders = bs.topo_orders(parts)
    # layout rules [a, b]: a before b (an arm's hand would sweep b, or hit it, placing a)
    rules = entry.get("before", [])
    orders = [o for o in orders if all(o.index(a) < o.index(b) for a, b in rules)] or orders
    order = orders[int(rng.integers(len(orders)))]

    def top_of(name):
        return robot.object(bs.thing(entry, name)["label"])[4]

    def site_top(names):
        """Highest top among these parts and the fixed objects (the trays, the box, the base cup)."""
        return max([top_of(n) for n in [*names, *fixed]] + [entry["table_z"]])

    # a part set into a container goes to the same spot relative to it however the container was pushed
    offsets = {}
    for p in parts:
        if p.into:
            ref = bs.thing(entry, p.into)["pos"]  # a container part's: where it is let go, over its own container
            offsets[p.name] = np.asarray(entry["parts"][p.name]["pos"][:2], float) - np.asarray(ref[:2], float)

    def goal(part, info):
        """(centre xyz, yaw) the part is released at: on its supports' measured tops, or into its container (its
        measured centre and top, release_dz)."""
        size = bs.SIZES[part.category]
        if part.into:
            centre = robot.object(bs.thing(entry, part.into)["label"])[0]
            return np.r_[centre[:2] + offsets[part.name], top_of(part.into) + part.release_dz + size[2] / 2], info["yaw_deg"]
        z0 = max((top_of(s) for s in part.supports), default=entry["table_z"])
        return np.r_[info["pos"][:2], z0 + size[2] / 2], info["yaw_deg"]

    def go(arm, point, rot, grip, step, speed, settle=False, lead=1.0, turn_step=TURN_STEP):
        here, turn = arm.grasp_point(), (rot * arm.pose()[1].inv()).magnitude()
        n = max(8, int(np.ceil(speed * max(np.linalg.norm(point - here) / step, turn / (turn_step * lead)))))
        yield from robot.move(arm, [(point, rot, grip, n, lead)])
        if settle:
            yield from robot.settle_at(arm, point, rot, grip, SETTLE_TOL, SETTLE_MAX)

    def hold(arm, grip, n):
        point, rot = arm.grasp_point(), arm.pose()[1]
        if n:
            yield from robot.move(arm, [(point, rot, grip, n)])

    # straight-down grasps put the wrist pitch at its limit, from where a Cartesian path back to the start pose's
    # rotation drives the hand into the table (v3e pilots 2-5): the retreat runs in joint space instead (home)
    vertical = min(entry.get("grasp_tilts", bs.TILTS)) >= bs.VERTICAL

    # v4 (layout frame "v4", build_house4_layouts --v4): start-pose jitter, blocks moved under the hand, target noise;
    # their own random streams, so the expert's draws stay those of the layout without them
    v4 = entry.get("v4")
    if v4:
        for side, a in arms.items():
            j = v4["init_jitter"][side]
            robot.jitter[a.arm_key] = (np.asarray(j["shift"], float), Rotation.from_rotvec(j["turn"]))
    v4_rng = np.random.default_rng(v4["noise"]["seed"]) if v4 else None
    from scripts.g1d import clearance

    clear = clearance.Clearance(cfg, urdf=arms["left"].robot.urdf_path) if clearance.HANDS_CLEAR is not None else None
    park_at = park_offsets(robot, arms, entry, parts, clear, entry["table_z"])

    def park(arm):
        point, rot = robot.home[arm.arm_key]
        shift, turn = robot.jitter.get(arm.arm_key, (np.zeros(3), Rotation.identity()))
        side = "left" if arm is arms["left"] else "right"
        return point + shift + park_at[side], turn * rot

    def move_block(label, part, side, tilt, hover, lift_up, house_top, how):
        """Set the block down `how["shift"]` m off where it lies (the first of 8 headings from how["heading_deg"]
        that keeps it clear of the other blocks and the house's places and reachable through the pick-and-place),
        turned how["yaw_deg"]; the new pose, or None."""
        lm, name = robot.lm, robot.lm.get_instance_name(0, label)
        pos, quat = (as_np(v) for v in lm.get_instance_pose(0, label=label))
        centre, rot, _, bottom, _ = robot.object(label)
        size = bs.SIZES[part.category]
        reach = np.linalg.norm(size[:2]) / 2
        others = [robot.object(e["label"])[0][:2] for n, e in entry["parts"].items() if e["label"] != label]
        others += [np.asarray(e["pos"][:2]) for n, e in entry["parts"].items() if e["label"] != label]
        tgt_name = next(n for n, e in entry["parts"].items() if e["label"] == label)
        tgt = goal(part, entry["parts"][tgt_name])
        turn = Rotation.from_euler("z", how["yaw_deg"], degrees=True)
        for k in range(8):
            a = np.deg2rad(how["heading_deg"] + 45 * k)
            d = how["shift"] * np.array([np.cos(a), np.sin(a)])
            xy = centre[:2] + d
            if any(np.linalg.norm(xy - o) < 2 * reach + V4_CLEAR for o in others):
                continue
            src = (np.r_[xy, bottom + size[2] / 2], heading(turn * rot))
            if not bs.feasible(model[side], part, src, tgt, side, house_top, entry["table_z"], tilts=(tilt,),
                               hovers=(hover,), lift_up=lift_up):  # fmt: skip
                continue
            import torch

            obj = lm.get_scene_object(0, name)
            x, y, z, w = (turn * from_wxyz(quat)).as_quat()
            world = pos + np.r_[d, 0.0] + robot.arms[0].origin
            obj.set_world_pose(position=torch.tensor(world, dtype=torch.float32),
                               orientation=torch.tensor([w, x, y, z], dtype=torch.float32))  # fmt: skip
            obj._rigid_prim_view.set_velocities(torch.zeros((1, 6)))
            return [float(v) for v in xy], float(how["yaw_deg"])
        return None

    def noise_for(rec, phase):
        """Robot.noise for one subtask's noise segment (v4 "noise"), starting now in `phase` (approach or transport):
        an Ornstein-Uhlenbeck offset of the tracked target for length_s, eased out over NOISE_EASE actions, cut when
        the phase changes (none in the descend / close / lower / release phases)."""
        cfg = v4["noise"]
        n = int(v4_rng.uniform(*cfg["length_s"]) * 30)
        dt, tau = 1 / 30, cfg["tau_s"]
        x, state = np.zeros(6), dict(left=n, phase=phase)
        sigma = np.r_[np.full(3, cfg["sigma_m"]), np.full(3, np.deg2rad(cfg["sigma_deg"]))]
        mark("noise_on", rec)

        def noise(arm):
            nonlocal x
            if state["left"] <= 0:
                return None
            if rec["events"] and rec["events"][-1][1] in bs.PHASES and rec["events"][-1][1] != state["phase"]:
                state["left"] = min(state["left"], NOISE_EASE)
            state["left"] -= 1
            if state["left"] < NOISE_EASE:
                x = x * 0.7
            else:
                x = x - x / tau * dt + sigma * np.sqrt(2 * dt / tau) * v4_rng.normal(size=6)
            if state["left"] == 0:
                mark("noise_off", rec)
                robot.noise = None
            return x[:3], x[3:]

        return noise

    def home(arm, speed=1.0):
        # turning back to the start pose swings the wrist and fingers through whatever is under the hand: leave the
        # house sideways first, hand as it is
        here, rot = arm.grasp_point(), arm.pose()[1]
        if np.linalg.norm(here[:2] - entry["origin"]) < LEAVE_NEAR:
            out = -1 if arm is arms["left"] else 1
            top = site_top(placed)
            up = max(here[2], top + bs.FINGER_TIPS + RETREAT_CLEAR + LEAVE_UP)
            yield from go(arm, np.r_[here[0], here[1], up], rot, 1.0, FREE_STEP, speed)
            yield from go(arm, np.r_[here[0] + out * LEAVE_OUT, here[1], up], rot, 1.0, FREE_STEP, speed)
        if vertical:
            # from the steep tilt a Cartesian path to the start pose's rotation drops the grip point to 2 cm above
            # the table on the way (v3e pilots 4-5); straight in joint space it stays >= 8 cm (URDF on their joints).
            # HOUSE4_HOME=start: back to the episode's start joints; default: the start hand at its park offset.
            robot.grip[arm.grip_key] = [1.0]
            if os.environ.get("HOUSE4_HOME") == "start":
                q_home = robot.start_q[arm.arm_key]
            else:
                side = "left" if arm is arms["left"] else "right"
                q_home = robot.joints_for(arm, park_at[side])
            yield from robot.move_joints(arm, q_home, JOINT_RATE / speed)
            yield from hold(arm, 1.0, 10)
            return
        point, rot = park(arm)
        yield from go(arm, point, rot, 1.0, FREE_STEP, speed, settle=True)

    placed, current = [], None
    for name in order:
        part, info = bs.by_name(parts, name), entry["parts"][name]
        label, size = info["label"], bs.SIZES[part.category]
        centre, rot, _, bottom, _ = robot.object(label)
        # the layout's arm (it may reach across for a block); older layouts: the arm on the block's side
        side = info.get("arm") or ("left" if centre[0] < entry["origin"][0] else "right")
        arm = arms[side]
        if current is not None and current is not arm:
            yield from home(current)  # out of the way of the other arm
        current = arm
        hover, speed = rng.uniform(*bs.HOVER), rng.uniform(*SPEED)
        aside, lift_up = rng.uniform(-DETOUR[0], DETOUR[0]), rng.uniform(0, DETOUR[1])
        table_z = entry["table_z"]
        src = (np.r_[centre[:2], bottom + size[2] / 2], heading(rot))
        tgt = goal(part, info)
        house_top = site_top(placed)
        tilts = bs.feasible(model[side], part, src, tgt, side, house_top, table_z, tilts=entry.get("grasp_tilts", bs.TILTS),
                            hovers=(hover,), lift_up=lift_up)  # fmt: skip
        tilts = tilts or info.get("tilts") or [60]
        tilt = min(tilts) if part.low_grip else float(rng.choice(tilts))
        rec = dict(
            index=len(log), part=name, label=label, arm=side, category=part.category, color=info["color"],
            supports=list(part.supports), tilt=tilt, hover=float(hover), speed=float(speed),
            detour=[float(aside), float(lift_up)],
            start_pos=centre.tolist(), start_yaw_deg=src[1], goal_pos=tgt[0].tolist(), goal_yaw_deg=tgt[1],
            frames=dict(start=robot.step), grasp_tries=0, events=[],
        )  # fmt: skip
        log.append(rec)

        noise_phase = str(v4_rng.choice(["approach", "transport"])) if v4 and v4["noise"]["on"] else None
        perturb = None if part.category in bs.UNMOVED else (v4 or {}).get("perturb", {}).get(name)
        # a tall light cup pinched by its rim swings out of the hand at the blocks' carry speed
        carry_step = FREE_STEP * (CUP_CARRY if part.category in bs.UNMOVED else 1.0)

        def mark(event, rec=rec):
            """[frame, event] in rec["events"]: a bs.PHASES phase starting there, or a bs.EVENTS event; a v4 noise
            segment starts with this subtask's noise phase (once)."""
            nonlocal noise_phase
            rec["events"].append([robot.step, event])
            if event == noise_phase:
                noise_phase = None
                robot.noise = noise_for(rec, event)

        print(f"[scripted_demo] build {name} ({label}) with {side} at tilt {tilt:g}, hover {hover:.3f}", flush=True)
        # HOUSE4_HOME=start: the other arm stays at the episode's start joints (the real rest pose), not the
        # outward park. Default: park it clear before the first reach.
        if rec["index"] == 0 and os.environ.get("HOUSE4_HOME") != "start":
            idle = next(a for a in robot.arms if a is not arm)
            yield from go(idle, *park(idle), 1.0, FREE_STEP, speed, settle=True)
        centred = part.low_grip or part.category in CENTRE_GRIP or part.category in bs.OBJECTS
        slide = 0.0 if centred else rng.uniform(-1, 1) * GRASP_SLIDE * max(0.0, size[0] - 0.03) / 2
        open_cmd = pad_open(size[1], arm.robot)
        sym = bs.KINDS[part.category][2]

        def aim():
            """Release pose (grip point, hand rotation) that sets the block, as it sits in the hand now, on its
            target. Only the yaw is corrected: turning the hand's tilt as well to undo the block's in-hand tilt puts
            the arm out of reach; a slightly tilted block settles on its support instead."""
            held, held_rot = robot.object(label)[:2]
            hand, grip = arm.pose()[1], arm.grasp_point()
            # into a container: the hand keeps its turn (build_shapes.waypoints)
            turn = 0.0 if part.into else (tgt[1] - heading(held_rot) + sym / 2) % sym - sym / 2
            cmd = Rotation.from_euler("z", turn, degrees=True) * hand
            tilt_now = float(np.degrees(np.arccos(np.clip(held_rot.apply([0, 0, 1])[2], -1, 1))))
            return tgt[0] + [0, 0, bs.PLACE_GAP] - cmd.apply(hand.inv().apply(held - grip)), cmd, turn, tilt_now

        def rejoin(target, rot, grip, what):
            """The warm-started tracking can slide into a branch against the joint limits on the way (task2 pilot:
            the hand 8-15 cm out to its side over a block, every retry the same; a roof carried 5 cm short of the
            house's top with the wrist roll at its limit): to `target` in joint space, solved from the start pose."""
            off = np.linalg.norm(arm.grasp_point() - target)
            if off > OFF_TARGET and (q := robot.joints_at(arm, target, rot)) is not None:
                print(f"[scripted_demo] {label} {what} {off * 1000:.0f} mm off: joint-space move", flush=True)
                rec["rejoined"] = rec.get("rejoined", 0) + 1
                yield from robot.move_joints(arm, q, JOINT_RATE / speed)
                yield from robot.settle_at(arm, target, rot, grip, SETTLE_TOL, SETTLE_MAX)

        def approach(above, shift):
            target = above[0] + shift
            if vertical:
                yield from go(arm, target, above[1], 1.0, FREE_STEP, speed, True, *VERTICAL_TURN)
            else:
                yield from go(arm, target, above[1], 1.0, FREE_STEP, speed, settle=True)
            yield from rejoin(target, above[1], 1.0, "approach")

        def descend(above, pick, shift):
            target = pick[0] + shift
            yield from go(arm, target, pick[1], 1.0, SLOW_STEP, speed, settle=True)
            off = np.linalg.norm(arm.grasp_point() - target)
            if off <= OFF_TARGET or (q_pick := robot.joints_at(arm, target, pick[1])) is None:
                return
            q_above = robot.joints_at(arm, above[0] + shift, above[1], seed=q_pick)
            if q_above is None:
                return
            # stalled on the way down (task2 pilot: 5 cm beside the block, 4 cm up): back over it and down again in
            # joint space, both poses on the pick's branch, so the fingers come down nearly straight
            print(f"[scripted_demo] {label} descend {off * 1000:.0f} mm off: joint-space move", flush=True)
            rec["rejoined"] = rec.get("rejoined", 0) + 1
            yield from robot.move_joints(arm, q_above, JOINT_RATE / speed)
            n = speed * np.linalg.norm(above[0] - pick[0]) / SLOW_STEP
            yield from robot.move_joints(arm, q_pick, np.abs(q_pick - q_above).max() / max(n, 10))
            yield from robot.settle_at(arm, target, pick[1], 1.0, SETTLE_TOL, SETTLE_MAX)

        rec["drops"], moved = 0, 0.0
        for carry_try in range(CARRY_TRIES):
            for _ in range(GRASP_TRIES):
                rec["grasp_tries"] += 1
                centre, rot, _, bottom, _ = robot.object(label)
                src = (np.r_[centre[:2], bottom + size[2] / 2], heading(rot))
                above, pick, lift, *_ = bs.waypoints(part, src, tgt, side, tilt, hover, house_top, table_z, lift_up)
                shift = np.r_[rot.apply([slide, 0, 0])[:2], 0.0]
                # fully open until the hand is at the grasp (the real gripper does not pre-close on the way)
                mark("approach")
                yield from approach(above, shift)
                if perturb and "perturbed" not in rec:
                    rec["perturbed"] = move_block(label, part, side, tilt, hover, lift_up, house_top, perturb)
                    print(f"[scripted_demo] {label} moved under the hand: {rec['perturbed']}", flush=True)
                    if rec["perturbed"]:
                        mark("perturbed")
                        yield from hold(arm, 1.0, int(v4_rng.integers(*NOTICE)))
                        centre, rot, _, bottom, _ = robot.object(label)
                        src = (np.r_[centre[:2], bottom + size[2] / 2], heading(rot))
                        above, pick, lift, *_ = bs.waypoints(part, src, tgt, side, tilt, hover, house_top, table_z, lift_up)
                        shift = np.r_[rot.apply([slide, 0, 0])[:2], 0.0]
                        mark("approach")
                        yield from approach(above, shift)
                mark("descend")
                yield from descend(above, pick, shift)
                yield from hold(arm, 1.0, int(rng.integers(*PAUSE)))
                rec["frames"]["grasp"] = robot.step
                mark("close")
                yield from robot.move(arm, [(pick[0] + shift, pick[1], 0, 18)])  # close on the table
                close_cmd = hold_cmd(arm)
                rec["frames"]["lift"] = robot.step
                mark("lift")
                yield from go(arm, lift[0] + shift, lift[1], close_cmd, SLOW_STEP * 1.5, speed)
                held, held_rot, _, held_bottom, _ = robot.object(label)
                lifted = held_bottom > bottom + LIFTED
                crooked = float(np.degrees(np.arccos(np.clip(held_rot.apply([0, 0, 1])[2], -1, 1))))
                if lifted and crooked < HELD_TILT:
                    break
                print(f"[scripted_demo] {label} lifted {lifted}, tilt {crooked:.1f} deg: regrasp", flush=True)
                mark("grasp_fail")
                if lifted:
                    yield from go(arm, pick[0] + shift, pick[1], close_cmd, SLOW_STEP, speed, settle=True)
                    yield from robot.move(arm, [(pick[0] + shift, pick[1], 1.0, 12)])
                yield from go(arm, above[0] + shift, above[1], 1.0, SLOW_STEP, speed)
            else:
                break

            hand, grip = arm.pose()[1], arm.grasp_point()
            place, cmd, turn, rec["in_hand_tilt_deg"] = aim()
            print(f"[scripted_demo] {label} in hand: tilt {rec['in_hand_tilt_deg']:.1f} deg, turn {turn:+.1f} deg", flush=True)
            up = grip[2] - held_bottom
            drop = bs.place_hover(place[2], hover, table_z)
            carry = max(grip[2], place[2] + drop, house_top + bs.OVER + up)
            via = np.r_[(grip[:2] + place[:2]) / 2 + [0, aside], max(carry, min(carry + lift_up, table_z + bs.CEIL))]
            mid = Slerp([0, 1], Rotation.concatenate([hand, cmd]))(0.5)
            mark("transport")
            yield from go(arm, via, mid, close_cmd, carry_step, speed)
            yield from go(arm, np.r_[place[:2], carry], cmd, close_cmd, carry_step, speed, settle=True)
            yield from rejoin(np.r_[place[:2], carry], cmd, close_cmd, "carry")
            planned = place
            place, cmd, turn, tilt_now = aim()  # the block shifts in the hand on the way over
            moved = np.linalg.norm(place - planned)
            rec["carry_shift_mm"], rec["carry_tilt_deg"] = float(moved * 1000), tilt_now
            print(f"[scripted_demo] {label} over target: shifted {moved * 1000:.1f} mm, tilt {tilt_now:.1f} deg", flush=True)
            if moved < DROPPED:
                break
            # out of the hand: chasing it with the release pose drives the arm through the house
            rec["drops"] += 1
            print(f"[scripted_demo] {label} dropped on the way: pick it up again", flush=True)
            mark("drop")
            here = arm.grasp_point()
            yield from robot.move(arm, [(here, cmd, open_cmd, 12)])
            clear = site_top(placed) + bs.FINGER_TIPS + RETREAT_CLEAR
            yield from go(arm, np.r_[here[:2], max(here[2], clear)], cmd, open_cmd, SLOW_STEP * 1.5, speed)
        if moved >= DROPPED or not (lifted and crooked < HELD_TILT):
            print(f"[scripted_demo] {label} not picked up ({rec['grasp_tries']} grasps, {rec['drops']} drops): give up", flush=True)
            break
        yield from go(arm, np.r_[place[:2], carry], cmd, close_cmd, carry_step, speed, settle=True)
        mark("lower")
        yield from go(arm, place + [0, 0, drop], cmd, close_cmd, carry_step, speed)
        yield from go(arm, place, cmd, close_cmd, SLOW_STEP, speed, settle=True)
        yield from hold(arm, close_cmd, int(rng.integers(*PAUSE)))
        rec["frames"]["release"] = robot.step
        mark("release")
        yield from robot.move(arm, [(place, cmd, open_cmd, 12)])
        mark("withdraw")
        # up clear of the house before moving sideways: the fingertips hang FINGER_TIPS below the grip point
        clear = site_top([*placed, name]) + bs.FINGER_TIPS + RETREAT_CLEAR
        yield from go(arm, np.r_[place[:2], max(place[2] + drop, clear)], cmd, open_cmd, SLOW_STEP * 1.5, speed)
        rec["frames"]["end"] = robot.step
        placed.append(name)
        labels = [entry["parts"][n]["label"] for n in placed] + [f["label"] for f in fixed.values()]
        poses = {lb: robot.lm.get_instance_pose(0, label=lb) for lb in labels}
        errs = bs.check(dict(entry, parts={n: entry["parts"][n] for n in placed}), {k: [as_np(v) for v in p] for k, p in poses.items()})
        print(
            f"[scripted_demo] placed {name}: "
            + ", ".join(f"{n} {e['xy'] * 1000:.0f}/{e['z'] * 1000:.0f} mm {e['yaw']:.0f}/{e['tilt']:.0f} deg" for n, e in errs.items()),
            flush=True,
        )
    if current is not None:
        yield from home(current)
    yield from hold(current or robot.arms[0], 1.0, 10)


# experts whose commands go through Limiter (the others predate it; their datasets were made without it)
LIMITED = {
    "pick_bucket",
    "g1d_pour_water_deliver_to_lounge",
    "g1d_build_block_pattern",
    "g1d_sort_blocks_by_color",
    "g1d_sort_blocks_by_shape",
    "g1d_pour_beans_plus",
    "g1d_pick_anything",
    *block_atomic.TASKS,
    "g1d_build_house",
    "g1d_sort_into_trays",
    "g1d_nest_cups",
    "g1d_put_fruit_in_box",
    "g1d_microwave_corn",
}

DEMOS = {
    "g1d_pour_water_into_cup": pour_actions,
    "g1d_pour_beans_cup_to_cup": cup_to_cup_actions,
    "g1d_pour_beans_measuring_cups": cup_to_cup_actions,
    "g1d_pour_beans_half_cup": half_cup_actions,
    "g1d_pour_beans_plus": plus_actions,
    "g1d_pour_beans_half_cup_rigid": half_cup_actions,
    "g1d_general_pickup": pickup_actions,
    "g1d_pick_anything": pick_anything_actions,
    "g1d_stack_blocks": stack_actions,
    "g1d_deliver_cup_to_lounge": deliver_cup_actions,
    "g1d_pour_water_deliver_to_lounge": pour_deliver_actions,
    "g1d_build_block_pattern": block_pattern_actions,
    "g1d_sort_blocks_by_color": block_sort_actions,
    "g1d_sort_blocks_by_shape": block_sort_actions,
    "g1d_place_block_on_mark": place_on_mark_actions,
    "g1d_stack_block_on_block": stack_on_block_actions,
    "g1d_pick_named_block": pick_named_actions,
    "g1d_put_block_in_tray": put_in_tray_actions,
    "pick_bucket": lambda env: __import__("scripts.g1d.pick_bucket_demo", fromlist=["actions"]).actions(env),
    "g1d_rotate_block_parallel": rotate_parallel_actions,
    "g1d_line_up_blocks": line_up_actions,
    "g1d_build_house": build_actions,
    "g1d_sort_into_trays": build_actions,
    "g1d_nest_cups": build_actions,
    "g1d_put_fruit_in_box": build_actions,
    "g1d_microwave_corn": lambda env, rng=None, log=None: __import__(
        "scripts.g1d.microwave_demo", fromlist=["actions"]
    ).actions(env, rng, log),
}


class Limiter:
    """Real-robot motion limits on every command: arm joints REAL_JOINT_MARGIN inside their limits and at most
    REAL_ARM_VMAX, the gripper command at most REAL_GRIP_RATE per action. Logs the fastest requested move."""

    def __init__(self, env):
        self.arms = [Arm(env, r) for r in env.robot_manager.robot_list if r.type == "target"]
        self.q = {a.arm_key: a.q() for a in self.arms}
        self.grip = {}
        self.lim = {
            a.arm_key: as_np(a.key.data.soft_joint_pos_limits[0, a.cols]).reshape(-1, 2)
            + [REAL_JOINT_MARGIN, -REAL_JOINT_MARGIN]
            for a in self.arms
        }
        self.dq_max = REAL_ARM_VMAX / 30  # rad per 30 Hz action
        self.asked = self.asked_grip = 0.0

    def __call__(self, action):
        out = dict(action)
        label = dict(out.pop(LABEL, {}))
        for a in self.arms:
            if a.arm_key in label:  # the noise-free command, limited from the same last command as the executed one
                q = np.clip(np.asarray(label[a.arm_key], dtype=float), *self.lim[a.arm_key].T)
                label[a.arm_key] = self.q[a.arm_key] + np.clip(q - self.q[a.arm_key], -self.dq_max, self.dq_max)
            if a.arm_key in out:
                q = np.clip(np.asarray(out[a.arm_key], dtype=float), *self.lim[a.arm_key].T)
                step = q - self.q[a.arm_key]
                self.asked = max(self.asked, float(np.abs(step).max()))
                self.q[a.arm_key] = self.q[a.arm_key] + np.clip(step, -self.dq_max, self.dq_max)
                out[a.arm_key] = self.q[a.arm_key]
            if a.grip_key in out:
                g = float(out[a.grip_key][0])
                prev = self.grip.get(a.grip_key, g)
                self.asked_grip = max(self.asked_grip, abs(g - prev))
                self.grip[a.grip_key] = prev + float(np.clip(g - prev, -REAL_GRIP_RATE, REAL_GRIP_RATE))
                out[a.grip_key] = [self.grip[a.grip_key]]
        if label:
            out[LABEL] = label
        return out


def eval_one_episode(TASK_ENV, model_client):
    TASK_ENV.get_obs()  # initial frame (also triggers the pre-action reward check so is_lift records table height)
    actions = DEMOS[TASK_ENV.task_name](TASK_ENV)
    limiter = Limiter(TASK_ENV) if TASK_ENV.task_name in LIMITED else None
    for action in actions:
        action = limiter(action) if limiter else action
        TASK_ENV.take_action({k: v for k, v in action.items() if k != LABEL})
        if TASK_ENV.is_episode_end():
            break
        TASK_ENV.get_obs()  # streams the camera frames into the episode videos
    if TASK_ENV.task_name == "pick_bucket":
        import json
        import os

        trace_path = ROOT / "output" / (os.environ.get("ROBODOJO_RUN_ID", "pick_bucket") + "_trace.json")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            json.dumps({"trace": TASK_ENV.trace[0], "history": TASK_ENV.progress[0].history}, indent=2)
        )
    if limiter:
        print(
            f"[scripted_demo] limiter: fastest requested arm step {limiter.asked:.3f} rad"
            f" ({limiter.asked * 30:.2f} rad/s, cap {REAL_ARM_VMAX}),"
            f" gripper {limiter.asked_grip:.3f} per action (cap {REAL_GRIP_RATE:.3f})"
        )
    if not TASK_ENV.end_flag[0]:  # EvalEnv starts each episode with success=True: running out of script is a fail
        TASK_ENV.success[0] = False
        TASK_ENV.end_flag[0] = True


def single_gpu_kit_args():
    """Kit args keeping the renderer on one GPU. Only with several GPUs visible: with one (CUDA_VISIBLE_DEVICES set to
    it, as run_batch.py and the house launcher do), --/renderer/multiGpu/enabled=false leaves the tiled camera
    annotators without a data window and the first capture fails."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return "" if visible and "," not in visible else "--/renderer/multiGpu/enabled=false"


def main(extra_kit_args="", policy_module="scripts.g1d.scripted_demo"):
    """Run src/eval_client/main.py in-process with `policy_module`'s eval_one_episode as the policy (no server).
    ROBODOJO_KIT_ARGS adds Kit arguments (the isaaclab_newton environment's extsDeprecated folder)."""
    extra_kit_args = f"{extra_kit_args} {os.environ.get('ROBODOJO_KIT_ARGS', '')}".strip()
    args = sys.argv[1:]
    defaults = {"--task_name": "g1d_pour_water_into_cup", "--seed": "0", "--device_id": "0"}
    args += [x for k, v in defaults.items() if k not in args for x in (k, v)]
    sys.argv = [str(ROOT / "src/eval_client/main.py"), *args, "--policy_name", POLICY, "--port", "0"]
    sys.argv += ["--policy_module", policy_module]
    sys.argv += ["--additional_info", "scripted", "--headless", "--enable_cameras"]
    sys.argv += [
        "--kit_args",
        f"--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera {extra_kit_args}".strip(),
    ]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
