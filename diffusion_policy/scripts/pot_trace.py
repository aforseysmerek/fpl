"""
pot_trace.py — scripted TwoArmLift (bimanual pot-carry) policy, mirroring the
wipe/peg pattern (absolute OSC_POSE waypoints + linear interpolation +
euler->axis_angle), with the two preference axes as knobs on top of a clean
grasp-lift-carry:

  speed_amount  (0..1): how fast the pot is carried (0 = slow, 1 = fast).
                        maps to the number of env steps spent in the carry phase.
  height_amount (0..1): how high the pot is carried above the table (0 = low
                        skim, 1 = high lift). maps to the lift height.

Both arms are driven in lockstep with a fixed downward gripper orientation; the
grippers close to grasp the two handles, then the pot is lifted and carried
laterally. Prints the REALIZED oracle metrics (mean carry speed + mean pot
height above the table) so we can check they track the knobs. --sweep renders
the 4 corners.

Run in robodiff:
    MUJOCO_GPU=<free> python scripts/pot_trace.py --sweep -o pot_sweep
    MUJOCO_GPU=<free> python scripts/pot_trace.py --speed_amount 0.8 --height_amount 0.3
"""
import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import click
import numpy as np
import robosuite

from diffusion_policy.model.common.rotation_transformer import RotationTransformer

CAMERAS = ["agentview"]   # trace video only needs agentview; 1 cam ~2x faster on CPU render
IMG = 128
CTRL_DT = 1.0 / 20.0      # control_freq=20 -> dt for m/s speed metric
OPEN, CLOSE = -1.0, 1.0

OSC_POSE_ABS = {
    "type": "OSC_POSE", "input_max": 1, "input_min": -1,
    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
    "kp": 300, "damping": 1, "impedance_mode": "fixed",   # stiff: needed for both arms to reach the side handles
    "kp_limits": [0, 300], "damping_limits": [0, 10],
    "position_limits": None, "orientation_limits": None,
    "uncouple_pos_ori": True, "control_delta": False,   # absolute, like the peg/wipe collection
    "interpolation": None, "ramp_ratio": 0.2,
}

# height_amount -> lift height of the pot above the table during the carry. Wide
# range so the knob is unmistakable: at 0 it's a low skim just above the table, at
# 1 it's a distinctly high carry. NOTE the low end is BELOW the env's success
# margin (pot bottom > table+0.10), so episodes are gated by our own carry check
# (grasped + reached goal + airborne; see GOAL_TOL / AIRBORNE_H) instead of
# env._check_success(). Max is 0.45 — the 0.60 top saturated (reach/sag-limited).
CARRY_H_MIN = 0.05      # height_amount=0 -> low skim carry
CARRY_H_RANGE = 0.40    # height_amount=1 -> 0.45 lift

# speed_amount -> number of env steps in the lateral carry phase (fewer = faster).
# The carry is the DOMINANT phase of the episode (the approach is compressed to a
# quick reach+grasp). Wide range so the knob really bites: fast end is a brisk carry,
# slow end a long deliberate one. (Per-step travel stays well under the controller's
# tracking limit even at the fast end, so the pot isn't jostled loose.)
CARRY_STEPS_SLOW = 400
CARRY_STEPS_FAST = 30

# +x is the symmetric carry axis for the opposed arms (both hands take the same
# delta), so it has the most reach headroom. The pot spawns at table center and the
# table is 0.8 m wide (half-width 0.4), so ~0.32 is a wide A->B that stays on the
# table; push --goal_x further and watch goal_dist/lifted if you want it wider.
GOAL_XY = (0.32, 0.0)   # world (x,y) the pot is carried to (the task destination)
APPROACH_H = 0.14       # m above the grasp point: height of the approach arc's
                        # control point — the arc crests near this clearance and
                        # arrives moving straight down onto the bar

# Fixed approach speed: the reach + descend (everything BEFORE the pot is held)
# run at this one medium speed for EVERY episode; the speed knob only applies
# from the lift onward (i.e. once the pot is grasped). Decoupling these also
# pins the wrist-yaw slerp to the same gentle rate every episode — previously
# it was tied to the carry speed, so fast episodes crammed the whole align+reach
# into a few steps (the visible twitch/twist). 0.075 m/s is the geometric mean
# of the carry-speed extremes (~0.021 and ~0.28 m/s) — the perceptual middle.
APPROACH_V = 0.075      # m/s

# Our own carry-success gate, replacing env._check_success() (which needs the pot
# above table+0.10 at episode end — the new low-carry end deliberately undershoots
# that). A kept episode must have BOTH handles grasped at the lift gate, ended
# within GOAL_TOL of the goal, and actually carried the pot airborne.
GOAL_TOL = 0.10         # m: final pot xy must be within this of the goal
AIRBORNE_H = 0.02       # m: mean pot-bottom height over the carry must exceed this

# SPAWN_BACK: shift the pot's spawn toward -x (away from the +x goal) so the carry
# is longer. The pot normally spawns at table center; 0.10 back makes the A->B carry
# ~0.10 m longer (and, at fixed carry steps, faster). Too far back can push the grasp
# out of a comfortable reach -> watch the grasp gate / dial back if attempts fail.
SPAWN_BACK = 0.10

# agentview camera: robosuite 1.2.0 has no CameraMover, so we edit the sim model's
# camera directly. CAM_PULLBACK dollies it backward along its own optical axis (true
# zoom-out that keeps the aim point); CAM_FOVY widens the field of view on top.
CAM_PULLBACK = 0.55     # m to pull the agentview camera back (0 = leave in place)
CAM_FOVY = 52.0         # agentview vertical FOV in deg (robosuite default ~45)


def create_pot_env(pot8qt=False):
    if pot8qt:
        import envs  # noqa: F401  registers TwoArmLift8QtPot with robosuite.make
    return robosuite.make(
        "TwoArmLift8QtPot" if pot8qt else "TwoArmLift",
        robots=["Panda", "Panda"], env_configuration="single-arm-opposed",
        controller_configs=OSC_POSE_ABS,
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=CAMERAS, camera_heights=IMG, camera_widths=IMG,
        control_freq=20, horizon=4000, ignore_done=True, hard_reset=False,
        render_gpu_device_id=int(os.environ.get("MUJOCO_GPU", "-1")),
    )


def grab(obs):
    a = np.asarray(obs["agentview_image"])
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8)
    return np.flipud(a).copy()


def pot_bottom_above_table(env):
    z = float(env.sim.data.site_xpos[env.pot_center_id][2] - env.pot.top_offset[2])
    return z - float(env.sim.data.site_xpos[env.table_top_id][2])


# ---- oracle metrics on the REALIZED trajectory (carry phase only) ----
def speed_metric(pot_xy, t0, t1):
    """Mean lateral pot speed over the carry window [t0, t1] in m/s."""
    seg = np.asarray(pot_xy[t0:t1])
    if len(seg) < 2:
        return 0.0
    dist = float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))
    return dist / (max(len(seg) - 1, 1) * CTRL_DT)


def height_metric(heights, t0, t1):
    """Mean pot-bottom height above the table over the carry window [t0, t1]."""
    seg = np.asarray(heights[t0:t1])
    return float(seg.mean()) if len(seg) else 0.0


# OSC controls the gripper's "grip_site" frame in WORLD coordinates (verified:
# controllers/osc.py reads site_xpos/site_xmat, absolute goals set directly).
# The grasp points the gripper straight DOWN (grip_site local z -> world -z). For
# yaw we DON'T just hold the arm's natural yaw: the handle bar rotates with the
# pot's random spawn yaw, so the finger-vs-bar angle varies per episode. When the
# finger-separation axis drifts toward parallel with the bar, the fingers close
# ALONG it (pinch air / shove the pot) instead of across it -- the pose-dependent
# failure. Instead we align the finger axis with the RADIAL direction (h_other ->
# h_self, i.e. perpendicular to the bar) so the fingers straddle the bar for every
# spawn yaw, but pick the branch NEAREST the arm's reachable natural yaw and clamp
# the twist to +-MAX_TWIST. The bar is 180-symmetric, so the needed twist is never
# more than +-90 deg and the clamp keeps us close to the natural yaw (an earlier
# "one fixed bar-aligned yaw" attempt over-twisted -> arm missed by 3-9 cm; this
# nearest-branch + clamp avoids that).
MAX_TWIST = np.deg2rad(60)   # max wrist twist away from the reachable natural yaw
TWIST_RATE = 15.0            # deg/s cap on the wrist-align slerp: the approach is
                             # stretched so even a worst-case align turns gently
                             # (60 deg -> >= 4 s) instead of flailing the arm

_RT_Q2M = RotationTransformer('quaternion', 'matrix')
_RT_M2AA = RotationTransformer('matrix', 'axis_angle')
_RT_M2Q = RotationTransformer('matrix', 'quaternion')
_RT_Q2AA = RotationTransformer('quaternion', 'axis_angle')


def _smoothstep(u):
    """Ease-in/ease-out on u in [0,1] (zero velocity at the ends). Applied to the
    interpolation fraction so the arms accelerate/decelerate smoothly instead of
    snapping toward each target (which the stiff OSC gains turn into overshoot/wiggle)."""
    u = min(max(float(u), 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _bezier(a, ctrls, b, u):
    """Bezier from a to b, bowed through control point(s) ctrls, at fraction u
    (de Casteljau). Used for the approach and the lift+carry so the hands rise +
    translate + arrive in ONE continuous curve — velocity never hits zero
    mid-motion, unlike chained linear waypoints that park at every corner."""
    pts = [np.asarray(p, float) for p in [a, *ctrls, b]]
    while len(pts) > 1:
        pts = [(1.0 - u) * p + u * q for p, q in zip(pts[:-1], pts[1:])]
    return pts[0]


def _arc_len(a, ctrls, b, n=32):
    """Approximate Bezier arc length by sampling (duration = length / speed)."""
    pts = np.stack([_bezier(a, ctrls, b, u) for u in np.linspace(0.0, 1.0, n + 1)])
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _slerp(q0, q1, t):
    """Shortest-arc quaternion slerp (wxyz). Used to ramp the wrist from its reset
    orientation to the grasp orientation over the approach, so it doesn't snap."""
    q0 = np.asarray(q0, float); q1 = np.asarray(q1, float)
    q0 = q0 / np.linalg.norm(q0); q1 = q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0.0:            # take the shorter arc
        q1 = -q1; d = -d
    if d > 0.9995:         # nearly aligned -> linear + renormalize
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th0 = np.arccos(d); th = th0 * t
    q2 = q1 - q0 * d; q2 = q2 / np.linalg.norm(q2)
    return q0 * np.cos(th) + q2 * np.sin(th)


def _natural_yaw(R_site):
    """The arm's reachable yaw: grip_site local x projected onto the table.
    @R_site is the grip_site rotation MATRIX read from the sim — NOT
    obs['..eef_quat'], which is the hand BODY frame, a fixed 90 deg away from the
    grip_site frame the OSC controls (feeding that in biased the yaw-branch pick
    and, as the slerp start, snapped the wrist at t=0)."""
    x = np.asarray(R_site, float)[:, 0].copy(); x[2] = 0.0   # local x projected horizontal
    return float(np.arctan2(x[1], x[0]))


def bar_perp_grasp_R(R_site, h_self, h_other, max_twist=MAX_TWIST):
    """World-frame rotation MATRIX for a straight-down grasp whose finger axis is
    aligned with the radial direction (perpendicular to the handle bar), choosing
    the branch nearest the arm's natural yaw and clamping the wrist twist.
    @R_site is the arm's grip_site rotation matrix at reset (from the sim, see
    _natural_yaw); @h_self / @h_other are this arm's handle and the other handle
    (xy defines the radial direction)."""
    nat = _natural_yaw(R_site)                                       # natural yaw
    r = np.asarray(h_other[:2], float) - np.asarray(h_self[:2], float)   # radial dir = perp bar
    want = float(np.arctan2(r[1], r[0]))
    d = (want - nat + np.pi / 2) % np.pi - np.pi / 2                 # wrap to (-90, 90]
    d = float(np.clip(d, -max_twist, max_twist))                    # limit wrist twist
    sx, sy = np.cos(nat + d), np.sin(nat + d)
    return np.array([[sx,  sy, 0.0],     # col0 = finger axis, now ~radial (straddles the bar)
                     [sy, -sx, 0.0],     # col1 = local y
                     [0.0, 0.0, -1.0]])  # col2 = approach (local z) -> world down


def finger_bar_deg(R_site, h_self, h_other):
    """Diagnostic: angle (deg) between the arm's NATURAL finger axis and the handle
    bar. Near 0 -> fingers near-parallel to the bar (the failure band); near 90 ->
    fingers straddle it. Uses the natural (un-twisted) yaw so it measures the raw
    per-spawn geometry."""
    nat = _natural_yaw(R_site)
    r = np.asarray(h_other[:2], float) - np.asarray(h_self[:2], float)   # radial dir
    want = float(np.arctan2(r[1], r[0]))
    a = (nat - want + np.pi / 2) % np.pi - np.pi / 2                 # finger-vs-radial, (-90, 90]
    return 90.0 - abs(np.rad2deg(a))                                # -> finger-vs-bar


def pot_yaw_deg(quat_xyzw):
    """Pot spawn yaw (deg) from obs['pot_quat'] — the per-episode variable that the
    handle bar rotates with; log it to see which yaw band the grasp fails in."""
    R = _RT_Q2M.forward(np.asarray(quat_xyzw, float)[[3, 0, 1, 2]])
    return float(np.rad2deg(np.arctan2(R[1, 0], R[0, 0])))


def set_agentview_camera(env, pullback=CAM_PULLBACK, fovy=CAM_FOVY):
    """Zoom the fixed agentview camera out so the full grasp->carry fits the frame.
    Dollies the camera backward along its own optical axis (keeps the aim point) and
    widens the FOV, editing the sim model directly (no CameraMover in robosuite 1.2).
    Idempotent: cam_pos/cam_fovy are persistent model fields (hard_reset=False keeps
    them across soft resets), so we always dolly from the cached ORIGINAL pose rather
    than accumulate — safe to call once per run (e.g. across the 4 sweep runs)."""
    cid = env.sim.model.camera_name2id("agentview")
    if not hasattr(env, "_agentview_pose0"):
        env._agentview_pose0 = (np.asarray(env.sim.model.cam_pos[cid], float).copy(),
                                float(env.sim.model.cam_fovy[cid]))
    pos0, fovy0 = env._agentview_pose0
    env.sim.model.cam_fovy[cid] = float(fovy) if fovy else fovy0
    R = _RT_Q2M.forward(np.asarray(env.sim.model.cam_quat[cid], float))   # cam_quat is wxyz
    env.sim.model.cam_pos[cid] = pos0 + R[:, 2] * pullback                # +z points back along the optical axis
    env.sim.forward()   # push the model edit into the rendered scene


def set_spawn_back(env, dx=SPAWN_BACK):
    """Shift the pot's spawn toward -x (away from the +x goal) by dx metres so the
    A->B carry is longer. Edits the placement sampler's reference_pos; idempotent
    (always offsets from the cached original), so it's safe to call before each run.
    Takes effect on the NEXT env.reset()."""
    s = env.placement_initializer
    if not hasattr(env, "_spawn_ref0"):
        env._spawn_ref0 = np.asarray(s.reference_pos, float).copy()
    ref = env._spawn_ref0.copy()
    ref[0] -= dx
    s.reference_pos = ref


class PotTracePolicy:
    def __init__(self, env, start_eef0, start_eef1,
                 speed_amount=0.5, height_amount=0.5, grasp_offset=0.0, goal_xy=GOAL_XY,
                 approach_grip=OPEN):
        self.env = env
        self.speed_amount = float(speed_amount)
        self.height_amount = float(height_amount)
        self.grasp_offset = float(grasp_offset)
        # gripper command held during the approach/descend (fingers close from the
        # grasp dwell regardless). OPEN (fully open, 8cm) gives ~2.4cm of radial
        # pad-to-bar clearance, well above the OSC's ~1.5cm lateral wander during
        # the twisting descent; a partial opening (e.g. 0.0 = hold the 4cm reset
        # width) shrinks that tolerance below the wander and fingers land ON the
        # bar -- only reach for it if an object's clearances truly demand it.
        self.approach_grip = float(approach_grip)
        self.goal_xy = np.asarray(goal_xy, float)
        # grip_site rotation matrices at reset, read STRAIGHT FROM THE SIM — the
        # frame the OSC actually controls. obs['..eef_quat'] is the hand BODY
        # frame, a fixed 90 deg away: used as the slerp start it commanded a pose
        # 90 deg from where the wrist already was, so the stiff OSC yanked the
        # whole arm at t=0 (and it biased the least-twist yaw-branch selection).
        # The grasp orientations are computed in generate_trajectory, AFTER the
        # handles are snapshotted and assigned to arms (the finger axis is set
        # relative to the radial/bar direction, so it needs to know which handle
        # each arm is going to).
        self.R_start0 = np.asarray(env.sim.data.site_xmat[
            env.sim.model.site_name2id("gripper0_grip_site")], float).reshape(3, 3).copy()
        self.R_start1 = np.asarray(env.sim.data.site_xmat[
            env.sim.model.site_name2id("gripper1_grip_site")], float).reshape(3, 3).copy()
        self.carry_h = CARRY_H_MIN + self.height_amount * CARRY_H_RANGE
        self.carry_steps = int(round(CARRY_STEPS_FAST +
                                     (1.0 - self.speed_amount) * (CARRY_STEPS_SLOW - CARRY_STEPS_FAST)))
        self.reset(start_eef0, start_eef1)

    def generate_trajectory(self):
        # SNAPSHOT the two handle positions (site_xpos is a live view).
        ha = np.array(self.env._handle0_xpos, float).copy()   # physical handle0
        hb = np.array(self.env._handle1_xpos, float).copy()   # physical handle1
        # Assign each arm its NEAREST handle (by xy from the arm's start eef).
        # Depending on the spawn yaw, physical handle0 can land on arm1's side;
        # sending arm0 there forces a long reach / near-crossed geometry. Swap so
        # arm0 always takes whichever handle is closer to it.
        e0 = self.start_eef0[:2]
        if np.linalg.norm(e0 - ha[:2]) <= np.linalg.norm(e0 - hb[:2]):
            h0, h1, self.arm0_hidx, self.arm1_hidx = ha, hb, 0, 1
        else:
            h0, h1, self.arm0_hidx, self.arm1_hidx = hb, ha, 1, 0

        # grasp orientation (rotation matrix -> quaternion): finger axis ~radial
        # (straddles the bar) at the branch nearest each arm's natural yaw. Computed
        # here, after handle assignment. Kept as quaternions so the wrist can be
        # SLERPed from its reset orientation to this over the approach (no snap).
        self.q_grasp0 = _RT_M2Q.forward(bar_perp_grasp_R(self.R_start0, h0, h1))
        self.q_grasp1 = _RT_M2Q.forward(bar_perp_grasp_R(self.R_start1, h1, h0))
        self.q_start0 = _RT_M2Q.forward(self.R_start0)   # slerp start = ACTUAL site pose (wxyz)
        self.q_start1 = _RT_M2Q.forward(self.R_start1)   #   -> zero orientation error at t=0
        # per-spawn diagnostics (natural finger-vs-bar angle; <~30 deg = failure band)
        self.finger_bar = (finger_bar_deg(self.R_start0, h0, h1),
                           finger_bar_deg(self.R_start1, h1, h0))

        gz = self.grasp_offset
        up = np.array([0.0, 0.0, 1.0])

        # carry translation: move the pot from its start (handle midpoint) to the
        # goal, in xy only. The SAME delta is applied to both hands, so the fixed
        # hand-to-hand distance set by the pot is preserved throughout the carry.
        pot_xy = ((h0 + h1) / 2.0)[:2]
        lat = np.array([self.goal_xy[0] - pot_xy[0], self.goal_xy[1] - pot_xy[1], 0.0])

        h0d, h1d = h0 + up * gz, h1 + up * gz                # grasp pose on the bar
        lift0, lift1 = h0d + up * self.carry_h, h1d + up * self.carry_h

        # Speeds: the approach runs at the FIXED medium APPROACH_V for every
        # episode, so the speed knob only shows up once the pot is grasped (the
        # lift+carry arc at the knob's per-step speed). Only the physical dwells
        # (finger close, goal settle) are fixed step counts — they aren't travel.
        carry_len = float(np.linalg.norm(lat))
        v = max(carry_len / max(self.carry_steps, 1), 1e-4)   # knob speed, m per env step
        v_app = APPROACH_V * CTRL_DT                          # fixed approach, m per env step

        GRASP_DWELL, SETTLE = 50, 40                          # fixed (physical, not travel)

        # ONE continuous approach arc per arm: quadratic Bezier from the reset eef
        # to the grasp point, bowed through a control point APPROACH_H above the
        # bar — rise + translate in a single sweep, arriving moving straight down
        # onto the bar. (Replaces reach-above -> descend, whose per-segment easing
        # parked the arms at the corner.) The wrist-align slerp shares this clock,
        # fully aligned exactly at bar arrival, and the approach is stretched so
        # the twist never turns faster than TWIST_RATE — a fast commanded twist
        # into the stiff OSC is what used to flail the whole arm.
        ctrl0, ctrl1 = h0d + up * APPROACH_H, h1d + up * APPROACH_H

        def _quat_deg(qa, qb):
            d = abs(float(np.dot(qa / np.linalg.norm(qa), qb / np.linalg.norm(qb))))
            return float(np.rad2deg(2.0 * np.arccos(min(d, 1.0))))
        twist = max(_quat_deg(self.q_start0, self.q_grasp0),
                    _quat_deg(self.q_start1, self.q_grasp1))
        twist_steps = int(np.ceil(twist / (TWIST_RATE * CTRL_DT)))   # min approach duration
        arc_app = max(_arc_len(self.start_eef0, [ctrl0], h0d),
                      _arc_len(self.start_eef1, [ctrl1], h1d))

        # waypoints: (t, pos0, pos1, grip[, ctrls0, ctrls1]). Positions interpolate
        # between waypoints (eased by smoothstep in predict_action); segments with
        # control points follow their Bezier arc instead of a straight line.
        wp = []
        t = max(int(round(arc_app / v_app)), twist_steps, 8)   # approach arc (twist-rate floor)
        wp.append((t, h0d, h1d, self.approach_grip, [ctrl0], [ctrl1]))
        self.t_align = t                          # slerp spans the whole approach: aligned at the bar

        t += GRASP_DWELL                          # close + seat (fingers close from the start of this dwell)
        wp.append((t, h0d, h1d, CLOSE))
        self.lift_t = t
        self.grasp_check_t = t - 5                # gate the lift just before it begins

        # ONE continuous lift+carry arc per arm: cubic Bezier that rises straight
        # off the bar, crests to the carry height early (both control points sit AT
        # carry height), then sweeps laterally, arriving at the goal horizontally —
        # replaces lift-up -> hold -> lateral carry, which paused at each corner.
        # Runs at the knob's per-step speed v. The metric window is the arc's
        # SECOND half (the cruise at carry height, the rise fully faded by then),
        # so the height/speed knobs still read out cleanly.
        end0, end1 = lift0 + lat, lift1 + lat
        cl0, cl1 = [lift0, end0 - 0.3 * lat], [lift1, end1 - 0.3 * lat]
        arc_carry = max(_arc_len(h0d, cl0, end0), _arc_len(h1d, cl1, end1))
        carry_steps = max(int(round(arc_carry / v)), 8)
        self.carry_t0 = t + carry_steps // 2      # metric window: cruise half of the arc
        t += carry_steps
        wp.append((t, end0, end1, CLOSE, cl0, cl1))
        self.carry_t1 = t

        t += SETTLE                               # settle at goal
        wp.append((t, end0, end1, CLOSE))

        self.trajectory = []
        for w in wp:
            d = {"t": w[0], "pos0": w[1], "pos1": w[2], "grip": w[3]}
            if len(w) > 4:                        # curved segment: Bezier control point(s)
                d["ctrl0"], d["ctrl1"] = w[4], w[5]
            self.trajectory.append(d)
        self.max_t = self.trajectory[-1]["t"]

    def reset(self, start_eef0, start_eef1):
        self.step_num = 0
        self.start_eef0 = np.asarray(start_eef0, float)   # needed for nearest-handle assignment
        self.start_eef1 = np.asarray(start_eef1, float)
        self.generate_trajectory()
        self.last_pos0 = np.asarray(start_eef0, float)
        self.last_pos1 = np.asarray(start_eef1, float)
        self.last_grip = self.approach_grip
        self.last_t = 0

    def predict_action(self):
        if len(self.trajectory) > 1 and self.step_num >= self.trajectory[0]["t"]:
            w = self.trajectory.pop(0)
            self.last_pos0, self.last_pos1, self.last_grip, self.last_t = \
                w["pos0"], w["pos1"], w["grip"], w["t"]
        nxt = self.trajectory[0]
        if nxt["t"] <= self.last_t:
            frac = 1.0
        else:
            frac = (self.step_num - self.last_t) / (nxt["t"] - self.last_t)
        ef = _smoothstep(frac)   # ease in/out -> smooth accel/decel, no overshoot at waypoints
        if "ctrl0" in nxt:       # curved segment: follow the Bezier arc
            pos0 = _bezier(self.last_pos0, nxt["ctrl0"], nxt["pos0"], ef)
            pos1 = _bezier(self.last_pos1, nxt["ctrl1"], nxt["pos1"], ef)
        else:
            pos0 = self.last_pos0 + (nxt["pos0"] - self.last_pos0) * ef
            pos1 = self.last_pos1 + (nxt["pos1"] - self.last_pos1) * ef
        grip = nxt["grip"]   # gripper is a discrete event, take the segment target
        # orientation ramps smoothly from the reset wrist pose to the grasp pose over
        # the reach (by t_align), then holds — so the wrist doesn't snap at t=0.
        of = _smoothstep(min(self.step_num / max(self.t_align, 1), 1.0))
        o0 = _RT_Q2AA.forward(_slerp(self.q_start0, self.q_grasp0, of)).reshape(3)
        o1 = _RT_Q2AA.forward(_slerp(self.q_start1, self.q_grasp1, of)).reshape(3)
        self.step_num += 1
        # per-arm action = [pos(3), axis-angle(3), gripper(1)] -> 14 total
        return np.concatenate([pos0, o0, [grip], pos1, o1, [grip]])


def run_one(env, speed_amount, height_amount, grasp_offset, goal_xy, out_mp4, max_tries=6,
            cam_pullback=CAM_PULLBACK, cam_fovy=CAM_FOVY, spawn_back=SPAWN_BACK,
            approach_grip=OPEN):
    """Run one grasp->lift->carry. Grasp reliability is pose-dependent, so retry
    on a fresh pot pose until the pot is actually lifted (so every rendered demo
    is a real A->B carry). Returns the last (successful, if found) attempt."""
    adim = env.action_dim
    handle_geoms = [env.pot.handle0_geoms, env.pot.handle1_geoms]   # indexed by arm{0,1}_hidx
    set_spawn_back(env, spawn_back)   # shift spawn back -> longer carry (takes effect on reset)
    result = None
    for attempt in range(1, max_tries + 1):
        obs = env.reset()
        set_agentview_camera(env, cam_pullback, cam_fovy)   # zoom out (idempotent; before any render)
        pol = PotTracePolicy(env, obs["robot0_eef_pos"], obs["robot1_eef_pos"],
                             speed_amount, height_amount, grasp_offset, goal_xy,
                             approach_grip=approach_grip)
        pyaw = pot_yaw_deg(obs["pot_quat"])   # per-episode spawn yaw (diagnostic)
        frames, pot_xy, heights = [], [], []
        grasped = (False, False)
        for i in range(pol.max_t):
            a = pol.predict_action()
            if adim != len(a):   # defensive: match env action dim
                a = np.resize(a, adim)
            obs, *_ = env.step(a)
            frames.append(grab(obs))
            pot_xy.append(np.asarray(obs["pot_pos"][:2]).copy())
            heights.append(pot_bottom_above_table(env))
            # gate the lift: check the SAME grasps the env reward uses, just before
            # the lift waypoint fires. If either handle isn't held, this run is
            # doomed -> abort now and skip the remaining post-grasp steps.
            if i == pol.grasp_check_t:
                grasped = (bool(env._check_grasp(env.robots[0].gripper, handle_geoms[pol.arm0_hidx])),
                           bool(env._check_grasp(env.robots[1].gripper, handle_geoms[pol.arm1_hidx])))
                if not all(grasped):
                    break
        spd = speed_metric(pot_xy, pol.carry_t0, pol.carry_t1)
        hgt = height_metric(heights, pol.carry_t0, pol.carry_t1)
        goal_dist = float(np.linalg.norm(pot_xy[-1] - pol.goal_xy))
        success = bool(all(grasped) and goal_dist < GOAL_TOL and hgt > AIRBORNE_H)
        result = dict(speed=spd, height=hgt, goal_dist=goal_dist, success=success, frames=frames)
        # per-attempt diagnostic: if the natural-yaw failure hypothesis holds,
        # failures cluster in a pot_yaw band where finger-bar goes near 0 deg.
        print(f"  attempt {attempt}: pot_yaw={pyaw:6.1f}deg  "
              f"finger-bar[a0={pol.finger_bar[0]:4.1f} a1={pol.finger_bar[1]:4.1f}]deg  "
              f"grasp[g0={grasped[0]} g1={grasped[1]}]  lifted={success}", flush=True)
        if success:
            break
    print(f"speed_knob={speed_amount:.1f} height_knob={height_amount:.1f}  ->  "
          f"carry_speed={result['speed']:5.3f} m/s  carry_height={result['height']:5.3f} m  "
          f"goal_dist={result['goal_dist']:5.3f} m  lifted={result['success']}", flush=True)
    try:
        import imageio
        imageio.mimwrite(out_mp4, result["frames"], fps=20)
    except Exception:
        pass
    return result


@click.command()
@click.option('-o', '--out_dir', default='pot_trace')
@click.option('--speed_amount', type=float, default=0.5)
@click.option('--height_amount', type=float, default=0.5)
@click.option('--grasp_offset', type=float, default=0.0, help='grip_site height above the bar (~fingertip gap; use pot_probe value)')
@click.option('--goal_x', type=float, default=GOAL_XY[0], help='world x the pot is carried to (wider = longer carry; watch reach)')
@click.option('--goal_y', type=float, default=GOAL_XY[1], help='world y the pot is carried to')
@click.option('--cam_pullback', type=float, default=CAM_PULLBACK, help='m to dolly the agentview camera back (zoom out)')
@click.option('--cam_fovy', type=float, default=CAM_FOVY, help='agentview vertical FOV in deg (wider = zoom out)')
@click.option('--spawn_back', type=float, default=SPAWN_BACK, help='m to shift the pot spawn back (-x) for a longer carry')
@click.option('--sweep', is_flag=True, help='render the 4 axis corners')
@click.option('--pot8qt', is_flag=True, help='use the textured 8-quart stockpot mesh instead of the stock pot')
def main(out_dir, speed_amount, height_amount, grasp_offset, goal_x, goal_y, cam_pullback, cam_fovy, spawn_back, sweep, pot8qt):
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    env = create_pot_env(pot8qt=pot8qt)
    goal_xy = (goal_x, goal_y)
    approach_grip = OPEN   # see PotTracePolicy.approach_grip; OPEN works for both pots
    if sweep:
        for s in (0.2, 0.9):
            for h in (0.2, 0.9):
                run_one(env, s, h, grasp_offset, goal_xy,
                        os.path.join(out_dir, f"speed{s:.1f}_height{h:.1f}.mp4"),
                        cam_pullback=cam_pullback, cam_fovy=cam_fovy, spawn_back=spawn_back,
                        approach_grip=approach_grip)
        print(f"\nsaved 4 corner videos to {out_dir}/")
    else:
        run_one(env, speed_amount, height_amount, grasp_offset, goal_xy,
                os.path.join(out_dir, "trace.mp4"),
                cam_pullback=cam_pullback, cam_fovy=cam_fovy, spawn_back=spawn_back,
                approach_grip=approach_grip)
        print(f"saved {out_dir}/trace.mp4")


if __name__ == '__main__':
    main()
