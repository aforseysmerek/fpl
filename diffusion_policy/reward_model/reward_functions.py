"""
Per-axis reward functions for the PickPlace + slow_fast benchmarks.

Each function takes a trajectory and returns one scalar reward for that
trajectory. They are the single source of truth for axis values — the
reward-model dataset builder calls them on every trajectory (demo or
rollout) instead of reading per-episode metadata from HDF5/npz attrs.

Function signature
------------------
    fn(obs, actions=None, reward_series=None) -> float

where:
- obs:           (T, D) numpy array of observations, trimmed to the real
                 episode length (caller is responsible for this — these
                 functions do NOT see padded data).
- actions:       (T, A) numpy array of actions, similarly trimmed (optional;
                 needed for smoothness).
- reward_series: (T,) per-step env reward (optional; only used by `success`
                 as a peg-task fallback).

Obs schema
----------
PickPlace obs (65 dims):
    [0..55]: object info — 14 dims × 4 objects in order [Milk, Bread, Cereal, Can]
              each object: obj_pos(3) obj_quat(4) obj_to_eef_pos(3) obj_to_eef_quat(4)
    [56..58]: robot0_eef_pos
    [59..62]: robot0_eef_quat
    [63..64]: robot0_gripper_qpos

Peg/square obs (23 dims):
    [0..2]:  nut_pos (world)
    [3..13]: rest of `object`
    [14..]:  robot state

Bin regions match robosuite's PickPlace `not_in_bin` check exactly:
bin2_pos=(0.1, 0.28, 0.8), bin_size=(0.39, 0.49, 0.82). z window = bin2_pos[2] .. bin2_pos[2]+0.1.
"""

import numpy as np


# ---------- shared constants ----------
PICKPLACE_OBJ_NAMES = ['Milk', 'Bread', 'Cereal', 'Can']
PICKPLACE_CANONICAL_ORDER = [1, 3, 0, 2]   # right-first

# bin_id → (x_lo, x_hi, y_lo, y_hi) for the target bin in bin2 (world coords).
PICKPLACE_BIN_BOUNDS = {
    0: (-0.095, 0.1,    0.035, 0.28),    # Milk   (x-low,  y-low)
    1: ( 0.1,   0.295,  0.035, 0.28),    # Bread  (x-high, y-low)
    2: (-0.095, 0.1,    0.28,  0.525),   # Cereal (x-low,  y-high)
    3: ( 0.1,   0.295,  0.28,  0.525),   # Can    (x-high, y-high)
}
PICKPLACE_BIN_Z_LO = 0.8
PICKPLACE_BIN_Z_HI = 0.9

# Drop-vs-careful threshold for the eef z at the moment the gripper opens
# above an object's target bin. Tracking lag in the OSC controller means
# "careful intent" (commanded z ~ 0.84) shows up as observed eef z ~ 0.90,
# while "drop intent" (commanded ~ 0.95+) shows up as observed ~ 0.96+.
# A threshold of 0.93 separates the two cleanly.
DROP_HEIGHT_THRESHOLD = 0.93


def _obj_pos(obs_t, obj_id):
    """3-vector of object obj_id's world position at one timestep."""
    base = obj_id * 14
    return obs_t[base:base + 3]


def _eef_pos(obs_t):
    return obs_t[56:59]


def _has_pickplace_obs(obs):
    """True if the obs schema looks like PickPlace (≥65 dims: 4 objects × 14)."""
    return obs is not None and obs.ndim >= 2 and obs.shape[-1] >= 65


def _object_in_bin(obs_t, obj_id):
    p = _obj_pos(obs_t, obj_id)
    if p.shape[0] < 3:
        return False
    x_lo, x_hi, y_lo, y_hi = PICKPLACE_BIN_BOUNDS[obj_id]
    return (x_lo < float(p[0]) < x_hi
            and y_lo < float(p[1]) < y_hi
            and PICKPLACE_BIN_Z_LO < float(p[2]) < PICKPLACE_BIN_Z_HI)


def _first_lift_times(obs):
    """For each of the 4 PickPlace objects, the first step where its z rises
    >5 cm above its starting z. Returns a list of length 4; len(obs) if the
    object was never lifted.

    Objects parked far off-table by robosuite's `clear_objects` (e.g. xy ~= 10)
    are ignored — they're not in the scene for this task so their z drift is
    not a meaningful "lift" event.
    """
    T = len(obs)
    times = [T] * 4
    for obj_id in range(4):
        x0 = float(obs[0, obj_id * 14])
        y0 = float(obs[0, obj_id * 14 + 1])
        # Sanity check: bin1 + workspace is roughly x∈[-0.5, 0.5], y∈[-0.7, 0.7].
        # Anything outside that is a parked/cleared object — skip it.
        if abs(x0) > 1.0 or abs(y0) > 1.0:
            continue
        z = obs[:, obj_id * 14 + 2]
        z0 = float(z[0])
        ups = np.where(z > z0 + 0.05)[0]
        if len(ups) > 0:
            times[obj_id] = int(ups[0])
    return times


def _gripper_width(obs_t):
    return float(obs_t[63] - obs_t[64])


def _release_eef_z(obs, obj_id, actions=None):
    """Gripper z at the moment this object's body first enters its bin region.

    For careful releases the gripper has lowered into ~0.84-0.91 by then;
    for drop releases the gripper is still hovering above 0.93+. Combined
    with the DROP_HEIGHT_THRESHOLD this cleanly separates the two modes
    even when the demo terminates before the gripper-open command fires
    (last few settle steps may keep the gripper closed).
    """
    if obs is None or len(obs) == 0:
        return None
    for t in range(len(obs)):
        if _object_in_bin(obs[t], obj_id):
            return float(_eef_pos(obs[t])[2])
    return None


# ----------------------------------------------------------------------------
# Per-axis reward functions
# ----------------------------------------------------------------------------


def success(obs, actions=None, reward_series=None):
    """1.0 if any goal condition is met:
      - PickPlace (obs ≥ 65 dims): any object in its bin at episode end, OR
      - Peg / square_twopeg (obs ≥ 23 dims): nut ends on either peg, OR
      - reward_series ever ≥ 1 (env-reward fallback).
    """
    if obs is None or len(obs) == 0:
        return 0.0
    # PickPlace check
    if _has_pickplace_obs(obs):
        if any(_object_in_bin(obs[-1], i) for i in range(4)):
            return 1.0
    # Peg-task check (nut on either peg).
    if peg_reward(obs) != 0.0:
        return 1.0
    if reward_series is not None and len(reward_series) > 0:
        return 1.0 if float(np.max(reward_series)) >= 1.0 else 0.0
    return 0.0


def speed_reward(obs, actions=None, reward_series=None, max_steps=2000):
    """Linear in episode length: 1.0 fast, 0.1 at max_steps. 0 if the
    trajectory didn't succeed."""
    L = len(obs) if obs is not None else 0
    base = 1.0 - 0.9 * (L / max_steps)
    return float(base) if success(obs, reward_series=reward_series) >= 1.0 else 0.0


def smoothness(obs, actions=None, reward_series=None):
    """exp(-10 * mean_jerk) over actions, gated by success."""
    if actions is None or len(actions) < 4:
        s = 1.0
    else:
        jerk = np.diff(actions, n=3, axis=0)
        jerk_mag = float(np.mean(np.linalg.norm(jerk, axis=-1)))
        s = float(np.exp(-10.0 * jerk_mag))
    if success(obs, reward_series=reward_series) < 1.0:
        s = 0.0
    return s


def _placed(obs, obj_id):
    if obs is None or len(obs) == 0 or not _has_pickplace_obs(obs):
        return 0.0
    return 1.0 if _object_in_bin(obs[-1], obj_id) else 0.0


def milk_placed(obs, actions=None, reward_series=None):    return _placed(obs, 0)
def bread_placed(obs, actions=None, reward_series=None):   return _placed(obs, 1)
def cereal_placed(obs, actions=None, reward_series=None):  return _placed(obs, 2)
def can_placed(obs, actions=None, reward_series=None):     return _placed(obs, 3)


def _placed_raw(obs, obj_id):
    """Continuous version of _placed: negative xy-distance from the object's
    final position to the target bin center. Range ~[-0.6, 0]:
      - object centered in target bin → ≈ 0
      - object at far edge of target  → ≈ -0.1
      - object still in bin1 / on table → ≈ -0.4 (workspace-scale)
    Returns 0 if obs schema isn't PickPlace.
    """
    if obs is None or len(obs) == 0 or not _has_pickplace_obs(obs):
        return 0.0
    final_pos = _obj_pos(obs[-1], obj_id)
    if final_pos.shape[0] < 2:
        return 0.0
    x_lo, x_hi, y_lo, y_hi = PICKPLACE_BIN_BOUNDS[obj_id]
    tgt_x = (x_lo + x_hi) / 2.0
    tgt_y = (y_lo + y_hi) / 2.0
    dx = float(final_pos[0]) - tgt_x
    dy = float(final_pos[1]) - tgt_y
    return -float(np.sqrt(dx * dx + dy * dy))


def milk_placed_raw(obs, actions=None, reward_series=None):    return _placed_raw(obs, 0)
def bread_placed_raw(obs, actions=None, reward_series=None):   return _placed_raw(obs, 1)
def cereal_placed_raw(obs, actions=None, reward_series=None):  return _placed_raw(obs, 2)
def can_placed_raw(obs, actions=None, reward_series=None):     return _placed_raw(obs, 3)


def order_reward(obs, actions=None, reward_series=None):
    """PickPlace-only axis. Returns 0 if the obs schema isn't PickPlace.

    +1 if objects were placed in the canonical right-first order
    (Bread → Can → Milk → Cereal, restricted to the active subset),
    -1 if the reversed-of-active order, 0 otherwise.

    Active objects are inferred from initial position: anything not parked
    off-table (|x|<1 and |y|<1 at t=0). For PickPlace_2 the active subset is
    [Bread, Can], so canonical = [1, 3] and reversed = [3, 1]; the 1-object
    case correctly resolves Bread→+1 and Can→-1.

    Detection event is object first-time-in-bin (placement), not first-lift —
    placement is the definitive "did this object actually get put in its bin".
    """
    if obs is None or len(obs) == 0 or not _has_pickplace_obs(obs):
        return 0.0
    T = len(obs)
    # Determine active object subset from initial positions.
    active_ids = []
    for obj_id in range(4):
        x0 = float(obs[0, obj_id * 14])
        y0 = float(obs[0, obj_id * 14 + 1])
        if abs(x0) <= 1.0 and abs(y0) <= 1.0:
            active_ids.append(obj_id)
    if not active_ids:
        return 0.0
    # active_canonical: canonical right-first order restricted to active set.
    active_canonical = [i for i in PICKPLACE_CANONICAL_ORDER if i in active_ids]
    active_reversed = list(reversed(active_canonical))

    # First-placement timestep for each active object (T if never placed).
    first_place = {obj_id: T for obj_id in active_ids}
    for obj_id in active_ids:
        for t in range(T):
            if _object_in_bin(obs[t], obj_id):
                first_place[obj_id] = t
                break
    placed_in_order = [i for i in sorted(active_ids, key=lambda x: first_place[x])
                       if first_place[i] < T]
    if len(placed_in_order) == 0:
        return 0.0
    canonical_prefix = active_canonical[:len(placed_in_order)]
    reversed_prefix = active_reversed[:len(placed_in_order)]
    if placed_in_order == canonical_prefix:
        return 1.0
    if placed_in_order == reversed_prefix:
        return -1.0
    return 0.0


def order_reward_raw(obs, actions=None, reward_series=None):
    """Continuous time-weighted version of order_reward.

    For the first two active objects (in right-first canonical order):
      `order_reward_raw = (t_b - t_a) / T`
    where `t_a` is the first time the first-canonical object enters its bin
    and `t_b` is the same for the second-canonical object, both clipped to
    T if never placed.

    Range: roughly [-1, +1].
      - +1: canonical-direction (a placed early, b placed late, or only a placed)
      -  0: simultaneous placement, OR no placements at all
      - -1: reversed-direction (b placed early, a placed late, or only b placed)

    Partial placements get extreme values (≈ ±0.8) which is what we want for
    a *direction* signal — the magnitude is informational. The composite
    reward still penalizes partial completion via the *_placed_raw axes.
    """
    if obs is None or len(obs) == 0 or not _has_pickplace_obs(obs):
        return 0.0
    T = len(obs)
    if T == 0:
        return 0.0
    # Determine active object subset from initial positions.
    active_ids = []
    for obj_id in range(4):
        x0 = float(obs[0, obj_id * 14])
        y0 = float(obs[0, obj_id * 14 + 1])
        if abs(x0) <= 1.0 and abs(y0) <= 1.0:
            active_ids.append(obj_id)
    # Need at least 2 active objects in canonical order to define direction.
    active_canonical = [i for i in PICKPLACE_CANONICAL_ORDER if i in active_ids][:2]
    if len(active_canonical) < 2:
        return 0.0
    obj_a, obj_b = active_canonical
    # First-placement times (T if never placed).
    t_a, t_b = T, T
    for t in range(T):
        if t_a == T and _object_in_bin(obs[t], obj_a):
            t_a = t
        if t_b == T and _object_in_bin(obs[t], obj_b):
            t_b = t
        if t_a < T and t_b < T:
            break
    # Neither placed → 0.
    if t_a == T and t_b == T:
        return 0.0
    return float(t_b - t_a) / float(T)


def _drop_value(obs, obj_id, actions=None):
    """+1 careful, -1 drop, 0 if the object never lands in its bin.
    PickPlace-only — returns 0 if the obs schema isn't PickPlace."""
    if obs is None or len(obs) == 0 or not _has_pickplace_obs(obs):
        return 0.0
    rz = _release_eef_z(obs, obj_id, actions=actions)
    if rz is None:
        return 0.0
    return 1.0 if rz < DROP_HEIGHT_THRESHOLD else -1.0


def milk_drop(obs, actions=None, reward_series=None):    return _drop_value(obs, 0, actions=actions)
def bread_drop(obs, actions=None, reward_series=None):   return _drop_value(obs, 1, actions=actions)
def cereal_drop(obs, actions=None, reward_series=None):  return _drop_value(obs, 2, actions=actions)
def can_drop(obs, actions=None, reward_series=None):     return _drop_value(obs, 3, actions=actions)


def _drop_value_raw(obs, obj_id, actions=None):
    """Continuous version of _drop_value: negative of the release-height
    (gripper z when the object first enters its bin). Smaller release
    height = more careful = larger (less negative) value.

    Fallback ordering when the object never enters its bin:
      1. If the object WAS lifted (z ↑ > 5 cm above start) but never reached
         the bin, use the max eef_z observed after the lift event. This
         captures "released mid-arc" demos with a value that smoothly extends
         the in-bin release-height distribution.
      2. If the object was never lifted at all (grasp totally failed), use
         -1.10 (just below the worst real drop) so failure ranks below any
         actual release without dominating the reward scale.

    Typical raw values:
      - careful release rz ~ 0.84-0.91 → raw ~ -0.84 to -0.91
      - high drop      rz ~ 0.96-1.04 → raw ~ -0.96 to -1.04
      - lifted but not placed (mid-arc release) → -0.95 to -1.10 (continuous)
      - never lifted at all → -1.10 (sentinel)
    PickPlace-only; returns 0 if obs schema isn't PickPlace.
    """
    if obs is None or len(obs) == 0 or not _has_pickplace_obs(obs):
        return 0.0
    rz = _release_eef_z(obs, obj_id, actions=actions)
    if rz is not None:
        return -float(rz)
    # Object never landed in its target bin. If it WAS lifted (grasp succeeded
    # at least once), fall back to the max eef_z observed during the lift
    # phase — corresponds to where the gripper was when it released into
    # mid-air. Smooths the failure tail rather than dumping every failure on
    # the same value.
    lift_times = _first_lift_times(obs)
    if lift_times[obj_id] < len(obs):
        eef_z_post_lift = [
            float(_eef_pos(obs[t])[2]) for t in range(lift_times[obj_id], len(obs))
        ]
        if eef_z_post_lift:
            # Clamp to -1.10 so the worst mid-arc drop never dips below the
            # never-lifted sentinel — keeps the failure cluster compact.
            return -float(min(max(eef_z_post_lift), 1.10))
    return -1.10


def milk_drop_raw(obs, actions=None, reward_series=None):    return _drop_value_raw(obs, 0, actions=actions)
def bread_drop_raw(obs, actions=None, reward_series=None):   return _drop_value_raw(obs, 1, actions=actions)
def cereal_drop_raw(obs, actions=None, reward_series=None):  return _drop_value_raw(obs, 2, actions=actions)
def can_drop_raw(obs, actions=None, reward_series=None):     return _drop_value_raw(obs, 3, actions=actions)


def drop_reward(obs, actions=None, reward_series=None):
    """Signed fraction-careful over placed objects, in [-1, +1].
    All careful → +1, all drop → -1, no placements → 0."""
    vals = [_drop_value(obs, i, actions=actions) for i in range(4)]
    n_careful = sum(1 for v in vals if v > 0)
    n_drop = sum(1 for v in vals if v < 0)
    n_placed = n_careful + n_drop
    if n_placed == 0:
        return 0.0
    return (n_careful - n_drop) / n_placed


def peg_reward(obs, actions=None, reward_series=None):
    """+1 if the nut ends on the right peg (y≈-0.1), -1 left (y≈+0.1),
    0 neither. Square/peg obs layout: nut_pos is at the first 3 dims of `obs`.
    The nut sits at z≈0.89–0.92 when on a peg (peg tops at ~0.95)."""
    if obs is None or len(obs) == 0:
        return 0.0
    nut_pos = obs[-1, :3]
    # Reasonable z window for "nut on a peg" — anywhere between the table
    # surface and a few cm above the peg top.
    on_table_or_peg = 0.85 < float(nut_pos[2]) < 0.95
    if abs(float(nut_pos[0]) - 0.23) < 0.04 and on_table_or_peg:
        if abs(float(nut_pos[1]) - 0.1) < 0.04:
            return -1.0   # left peg
        if abs(float(nut_pos[1]) - (-0.1)) < 0.04:
            return 1.0    # right peg
    return 0.0


# Right (correct) peg location for slow_fast / square_twopeg.
PEG_RIGHT_XY = np.array([0.23, -0.1], dtype=np.float32)
# Workspace-scale max distance used to normalise peg_reward_raw into [0, 1].
# Nuts start somewhere in the central region and need to travel <= ~0.4 m
# to reach the right peg; using 0.4 means any nut on the wrong peg / on the
# floor gets a value close to 0.
PEG_MAX_DISTANCE = 0.4


def peg_reward_raw(obs, actions=None, reward_series=None):
    """Continuous version of peg_reward: how close the nut's final xy is to
    the RIGHT peg, normalised so the result sits roughly in [0, 1] and
    matches the scale of `speed_reward`.

      raw = clamp(1 - distance_to_right_peg / PEG_MAX_DISTANCE, 0, 1)

    Values:
      - 1.0  : nut exactly on the right peg
      - ~0.9 : nut on the peg with a small offset
      - ~0.5 : nut on the left peg (~0.2 m away from the right one)
      - ~0.0 : nut on the floor far from either peg

    This collapses the discrete {-1, 0, +1} peg_reward into a single
    continuous "closeness to target" signal that combines cleanly with
    `speed_reward` (also ~[0, 1]) in a composite-mean reward.
    """
    if obs is None or len(obs) == 0:
        return 0.0
    nut_xy = obs[-1, :2].astype(np.float32)
    distance = float(np.linalg.norm(nut_xy - PEG_RIGHT_XY))
    return float(max(0.0, 1.0 - distance / PEG_MAX_DISTANCE))


# ----------------------------------------------------------------------------
# Wipe axes. These read the wipe collector's state_lowdim layout:
#   [robot0_eef_pos(3), robot0_eef_quat(4), joints(7), proportion_wiped(1)]
# eef xy is the first two dims; proportion_wiped is the last dim. (Same layout
# whether obs comes straight from state_lowdim or is reconstructed from the
# per-key split, since the split preserves column order.)
# ----------------------------------------------------------------------------
# --- circle-detection gates for _circle_fraction ---------------------------
# Radius band matches generation (wipe_trace CIRC_RADIUS_RANGE = 3-5 cm loops)
# with margin; the upper gate also rejects straight-ish segments, which fit
# circles of huge radius. RMS gate: points must lie on the fitted circle to
# within this fraction of its radius (rejects zigzags / scribbles of the right
# amplitude but wrong shape).
CIRC_SMOOTH_WINDOW = 15   # moving-average steps (~0.75 s at 20 Hz): kills mm tracking jitter, keeps 3 cm loops
CIRC_MIN_STEP = 1e-3      # m, drop sub-mm displacements before measuring heading
CIRC_R_MIN = 0.02         # m, smallest valid scrub-circle radius
CIRC_R_MAX = 0.07         # m, largest valid scrub-circle radius
CIRC_RMS_FRAC = 0.20      # fit RMS residual must be < this fraction of the fitted radius
CIRC_ROUNDNESS_MIN = 0.55  # minor/major harmonic amplitude ratio: 1 = perfect circle, ~0 = line oscillation
# Gate calibration (shared_data_wipe, old-knob labels): genuine steady scrub
# loops fit at rms/r 0.02-0.15 and roundness 0.6-0.95; the approach-phase loop
# and straight/wandering paths land at roundness 0.14-0.5 and/or rms/r > 0.2.


def _fit_drifting_circle(seg):
    """Fit one closed scrub loop as a circle whose center drifts linearly along
    the sweep: regress each coordinate on [1, t, cos th, sin th] with th one
    full turn over the segment (linear least squares — no detrending pass,
    the drift is part of the model). Returns (radius, roundness, rms):
    the 2x2 harmonic amplitude matrix's singular values are the trace
    ellipse's semi-axes, so radius = their geometric mean and roundness =
    minor/major (a circle has equal x/y amplitudes in quadrature; any
    line-ish oscillation collapses one axis)."""
    n = len(seg)
    tt = np.arange(n)
    th = 2 * np.pi * tt / (n - 1)
    X = np.column_stack([np.ones(n), tt / n, np.cos(th), np.sin(th)])
    coef, *_ = np.linalg.lstsq(X, seg, rcond=None)
    resid = seg - X @ coef
    rms = float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1) / 2)))
    sv = np.linalg.svd(coef[2:4].T, compute_uv=False)
    r = float(np.sqrt(max(sv[0] * sv[1], 0.0)))
    return r, float(sv[1] / max(sv[0], 1e-12)), rms


def _circle_fraction(path2d, window=CIRC_SMOOTH_WINDOW, min_step=CIRC_MIN_STEP,
                     r_min=CIRC_R_MIN, r_max=CIRC_R_MAX, rms_frac=CIRC_RMS_FRAC,
                     roundness_min=CIRC_ROUNDNESS_MIN):
    """Fraction of the trajectory spent drawing valid scrub circles (0..1).

    Segments the in-plane eef path into closed loops by CUMULATIVE SIGNED
    TURNING (every +-2pi closes one candidate loop — jitter and zigzags have
    symmetric turning that cancels, so only sustained same-handed rotation
    produces candidates). Each candidate gets a drifting-circle fit
    (_fit_drifting_circle) and counts only if it passes ALL gates: the fit is
    tight (rms < rms_frac * r), the loop is round (roundness >= roundness_min,
    rejecting line-ish oscillation), AND the radius is in [r_min, r_max]
    (too-small scrubbing fails low; straight-ish arcs fit huge radii and fail
    high). Score = timesteps inside passing loops / total timesteps: bounded,
    count-invariant, and one lone circle in a long wipe scores low.

    Replaces the retired _total_turning oracle (turning counts loops but is
    radius-blind — see _total_turning docstring). NOTE: pairs/labels generated
    with the old oracle are not comparable with this one.
    """
    raw = np.asarray(path2d, dtype=float)
    T = len(raw)
    if T < 3:
        return 0.0
    p = raw
    if window > 1 and T > window:
        k = np.ones(window) / window
        p = np.stack([np.convolve(raw[:, i], k, mode='valid') for i in range(2)], 1)
    d = np.diff(p, axis=0)
    keep = np.linalg.norm(d, axis=1) > min_step
    if keep.sum() < 3:
        return 0.0
    kidx = np.nonzero(keep)[0]          # kept displacement k starts at point kidx[k]
    dk = d[keep]
    ang = np.arctan2(dk[:, 1], dk[:, 0])
    turn = (np.diff(ang) + np.pi) % (2 * np.pi) - np.pi
    covered = 0
    cum, start = 0.0, 0                 # start: index into kidx of current candidate
    for j, t in enumerate(turn):
        cum += t
        if abs(cum) < 2 * np.pi:
            continue
        s, e = kidx[start], kidx[j + 1] + 1   # point-index span of the closed loop
        seg = p[s:e + 1]
        if len(seg) >= 8:
            r, roundness, rms = _fit_drifting_circle(seg)
            if r_min <= r <= r_max and roundness >= roundness_min and rms <= rms_frac * r:
                covered += e - s + 1
        cum, start = 0.0, j + 1
    return covered / T


def _total_turning(path2d, window=31, min_step=1e-3):
    """RETIRED oracle (kept for reference / reproducing old labels; nothing
    imports it anymore). Total turning (radians) of a 2D path — high for
    circular scrubbing, low for straight wipes. Retired because turning is
    RADIUS-BLIND (any closed loop is 2pi regardless of size): its knob
    correlation came from the smoothing window erasing small loops, i.e. a
    filtering artifact, not a measurement. Superseded by _circle_fraction.

    The path is smoothed (moving average, ~1.5 s at 20 Hz) and sub-@min_step
    displacements are dropped before measuring turning: raw per-step turning
    is dominated by mm-scale tracking jitter, which buried the scrub-circle
    signal in a large noisy baseline. Calibrated on the 200-episode
    shared_data_wipe set: knob<->score Spearman 0.68 -> 0.91, flipped
    preference labels on knob-separated pairs 16.3% -> 2.7% (vertical wipe:
    0.70 -> 0.93, clean low/high separation). NOTE: pairs.npz generated
    before this change used the noisier raw-turning labels.
    """
    p = np.asarray(path2d, dtype=float)
    if window > 1 and len(p) > window:
        k = np.ones(window) / window
        p = np.stack([np.convolve(p[:, i], k, mode='valid') for i in range(2)], 1)
    d = np.diff(p, axis=0)
    d = d[np.linalg.norm(d, axis=1) > min_step]
    if len(d) < 2:
        return 0.0
    ang = np.arctan2(d[:, 1], d[:, 0])
    return float(np.sum(np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)))


def circularity(obs, actions=None, reward_series=None):
    """Fraction of the eef xy path spent in valid scrub circles — the table
    (horizontal) wiping plane. See _circle_fraction for the gates."""
    if obs is None or len(obs) < 2:
        return 0.0
    return _circle_fraction(np.asarray(obs)[:, 0:2])


def circularity_vertical(obs, actions=None, reward_series=None):
    """Fraction of the eef (y, z) path spent in valid scrub circles —
    VerticalWipe's wall plane (normal is world x). Same state_lowdim layout;
    only the projection plane differs."""
    if obs is None or len(obs) < 2:
        return 0.0
    return _circle_fraction(np.asarray(obs)[:, 1:3])


def wiped_frac(obs, actions=None, reward_series=None):
    """Fraction of the spill wiped clean by the end of the trajectory, read from
    the per-step proportion_wiped sensor stored as the last state dim. Monotonic
    non-decreasing, so the final step is the max."""
    if obs is None or len(obs) == 0:
        return 0.0
    return float(np.asarray(obs)[-1, -1])


# ---- TwoArmLift pot-carry axes (state_lowdim = [eef0 pose7, eef1 pose7,
# pot pose7]; pot xy = cols 14:16, pot z = col 16). Carry window = frames where
# the pot is lifted >2cm above its resting z AND moving laterally — mirrors the
# collector's scripted carry phase (pair-label parity vs stored per-episode
# attrs: 96-99.7%; build_pairs prefers the stored attrs when present, these
# fns cover attr-less data such as policy rollouts). CTRL_DT = 1/20 (20 Hz).
_POT_CTRL_DT = 1.0 / 20.0

def _pot_carry_window(obs, lift_thresh=0.02, lat_eps=5e-4):
    st = np.asarray(obs)
    if st is None or len(st) < 2 or st.shape[1] < 17:
        return None
    lift = st[:, 16] - st[0, 16]
    step = np.zeros(len(st))
    step[1:] = np.linalg.norm(np.diff(st[:, 14:16], axis=0), axis=1)
    idx = np.nonzero((lift > lift_thresh) & (step > lat_eps))[0]
    return (int(idx[0]), int(idx[-1]) + 1) if len(idx) else None


def _pot_speed_ms(obs):
    """Mean lateral pot speed (m/s) over the carry window (physical value)."""
    w = _pot_carry_window(obs)
    if w is None:
        return 0.0
    t0, t1 = w
    seg = np.asarray(obs)[t0:t1, 14:16]
    if len(seg) < 2:
        return 0.0
    dist = float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))
    return dist / (max(len(seg) - 1, 1) * _POT_CTRL_DT)


def _pot_lift_m(obs):
    """Mean pot lift above its resting z over the carry window (physical value;
    == pot-bottom height above the table up to a per-episode-constant offset)."""
    w = _pot_carry_window(obs)
    if w is None:
        return 0.0
    t0, t1 = w
    st = np.asarray(obs)
    return float((st[t0:t1, 16] - st[0, 16]).mean())


def slow_carry(obs, actions=None, reward_series=None):
    """Rewarded direction: SLOW carrying wins -> negated lateral speed."""
    return -_pot_speed_ms(obs)


def low_carry(obs, actions=None, reward_series=None):
    """Rewarded direction: LOW carrying wins -> negated lift height."""
    return -_pot_lift_m(obs)


# Axes whose value genuinely depends on `actions`. Callers that cannot supply
# actions must refuse to compute these (raise) rather than let the function
# fall through to its actions-None default — that default is a silently wrong
# ground-truth value, which is worse than a crash.
ACTION_DEPENDENT_AXES = {'smoothness'}

# ----------------------------------------------------------------------------
# Registry: axis name -> function. Add new axes by adding a function and an
# entry here.
# ----------------------------------------------------------------------------
AXIS_FUNCTIONS = {
    'success':        success,
    'speed_reward':   speed_reward,
    'smoothness':     smoothness,
    'circularity':    circularity,
    'circularity_vertical': circularity_vertical,
    'wiped_frac':     wiped_frac,
    'slow_carry':     slow_carry,
    'low_carry':      low_carry,
    'order_reward':   order_reward,
    'order_reward_raw': order_reward_raw,
    'milk_placed':    milk_placed,
    'bread_placed':   bread_placed,
    'cereal_placed':  cereal_placed,
    'can_placed':     can_placed,
    'milk_drop':      milk_drop,
    'bread_drop':     bread_drop,
    'cereal_drop':    cereal_drop,
    'can_drop':       can_drop,
    'drop_reward':    drop_reward,
    'peg_reward':     peg_reward,
    'peg_reward_raw': peg_reward_raw,
    # Continuous (distance-based) versions of placed/drop. Use these as
    # conditioning axes when you want a denser reward signal than the
    # discrete {-1, 0, +1} versions; the discrete ones remain available for
    # logging/comparison.
    'milk_placed_raw':    milk_placed_raw,
    'bread_placed_raw':   bread_placed_raw,
    'cereal_placed_raw':  cereal_placed_raw,
    'can_placed_raw':     can_placed_raw,
    'milk_drop_raw':      milk_drop_raw,
    'bread_drop_raw':     bread_drop_raw,
    'cereal_drop_raw':    cereal_drop_raw,
    'can_drop_raw':       can_drop_raw,
}


def compute_axes(axis_names, obs, actions=None, reward_series=None):
    """Compute a dict {axis_name: value} for one trajectory."""
    out = {}
    for name in axis_names:
        fn = AXIS_FUNCTIONS.get(name)
        if fn is None:
            raise KeyError(f"Unknown reward axis '{name}'. Known: {list(AXIS_FUNCTIONS.keys())}")
        out[name] = float(fn(obs, actions=actions, reward_series=reward_series))
    return out


# ----------------------------------------------------------------------------
# PickPlace per-axis eval-logging helpers
# ----------------------------------------------------------------------------
# Object id → name (matches the slot order in the 56-dim 'object' obs block).
PICKPLACE_OBJECT_NAMES = {0: 'milk', 1: 'bread', 2: 'cereal', 3: 'can'}


def get_pickplace_eval_axes(n_active_objects):
    """Canonical (discrete) per-axis eval set for a PickPlace variant:
    order_reward + per-active-object placed + per-active-object drop. Active
    subset is the right-first canonical prefix (Bread, Can, Milk, Cereal).

    Used for mean_score (sum) and mean_strict_success — both rely on the
    discrete {-1, 0, +1} semantics. For the larger set that also includes
    the continuous raw axes (for logging), use `get_pickplace_logging_axes`.
    """
    n = max(1, min(int(n_active_objects), 4))
    active_ids = PICKPLACE_CANONICAL_ORDER[:n]
    axes = ['order_reward']
    axes += [f'{PICKPLACE_OBJECT_NAMES[i]}_placed' for i in active_ids]
    axes += [f'{PICKPLACE_OBJECT_NAMES[i]}_drop' for i in active_ids]
    return axes


def get_slow_fast_logging_axes():
    """Canonical per-axis eval-logging set for the slow_fast (square_twopeg)
    benchmark. Includes both the discrete and continuous versions of the
    peg-position signal so eval shows both at once.
      - success    (0 / 1)
      - speed_reward (continuous in [0, 1])
      - smoothness   (continuous in [0, 1])
      - peg_reward     (discrete -1 / 0 / +1)
      - peg_reward_raw (continuous in [0, 1], distance to right peg)
    """
    return [
        'success',
        'speed_reward',
        'smoothness',
        'peg_reward',
        'peg_reward_raw',
    ]


def get_pickplace_logging_axes(n_active_objects):
    """Full set of per-axis values to log to wandb during eval: the discrete
    axes from `get_pickplace_eval_axes` plus their continuous `_raw`
    counterparts. The raw axes have a fundamentally different scale (xy
    distance / release-height in meters, all ≤ 0), so they're kept out of
    the discrete-only mean_score / strict_success calculations.
    """
    n = max(1, min(int(n_active_objects), 4))
    active_ids = PICKPLACE_CANONICAL_ORDER[:n]
    axes = ['order_reward', 'order_reward_raw']
    for i in active_ids:
        name = PICKPLACE_OBJECT_NAMES[i]
        axes.append(f'{name}_placed')
        axes.append(f'{name}_placed_raw')
    for i in active_ids:
        name = PICKPLACE_OBJECT_NAMES[i]
        axes.append(f'{name}_drop')
        axes.append(f'{name}_drop_raw')
    return axes


def compute_pickplace_eval_log(obs_seqs, action_seqs, prefixes,
                               n_active_objects, axis_names=None):
    """Aggregate per-axis reward values + strict-success across rollouts.

    Args:
      obs_seqs:    list of (T_i, D) np.ndarray per rollout
      action_seqs: list of (T_i, A) np.ndarray per rollout
      prefixes:    list of str per rollout (e.g. 'test/', 'train/')
      n_active_objects: used for strict_success bookkeeping
      axis_names: list of base axes to log. If None, derived from
                  get_pickplace_logging_axes(n_active_objects), which
                  includes both discrete and continuous (_raw) variants.

    Returns dict {prefix + key: mean_value} with one entry per (prefix, axis)
    plus '{prefix}mean_strict_success' (positive order AND every DISCRETE
    per-object *_drop axis > 0). Raw axes are logged but excluded from
    strict_success since they're always ≤ 0.
    """
    import collections
    if axis_names is None:
        axis_names = get_pickplace_logging_axes(n_active_objects)

    axis_accum = {ax: collections.defaultdict(list) for ax in axis_names}
    # Strict success ignores raw axes — they're always ≤ 0 and would prevent
    # the criterion from ever firing.
    strict_drop_axes = [ax for ax in axis_names
                        if ax.endswith('_drop') and ax != 'drop_reward']
    strict_axes_available = ('order_reward' in axis_names
                             and len(strict_drop_axes) > 0)
    prefix_strict = collections.defaultdict(list)

    for obs_seq, act_seq, prefix in zip(obs_seqs, action_seqs, prefixes):
        if obs_seq is None or len(obs_seq) == 0:
            continue
        rollout_vals = {}
        for ax in axis_names:
            fn = AXIS_FUNCTIONS.get(ax)
            if fn is None:
                continue
            try:
                v = float(fn(obs_seq, actions=act_seq))
            except Exception:
                v = 0.0
            axis_accum[ax][prefix].append(v)
            rollout_vals[ax] = v
        if strict_axes_available:
            order_ok = rollout_vals.get('order_reward', 0.0) >= 1.0 - 1e-6
            drops_ok = all(rollout_vals.get(ax, 0.0) > 0
                           for ax in strict_drop_axes)
            prefix_strict[prefix].append(float(order_ok and drops_ok))

    log_data = {}
    for ax, by_prefix in axis_accum.items():
        for prefix, vals in by_prefix.items():
            if vals:
                log_data[prefix + ax] = float(np.mean(vals))
    for prefix, vals in prefix_strict.items():
        if vals:
            log_data[prefix + 'mean_strict_success'] = float(np.mean(vals))
    return log_data
