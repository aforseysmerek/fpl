"""
Wipe task collector — scripted table-wiping rollouts. Two knobs vary behavior:
  - circ_amount  (straight trace <-> circular scrub)  -> drives the circularity axis
  - press_amount (light contact  <-> hard press)      -> generative only, not scored

Mirrors collect_initial_scripted_rollouts_images.py (the square-with-images
collector): reuses the validated scripted policy (WipeTracePolicy from
wipe_trace.py — peg-style absolute OSC_POSE waypoints + interpolation +
euler->axis_angle) and captures both cameras + low-dim state (eef pose + joints
+ per-step proportion_wiped). Axis scores (circularity, wiped_frac) are NOT
computed here — reward_functions.py is the single oracle, applied downstream by
generate_preferences.py, which reuses this episodes.hdf5 layout.

Run in robodiff:
    MUJOCO_GPU=<free> python scripts/collect_initial_scripted_rollouts_wipe_images.py \
        -o shared_data_wipe -n 200

--vertical collects the same demos on the VerticalWipe wall env (whiteboard-
style vertical plane): same knobs/sampling/HDF5 layout, policy swapped to
VerticalWipeTracePolicy and the circularity axis to circularity_vertical
(the wall's (y, z) plane instead of the table's (x, y)):
    MUJOCO_GPU=<free> python scripts/collect_initial_scripted_rollouts_wipe_images.py \
        --vertical -o shared_data_vertical_wipe -n 200
"""
import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, str(pathlib.Path(ROOT_DIR) / "reward_model"))
os.chdir(ROOT_DIR)

import json
import click
import numpy as np
import h5py
import robosuite

# Reuse the validated wipe policy + controller config (mirrors how the square
# collector imports SquareSideScriptedPolicy).
from scripts.wipe_trace import WipeTracePolicy, VerticalWipeTracePolicy, OSC_POSE_ABS
# Axis oracle — the SAME functions generate_preferences/train_reward_model use,
# so the collection-time sanity print never diverges from the scored labels.
from reward_functions import circularity, circularity_vertical, wiped_frac

CAMERAS = ["agentview", "robot0_eye_in_hand"]   # third-person + wrist (Qwen needs both)
IMG_HW = 128


def create_wipe_env(vertical=False):
    """Direct robosuite Wipe env with the SAME controller as the peg collection
    (OSC_POSE, control_delta=False) + offscreen two-camera rendering."""
    if vertical:
        import envs.vertical_wipe  # noqa: F401  registers VerticalWipe with robosuite
    return robosuite.make(
        "VerticalWipe" if vertical else "Wipe", robots="Panda", controller_configs=OSC_POSE_ABS,
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=CAMERAS, camera_heights=IMG_HW, camera_widths=IMG_HW,
        control_freq=20, horizon=4000, ignore_done=True, hard_reset=False,
        render_gpu_device_id=int(os.environ.get("MUJOCO_GPU", "-1")),
    )


def to_uint8_hwc(img):
    """robosuite camera obs (H,W,3) uint8, vertically flipped -> upright uint8."""
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] == 3:
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (a * 255.0).clip(0, 255).astype(np.uint8)
    return np.flipud(a).copy()


@click.command()
@click.option('-o', '--output_dir', default='shared_data_wipe')
@click.option('-n', '--num_episodes', type=int, default=200)
@click.option('--seed', type=int, default=0)
@click.option('--n_waypoints', type=int, default=30)
@click.option('--seg_steps', type=int, default=12)
@click.option('--bimodal', is_flag=True,
              help='Sample each knob from only the LOW or HIGH extreme band (no middle values), '
                   'balanced across the 4 (circ x press) quadrants. Still continuous within a band '
                   'so trajectories differ. Appends "_bimodal" to the output dir name.')
@click.option('--bimodal_band', type=float, default=0.2,
              help='Band width: LOW=[0, band], HIGH=[1-band, 1] (bimodal); '
                   'the complement [band, 1-band] is the MIDDLE region (--middle).')
@click.option('--middle', is_flag=True,
              help='EVAL set for a bimodal-trained model: sample both knobs from the MIDDLE region '
                   '[band, 1-band] — the gap the bimodal set never covers ("unseen" range). '
                   'Appends "_middle" to the output dir name.')
@click.option('--vertical', is_flag=True,
              help='Collect on the VerticalWipe wall env instead of the table: same knobs/sampling/'
                   'layout, policy and circularity axis swapped to the wall (y, z) plane.')
@click.option('--circular/--straight', 'circular', default=None,
              help='CURRENT circularity interface (binary): --circular scrubs in circles with per-loop '
                   'radius sampled 3-5 cm, --straight traces the spill dead-straight. Applies to ALL '
                   'episodes; they still differ via random marker placement and per-loop radii. '
                   'Leaving it unset falls back to the retired continuous-knob sampling below.')
@click.option('--circ_knob', type=float, default=None,
              help='RETIRED: fix the old continuous circ_amount [0,1] (radius = knob * 0.05 m) for all '
                   'episodes. Only for reproducing old datasets — use --circular/--straight instead.')
@click.option('--press_knob', type=float, default=None,
              help='Fix press_amount to this value [0,1] for ALL episodes instead of sampling it.')
def main(output_dir, num_episodes, seed, n_waypoints, seg_steps, bimodal, bimodal_band, middle,
         vertical, circular, circ_knob, press_knob):
    if bimodal and middle:
        raise click.UsageError("--bimodal and --middle are mutually exclusive (extremes vs. the gap).")
    if (circ_knob is not None or press_knob is not None) and (bimodal or middle):
        raise click.UsageError("--circ_knob/--press_knob fix the knobs and cannot combine with "
                               "--bimodal/--middle sampling.")
    if circular is not None and (bimodal or middle or circ_knob is not None):
        raise click.UsageError("--circular/--straight replaces the retired circ_amount knob and cannot "
                               "combine with --bimodal/--middle/--circ_knob.")
    if vertical and output_dir == 'shared_data_wipe':
        output_dir = 'shared_data_vertical_wipe'   # keep table/wall datasets separate by default
    policy_cls = VerticalWipeTracePolicy if vertical else WipeTracePolicy
    circ_fn, circ_axis = (circularity_vertical, "circularity_vertical") if vertical \
        else (circularity, "circularity")
    rng = np.random.default_rng(seed)
    # Tag the dir so different sampling regimes are never confused.
    tag = "bimodal" if bimodal else ("middle" if middle else "")
    if tag and tag not in os.path.basename(output_dir):
        output_dir = f"{output_dir}_{tag}"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    fixed = circ_knob is not None or press_knob is not None or circular is not None
    mode = "BIMODAL" if bimodal else ("MIDDLE" if middle else (
        f"FIXED circ={('circular' if circular else 'straight') if circular is not None else circ_knob} "
        f"press={press_knob}" if fixed else "uniform"))
    print(f"Sampling: {mode}{(' band=%.2f' % bimodal_band) if (bimodal or middle) else ''}  ->  {output_dir}",
          flush=True)

    env = create_wipe_env(vertical=vertical)
    adim = env.action_dim

    out = h5py.File(pathlib.Path(output_dir) / "episodes.hdf5", "w")
    data_grp = out.create_group("data")

    # Bimodal: each knob drawn from LOW=[0, band] or HIGH=[1-band, 1], with the 4
    # (circ, press) quadrants cycled round-robin so all are balanced (50 each at
    # n=200). Continuous within a band -> distinct trajectories, but no middle.
    def _band(is_high):
        return float(rng.uniform(1.0 - bimodal_band, 1.0) if is_high else rng.uniform(0.0, bimodal_band))
    quadrants = [(False, False), (False, True), (True, False), (True, True)]  # (circ_high, press_high)

    for ep in range(num_episodes):
        if bimodal:
            circ_high, press_high = quadrants[ep % 4]
            circ_amount = _band(circ_high)
            press_amount = _band(press_high)
        elif middle:
            circ_amount = float(rng.uniform(bimodal_band, 1.0 - bimodal_band))
            press_amount = float(rng.uniform(bimodal_band, 1.0 - bimodal_band))
        else:
            circ_amount = float(rng.uniform(0.0, 1.0))
            press_amount = float(rng.uniform(0.0, 1.0))
        # Fixed overrides: --circular/--straight (current binary interface) or
        # the retired --circ_knob / --press_knob; every episode gets the same
        # setting, only the marker layout (and per-loop radii) vary.
        if circular is not None:
            circ_amount = None   # binary mode: the retired knob stays out of the policy
        elif circ_knob is not None:
            circ_amount = float(circ_knob)
        if press_knob is not None:
            press_amount = float(press_knob)
        obs = env.reset()
        policy = policy_cls(env, obs["robot0_eef_pos"], circ_amount=circ_amount,
                            press_amount=press_amount, n_waypoints=n_waypoints, seg_steps=seg_steps,
                            circular=bool(circular), rng=rng)

        tp_l, wr_l, jp_l, low_l, act_l = [], [], [], [], []
        for _ in range(policy.max_t):
            tp_l.append(to_uint8_hwc(obs["agentview_image"]))          # pre-action state
            wr_l.append(to_uint8_hwc(obs["robot0_eye_in_hand_image"]))
            ep_pos = np.asarray(obs["robot0_eef_pos"], np.float32)
            eq = np.asarray(obs["robot0_eef_quat"], np.float32)
            jp = np.asarray(env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes], np.float32)  # obs has no robot0_joint_pos in direct robosuite
            jp_l.append(jp)
            wiped_t = len(env.wiped_markers) / max(env.num_markers, 1)  # cumulative % wiped at this (pre-action) state
            low_l.append(np.concatenate([ep_pos, eq, jp, [wiped_t]]))  # state_lowdim: eef pose + joints + proportion_wiped (NO force)

            action = policy.predict_action()
            if adim > 6:
                action = np.concatenate([action, np.zeros(adim - 6)])
            act_l.append(action[:6].copy())
            obs, *_ = env.step(action)

        state = np.stack(low_l, 0)
        circ = circ_fn(state)   # oracle fns — no separate collection-time metric
        wiped = wiped_frac(state)

        g = data_grp.create_group(f"demo_{ep}")
        og = g.create_group("obs")
        og.create_dataset("agent_view", data=np.stack(tp_l, 0))        # (T,128,128,3) uint8
        og.create_dataset("wrist", data=np.stack(wr_l, 0))             # (T,128,128,3) uint8
        og.create_dataset("JOINT_POS", data=np.stack(jp_l, 0))         # (T,7) float32
        og.create_dataset("state_lowdim", data=state)                  # (T,15) float32 (eef pose + joints + proportion_wiped)
        g.create_dataset("actions", data=np.stack(act_l, 0))
        # Only generative knobs + length are stored; axis SCORES are computed
        # downstream by reward_functions (single source of truth), not here.
        ep_attrs = dict(press_amount=press_amount, n_steps=len(act_l))
        if circular is not None:
            ep_attrs["circular"] = bool(circular)
        else:
            ep_attrs["circ_amount"] = circ_amount
        g.attrs.update(ep_attrs)
        circ_lbl = ("circular" if circular else "straight") if circular is not None else f"{circ_amount:.2f}"
        print(f"ep {ep:3d}: circ={circ_lbl} press_knob={press_amount:.2f}  ->  "
              f"circularity={circ:5.2f} wiped={wiped:.2f}", flush=True)

    data_grp.attrs["cameras"] = json.dumps(CAMERAS)
    data_grp.attrs["axes"] = json.dumps([circ_axis, "wiped_frac"])
    data_grp.attrs["sampling"] = "bimodal" if bimodal else ("middle" if middle else ("fixed" if fixed else "uniform"))
    if fixed:
        data_grp.attrs["fixed_knobs"] = json.dumps(dict(circular=circular, circ_knob=circ_knob,
                                                        press_knob=press_knob))
    if bimodal or middle:
        data_grp.attrs["bimodal_band"] = float(bimodal_band)
    out.close()
    print(f"\nWrote {num_episodes} episodes to {output_dir}/episodes.hdf5")


if __name__ == '__main__':
    main()
