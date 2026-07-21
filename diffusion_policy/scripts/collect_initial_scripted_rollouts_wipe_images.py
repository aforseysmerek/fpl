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
from scripts.wipe_trace import WipeTracePolicy, OSC_POSE_ABS
# Axis oracle — the SAME functions generate_preferences/train_reward_model use,
# so the collection-time sanity print never diverges from the scored labels.
from reward_functions import circularity, wiped_frac

CAMERAS = ["agentview", "robot0_eye_in_hand"]   # third-person + wrist (Qwen needs both)
IMG_HW = 128


def create_wipe_env():
    """Direct robosuite Wipe env with the SAME controller as the peg collection
    (OSC_POSE, control_delta=False) + offscreen two-camera rendering."""
    return robosuite.make(
        "Wipe", robots="Panda", controller_configs=OSC_POSE_ABS,
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
def main(output_dir, num_episodes, seed, n_waypoints, seg_steps, bimodal, bimodal_band, middle):
    if bimodal and middle:
        raise click.UsageError("--bimodal and --middle are mutually exclusive (extremes vs. the gap).")
    rng = np.random.default_rng(seed)
    # Tag the dir so different sampling regimes are never confused.
    tag = "bimodal" if bimodal else ("middle" if middle else "")
    if tag and tag not in os.path.basename(output_dir):
        output_dir = f"{output_dir}_{tag}"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    mode = "BIMODAL" if bimodal else ("MIDDLE" if middle else "uniform")
    print(f"Sampling: {mode}{(' band=%.2f' % bimodal_band) if (bimodal or middle) else ''}  ->  {output_dir}",
          flush=True)

    env = create_wipe_env()
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
        obs = env.reset()
        policy = WipeTracePolicy(env, obs["robot0_eef_pos"], circ_amount=circ_amount,
                                 press_amount=press_amount, n_waypoints=n_waypoints, seg_steps=seg_steps)

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
        circ = circularity(state)   # oracle fns — no separate collection-time metric
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
        g.attrs.update(dict(circ_amount=circ_amount, press_amount=press_amount, n_steps=len(act_l)))
        print(f"ep {ep:3d}: circ_knob={circ_amount:.2f} press_knob={press_amount:.2f}  ->  "
              f"circularity={circ:6.2f} wiped={wiped:.2f}", flush=True)

    data_grp.attrs["cameras"] = json.dumps(CAMERAS)
    data_grp.attrs["axes"] = json.dumps(["circularity", "wiped_frac"])
    data_grp.attrs["sampling"] = "bimodal" if bimodal else ("middle" if middle else "uniform")
    if bimodal or middle:
        data_grp.attrs["bimodal_band"] = float(bimodal_band)
    out.close()
    print(f"\nWrote {num_episodes} episodes to {output_dir}/episodes.hdf5")


if __name__ == '__main__':
    main()
