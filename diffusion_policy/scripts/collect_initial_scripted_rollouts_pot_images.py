"""
TwoArmLift (bimanual pot-carry) collector — scripted rollouts spanning two axes:
  - "speed":  speed_amount knob (slow carry <-> fast carry)
  - "height": height_amount knob (low skim <-> high lift of the pot)

Mirrors collect_initial_scripted_rollouts_wipe_images.py: reuses the validated
scripted policy (PotTracePolicy from pot_trace.py — peg-style absolute OSC_POSE
waypoints + interpolation + euler->axis_angle, for two arms), captures agentview
+ wrist cameras + low-dim state, scores each episode on the REALIZED trajectory
(carry speed + pot height above the table), and writes episodes.hdf5 in the same
layout so generate_preferences.py reuses it.

Run in robodiff:
    MUJOCO_GPU=<free> python scripts/collect_initial_scripted_rollouts_pot_images.py \
        -o shared_data_pot -n 200
"""
import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import json
import click
import numpy as np
import h5py
import robosuite

# Reuse the validated pot policy + oracle metrics + controller config
# (mirrors how the wipe collector imports WipeTracePolicy).
from scripts.pot_trace import (
    PotTracePolicy, speed_metric, height_metric, pot_bottom_above_table, OSC_POSE_ABS, GOAL_XY,
)

CAMERAS = ["agentview", "robot0_eye_in_hand"]   # third-person + wrist (Qwen needs both)
IMG_HW = 128


def create_pot_env():
    """Direct robosuite TwoArmLift env with the SAME controller as pot_trace
    (OSC_POSE, control_delta=False) + offscreen two-camera rendering."""
    return robosuite.make(
        "TwoArmLift", robots=["Panda", "Panda"], env_configuration="single-arm-opposed",
        controller_configs=OSC_POSE_ABS,
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


def low_dim_state(env, obs):
    """[eef0 pose(7), eef1 pose(7), pot pose(7)] = 21-dim (NO joints; those go in JOINT_POS)."""
    return np.concatenate([
        np.asarray(obs["robot0_eef_pos"], np.float32), np.asarray(obs["robot0_eef_quat"], np.float32),
        np.asarray(obs["robot1_eef_pos"], np.float32), np.asarray(obs["robot1_eef_quat"], np.float32),
        np.asarray(obs["pot_pos"], np.float32),        np.asarray(obs["pot_quat"], np.float32),
    ]).astype(np.float32)


@click.command()
@click.option('-o', '--output_dir', default='shared_data_pot')
@click.option('-n', '--num_episodes', type=int, default=200)
@click.option('--seed', type=int, default=0)
@click.option('--grasp_offset', type=float, default=0.0, help='grip_site height above the bar (~fingertip gap; use pot_probe value)')
@click.option('--goal_x', type=float, default=GOAL_XY[0], help='world x the pot is carried to')
@click.option('--goal_y', type=float, default=GOAL_XY[1], help='world y the pot is carried to')
def main(output_dir, num_episodes, seed, grasp_offset, goal_x, goal_y):
    rng = np.random.default_rng(seed)
    goal_xy = (goal_x, goal_y)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    env = create_pot_env()
    adim = env.action_dim
    j0 = env.robots[0]._ref_joint_pos_indexes
    j1 = env.robots[1]._ref_joint_pos_indexes

    out = h5py.File(pathlib.Path(output_dir) / "episodes.hdf5", "w")
    data_grp = out.create_group("data")

    # Grasp reliability is pose-dependent (random pot spawn yaw vs two opposed
    # arms), so — like the square collector — we KEEP only successful carries:
    # attempt episodes until we have num_episodes that lifted the pot, capped.
    kept = 0
    attempt = 0
    max_attempts = num_episodes * 4
    while kept < num_episodes and attempt < max_attempts:
        attempt += 1
        speed_amount = float(rng.uniform(0.0, 1.0))
        height_amount = float(rng.uniform(0.0, 1.0))
        obs = env.reset()
        policy = PotTracePolicy(env, obs["robot0_eef_pos"], obs["robot1_eef_pos"],
                                obs["robot0_eef_quat"], obs["robot1_eef_quat"],
                                speed_amount=speed_amount, height_amount=height_amount,
                                grasp_offset=grasp_offset, goal_xy=goal_xy)

        tp_l, wr_l, jp_l, low_l, pot_xy, heights, act_l = [], [], [], [], [], [], []
        for _ in range(policy.max_t):
            tp_l.append(to_uint8_hwc(obs["agentview_image"]))          # pre-action state
            wr_l.append(to_uint8_hwc(obs["robot0_eye_in_hand_image"]))
            jp = np.concatenate([env.sim.data.qpos[j0], env.sim.data.qpos[j1]]).astype(np.float32)
            jp_l.append(jp)                                            # (14,) both arms
            low_l.append(low_dim_state(env, obs))                      # (21,)
            pot_xy.append(np.asarray(obs["pot_pos"][:2]).copy())

            action = policy.predict_action()
            if adim != len(action):
                action = np.resize(action, adim)
            act_l.append(action.copy())
            obs, *_ = env.step(action)
            heights.append(pot_bottom_above_table(env))

        spd = speed_metric(pot_xy, policy.carry_t0, policy.carry_t1)
        hgt = height_metric(heights, policy.carry_t0, policy.carry_t1)
        goal_dist = float(np.linalg.norm(pot_xy[-1] - policy.goal_xy))
        success = bool(env._check_success())
        if not success:
            print(f"  attempt {attempt:3d}: NOT lifted (speed_knob={speed_amount:.2f} "
                  f"height_knob={height_amount:.2f}) — skipped", flush=True)
            continue

        g = data_grp.create_group(f"demo_{kept}")
        og = g.create_group("obs")
        og.create_dataset("agent_view", data=np.stack(tp_l, 0))        # (T,128,128,3) uint8
        og.create_dataset("wrist", data=np.stack(wr_l, 0))             # (T,128,128,3) uint8
        og.create_dataset("JOINT_POS", data=np.stack(jp_l, 0))         # (T,14) float32 (both arms)
        og.create_dataset("state_lowdim", data=np.stack(low_l, 0))     # (T,21) float32
        g.create_dataset("actions", data=np.stack(act_l, 0))           # (T,14) float32
        g.attrs.update(dict(carry_speed=float(spd), carry_height=float(hgt), goal_dist=goal_dist,
                            success=success, speed_amount=speed_amount, height_amount=height_amount,
                            goal_x=float(goal_x), goal_y=float(goal_y), n_steps=len(act_l)))
        print(f"kept {kept:3d} (attempt {attempt:3d}): speed_knob={speed_amount:.2f} height_knob={height_amount:.2f}  ->  "
              f"carry_speed={spd:5.3f} m/s carry_height={hgt:5.3f} m goal_dist={goal_dist:5.3f} m", flush=True)
        kept += 1

    data_grp.attrs["n_demos"] = kept
    data_grp.attrs["n_attempts"] = attempt
    data_grp.attrs["cameras"] = json.dumps(CAMERAS)
    data_grp.attrs["axes"] = json.dumps(["speed", "carry_height"])
    out.close()
    print(f"\nWrote {kept} successful demos ({attempt} attempts, "
          f"{kept/max(attempt,1)*100:.0f}% success) to {output_dir}/episodes.hdf5")


if __name__ == '__main__':
    main()
