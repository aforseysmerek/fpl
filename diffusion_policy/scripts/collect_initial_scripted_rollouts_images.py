"""
2a: collect scripted square rollouts WITH two-camera images, low-dim state, and
oracle per-axis scores — in one pass. The output feeds BOTH reward models:
  - Qwen (qwen_open): obs/agent_view, obs/wrist  (images)
  - state model:      obs/state_lowdim           (the 23-dim low-dim vector)
and 2b (make_preferences.py) turns it into preference pairs.

Reuses the scripted policy + wrappers from collect_initial_scripted_rollouts.py;
the only change is enabling image obs on the underlying robomimic env.

Run in the `robodiff` env:
    cd ~/Desktop/fpl/diffusion_policy
    python scripts/collect_with_images.py -o shared_data_square_images -n 8      # small validation
    python scripts/collect_with_images.py -o shared_data_square_images -n 120    # real run
"""
import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import json
import random
import click
import numpy as np
import h5py

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper

# Reuse the scripted policy, env wrapper, obs keys, and oracle metrics.
from scripts.collect_initial_scripted_rollouts import (
    MultimodalSquareLowdimWrapper,
    SquareSideScriptedPolicy,
    OBS_KEYS,
    compute_smoothness,
    compute_speed_reward,
)

DATASET = "data/robomimic/datasets/square/mh/low_dim.hdf5"
CAMERAS = ["agentview", "robot0_eye_in_hand"]   # third-person + wrist
IMG_HW = 128


def create_image_env():
    """Same env chain as the original collector, but with offscreen rendering +
    image obs turned on so we can capture the two cameras. Returns the wrapped
    env (for the policy) and a handle to the raw robomimic env (for images)."""
    env_meta = FileUtils.get_env_metadata_from_dataset(DATASET)
    env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
    env_meta["env_kwargs"]["camera_names"] = CAMERAS
    env_meta["env_kwargs"]["camera_heights"] = IMG_HW
    env_meta["env_kwargs"]["camera_widths"] = IMG_HW

    ObsUtils.initialize_obs_modality_mapping_from_dict({
        "low_dim": OBS_KEYS + ["robot0_joint_pos"],
        "rgb": [f"{c}_image" for c in CAMERAS],
    })
    robomimic_env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=True,
    )

    env = MultiStepWrapper(
        VideoRecordingWrapper(
            MultimodalSquareLowdimWrapper(
                RobomimicLowdimWrapper(env=robomimic_env, obs_keys=OBS_KEYS, init_state=None),
            ),
            video_recoder=VideoRecorder.create_h264(
                fps=10, codec='h264', input_pix_fmt='rgb24', crf=22,
                thread_type='FRAME', thread_count=1),
            file_path=None,
            steps_per_render=2,
        ),
        n_obs_steps=1, n_action_steps=1, max_episode_steps=600,
    )
    return env, robomimic_env


def to_uint8_hwc(img):
    """robomimic camera obs (3,H,W) float32 [0,1] -> (H,W,3) uint8."""
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] == 3:          # (C,H,W) -> (H,W,C)
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (a * 255.0).clip(0, 255).astype(np.uint8)
    return a


@click.command()
@click.option('-o', '--output_dir', default='shared_data_square_images')
@click.option('-n', '--num_episodes', type=int, default=8)
@click.option('--seed', type=int, default=0)
@click.option('--noise_min', type=float, default=0.0)
@click.option('--noise_max', type=float, default=0.12)
@click.option('--speed_range_left', type=(float, float), default=(1.0, 4.0),
              help='Left-peg speed_factor range (fast).')
@click.option('--speed_range_right', type=(float, float), default=(1.0, 2.0),
              help='Right-peg speed_factor range (slow).')
def main(output_dir, num_episodes, seed, noise_min, noise_max, speed_range_left, speed_range_right):
    random.seed(seed)
    np.random.seed(seed)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    env, robomimic_env = create_image_env()

    out = h5py.File(pathlib.Path(output_dir) / "episodes.hdf5", "w")
    data_grp = out.create_group("data")

    noise_levels = np.linspace(noise_min, noise_max, num_episodes)
    np.random.shuffle(noise_levels)

    def grab_images():
        full = robomimic_env.get_observation()
        jp = full.get("robot0_joint_pos")
        jp = np.asarray(jp, dtype=np.float32) if jp is not None else np.zeros(7, dtype=np.float32)
        return (to_uint8_hwc(full["agentview_image"]),
                to_uint8_hwc(full["robot0_eye_in_hand_image"]), jp)

    kept = 0
    for ep in range(num_episodes):
        noise = float(noise_levels[ep])
        policy = SquareSideScriptedPolicy(env, target_peg='random', noise_level=noise, speed_factor=1.0)
        policy.speed_factor = float(np.random.uniform(
            *(speed_range_left if policy.target_peg == 'left' else speed_range_right)))

        env.env.video_recoder.stop()
        env.env.file_path = None
        env.env.env.env.init_state = None
        env.seed(np.random.randint(0, 10_000_000))
        obs = env.reset()
        policy.replan()

        tp_list, wr_list, jp_list, low_list, act_list = [], [], [], [], []
        success = False
        for step in range(600):
            tp, wr, jp = grab_images()        # current (pre-action) state
            tp_list.append(tp); wr_list.append(wr); jp_list.append(jp)
            low_list.append(obs[-1].copy())

            action = policy.predict_action(obs)
            act_list.append(action[0].copy())
            obs, reward, done, info = env.step(action)
            if reward >= 1.0:
                success = True
                break

        n_steps = len(act_list)
        actions = np.stack(act_list, axis=0)
        smooth = compute_smoothness(actions) if success else 0.0
        speed = compute_speed_reward(n_steps)
        peg = -1.0 if policy.target_peg == 'left' else 1.0

        g = data_grp.create_group(f"demo_{kept}")
        og = g.create_group("obs")
        og.create_dataset("agent_view", data=np.stack(tp_list, 0))     # (T,128,128,3) uint8
        og.create_dataset("wrist", data=np.stack(wr_list, 0))          # (T,128,128,3) uint8
        og.create_dataset("JOINT_POS", data=np.stack(jp_list, 0))      # (T,7) float32
        og.create_dataset("state_lowdim", data=np.stack(low_list, 0))  # (T,23) float32
        g.create_dataset("actions", data=actions)
        g.attrs.update(dict(
            target_peg=policy.target_peg, success=bool(success), n_steps=int(n_steps),
            speed_reward=float(speed), smoothness=float(smooth), peg_reward=float(peg),
            noise=noise, speed_factor=float(policy.speed_factor)))
        kept += 1
        print(f"ep {ep}: peg={policy.target_peg} success={success} steps={n_steps} "
              f"speed={speed:.3f} smooth={smooth:.3f} peg={peg:+.0f}", flush=True)

    data_grp.attrs["cameras"] = json.dumps(CAMERAS)
    out.close()
    print(f"\nWrote {kept} episodes to {output_dir}/episodes.hdf5")


if __name__ == '__main__':
    main()
