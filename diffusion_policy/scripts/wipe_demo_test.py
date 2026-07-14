"""
Minimal BASE wiping motion — get one clean wipe with proper table contact and
render it to mp4. No axes/knobs yet (that comes after the motion looks right).

Key vs the flailing v1: OSC_POSITION with control_delta=True, and a proportional
delta-controller toward waypoints (clipped to the action range, so it moves
smoothly toward targets and physically cannot flail).

Run in robodiff (free GPU):
    MUJOCO_GPU=<free> python scripts/wipe_demo_test.py -o wipe_test
Then watch wipe_test/wipe.mp4.
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
try:
    from robosuite.controllers import load_controller_config
except ImportError:
    from robosuite.controllers.controller_factory import load_controller_config

CAMERAS = ["agentview", "robot0_eye_in_hand"]
IMG = 128
# Table surface (Wipe defaults): full [0.5, 0.8], offset (0.15, 0, 0.9).
CX, CY = 0.15, 0.0
EX, EY = 0.14, 0.20        # sweep half-extents (keep inside the dirt region + Panda reach)
TABLE_Z = 0.90


def make_env():
    ctrl = load_controller_config(default_controller="OSC_POSITION")
    ctrl["control_delta"] = True      # deltas: action in [-1,1] -> small bounded step
    return robosuite.make(
        "Wipe", robots="Panda", controller_configs=ctrl,
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=CAMERAS, camera_heights=IMG, camera_widths=IMG,
        control_freq=20, horizon=3000, ignore_done=True, hard_reset=False,
        render_gpu_device_id=int(os.environ.get("MUJOCO_GPU", "-1")),
    )


def ee_force(env):
    return float(np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:3])))


def grab(obs):
    a = np.asarray(obs["agentview_image"])
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8)
    return np.flipud(a).copy()


def move_to(env, obs, target, frames, gain=15.0, tol=0.008, max_steps=150):
    """Proportional delta control toward target xyz (position DOF only)."""
    target = np.asarray(target, dtype=float)
    for _ in range(max_steps):
        err = target - obs["robot0_eef_pos"]
        obs, *_ = env.step(np.clip(err * gain, -1.0, 1.0))
        frames.append(grab(obs))
        if np.linalg.norm(err) < tol:
            break
    return obs


@click.command()
@click.option('-o', '--out_dir', default='wipe_test')
@click.option('--press', type=float, default=0.012, help='m to push target below contact (holds gentle contact)')
@click.option('--n_lanes', type=int, default=5)
def main(out_dir, press, n_lanes):
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    env = make_env()
    obs = env.reset()
    frames = []

    # A) move above the table centre
    obs = move_to(env, obs, [CX, CY, TABLE_Z + 0.15], frames, gain=15)

    # B) descend gently until contact -> contact_z
    contact_z = TABLE_Z + 0.05
    for _ in range(150):
        obs, *_ = env.step(np.array([0.0, 0.0, -0.25]))
        frames.append(grab(obs))
        if env._has_gripper_contact or ee_force(env) > 2.0:
            contact_z = float(obs["robot0_eef_pos"][2])
            break
    print(f"[contact] eef z={contact_z:.3f}  force={ee_force(env):.1f}N", flush=True)
    z = contact_z - press          # press target slightly below contact to maintain pressure

    # C) slow lawnmower sweep, holding downward contact
    forces = []
    wiped0 = len(env.wiped_markers)
    for li, lx in enumerate(np.linspace(CX - EX, CX + EX, n_lanes)):
        y0, y1 = (CY - EY, CY + EY) if li % 2 == 0 else (CY + EY, CY - EY)
        obs = move_to(env, obs, [lx, y0, z], frames, gain=10, tol=0.01)   # to lane start
        obs = move_to(env, obs, [lx, y1, z], frames, gain=6, tol=0.01)    # slow wipe across
        forces.append(ee_force(env))
    wiped = len(env.wiped_markers) - wiped0
    print(f"[sweep] mean contact force ~{np.mean(forces):.1f}N  |  wiped {wiped}/{env.num_markers}  "
          f"|  {len(frames)} frames", flush=True)

    try:
        import imageio
        imageio.mimwrite(os.path.join(out_dir, "wipe.mp4"), frames, fps=20)
        print(f"saved {out_dir}/wipe.mp4")
    except Exception as e:
        from PIL import Image
        idx = np.linspace(0, len(frames) - 1, 10).astype(int)
        Image.fromarray(np.concatenate([frames[i] for i in idx], axis=1)).save(os.path.join(out_dir, "wipe_strip.png"))
        print(f"saved contact sheet instead ({e})")


if __name__ == '__main__':
    main()
