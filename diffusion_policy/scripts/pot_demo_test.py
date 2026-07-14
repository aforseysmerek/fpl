"""
Minimal BASE two-arm pot-carry motion — grasp both handles, LIFT the pot, and
CARRY it to a specific goal location in the scene, then render to mp4. No
axes/knobs yet (that's pot_trace).

Key fix vs the flailing v1: every phase moves both hands by an IDENTICAL small
delta per step (rigid-pair interpolation toward waypoints), so the fixed
hand-to-hand distance set by the pot is preserved and the grasp holds. Jumping
straight to a far absolute target (what v1 did) desyncs the arms and pulls them
apart. Uses the same absolute OSC_POSE controller + computed grasp orientation
as pot_trace.

Run in robodiff (free GPU):
    MUJOCO_GPU=<free> python scripts/pot_demo_test.py -o pot_test
Then watch pot_test/carry.mp4.
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

from scripts.pot_trace import OSC_POSE_ABS, down_grasp_aa, pot_bottom_above_table

CAMERAS = ["agentview", "robot0_eye_in_hand"]
IMG = 128
OPEN, CLOSE = -1.0, 1.0
UP = np.array([0.0, 0.0, 1.0])


def make_env():
    return robosuite.make(
        "TwoArmLift", robots=["Panda", "Panda"], env_configuration="single-arm-opposed",
        controller_configs=OSC_POSE_ABS,
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=CAMERAS, camera_heights=IMG, camera_widths=IMG,
        control_freq=20, horizon=3000, ignore_done=True, hard_reset=False,
        render_gpu_device_id=int(os.environ.get("MUJOCO_GPU", "-1")),
    )


def grab(obs):
    a = np.asarray(obs["agentview_image"])
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8)
    return np.flipud(a).copy()


def step_both(env, obs, t0, t1, o0, o1, grip, frames):
    a = np.concatenate([np.asarray(t0, float), o0, [grip],
                        np.asarray(t1, float), o1, [grip]])
    obs, *_ = env.step(a)
    frames.append(grab(obs))
    return obs


def settle_move(env, obs, t0, t1, o0, o1, grip, frames, tol=0.008, max_steps=150):
    """Pre-grasp move: command the absolute target and loop until BOTH hands
    actually arrive (arms reach independently to their own handles; no rigid
    constraint yet). Ensures the arm is fully at grasp depth with fingers OPEN
    before any closing — closing early makes the fingers hit the bar from above
    and block the descent."""
    for _ in range(max_steps):
        obs = step_both(env, obs, t0, t1, o0, o1, grip, frames)
        err = max(np.linalg.norm(np.asarray(t0) - obs["robot0_eef_pos"]),
                  np.linalg.norm(np.asarray(t1) - obs["robot1_eef_pos"]))
        if err < tol:
            break
    return obs


def interp_move(env, obs, e0, e1, o0, o1, grip, frames, n_steps):
    """Post-grasp rigid-pair move: interpolate BOTH hands from their current pose
    to (e0,e1) over n_steps small steps. If (e0-s0)==(e1-s1) the hand separation
    is preserved every step; small per-step deltas keep OSC tracking tight so the
    arms stay in sync and the grasp holds."""
    s0 = np.asarray(obs["robot0_eef_pos"], float)
    s1 = np.asarray(obs["robot1_eef_pos"], float)
    for k in range(1, n_steps + 1):
        f = k / n_steps
        obs = step_both(env, obs, s0 + (e0 - s0) * f, s1 + (e1 - s1) * f, o0, o1, grip, frames)
    return obs


def close_until_grasp(env, obs, t0, t1, o0, o1, frames, max_steps=250):
    for _ in range(max_steps):
        obs = step_both(env, obs, t0, t1, o0, o1, CLOSE, frames)
        if (env._check_grasp(env.robots[0].gripper, env.pot.handle0_geoms) and
                env._check_grasp(env.robots[1].gripper, env.pot.handle1_geoms)):
            break
    return obs


@click.command()
@click.option('-o', '--out_dir', default='pot_test')
@click.option('--grasp_offset', type=float, default=0.0, help='grip_site height above the bar (~fingertip gap; use pot_probe value)')
@click.option('--carry_h', type=float, default=0.15, help='lift height of the pot above its grasp point')
@click.option('--goal_x', type=float, default=0.20, help='world x the pot is carried to')
@click.option('--goal_y', type=float, default=0.00, help='world y the pot is carried to')
def main(out_dir, grasp_offset, carry_h, goal_x, goal_y):
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    env = make_env()
    obs = env.reset()
    frames = [grab(obs)]

    h0 = np.array(env._handle0_xpos, float).copy()   # SNAPSHOT (site_xpos is a live view)
    h1 = np.array(env._handle1_xpos, float).copy()
    o0 = down_grasp_aa(obs["robot0_eef_quat"])       # straight-down at each arm's own reachable yaw
    o1 = down_grasp_aa(obs["robot1_eef_quat"])
    gz = grasp_offset
    goal = np.array([goal_x, goal_y])

    # A) approach above the handles, grippers open (closed-loop: arms reach freely).
    # 0.14 above so the arms descend into a reachable elbow config (verified headless).
    obs = settle_move(env, obs, h0 + UP * (0.14 + gz), h1 + UP * (0.14 + gz), o0, o1, OPEN, frames)
    # B) descend onto the handle bars (arrive fully with fingers OPEN before closing)
    obs = settle_move(env, obs, h0 + UP * gz, h1 + UP * gz, o0, o1, OPEN, frames)
    # C) close until both handles are grasped
    obs = close_until_grasp(env, obs, h0 + UP * gz, h1 + UP * gz, o0, o1, frames)
    g0 = env._check_grasp(env.robots[0].gripper, env.pot.handle0_geoms)
    g1 = env._check_grasp(env.robots[1].gripper, env.pot.handle1_geoms)
    print(f"[grasp] handle0={g0} handle1={g1}", flush=True)

    # anchor the carry to the ACTUAL grasped hand positions (rigid pair from here)
    e0 = np.asarray(obs["robot0_eef_pos"], float)
    e1 = np.asarray(obs["robot1_eef_pos"], float)

    # D) lift straight up to carry height
    obs = interp_move(env, obs, e0 + UP * carry_h, e1 + UP * carry_h, o0, o1, CLOSE, frames, 60)
    print(f"[lift]  pot bottom above table = {pot_bottom_above_table(env):.3f} m", flush=True)

    # E) carry to the goal (translate both hands by pot_goal - pot_now, in xy)
    pot_now = np.asarray(obs["pot_pos"][:2], float)
    d = np.array([goal[0] - pot_now[0], goal[1] - pot_now[1], 0.0])
    obs = interp_move(env, obs, e0 + UP * carry_h + d, e1 + UP * carry_h + d, o0, o1, CLOSE, frames, 120)

    pot_final = np.asarray(obs["pot_pos"][:2], float)
    print(f"[carry] pot xy={np.round(pot_final,3)} goal={np.round(goal,3)} "
          f"dist={np.linalg.norm(pot_final - goal):.3f} m  |  height={pot_bottom_above_table(env):.3f} m  "
          f"|  lifted={bool(env._check_success())}  |  {len(frames)} frames", flush=True)

    try:
        import imageio
        imageio.mimwrite(os.path.join(out_dir, "carry.mp4"), frames, fps=20)
        print(f"saved {out_dir}/carry.mp4")
    except Exception as e:
        from PIL import Image
        idx = np.linspace(0, len(frames) - 1, 10).astype(int)
        Image.fromarray(np.concatenate([frames[i] for i in idx], axis=1)).save(os.path.join(out_dir, "carry_strip.png"))
        print(f"saved contact sheet instead ({e})")


if __name__ == '__main__':
    main()
