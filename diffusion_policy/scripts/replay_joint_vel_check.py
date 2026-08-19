"""
Replay check for the pi0.5 -> robosuite action bridge (JOINT_VELOCITY control).

The trained pi05-droid policy emits 7 joint velocities (rad/s @ 20 Hz). Its
training actions were derived from collected demos as diff(JOINT_POS) * fps
(convert_custom_droid_to_lerobot.py, sim branch). Before building the policy
rollout loop, this script answers: can a JOINT_VELOCITY-controlled Wipe env
reproduce a demo when fed those exact velocities? If joint tracking holds and
the eef keeps wiping contact, the action bridge is sound; if not, the fallback
is a JOINT_POSITION controller fed per-step deltas.

Env construction mirrors collect_initial_scripted_rollouts_wipe_images.
create_wipe_env (same cameras/128px/20 Hz/horizon/ignore_done/hard_reset),
only the controller differs. Oracle metrics come from reward_functions —
never recomputed here.

Run in robodiff:
    MUJOCO_GPU=<free> python scripts/replay_joint_vel_check.py --demo demo_0

--selftest closes the loop that recorded demos cannot (their spill layouts were
never seeded): it scripts a FRESH demo (WipeTracePolicy on the OSC env, same
recording as the collector), snapshots its spill marker layout, copies that
layout into the replay env, and replays the velocity-converted actions on it —
making wiped_frac directly comparable between demo and replay:
    MUJOCO_GPU=<free> python scripts/replay_joint_vel_check.py --selftest --controller joint_pos_integrated
"""
import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, str(pathlib.Path(ROOT_DIR) / "reward_model"))
os.chdir(ROOT_DIR)

import click
import numpy as np
import h5py
import robosuite
import robosuite.utils.transform_utils as T

from scripts.collect_initial_scripted_rollouts_wipe_images import CAMERAS, IMG_HW, to_uint8_hwc, create_wipe_env
from scripts.wipe_trace import ee_force, WipeTracePolicy, VerticalWipeTracePolicy
from reward_functions import circularity, circularity_vertical, wiped_frac

# fps used by the LeRobot conversion AND the env control_freq — the velocity
# scale only round-trips if these agree (dataset meta/info.json says 20).
FPS = 20

# Base: robosuite 1.2.0 controllers/config/joint_velocity.json. Changed vs
# default: input/output/velocity_limits widened from (+-1 -> +-0.5 rad/s,
# clip +-1) to a 1:1 pass-through clipped at +-3 rad/s — demo velocities are
# RMS ~0.2 rad/s but single-step diff spikes reach ~4.2, and the default
# mapping would halve-and-clip every command the policy sends.
JOINT_VEL_ABS = {
    "type": "JOINT_VELOCITY",
    "input_max": 3.0, "input_min": -3.0,
    "output_max": 3.0, "output_min": -3.0,
    "kp": 0.03,
    "velocity_limits": [-3.0, 3.0],
    "interpolation": None, "ramp_ratio": 0.2,
}

# Fallback bridge: robosuite 1.2.0 controllers/config/joint_position.json with
# input/output widened from (+-1 -> +-0.05 rad) to a 1:1 per-step delta
# pass-through at +-0.5 rad (replay deltas = vel/20, max ~0.21 rad). The
# controller adds the delta to the CURRENT joint pos each step (stiff PD,
# kp=50), i.e. exactly the position differences the velocities were derived
# from at conversion time.
JOINT_POS_DELTA = {
    "type": "JOINT_POSITION",
    "input_max": 0.5, "input_min": -0.5,
    "output_max": 0.5, "output_min": -0.5,
    "kp": 50, "damping_ratio": 1, "impedance_mode": "fixed",
    "kp_limits": [0, 300], "damping_ratio_limits": [0, 10],
    "qpos_limits": None,
    "interpolation": None, "ramp_ratio": 0.2,
}

# joint_pos_integrated: same PD config, but commanded velocities are first
# integrated into an ABSOLUTE joint target (q_tgt += vel/fps) and the action
# each step is clip(q_tgt - q_current) — what the real DROID velocity
# controller does. Sustained press force then comes from the integrated
# target sitting below the surface, which raw velocity replay cannot express
# (steady pressing differentiates to ~zero velocity), and which per-step
# relative deltas cannot either (goal re-anchors to current pose, capping
# torque at kp*delta).
CONTROLLERS = {
    "joint_vel": JOINT_VEL_ABS,
    "joint_pos_delta": JOINT_POS_DELTA,
    "joint_pos_integrated": JOINT_POS_DELTA,
}


def create_wipe_env_joint_space(controller_config, vertical=False):
    """create_wipe_env with the controller swapped to a joint-space one (action
    space becomes 7 joint targets; WipingGripper has no gripper DOF)."""
    if vertical:
        import envs.vertical_wipe  # noqa: F401  registers VerticalWipe with robosuite
    return robosuite.make(
        "VerticalWipe" if vertical else "Wipe", robots="Panda", controller_configs=controller_config,
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=CAMERAS, camera_heights=IMG_HW, camera_widths=IMG_HW,
        control_freq=20, horizon=4000, ignore_done=True, hard_reset=False,
        render_gpu_device_id=int(os.environ.get("MUJOCO_GPU", "-1")),
    )


@click.command()
@click.option('--episodes', default='shared_data_wipe/episodes.hdf5',
              help='Collected episodes.hdf5 to replay from.')
@click.option('--demo', default='demo_0', help='Demo key under data/.')
@click.option('-o', '--output_dir', default='pi05_eval_out/replay_check')
@click.option('--vertical', is_flag=True,
              help='Replay on the VerticalWipe wall env (use with a vertical episodes.hdf5).')
@click.option('--controller', type=click.Choice(sorted(CONTROLLERS)), default='joint_vel',
              help='Action bridge under test: velocity pass-through vs per-step position deltas.')
@click.option('--selftest', is_flag=True,
              help='Script a fresh demo, copy its spill layout into the replay env, and replay '
                   'against it -> wiped_frac becomes directly comparable (see module docstring).')
@click.option('--seed', type=int, default=0, help='Selftest: seeds knobs AND the spill layout.')
@click.option('--circ_knob', type=float, default=0.5, help='Selftest scripted-policy knob.')
@click.option('--press_knob', type=float, default=0.5, help='Selftest scripted-policy knob.')
@click.option('--kp', type=float, default=50.0,
              help='PD stiffness for the joint_pos_* bridges (robosuite default 50, limit 300). '
                   'Higher -> more contact force at the same integrated target.')
def main(episodes, demo, output_dir, vertical, controller, selftest, seed, circ_knob, press_knob, kp):
    circ_fn = circularity_vertical if vertical else circularity

    markers = None
    if selftest:
        demo = f"selftest_seed{seed}"
        np.random.seed(seed)  # global np.random drives WipeArena's spill sampling
        env_src = create_wipe_env(vertical=vertical)
        obs = env_src.reset()
        pol_cls = VerticalWipeTracePolicy if vertical else WipeTracePolicy
        pol = pol_cls(env_src, obs["robot0_eef_pos"], circ_knob, press_knob, 30, 12)
        adim = env_src.action_dim
        src_robot = env_src.robots[0]
        jp_l0, low_l0, tp_l0 = [], [], []
        for _ in range(pol.max_t):
            # Same pre-action recording as the collector.
            tp_l0.append(to_uint8_hwc(obs["agentview_image"]))
            ep_pos = np.asarray(obs["robot0_eef_pos"], np.float32)
            eq = np.asarray(obs["robot0_eef_quat"], np.float32)
            jp0 = np.asarray(env_src.sim.data.qpos[src_robot._ref_joint_pos_indexes], np.float32)
            wiped_t0 = len(env_src.wiped_markers) / max(env_src.num_markers, 1)
            jp_l0.append(jp0)
            low_l0.append(np.concatenate([ep_pos, eq, jp0, [wiped_t0]]))
            a = pol.predict_action()
            if adim > 6:
                a = np.concatenate([a, np.zeros(adim - 6)])
            obs, *_ = env_src.step(a)
        jp_demo, state_demo, agent_demo = np.stack(jp_l0), np.stack(low_l0), np.stack(tp_l0)
        # Snapshot the spill: reset_arena places markers via model body_pos.
        markers = []
        for m in env_src.model.mujoco_arena.markers:
            bid = env_src.sim.model.body_name2id(m.root_body)
            markers.append((m.root_body, env_src.sim.model.body_pos[bid].copy(),
                            env_src.sim.model.body_quat[bid].copy()))
        env_src.close()
        print(f"[replay] selftest demo scripted: knobs circ={circ_knob} press={press_knob} seed={seed}",
              flush=True)
    else:
        with h5py.File(episodes, 'r') as f:
            g = f[f"data/{demo}"]
            jp_demo = g["obs/JOINT_POS"][:]            # (T,7) pre-action joint positions
            state_demo = g["obs/state_lowdim"][:]      # (T,15)
            agent_demo = g["obs/agent_view"][:]        # (T,128,128,3) uint8, upright
    T_steps = jp_demo.shape[0]

    # Same formula as convert_custom_droid_to_lerobot.py (sim branch): rad/s,
    # final step repeats position -> zero velocity.
    vels = (np.diff(jp_demo, axis=0, append=jp_demo[-1:]) * float(FPS)).astype(np.float64)
    n_clip = int((np.abs(vels) > JOINT_VEL_ABS["velocity_limits"][1]).sum())
    print(f"[replay] {demo}: T={T_steps}  controller={controller}  |vel| rms={np.sqrt((vels**2).mean()):.3f} "
          f"max={np.abs(vels).max():.2f} rad/s  entries clipped at ±{JOINT_VEL_ABS['velocity_limits'][1]}: {n_clip}",
          flush=True)

    # Under joint_pos_delta the action is the per-step position delta = vel/fps
    # (identical numbers to the position diffs the velocities came from).
    # joint_pos_integrated computes its action inside the loop (needs sim state).
    acts = vels if controller == 'joint_vel' else vels / float(FPS)

    ctrl_cfg = dict(CONTROLLERS[controller])
    if controller != 'joint_vel':
        ctrl_cfg["kp"] = kp
    env = create_wipe_env_joint_space(ctrl_cfg, vertical=vertical)
    assert env.action_dim == 7, f"expected 7-dim joint-space action, got {env.action_dim}"
    env.reset()

    # Teleport to the demo's first joint pose (removes reset init-noise so the
    # replay starts exactly where the demo did), then let mujoco recompute.
    robot = env.robots[0]
    env.sim.data.qpos[robot._ref_joint_pos_indexes] = jp_demo[0]
    env.sim.data.qvel[robot._ref_joint_vel_indexes] = 0.0
    if markers is not None:
        # Overwrite the freshly-randomized spill with the selftest demo's layout
        # so wiped_frac compares like-for-like.
        for name, pos, quat in markers:
            bid = env.sim.model.body_name2id(name)
            env.sim.model.body_pos[bid] = pos
            env.sim.model.body_quat[bid] = quat
    env.sim.forward()

    q_tgt = jp_demo[0].astype(np.float64).copy()   # integrated velocity target
    jp_l, low_l, frames, force_l = [], [], [], []
    for t in range(T_steps):
        # Pre-action state straight from sim (matches the collector's timing;
        # eef pose read from the grip site, not the hand-body obs frame).
        jp = np.asarray(env.sim.data.qpos[robot._ref_joint_pos_indexes], np.float32)
        eef_pos = np.asarray(env.sim.data.site_xpos[robot.eef_site_id], np.float32)
        eef_quat = np.asarray(
            T.mat2quat(env.sim.data.site_xmat[robot.eef_site_id].reshape(3, 3)), np.float32)
        wiped_t = len(env.wiped_markers) / max(env.num_markers, 1)
        jp_l.append(jp)
        low_l.append(np.concatenate([eef_pos, eef_quat, jp, [wiped_t]]))

        if controller == 'joint_pos_integrated':
            q_tgt += vels[t] / float(FPS)
            hi = JOINT_POS_DELTA["input_max"]
            act = np.clip(q_tgt - jp.astype(np.float64), -hi, hi)
        else:
            act = acts[t]
        obs, *_ = env.step(act)
        frames.append(to_uint8_hwc(obs["agentview_image"]))
        force_l.append(ee_force(env))  # tool force during step t (contact check)

    jp_replay = np.stack(jp_l, 0)
    state_replay = np.stack(low_l, 0)

    q_err = np.abs(jp_replay - jp_demo)
    z_col = 2  # eef height above the table (press/contact proxy); vertical env presses along x
    press_col = 0 if vertical else z_col
    press_err = np.abs(state_replay[:, press_col] - state_demo[:, press_col])
    circ_demo, circ_rep = float(circ_fn(state_demo)), float(circ_fn(state_replay))
    wiped_demo, wiped_rep = float(wiped_frac(state_demo)), float(wiped_frac(state_replay))

    # Contact check on the steps where the DEMO was provably wiping (its
    # recorded proportion_wiped increased during step t). wiped_frac itself is
    # not comparable — the demo's spill layout was unseeded and the scripted
    # policy aimed at ITS markers — but on active-wipe steps the tool must be
    # pressed on the surface, so replay force there is the honest signal.
    forces = np.asarray(force_l)
    active = np.diff(state_demo[:, -1]) > 0            # step t: wiped counter rose
    thr = float(getattr(env, "contact_threshold", 1.0))
    contact_frac = float((forces[:-1][active] > thr).mean()) if active.any() else float("nan")
    mean_force = float(forces[:-1][active].mean()) if active.any() else float("nan")

    print(f"[replay] joint tracking |err| rad: mean={q_err.mean():.4f}  max={q_err.max():.4f}  "
          f"per-joint max={np.array2string(q_err.max(0), precision=3)}", flush=True)
    print(f"[replay] eef press-axis |err| m:   mean={press_err.mean():.4f}  max={press_err.max():.4f}  "
          f"(at step {int(press_err.argmax())}/{T_steps})", flush=True)
    print(f"[replay] contact on demo's active-wipe steps (n={int(active.sum())}): "
          f"force>{thr:.0f}N on {contact_frac:.0%}, mean {mean_force:.1f}N", flush=True)
    print(f"[replay] circularity: demo={circ_demo:6.2f}  replay={circ_rep:6.2f}", flush=True)
    wiped_note = "MATCHED spill -> directly comparable" if markers is not None \
        else "different spill layout -> NOT comparable, see contact line"
    print(f"[replay] wiped_frac:  demo={wiped_demo:.2f}  replay={wiped_rep:.2f} ({wiped_note})", flush=True)

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mp4 = out / f"{demo}_{controller}_replay_vs_demo.mp4"
    # frames[t] is the post-step image (state t+1) -> pair with demo frame t+1.
    pairs = [np.hstack([frames[t], agent_demo[t + 1]]) for t in range(T_steps - 1)]
    try:
        import imageio
        imageio.mimwrite(str(mp4), pairs, fps=FPS)
        print(f"[replay] wrote {mp4}  (left: replay, right: demo)", flush=True)
    except Exception as e:
        print(f"[replay] video skipped ({e})", flush=True)


if __name__ == '__main__':
    main()
