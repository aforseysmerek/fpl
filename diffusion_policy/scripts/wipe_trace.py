"""
wipe_trace.py — scripted Wipe policy (mirrors their peg SquareSideScriptedPolicy:
absolute OSC_POSE waypoints + linear interpolation + euler->axis_angle), now with
the two preference axes as knobs on top of the base spill-trace:

  circ_amount (0..1): overlay a circular scrub on the traced path (0 = straight)
  press_amount (0..1): how hard to press (maps to eef height below the surface)

Prints the REALIZED oracle metrics (path circularity + mean contact force) so we
can check they track the knobs. --sweep renders the 4 corners.

Run in robodiff:
    MUJOCO_GPU=<free> python scripts/wipe_trace.py --sweep -o wipe_sweep
    MUJOCO_GPU=<free> python scripts/wipe_trace.py --circ_amount 0.7 --press_amount 0.8
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

OSC_POSE_ABS = {
    "type": "OSC_POSE", "input_max": 1, "input_min": -1,
    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
    "kp": 150, "damping": 1, "impedance_mode": "fixed",
    "kp_limits": [0, 300], "damping_limits": [0, 10],
    "position_limits": None, "orientation_limits": None,
    "uncouple_pos_ori": True, "control_delta": False,   # absolute, like the peg collection
    "interpolation": None, "ramp_ratio": 0.2,
}

# press_amount -> wiper_offset (eef height above surface). Calibrated from the probe:
# tool tip is 0.018 m below eef; offset 0.010 gave ~9 N. So a light->hard span:
PRESS_TOUCH = 0.016     # ~just contacting (light)
PRESS_RANGE = 0.012     # press_amount=1 -> offset 0.004 (hard)

# circular knob: circ_amount scales the loop RADIUS of a fixed number of scrub
# circles overlaid on the straight spill-trace. 0 -> dead-straight over the spill,
# small -> gentle circular scrubbing, 1 -> big loops (very circular). The circles
# are centered on the spill markers, so the wiper stays over the spill either way.
CIRC_RADIUS = 0.05      # m, scrub-circle radius at circ_amount=1
CIRC_LOOPS  = 4         # number of scrub loops along the sweep


def create_wipe_env():
    return robosuite.make(
        "Wipe", robots="Panda", controller_configs=OSC_POSE_ABS,
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=CAMERAS, camera_heights=IMG, camera_widths=IMG,
        control_freq=20, horizon=4000, ignore_done=True, hard_reset=False,
        render_gpu_device_id=int(os.environ.get("MUJOCO_GPU", "-1")),
    )


def marker_positions(env):
    return np.array([env.sim.data.body_xpos[env.sim.model.body_name2id(m.root_body)].copy()
                     for m in env.model.mujoco_arena.markers])


def ee_force(env):
    return float(np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:3])))


def grab(obs):
    a = np.asarray(obs["agentview_image"])
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8)
    return np.flipud(a).copy()


# ---- oracle metrics on the REALIZED trajectory ----
def circularity_metric(eef_xy):
    d = np.diff(np.asarray(eef_xy), axis=0)
    d = d[np.linalg.norm(d, axis=1) > 1e-5]
    if len(d) < 2:
        return 0.0
    ang = np.arctan2(d[:, 1], d[:, 0])
    return float(np.sum(np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)))  # total turning (rad)


def pressing_metric(forces):
    f = np.asarray(forces)
    c = f > 0.5
    return float(f[c].mean()) if c.any() else float(f.mean())


class WipeTracePolicy:
    DOWN = np.array([0.0, np.pi, 0.0])

    def __init__(self, env, start_eef, circ_amount=0.0, press_amount=0.5,
                 n_waypoints=30, seg_steps=12, approach_h=0.10):
        self.env = env
        self.rt = RotationTransformer('euler_angles', 'axis_angle', from_convention='XYZ')
        self.circ_amount = circ_amount
        self.wiper_offset = PRESS_TOUCH - press_amount * PRESS_RANGE
        self.n_waypoints = n_waypoints
        self.seg_steps = seg_steps
        self.approach_h = approach_h
        self.reset(start_eef)

    def generate_trajectory(self):
        m = marker_positions(self.env)
        idx = np.linspace(0, len(m) - 1, min(self.n_waypoints, len(m))).astype(int)
        path = m[idx]
        z = float(path[0, 2]) + self.wiper_offset
        self.surface_z = float(path[0, 2])

        n = len(path)
        r = self.circ_amount * CIRC_RADIUS       # 0 -> straight; larger -> more circular
        def circ_xy(i):
            # continuous circular scrub (CIRC_LOOPS loops over the sweep) centered on
            # the spill marker, so we stay over the spill; radius scales with circ_amount.
            th = 2 * np.pi * CIRC_LOOPS * i / max(n - 1, 1)
            return path[i, :2] + r * np.array([np.cos(th), np.sin(th)])

        x0, y0 = circ_xy(0)
        traj = [{"t": 40, "action": np.concatenate([[x0, y0, z + self.approach_h], self.DOWN])}]
        t = 65
        traj.append({"t": t, "action": np.concatenate([[x0, y0, z], self.DOWN])})
        for i in range(n):
            t += self.seg_steps
            xy = circ_xy(i)
            traj.append({"t": t, "action": np.concatenate([[xy[0], xy[1], z], self.DOWN])})
        xe, ye = circ_xy(n - 1)
        traj.append({"t": t + 60, "action": np.concatenate([[xe, ye, z], self.DOWN])})
        self.trajectory = traj
        self.max_t = traj[-1]["t"]

    def reset(self, start_eef):
        self.step_num = 0
        self.generate_trajectory()
        self.last_action = np.concatenate([np.asarray(start_eef, float), self.DOWN])
        self.last_t = 0

    def predict_action(self):
        if len(self.trajectory) > 1 and self.step_num >= self.trajectory[0]["t"]:
            self.last_action = self.trajectory[0]["action"]
            self.last_t = self.trajectory[0]["t"]
            self.trajectory.pop(0)
        nxt = self.trajectory[0]
        if nxt["t"] <= self.last_t:
            a = nxt["action"].copy()
        else:
            frac = (self.step_num - self.last_t) / (nxt["t"] - self.last_t)
            a = (self.last_action + (nxt["action"] - self.last_action) * frac).copy()
        a[3:6] = self.rt.forward(a[3:6].reshape((1, 3))).reshape(3)
        self.step_num += 1
        return a


def run_one(env, circ_amount, press_amount, n_waypoints, seg_steps, out_mp4):
    obs = env.reset()
    adim = env.action_dim
    pol = WipeTracePolicy(env, obs["robot0_eef_pos"], circ_amount, press_amount, n_waypoints, seg_steps)
    frames, eef_xy, forces = [], [], []
    for _ in range(pol.max_t):
        a = pol.predict_action()
        if adim > 6:
            a = np.concatenate([a, np.zeros(adim - 6)])
        obs, *_ = env.step(a)
        frames.append(grab(obs)); eef_xy.append(obs["robot0_eef_pos"][:2].copy()); forces.append(ee_force(env))
    circ = circularity_metric(eef_xy)
    press = pressing_metric(forces)
    wiped = len(env.wiped_markers)
    print(f"circ_knob={circ_amount:.2f} press_knob={press_amount:.2f}  ->  "
          f"circularity={circ:6.2f}  pressing={press:5.1f}N  wiped={wiped}/{env.num_markers}", flush=True)
    try:
        import imageio
        imageio.mimwrite(out_mp4, frames, fps=20)
    except Exception:
        pass
    return dict(circ=circ, press=press, wiped=wiped)


@click.command()
@click.option('-o', '--out_dir', default='wipe_trace')
@click.option('--circ_amount', type=float, default=0.0)
@click.option('--press_amount', type=float, default=0.5)
@click.option('--n_waypoints', type=int, default=30)
@click.option('--seg_steps', type=int, default=12)
@click.option('--sweep', is_flag=True, help='render the 4 axis corners')
def main(out_dir, circ_amount, press_amount, n_waypoints, seg_steps, sweep):
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    env = create_wipe_env()
    if sweep:
        for c in (0.2, 0.9):
            for p in (0.2, 0.9):
                run_one(env, c, p, n_waypoints, seg_steps,
                        os.path.join(out_dir, f"circ{c:.1f}_press{p:.1f}.mp4"))
        print(f"\nsaved 4 corner videos to {out_dir}/")
    else:
        run_one(env, circ_amount, press_amount, n_waypoints, seg_steps,
                os.path.join(out_dir, "trace.mp4"))
        print(f"saved {out_dir}/trace.mp4")


if __name__ == '__main__':
    main()
