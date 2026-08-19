"""
wipe_trace.py — scripted Wipe policy (mirrors their peg SquareSideScriptedPolicy:
absolute OSC_POSE waypoints + linear interpolation + euler->axis_angle), now with
the two preference axes as knobs on top of the base spill-trace:

  circular (bool): overlay circular scrub loops on the traced path, each loop's
      radius sampled from CIRC_RADIUS_RANGE (3-5 cm, varies within a traj).
      False = dead-straight trace. This is the CURRENT circularity interface.
  press_amount (0..1): how hard to press (maps to eef height below the surface)
  circ_amount (0..1): RETIRED continuous knob (radius = knob * CIRC_RADIUS,
      fixed within a traj). Kept only to reproduce old datasets/sweeps.

Prints the REALIZED oracle metrics (circle-fraction circularity + mean contact
force) so we can check they track the knobs. --sweep renders the 4 corners.

Run in robodiff:
    MUJOCO_GPU=<free> python scripts/wipe_trace.py --sweep -o wipe_sweep
    MUJOCO_GPU=<free> python scripts/wipe_trace.py --circular --press_amount 0.8

--vertical runs the same thing on the VerticalWipe env (wall instead of table):
same knobs, press along x (wall normal), scrub circles in the wall's (y, z)
plane, circularity measured in that plane:
    MUJOCO_GPU=<free> python scripts/wipe_trace.py --vertical --sweep -o vertical_wipe_sweep
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
# The preference oracle's own circularity metric — imported (not copied) so the
# sweep prints can never drift from what reward_functions scores.
from reward_model.reward_functions import _circle_fraction as circularity_metric

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

# circular mode: scrub circles overlaid on the straight spill-trace, centered
# on the spill markers so the wiper stays over the spill either way. Binary:
# circular=True draws CIRC_LOOPS loops with each loop's radius sampled from
# CIRC_RADIUS_RANGE (varies within a traj), False is dead-straight. Matches the
# eval oracle (reward_functions._circle_fraction gates radius to 2-7 cm).
CIRC_RADIUS_RANGE = (0.03, 0.05)   # m, per-loop radius band for circular=True
CIRC_LOOPS  = 4         # number of scrub loops along the sweep
# RETIRED continuous knob's radius scale (radius = circ_amount * CIRC_RADIUS).
CIRC_RADIUS = 0.05


def create_wipe_env(vertical=False):
    if vertical:
        import envs.vertical_wipe  # noqa: F401  registers VerticalWipe with robosuite
    return robosuite.make(
        "VerticalWipe" if vertical else "Wipe", robots="Panda", controller_configs=OSC_POSE_ABS,
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
def pressing_metric(forces):
    f = np.asarray(forces)
    c = f > 0.5
    return float(f[c].mean()) if c.any() else float(f.mean())


class WipeTracePolicy:
    DOWN = np.array([0.0, np.pi, 0.0])
    CIRC_RADIUS = CIRC_RADIUS   # class attr so subclasses can shrink the scrub loops

    def __init__(self, env, start_eef, circ_amount=None, press_amount=0.5,
                 n_waypoints=30, seg_steps=12, approach_h=0.10, circular=False, rng=None):
        self.env = env
        self.rt = RotationTransformer('euler_angles', 'axis_angle', from_convention='XYZ')
        # Per-loop scrub radii (CIRC_LOOPS+1 knots, interpolated across loop
        # boundaries by _radius_at so consecutive loops of different sizes
        # connect smoothly). circ_amount is the RETIRED knob: when given it
        # reproduces the old constant radius; otherwise `circular` decides.
        if circ_amount is not None:
            self.loop_radii = np.full(CIRC_LOOPS + 1, float(circ_amount) * self.CIRC_RADIUS)
        elif circular:
            rng = np.random.default_rng() if rng is None else rng
            self.loop_radii = rng.uniform(*CIRC_RADIUS_RANGE, size=CIRC_LOOPS + 1)
        else:
            self.loop_radii = np.zeros(CIRC_LOOPS + 1)
        self.wiper_offset = PRESS_TOUCH - press_amount * PRESS_RANGE
        self.n_waypoints = n_waypoints
        self.seg_steps = seg_steps
        self.approach_h = approach_h
        self.reset(start_eef)

    def _radius_at(self, i, n):
        """Scrub radius at waypoint i: linear interp of the per-loop radii over
        the loop phase, so radius varies smoothly between differently-sized loops."""
        phase = CIRC_LOOPS * i / max(n - 1, 1)
        return float(np.interp(phase, np.arange(CIRC_LOOPS + 1), self.loop_radii))

    def generate_trajectory(self):
        m = marker_positions(self.env)
        idx = np.linspace(0, len(m) - 1, min(self.n_waypoints, len(m))).astype(int)
        path = m[idx]
        z = float(path[0, 2]) + self.wiper_offset
        self.surface_z = float(path[0, 2])

        n = len(path)
        def circ_xy(i):
            # continuous circular scrub (CIRC_LOOPS loops over the sweep) centered on
            # the spill marker, so we stay over the spill; radius per loop from _radius_at.
            th = 2 * np.pi * CIRC_LOOPS * i / max(n - 1, 1)
            return path[i, :2] + self._radius_at(i, n) * np.array([np.cos(th), np.sin(th)])

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


class VerticalWipeTracePolicy(WipeTracePolicy):
    """Same trace policy on the VerticalWipe wall: the wiping plane's normal is
    world -x instead of +z, so press_amount maps to an x stand-off and the
    circular scrub lives in the wall's (y, z) plane. Waypoint interpolation,
    knob->offset calibration constants, and euler->axis_angle are inherited.
    """

    # Kept under the base class's attribute name so reset()/interp reuse it.
    # This is the realized grip_site orientation of envs.vertical_wipe's
    # WALL_INIT_QPOS (tool at world +x = the wall): commanding the SAME
    # orientation the init pose realizes keeps the OSC null-space and the
    # task target consistent, which is what keeps the wrist off its limits.
    DOWN = np.array([-0.8461, 1.5455, -0.7336])

    # Full-size scrub loops (inherited 0.05) so high-circ demos are VISIBLY
    # circular at 128px. With the spill confined to the flush-reach envelope
    # (coverage 0.35) this is validated: high-press episodes wipe 93-100/100;
    # the arm may briefly enter robosuite's 0.1-rad limit-warning zone on
    # some high-circ episodes (prints, no failures).

    def generate_trajectory(self):
        m = marker_positions(self.env)
        idx = np.linspace(0, len(m) - 1, min(self.n_waypoints, len(m))).astype(int)
        path = m[idx]
        # Markers share the wall plane; stand off from it along -x (robot side).
        x = float(path[0, 0]) - self.wiper_offset
        self.surface_x = float(path[0, 0])

        n = len(path)
        def circ_yz(i):
            th = 2 * np.pi * CIRC_LOOPS * i / max(n - 1, 1)
            return path[i, 1:3] + self._radius_at(i, n) * np.array([np.cos(th), np.sin(th)])

        y0, z0 = circ_yz(0)
        traj = [{"t": 40, "action": np.concatenate([[x - self.approach_h, y0, z0], self.DOWN])}]
        t = 65
        traj.append({"t": t, "action": np.concatenate([[x, y0, z0], self.DOWN])})
        for i in range(n):
            t += self.seg_steps
            yz = circ_yz(i)
            traj.append({"t": t, "action": np.concatenate([[x, yz[0], yz[1]], self.DOWN])})
        ye, ze = circ_yz(n - 1)
        traj.append({"t": t + 60, "action": np.concatenate([[x, ye, ze], self.DOWN])})
        self.trajectory = traj
        self.max_t = traj[-1]["t"]


def run_one(env, circ_amount, press_amount, n_waypoints, seg_steps, out_mp4, vertical=False, circular=False):
    obs = env.reset()
    adim = env.action_dim
    pol_cls = VerticalWipeTracePolicy if vertical else WipeTracePolicy
    plane = [1, 2] if vertical else [0, 1]   # wiping plane: wall (y,z) vs table (x,y)
    pol = pol_cls(env, obs["robot0_eef_pos"], circ_amount, press_amount, n_waypoints, seg_steps,
                  circular=circular)
    frames, eef_xy, forces = [], [], []
    for _ in range(pol.max_t):
        a = pol.predict_action()
        if adim > 6:
            a = np.concatenate([a, np.zeros(adim - 6)])
        obs, *_ = env.step(a)
        frames.append(grab(obs)); eef_xy.append(obs["robot0_eef_pos"][plane].copy()); forces.append(ee_force(env))
    circ = circularity_metric(eef_xy)
    press = pressing_metric(forces)
    wiped = len(env.wiped_markers)
    circ_lbl = ("circular" if circular else "straight") if circ_amount is None else f"legacy {circ_amount:.2f}"
    print(f"circ={circ_lbl} press_knob={press_amount:.2f}  ->  "
          f"circularity={circ:5.2f}  pressing={press:5.1f}N  wiped={wiped}/{env.num_markers}", flush=True)
    try:
        import imageio
        imageio.mimwrite(out_mp4, frames, fps=20)
    except Exception:
        pass
    return dict(circ=circ, press=press, wiped=wiped)


@click.command()
@click.option('-o', '--out_dir', default='wipe_trace')
@click.option('--circular', is_flag=True, help='scrub in circles (per-loop radius 3-5 cm); default is straight')
@click.option('--circ_amount', type=float, default=None,
              help='RETIRED continuous knob (radius = knob * 0.05 m); only for reproducing old runs')
@click.option('--press_amount', type=float, default=0.5)
@click.option('--n_waypoints', type=int, default=30)
@click.option('--seg_steps', type=int, default=12)
@click.option('--sweep', is_flag=True, help='render the 4 axis corners (straight/circular x light/hard press)')
@click.option('--vertical', is_flag=True, help='run on the VerticalWipe wall env instead of the table')
def main(out_dir, circular, circ_amount, press_amount, n_waypoints, seg_steps, sweep, vertical):
    if vertical and out_dir == 'wipe_trace':
        out_dir = 'vertical_wipe_trace'
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    env = create_wipe_env(vertical=vertical)
    if sweep:
        for circ in (False, True):
            for p in (0.2, 0.9):
                run_one(env, None, p, n_waypoints, seg_steps,
                        os.path.join(out_dir, f"{'circular' if circ else 'straight'}_press{p:.1f}.mp4"),
                        vertical=vertical, circular=circ)
        print(f"\nsaved 4 corner videos to {out_dir}/")
    else:
        run_one(env, circ_amount, press_amount, n_waypoints, seg_steps,
                os.path.join(out_dir, "trace.mp4"), vertical=vertical, circular=circular)
        print(f"saved {out_dir}/trace.mp4")


if __name__ == '__main__':
    main()
