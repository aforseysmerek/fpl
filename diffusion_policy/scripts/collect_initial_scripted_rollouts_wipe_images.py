"""
Wipe task collector (v1) — scripted table-wiping rollouts that span two axes:
  - "circular motion": sweep shape blends straight <-> circular (circ_amount knob)
  - "pressing":        downward push into the table (press_depth knob)

Captures images + state + per-step signals, and scores each episode with oracle
reward functions on the REALIZED trajectory (not the knobs). Output episodes.hdf5
matches the square collector's layout so 2b / the state trainer reuse unchanged.

NOTE (v1): the exact controller action dim, image dtype, and contact-z want the
render check to confirm — this is a first draft to VISUALIZE and then tune. Run
`scripts/check_render_wipe.py` first (after the square collection frees the GPU).

Run in robodiff:
    cd ~/Desktop/fpl/diffusion_policy
    python scripts/collect_wipe.py -o shared_data_wipe -n 8      # small, to visualize
"""
import sys, os, pathlib
ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR); os.chdir(ROOT_DIR)

import json
import click
import numpy as np
import h5py
import robosuite
try:
    from robosuite.controllers import load_controller_config
except ImportError:
    from robosuite.controllers.controller_factory import load_controller_config

CAMERAS = ["agentview", "robot0_eye_in_hand"]
IMG_HW = 128
# Table surface (from Wipe DEFAULT_WIPE_CONFIG): full [0.5,0.8], offset (0.15,0,0.9).
TABLE_CENTER_XY = np.array([0.15, 0.0])
SWEEP_EXTENT = np.array([0.18, 0.28])   # stay inside the dirt region
TABLE_TOP_Z = 0.90


def create_wipe_env():
    ctrl = load_controller_config(default_controller="OSC_POSITION")
    ctrl["control_delta"] = False       # absolute xyz targets — easy to script
    return robosuite.make(
        "Wipe", robots="Panda", controller_configs=ctrl,
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=CAMERAS, camera_heights=IMG_HW, camera_widths=IMG_HW,
        control_freq=20, horizon=600, ignore_done=True, hard_reset=False,
        render_gpu_device_id=int(os.environ.get("MUJOCO_GPU", "-1")),  # pin EGL render GPU
    )


def to_uint8_hwc(img):
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] == 3:          # (C,H,W) -> (H,W,C)
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (a * 255.0).clip(0, 255).astype(np.uint8)
    return np.flipud(a).copy()                    # robosuite renders vertically flipped


def sweep_xy(circ_amount, n_points, rng):
    """Lawnmower sweep across the table, with a circular oscillation blended in.
    circ_amount=0 -> straight raster; higher -> loopy/circular motion."""
    cx, cy = TABLE_CENTER_XY
    ex, ey = SWEEP_EXTENT
    n_lanes = 4
    pts = []
    lane_xs = np.linspace(cx - ex, cx + ex, n_lanes)
    per_lane = max(n_points // n_lanes, 4)
    for li, lx in enumerate(lane_xs):
        ys = np.linspace(cy - ey, cy + ey, per_lane)
        if li % 2 == 1:
            ys = ys[::-1]
        for j, y in enumerate(ys):
            t = 2 * np.pi * (j / per_lane) * 3.0          # circular phase
            ox = circ_amount * 0.06 * np.cos(t)
            oy = circ_amount * 0.06 * np.sin(t)
            pts.append([lx + ox, y + oy])
    return np.array(pts)


def ee_force(env):
    return float(np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:3])))


# ---------- oracle reward functions (score the REALIZED trajectory) ----------
def circularity_score(eef_xy):
    """Total absolute turning (radians) of the eef xy-path. Straight ~ low, loopy ~ high."""
    d = np.diff(eef_xy, axis=0)
    seg = np.linalg.norm(d, axis=1) > 1e-5
    d = d[seg]
    if len(d) < 2:
        return 0.0
    ang = np.arctan2(d[:, 1], d[:, 0])
    dang = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)
    return float(np.sum(dang))


def pressing_score(forces, contacts):
    """Mean ee force while in contact (falls back to mean over all steps)."""
    f = np.asarray(forces)
    c = np.asarray(contacts, dtype=bool)
    return float(f[c].mean()) if c.any() else float(f.mean())


@click.command()
@click.option('-o', '--output_dir', default='shared_data_wipe')
@click.option('-n', '--num_episodes', type=int, default=8)
@click.option('--seed', type=int, default=0)
def main(output_dir, num_episodes, seed):
    rng = np.random.default_rng(seed)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    env = create_wipe_env()
    try:
        adim = env.action_dim
    except AttributeError:
        adim = len(env.action_spec[0])

    out = h5py.File(pathlib.Path(output_dir) / "episodes.hdf5", "w")
    data_grp = out.create_group("data")

    def obs_pack(obs):
        jp = np.asarray(obs.get("robot0_joint_pos", np.zeros(7)), np.float32)
        ep = np.asarray(obs["robot0_eef_pos"], np.float32)
        eq = np.asarray(obs.get("robot0_eef_quat", np.zeros(4)), np.float32)
        return (to_uint8_hwc(obs["agentview_image"]), to_uint8_hwc(obs["robot0_eye_in_hand_image"]),
                jp, np.concatenate([ep, eq, jp]).astype(np.float32), ep)  # state_lowdim = eef_pos+quat+joints (NO force)

    for ep in range(num_episodes):
        circ_amount = float(rng.uniform(0.0, 1.0))
        press_depth = float(rng.uniform(0.0, 0.02))    # metres pushed below contact
        obs = env.reset()

        tp_l, wr_l, jp_l, low_l, force_l, contact_l, act_l = [], [], [], [], [], [], []

        # Phase 1: move above table centre, descend until contact -> contact_z
        target = np.array([*TABLE_CENTER_XY, TABLE_TOP_Z + 0.12])
        for _ in range(40):
            obs, *_ = env.step(np.clip(target - obs["robot0_eef_pos"], -1, 1)[:adim] if adim == 3 else np.zeros(adim))
        contact_z = TABLE_TOP_Z + 0.05
        for _ in range(60):
            act = np.zeros(adim); act[2] = -0.4                # push down
            obs, *_ = env.step(act)
            if env._has_gripper_contact or ee_force(env) > 1.0:
                contact_z = float(obs["robot0_eef_pos"][2]); break

        # Phase 2: sweep at (contact_z - press_depth), following the shaped path
        z = contact_z - press_depth
        waypts = sweep_xy(circ_amount, n_points=200, rng=rng)
        for wx, wy in waypts:
            tgt = np.array([wx, wy, z])
            act = np.clip((tgt - obs["robot0_eef_pos"]) * 10.0, -1, 1)[:adim]
            tp, wr, jp, low, ep_pos = obs_pack(obs)
            tp_l.append(tp); wr_l.append(wr); jp_l.append(jp); low_l.append(low)
            force_l.append(ee_force(env)); contact_l.append(bool(env._has_gripper_contact)); act_l.append(act.copy())
            obs, *_ = env.step(act)

        eef_xy = np.stack(low_l)[:, :2]
        circ = circularity_score(eef_xy)
        press = pressing_score(force_l, contact_l)
        wiped = len(env.wiped_markers) / max(env.num_markers, 1)

        g = data_grp.create_group(f"demo_{ep}")
        og = g.create_group("obs")
        og.create_dataset("agent_view", data=np.stack(tp_l))
        og.create_dataset("wrist", data=np.stack(wr_l))
        og.create_dataset("JOINT_POS", data=np.stack(jp_l))
        og.create_dataset("state_lowdim", data=np.stack(low_l))
        g.create_dataset("actions", data=np.stack(act_l))
        g.attrs.update(dict(circularity=float(circ), pressing=float(press), wiped_frac=float(wiped),
                            circ_amount=circ_amount, press_depth=press_depth, n_steps=len(act_l)))
        print(f"ep {ep}: circ_knob={circ_amount:.2f} press_knob={press_depth*1e3:.0f}mm | "
              f"circularity={circ:.2f} pressing={press:.2f}N wiped={wiped:.2f}", flush=True)

    data_grp.attrs["axes"] = json.dumps(["circularity", "pressing"])
    out.close()
    print(f"\nWrote {num_episodes} episodes to {output_dir}/episodes.hdf5")


if __name__ == '__main__':
    main()
