"""
Steering eval for the pi05-droid finetune in robosuite: roll out the policy at
different prompted reward values and score achieved behavior with the oracle
(reward_functions.compute_axes — never recomputed here).

Requires the openpi policy server running (openpi venv, py3.11):
    cd real_world/openpi && uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_droid_finetune --policy.dir=<ckpt>/<step>

Action bridge (validated by replay_joint_vel_check.py --selftest): the policy's
joint velocities are integrated into an absolute joint target (q_tgt += v/fps)
tracked by a JOINT_POSITION PD (kp=50 default) — what the real DROID velocity
controller does. Executes the first --open_loop_horizon of each 16-step chunk.

Spill layouts are seeded per episode index, so every setting sees the SAME
sequence of spills — settings differ only by their prompt.

Run in robodiff (needs `pip install -e real_world/openpi/packages/openpi-client`):
    MUJOCO_GPU=<free> python scripts/eval_pi05_steering.py -o pi05_eval_out/circ_steer
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

from scripts.collect_initial_scripted_rollouts_wipe_images import CAMERAS, to_uint8_hwc
from scripts.replay_joint_vel_check import FPS, JOINT_POS_DELTA, create_wipe_env_joint_space
from reward_functions import compute_axes
from preferences import AXIS_PROMPT

# Task registry: env + oracle axes + base prompt. The z vector in --settings is
# ordered like `axes`. vertical_wipe shares machinery; pot needs a bimanual
# policy (pi05-droid is single-arm) and is not wired up yet.
TASKS = {
    "wipe": dict(vertical=False, axes=["circularity", "wiped_frac"], task_prompt="wipe the table"),
    "vertical_wipe": dict(vertical=True, axes=["circularity_vertical", "wiped_frac"], task_prompt="wipe the table"),
}


def build_prompt(task_prompt, axes, z, decimal_places=1):
    """MUST match real_world/convert_custom_droid_to_lerobot.py build_prompt
    (that module needs lerobot/torch, not importable in robodiff): task prompt
    + ", " + comma-joined "<axis phrase>: <value>" at the dataset's 1 dp."""
    parts = [f"{AXIS_PROMPT[a]}: {v:.{decimal_places}f}" for a, v in zip(axes, z)]
    return task_prompt + ", " + ", ".join(parts)


def rollout(env, client, image_tools, prompt, ep_len, open_loop_horizon, seed, kp_clip):
    np.random.seed(seed)   # global np.random drives WipeArena's spill sampling
    obs = env.reset()
    robot = env.robots[0]
    q_tgt = np.asarray(env.sim.data.qpos[robot._ref_joint_pos_indexes], np.float64).copy()

    chunk, chunk_i = None, 0
    tp_l, wr_l, jp_l, low_l, act_l = [], [], [], [], []
    for _ in range(ep_len):
        # Pre-action obs, recorded exactly like the collector.
        tp = to_uint8_hwc(obs["agentview_image"])
        wr = to_uint8_hwc(obs["robot0_eye_in_hand_image"])
        ep_pos = np.asarray(obs["robot0_eef_pos"], np.float32)
        eq = np.asarray(obs["robot0_eef_quat"], np.float32)
        jp = np.asarray(env.sim.data.qpos[robot._ref_joint_pos_indexes], np.float32)
        wiped_t = len(env.wiped_markers) / max(env.num_markers, 1)
        tp_l.append(tp)
        wr_l.append(wr)
        jp_l.append(jp)
        low_l.append(np.concatenate([ep_pos, eq, jp, [wiped_t]]))

        if chunk is None or chunk_i >= open_loop_horizon:
            # The exact per-step contract the policy was trained on (DROID keys).
            request = {
                "observation/exterior_image_1_left": image_tools.resize_with_pad(tp, 224, 224),
                "observation/wrist_image_left": image_tools.resize_with_pad(wr, 224, 224),
                "observation/joint_position": jp.astype(np.float64),
                "observation/gripper_position": np.zeros(1),
                "prompt": prompt,
            }
            chunk = np.asarray(client.infer(request)["actions"])   # (16, 8)
            chunk_i = 0
        vel = chunk[chunk_i, :7].astype(np.float64)   # dim 7 = gripper, WipingGripper has no DOF
        chunk_i += 1

        q_tgt = q_tgt + vel / float(FPS)
        act = np.clip(q_tgt - jp.astype(np.float64), -kp_clip, kp_clip)
        obs, *_ = env.step(act)
        act_l.append(vel.astype(np.float32))   # store policy-native actions (rad/s)

    return dict(agent_view=np.stack(tp_l), wrist=np.stack(wr_l), JOINT_POS=np.stack(jp_l),
                state_lowdim=np.stack(low_l), actions=np.stack(act_l))


@click.command()
@click.option('-o', '--output_dir', default='pi05_eval_out/steering')
@click.option('--task', type=click.Choice(sorted(TASKS)), default='wipe')
@click.option('--settings', default='{"circ_pos": [1.0, 0.0], "circ_neg": [-1.0, 0.0]}',
              help='JSON {name: [z per axis]}, z ordered like the task axes, '
                   'formatted into the prompt at 1 dp (the dataset convention).')
@click.option('--n_rollouts', type=int, default=10)
@click.option('--ep_len', type=int, default=485, help='Training episodes were 485 steps @ 20 Hz.')
@click.option('--open_loop_horizon', type=int, default=8, help='Steps executed per 16-step chunk.')
@click.option('--seed', type=int, default=0, help='Episode k uses seed+k in EVERY setting.')
@click.option('--n_videos', type=int, default=3)
@click.option('--host', default='localhost')
@click.option('--port', type=int, default=8000)
@click.option('--kp', type=float, default=50.0, help='Bridge PD stiffness (frozen default 50).')
def main(output_dir, task, settings, n_rollouts, ep_len, open_loop_horizon, seed,
         n_videos, host, port, kp):
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy

    spec = TASKS[task]
    settings = json.loads(settings)
    for name, z in settings.items():
        if len(z) != len(spec["axes"]):
            raise click.UsageError(f"setting {name}: {len(z)} values for axes {spec['axes']}")

    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    print(f"[eval] policy server: {host}:{port}  metadata={client.get_server_metadata()}", flush=True)

    ctrl_cfg = dict(JOINT_POS_DELTA, kp=kp)
    env = create_wipe_env_joint_space(ctrl_cfg, vertical=spec["vertical"])
    assert env.action_dim == 7, f"expected 7-dim joint-space action, got {env.action_dim}"

    out_root = pathlib.Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, z in settings.items():
        prompt = build_prompt(spec["task_prompt"], spec["axes"], z)
        print(f"\n[eval] setting {name}: z={z}  prompt={prompt!r}", flush=True)
        set_dir = out_root / name
        (set_dir / "videos").mkdir(parents=True, exist_ok=True)

        per_ep = {a: [] for a in spec["axes"]}
        with h5py.File(set_dir / "episodes.hdf5", "w") as f:
            data_grp = f.create_group("data")
            data_grp.attrs["cameras"] = json.dumps(CAMERAS)
            data_grp.attrs["axes"] = json.dumps(spec["axes"])
            data_grp.attrs["prompt"] = prompt
            data_grp.attrs["z"] = json.dumps(list(map(float, z)))
            for ep in range(n_rollouts):
                d = rollout(env, client, image_tools, prompt, ep_len, open_loop_horizon,
                            seed + ep, kp_clip=JOINT_POS_DELTA["input_max"])
                vals = compute_axes(spec["axes"], d["state_lowdim"], actions=d["actions"])
                for a in spec["axes"]:
                    per_ep[a].append(float(vals[a]))
                g = data_grp.create_group(f"demo_{ep}")
                og = g.create_group("obs")
                for k in ("agent_view", "wrist", "JOINT_POS", "state_lowdim"):
                    og.create_dataset(k, data=d[k])
                g.create_dataset("actions", data=d["actions"])
                g.attrs.update(dict(n_steps=ep_len, ep_seed=seed + ep,
                                    **{a: float(vals[a]) for a in spec["axes"]}))
                print(f"[eval]   ep {ep:2d} (seed {seed + ep}): " +
                      "  ".join(f"{a}={vals[a]:6.2f}" for a in spec["axes"]), flush=True)
                if ep < n_videos:
                    try:
                        import imageio
                        imageio.mimwrite(str(set_dir / "videos" / f"ep{ep:03d}.mp4"),
                                         list(d["agent_view"]), fps=FPS)
                    except Exception as e:
                        print(f"[eval]   video skipped ({e})", flush=True)

        results[name] = dict(
            z=list(map(float, z)), prompt=prompt, n_rollouts=n_rollouts,
            per_episode=per_ep,
            **{f"mean_{a}": float(np.mean(v)) for a, v in per_ep.items()},
            **{f"std_{a}": float(np.std(v)) for a, v in per_ep.items()},
        )

    results_path = out_root / "results.json"
    with open(results_path, "w") as f:
        json.dump(dict(task=task, ep_len=ep_len, open_loop_horizon=open_loop_horizon,
                       kp=kp, seed=seed, settings=results), f, indent=2)
    print("\n[eval] summary:", flush=True)
    for name, r in results.items():
        print(f"[eval]   {name:12s} z={r['z']}  " +
              "  ".join(f"{a}={r[f'mean_{a}']:6.2f}±{r[f'std_{a}']:.2f}" for a in spec["axes"]),
              flush=True)
    print(f"[eval] wrote {results_path}", flush=True)


if __name__ == '__main__':
    main()
