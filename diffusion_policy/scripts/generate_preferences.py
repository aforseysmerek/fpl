"""
generate_preferences.py — build preference pairs from collected episodes using
THEIR code, and save for the state model and/or the Qwen model (same pairs).

All the actual preference logic is theirs:
  - metrics  : reward_model.reward_functions.compute_axes  (per-trajectory oracle scores)
  - pairs    : reward_model.train_reward_model.PreferencePairDataset  (sampling + labeling)

Outputs:
  --save_state DIR : DIR/demos.hdf5 (their per-key layout, so their trainer loads it)
                     + DIR/pairs.npz (idx_a, idx_b, labels)  ->  train_reward_model.py --load_prefs
  --save_qwen  DIR : co-located pair dirs (rollout_A/B.hdf5 with images + preference.json)
                     ->  real_world/train_reward_model.py --model qwen_open --task auto

Run in robodiff:
  python scripts/generate_preferences.py --episodes shared_data_square_images/episodes.hdf5 \
      --reward_axes speed_reward,peg_reward --n_pairs 300 \
      --save_state shared_data_square_images/prefs_state \
      --save_qwen  shared_data_square_images/preferences
"""
import sys
import os
import json
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, str(pathlib.Path(ROOT_DIR) / "reward_model"))
os.chdir(ROOT_DIR)

import click
import numpy as np
import h5py

from reward_functions import compute_axes                     # THEIR metric functions
from preferences import AXIS_PROMPT                           # axis → prompt phrase map
from train_reward_model import PreferencePairDataset          # THEIR pair generator

# state_lowdim = concat(object, robot0_eef_pos(3), robot0_eef_quat(4), robot0_gripper_qpos(2))
# — split it back into their per-key demo layout (object = everything before the last 9).
OBS_TAIL = [("robot0_eef_pos", 3), ("robot0_eef_quat", 4), ("robot0_gripper_qpos", 2)]


def load_episodes(path, n_episodes):
    demos = []
    with h5py.File(path, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
        if n_episodes is not None:
            assert n_episodes <= len(keys), "n_episodes exceeds total episodes available"
            keys = keys[:n_episodes]
        for k in keys:
            g = f["data"][k]
            demos.append(dict(
                key=k,   # HDF5 group name, for in-place '#demo_key' references
                state=g["obs"]["state_lowdim"][:].astype(np.float32),
                actions=g["actions"][:].astype(np.float32),
                agent_view=g["obs"]["agent_view"][:],
                wrist=g["obs"]["wrist"][:],
                joint_pos=g["obs"]["JOINT_POS"][:],
                success=bool(g.attrs.get("success", True)),
            ))
    return demos


def split_state(state):
    """(T, D) state_lowdim -> per-key dict matching their OBS_KEYS."""
    D = state.shape[-1]
    obj = D - sum(d for _, d in OBS_TAIL)
    out = {"object": state[:, :obj]}
    i = obj
    for name, d in OBS_TAIL:
        out[name] = state[:, i:i + d]
        i += d
    return out


@click.command()
@click.option("--episodes", required=True)
@click.option("--reward_axes", default="speed_reward,peg_reward", help="THEIR reward_functions axes")
@click.option("--n_pairs", type=int, default=300)
@click.option("--n_episodes", type=int, default=None)
@click.option("--seed", type=int, default=42)
@click.option("--max_seq_len", type=int, default=512)
@click.option("--stride", type=int, default=1)
@click.option("--save_state", default=None, help="dir for demos.hdf5 + pairs.npz (state model)")
@click.option("--save_qwen", default=None, help="dir for co-located pair dirs (qwen)")
@click.option("--save_qwen_indexed", default=None,
              help="dir for the INDEXED qwen layout: writes ONLY tiny DIR/pairs/preference_*.json "
                   "manifests that reference episodes IN PLACE inside the source episodes.hdf5 "
                   "(relative path + '#demo_key'). No HDF5 is re-saved. Upload episodes.hdf5 + "
                   "DIR/pairs keeping relative positions. Train with: "
                   "--preferences_dir '' --cross_preferences_dir DIR/pairs")
@click.option("--task_prompt", default="place the square nut on the peg")
def main(episodes, reward_axes, n_pairs, n_episodes, seed, max_seq_len, stride, save_state, save_qwen, save_qwen_indexed, task_prompt):
    axes = [a.strip() for a in reward_axes.split(",")]
    demos = load_episodes(episodes, n_episodes)
    N = len(demos)
    D = demos[0]["state"].shape[-1]
    A = demos[0]["actions"].shape[-1]
    maxT = max(len(d["state"]) for d in demos)

    obs = np.zeros((N, maxT, D), np.float32)
    lengths = np.zeros(N, np.int32)
    metrics = np.zeros((N, len(axes)), np.float32)
    for i, d in enumerate(demos):
        L = len(d["state"])
        lengths[i] = L
        obs[i, :L] = d["state"]
        vals = compute_axes(axes, d["state"], actions=d["actions"])   # THEIR metrics — single oracle
        metrics[i] = [vals[a] for a in axes]

    ds = PreferencePairDataset(obs, lengths, metrics, max_seq_len=max_seq_len,
                               stride=stride, n_pairs=n_pairs, seed=seed)   # THEIR pairs+labels
    idx_a, idx_b, labels = np.asarray(ds.idx_a), np.asarray(ds.idx_b), np.asarray(ds.labels)

    print(f"Generated {len(idx_a)} pairs over axes {axes}")
    print(f"  {'axis':14s}  {'A(pos)':>7s} {'B(neg)':>7s} {'Equal':>7s}")
    for k, ax in enumerate(axes):
        a = int((labels[:, k] == 1.0).sum()); b = int((labels[:, k] == 0.0).sum()); e = int((labels[:, k] == 0.5).sum())
        print(f"  {ax:14s}  {a:7d} {b:7d} {e:7d}")
    tot_a = int((labels == 1.0).sum()); tot_b = int((labels == 0.0).sum()); tot_e = int((labels == 0.5).sum())
    print(f"  {'TOTAL':14s}  {tot_a:7d} {tot_b:7d} {tot_e:7d}   "
          f"({len(idx_a)} pairs x {len(axes)} axes = {len(idx_a) * len(axes)} labels)")

    # ---- state: their demos.hdf5 layout + the exact pairs ----
    if save_state:
        d = pathlib.Path(save_state); d.mkdir(parents=True, exist_ok=True)
        with h5py.File(d / "demos.hdf5", "w") as f:
            grp = f.create_group("data")
            for i, dm in enumerate(demos):
                og = grp.create_group(f"demo_{i}").create_group("obs")
                for key, arr in split_state(dm["state"]).items():
                    og.create_dataset(key, data=arr)
                grp[f"demo_{i}"].create_dataset("actions", data=dm["actions"])
        np.savez(d / "pairs.npz", idx_a=idx_a, idx_b=idx_b, labels=labels,
                 axes=np.array(axes, dtype=object))
        print(f"[state] -> {d}/demos.hdf5 + {d}/pairs.npz")
        print(f"        train: python reward_model/train_reward_model.py --rollout_data none "
              f"--demo_hdf5 {d}/demos.hdf5 --reward_axes {reward_axes} --load_prefs {d}/pairs.npz --output_dir <out>")

    # ---- qwen: shared helpers for both layouts ----
    if save_qwen or save_qwen_indexed:
        prompts = [AXIS_PROMPT.get(a, a) for a in axes]

        def write_rollout(p, dm):
            with h5py.File(p, "w") as f:
                og = f.create_group("data/demo_0/obs")
                og.create_dataset("agent_view", data=dm["agent_view"])
                og.create_dataset("wrist", data=dm["wrist"])
                og.create_dataset("JOINT_POS", data=dm["joint_pos"])

        def rollout_info(idx):
            """Per-rollout block for preference.json: success flag + the source
            demo index (for trajectory dedup) + the oracle axis values (for
            --value_metrics in the qwen trainer), keyed by the axis prompt."""
            return {"succeeded": demos[idx]["success"],
                    "demo_idx": int(idx),
                    "gt_metrics": {prompts[k]: float(metrics[idx, k]) for k in range(len(axes))}}

        def pair_prefs(i):
            return {prompts[k]: ("A" if labels[i, k] == 1.0 else "B" if labels[i, k] == 0.0 else "Equal")
                    for k in range(len(axes))}

    # ---- qwen (co-located): pair dirs, each with its own copies of the images ----
    if save_qwen:
        outroot = pathlib.Path(save_qwen); outroot.mkdir(parents=True, exist_ok=True)
        for i in range(len(idx_a)):
            a, b = int(idx_a[i]), int(idx_b[i])
            pd = outroot / f"pair_{i:05d}"; pd.mkdir(exist_ok=True)
            write_rollout(pd / "rollout_A.hdf5", demos[a])
            write_rollout(pd / "rollout_B.hdf5", demos[b])
            json.dump(dict(preferences=pair_prefs(i),
                           rollout_A=rollout_info(a),
                           rollout_B=rollout_info(b),
                           instruction=task_prompt, session_timestamp=f"pair_{i:05d}"),
                      open(pd / "preference.json", "w"), indent=2)
        print(f"[qwen]  -> {outroot} ({len(idx_a)} pair dirs)")
        print(f"        train (qwen_rl): python train_reward_model.py --model qwen_open --use_lora "
              f"--task auto --preferences_dir {outroot.resolve()} --epochs 30 --batch_size 1")

    # ---- qwen (indexed): manifests ONLY — episodes referenced in place ----
    # No HDF5 is written at all: each manifest points into the source
    # episodes.hdf5 with a '#demo_key' fragment. Upload episodes.hdf5 + this
    # pairs/ dir, keeping their relative positions (paths are relative).
    if save_qwen_indexed:
        pairs_dir = pathlib.Path(save_qwen_indexed) / "pairs"
        pairs_dir.mkdir(parents=True, exist_ok=True)
        ep_rel = os.path.relpath(os.path.abspath(episodes), start=str(pairs_dir))

        def ref(idx):
            return f"{ep_rel}#{demos[idx]['key']}"

        for i in range(len(idx_a)):
            a, b = int(idx_a[i]), int(idx_b[i])
            json.dump(dict(preferences=pair_prefs(i),
                           rollout_A_id=ref(a),
                           rollout_B_id=ref(b),
                           rollout_A=rollout_info(a),
                           rollout_B=rollout_info(b),
                           instruction=task_prompt, session_timestamp=f"pair_{i:05d}"),
                      open(pairs_dir / f"preference_{i:05d}.json", "w"), indent=2)
        print(f"[qwen_indexed] -> {pairs_dir} ({len(idx_a)} pair manifests; episodes "
              f"referenced in place as '{ep_rel}#<demo>' — no HDF5 re-saved)")
        print(f"        train (qwen_rl): python train_reward_model.py --model qwen_open --use_lora "
              f"--task auto --preferences_dir '' --cross_preferences_dir {pairs_dir.resolve()} "
              f"--epochs 30 --batch_size 1   # split is ordered automatically for generated manifests")


if __name__ == "__main__":
    main()
