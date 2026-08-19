"""
Train a state-space reward model on rollout data using Bradley-Terry preferences.

After training, scores all rollouts + original demos and saves z-score normalized scores.

Usage:
  python reward_model/train_reward_model.py \
    --rollout_data rollouts.npz \
    --demo_hdf5 data/robomimic/datasets/square/mh/low_dim.hdf5 \
    --output_dir reward_model_output
"""

import sys
import os
import pathlib

# Add repo root and reward_model dir to path
ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, str(pathlib.Path(__file__).parent))
os.chdir(ROOT_DIR)

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import json
import click
import torch
import numpy as np
import h5py
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from state_reward_model import StateRewardModel, bradley_terry_loss
from reward_functions import AXIS_FUNCTIONS, ACTION_DEPENDENT_AXES, compute_axes
from preferences import sample_pair_indices, preference_pair_labels


def stride_indices(episode_len: int, max_seq_len: int, stride: int = 1) -> np.ndarray:
    """Return at most `max_seq_len` indices into [0, episode_len), spaced by `stride`.

    stride=1 (default): indices 0,1,...,min(L,max_seq_len)-1 — same prefix-only
    behavior as before; long episodes get truncated at max_seq_len.
    stride>1: indices 0,stride,2*stride,... capped to max_seq_len entries; the
    reward model then sees the whole trajectory at lower temporal resolution.
    """
    L = int(episode_len)
    if L <= 0:
        return np.zeros(0, dtype=np.int64)
    stride = max(int(stride), 1)
    if stride == 1:
        return np.arange(min(L, max_seq_len), dtype=np.int64)
    idx = np.arange(0, L, stride, dtype=np.int64)
    return idx[:max_seq_len]


class PreferencePairDataset(Dataset):
    """Generate preference pairs from rollout metrics."""

    def __init__(self, obs, episode_lengths, metrics, max_seq_len=512, stride=1,
                 n_pairs=None, seed=42):
        """
        Args:
            obs: (N, T, D) padded observations
            episode_lengths: (N,) actual lengths
            metrics: (N, K) ground truth metric values per episode
            max_seq_len: truncate sequences to this length
            stride: 1 = take consecutive prefix (default), >1 = stride-subsample
                    so the whole trajectory is seen at lower temporal resolution.
            n_pairs: number of preference pairs to generate.
                     None (default) = all unique pairs N*(N-1)/2.
                     An integer = randomly sample that many pairs.
        """
        self.obs = obs
        self.episode_lengths = episode_lengths
        self.metrics = metrics
        self.max_seq_len = max_seq_len
        self.stride = int(stride)

        # Sampling + labeling live in reward_functions so the qwen trainer's
        # on-the-fly --episodes mode draws the IDENTICAL pairs from the same seed.
        N = len(obs)
        idx_a, idx_b = sample_pair_indices(N, n_pairs, seed)
        labels = preference_pair_labels(metrics, idx_a, idx_b)

        self.idx_a = idx_a
        self.idx_b = idx_b
        self.labels = labels

    def __len__(self):
        return len(self.idx_a)

    def __getitem__(self, idx):
        ia, ib = self.idx_a[idx], self.idx_b[idx]
        L = self.max_seq_len

        idx_a_steps = stride_indices(int(self.episode_lengths[ia]), L, self.stride)
        idx_b_steps = stride_indices(int(self.episode_lengths[ib]), L, self.stride)

        obs_a = np.zeros((L, self.obs.shape[-1]), dtype=np.float32)
        obs_b = np.zeros((L, self.obs.shape[-1]), dtype=np.float32)
        mask_a = np.ones(L, dtype=bool)  # True = padded
        mask_b = np.ones(L, dtype=bool)

        obs_a[:len(idx_a_steps)] = self.obs[ia, idx_a_steps]
        obs_b[:len(idx_b_steps)] = self.obs[ib, idx_b_steps]
        mask_a[:len(idx_a_steps)] = False
        mask_b[:len(idx_b_steps)] = False

        return {
            'obs_a': obs_a,
            'obs_b': obs_b,
            'mask_a': mask_a,
            'mask_b': mask_b,
            'labels': self.labels[idx],
        }


def load_demo_obs(demo_hdf5, obs_keys, max_demos=None):
    """Load obs from demo HDF5, return list of (T, D) arrays."""
    episodes = []
    with h5py.File(demo_hdf5, 'r') as f:
        demos = f['data']
        n_demos = len(demos)
        if max_demos is not None:
            n_demos = min(n_demos, max_demos)
        for i in range(n_demos):
            demo = demos[f'demo_{i}']
            obs_parts = [demo['obs'][key][:].astype(np.float32) for key in obs_keys]
            obs = np.concatenate(obs_parts, axis=-1)
            episodes.append(obs)
    return episodes


@click.command()
@click.option('--rollout_data', required=True, help='Path(s) to .npz from collect_rollouts.py (comma-separated for multiple files)')
@click.option('--demo_hdf5', required=True, help='Path to demo HDF5 for scoring original demos')
@click.option('--output_dir', required=True)
@click.option('--obs_keys', default='object,robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos')
@click.option('--epochs', default=100, type=int)
@click.option('--batch_size', default=64, type=int)
@click.option('--lr', default=1e-4, type=float)
@click.option('--n_pairs', default=None, type=int, help='Number of preference pairs. Default: all unique pairs N*(N-1)/2')
@click.option('--max_seq_len', default=512, type=int)
@click.option('--stride', default=1, type=int,
              help='1=prefix only (default, preserves prior behavior); >1 stride-subsamples so the whole trajectory fits in max_seq_len.')
@click.option('--max_demos', default=None, type=int, help='Max original demos to include')
@click.option('--device', default='cuda:0')
@click.option('--wandb_project', default='reward_cond_pipeline', help='wandb project name')
@click.option('--wandb_run_name', default='phase2_reward_model', help='wandb run name (set per-task so runs are distinguishable)')
@click.option('--reward_axes', default=None,
              help='Comma-separated reward axes to use. Any combination of: success,speed_reward,smoothness,peg_reward,order_reward,milk_placed,bread_placed,cereal_placed,can_placed,drop_reward,composite(...)')
@click.option('--load_prefs', default=None,
              help='Path to pairs.npz from generate_preferences.py — train on those exact '
                   '(idx_a, idx_b, labels) instead of sampling fresh, so the state model uses '
                   'the SAME pairs as the Qwen model. idx index into demos in demo_0..N order.')
@click.option('--max_train_prefs', default=None, type=int,
              help='Data-efficiency sweep (only with --load_prefs): cap TRAINING pairs to the '
                   'first N after the fixed val holdout. Subsets are nested (first-10 ⊂ first-50 ⊂ …) '
                   'and every N evaluates on the SAME val set. Default: use all remaining pairs.')
@click.option('--eval_prefs', default=None,
              help='OPTIONAL out-of-band eval set: path to a second pairs.npz (e.g. from --middle '
                   'demos). Win rate on it is logged each epoch as reward_model/val_acc_* — the '
                   '"unseen" range, alongside the in-band held-out val_acc ("seen" range). '
                   'Requires --eval_demo_hdf5.')
@click.option('--eval_demo_hdf5', default=None,
              help='demos.hdf5 the --eval_prefs idx index into (the middle/unseen demo set). '
                   'Can also be given WITHOUT --eval_prefs: then no oos pair accuracy is '
                   'computed, but --value_metrics still gets its test set.')
@click.option('--val_ratio', default=0.1, type=float,
              help='Fraction of the loaded pairs held out for the in-band "seen" val set. '
                   'e.g. 0.5 on a 200-pair set → 100 val / 100 train pool. Default 0.1.')
@click.option('--value_metrics', is_flag=True,
              help='Also log per-trajectory reward-VALUE fidelity (not just pref accuracy): '
                   'Spearman rho + normalized MSE + a predicted-vs-GT scatter, computed on the '
                   'FULL train demo set and (if --eval_demo_hdf5 given) the test demo set, each '
                   'separately. MSE is z-scored on a SHARED scale pooled across both sets, so '
                   'test/middle values are comparable to the training clusters (interpolation vs '
                   'collapse). Same every-10-epoch cadence as the distribution plots.')
def main(rollout_data, demo_hdf5, output_dir, obs_keys, epochs, batch_size, lr,
         n_pairs, max_seq_len, stride, max_demos, device, wandb_project, wandb_run_name, reward_axes,
         load_prefs, max_train_prefs, eval_prefs, eval_demo_hdf5, val_ratio, value_metrics):
    os.makedirs(output_dir, exist_ok=True)
    obs_keys = obs_keys.split(',')
    device = torch.device(device)

    # Init wandb
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            'rollout_data': rollout_data,
            'demo_hdf5': demo_hdf5,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'n_pairs': n_pairs if n_pairs is not None else 'all',
            'max_seq_len': max_seq_len,
        },
    )

    # Load rollout data (skip if path is "none" — demos-only mode)
    # Supports comma-separated list of npz files
    rollout_paths = [p.strip() for p in rollout_data.split(',') if p.strip() and p.strip() != 'none']
    has_rollouts = len(rollout_paths) > 0
    # --------------------------------------------------------------------- #
    # Load obs/actions/lengths from rollouts + demos. Axis values are computed
    # at the end via reward_functions.AXIS_FUNCTIONS — one source of truth.
    # --------------------------------------------------------------------- #
    if has_rollouts:
        all_obs_chunks, all_action_chunks, all_lengths_chunks = [], [], []
        all_reward_chunks = []   # per-step env rewards (for axis fns that need them)
        for rpath in rollout_paths:
            data = np.load(rpath)
            n = len(data['episode_lengths'])
            all_obs_chunks.append(data['obs'][:n])
            all_lengths_chunks.append(data['episode_lengths'])
            if 'actions' in data:
                all_action_chunks.append(data['actions'][:n])
            else:
                # Some legacy npz files don't store actions — substitute zeros
                # so the smoothness axis just returns 0 for those trajectories.
                all_action_chunks.append(np.zeros((n, data['obs'].shape[1], 1), dtype=np.float32))
            if 'rewards' in data:
                all_reward_chunks.append(data['rewards'][:n])
            else:
                all_reward_chunks.append(None)
            print(f"  Loaded {n} rollouts from {rpath}")

        # Pad obs/actions to a common max T before concatenating.
        max_t = max(a.shape[1] for a in all_obs_chunks)
        obs_dim = all_obs_chunks[0].shape[-1]
        act_dim = all_action_chunks[0].shape[-1] if all_action_chunks[0] is not None else 1
        for i, obs_arr in enumerate(all_obs_chunks):
            if obs_arr.shape[1] < max_t:
                pad = np.zeros((obs_arr.shape[0], max_t - obs_arr.shape[1], obs_dim), dtype=np.float32)
                all_obs_chunks[i] = np.concatenate([obs_arr, pad], axis=1)
        for i, act_arr in enumerate(all_action_chunks):
            if act_arr.shape[1] < max_t:
                pad = np.zeros((act_arr.shape[0], max_t - act_arr.shape[1], act_arr.shape[-1]), dtype=np.float32)
                all_action_chunks[i] = np.concatenate([act_arr, pad], axis=1)

        rollout_obs = np.concatenate(all_obs_chunks, axis=0)
        rollout_actions = np.concatenate(all_action_chunks, axis=0)
        rollout_lengths = np.concatenate(all_lengths_chunks)
        n_rollouts = len(rollout_obs)
        print(f"Total: {n_rollouts} rollouts from {len(rollout_paths)} file(s), obs_dim={obs_dim}")
    else:
        n_rollouts = 0
        rollout_obs = None
        rollout_actions = None
        rollout_lengths = None
        obs_dim = None
        print("No rollout data — training on demos only.")

    # Load demo trajectories from HDF5.
    demo_episodes = load_demo_obs(demo_hdf5, obs_keys, max_demos=max_demos)
    n_demos = len(demo_episodes)
    demo_actions_list = []
    demo_lengths_list = []
    with h5py.File(demo_hdf5, 'r') as f:
        demos_group = f['data']
        for i in range(n_demos):
            demo = demos_group[f'demo_{i}']
            actions = demo['actions'][:].astype(np.float32)
            demo_actions_list.append(actions)
            demo_lengths_list.append(len(demo_episodes[i]))

    # Infer obs_dim from demos if no rollouts
    if obs_dim is None:
        obs_dim = demo_episodes[0].shape[-1]

    # Pad demo obs/actions to same T as rollouts. max_T = max over the
    # longest trajectory in any of: rollout obs, rollout actions, demos.
    # Rollout obs and actions can have different padded T across iteration-
    # rollout files (one file's actions may be longer than another file's
    # obs), so we need to size the combined array to the absolute longest.
    max_demo_len = max(demo_lengths_list) if demo_lengths_list else 0
    if has_rollouts:
        max_T = max(rollout_obs.shape[1], rollout_actions.shape[1], max_demo_len)
    else:
        max_T = max_demo_len
    act_dim_demo = demo_actions_list[0].shape[-1] if demo_actions_list else (
        rollout_actions.shape[-1] if has_rollouts else 1)

    demo_obs_padded = np.zeros((n_demos, max_T, obs_dim), dtype=np.float32)
    demo_actions_padded = np.zeros((n_demos, max_T, act_dim_demo), dtype=np.float32)
    for i, ep in enumerate(demo_episodes):
        demo_obs_padded[i, :len(ep)] = ep
        a = demo_actions_list[i]
        demo_actions_padded[i, :len(a), :a.shape[-1]] = a
    demo_lengths = np.array(demo_lengths_list, dtype=np.int32)

    # Pad rollout obs/actions to max_T if either is shorter. (Either dimension
    # could be the shorter one — obs typically T+1 vs action T, plus per-file
    # padding can leave them inconsistent across files.)
    if has_rollouts and rollout_obs.shape[1] < max_T:
        new_obs = np.zeros((n_rollouts, max_T, obs_dim), dtype=np.float32)
        new_obs[:, :rollout_obs.shape[1]] = rollout_obs
        rollout_obs = new_obs
    if has_rollouts and rollout_actions.shape[1] < max_T:
        new_act = np.zeros((n_rollouts, max_T, rollout_actions.shape[-1]), dtype=np.float32)
        new_act[:, :rollout_actions.shape[1]] = rollout_actions
        rollout_actions = new_act

    print(f"Loaded {n_demos} demos")

    # Concatenate rollouts + demos. Both halves go through the same axis
    # computation below.
    if has_rollouts:
        all_obs = np.concatenate([rollout_obs, demo_obs_padded], axis=0)
        all_actions_arr = np.concatenate([rollout_actions, demo_actions_padded], axis=0) \
            if rollout_actions.shape[-1] == demo_actions_padded.shape[-1] else None
        all_lengths = np.concatenate([rollout_lengths, demo_lengths], axis=0)
    else:
        all_obs = demo_obs_padded
        all_actions_arr = demo_actions_padded
        all_lengths = demo_lengths

    # --------------------------------------------------------------------- #
    # Select axes and compute their values per trajectory by calling the
    # functions in reward_model/reward_functions.py.
    # --------------------------------------------------------------------- #
    import re
    if reward_axes is not None:
        requested_axes = [a.strip() for a in reward_axes.split(',')]
    else:
        # Default: success + speed + smoothness only; the user opts into the
        # task-specific axes via --reward_axes.
        requested_axes = ['success', 'speed_reward', 'smoothness']

    # Expand composite(...) entries — collect the unique base axes we need to
    # compute, then average them back together as composites.
    base_axes_needed = set()
    expanded = []   # list of (kind, payload) — kind in {'plain', 'composite'}
    for ax in requested_axes:
        m = re.match(r'^composite\((.+)\)$', ax)
        if m:
            sub = [s.strip() for s in m.group(1).split('+')]
            base_axes_needed.update(sub)
            expanded.append(('composite', sub))
        else:
            base_axes_needed.add(ax)
            expanded.append(('plain', ax))
    unknown = [a for a in base_axes_needed if a not in AXIS_FUNCTIONS]
    if unknown:
        raise ValueError(f"Unknown reward axis(es): {unknown}. Available: {list(AXIS_FUNCTIONS.keys())}")
    _needs_actions = sorted(base_axes_needed & ACTION_DEPENDENT_AXES)
    if _needs_actions and all_actions_arr is None:
        raise ValueError(
            f"Axes {_needs_actions} require actions, but actions are unavailable "
            f"(rollout action dim != demo action dim, so they were not concatenated). "
            f"Their GT values would be silently wrong — refusing to continue.")

    # Compute base axis values per trajectory. Trim padded obs/actions down
    # to the actual episode length BEFORE handing them to the reward fns —
    # the fns operate on whole-trajectory data and don't accept a length arg.
    print(f"Computing per-axis rewards for {len(all_obs)} trajectories "
          f"over {len(base_axes_needed)} base axes: {sorted(base_axes_needed)}")
    base_axis_values = {name: np.zeros(len(all_obs), dtype=np.float32) for name in base_axes_needed}
    for i in range(len(all_obs)):
        L_i = int(all_lengths[i])
        obs_i = all_obs[i][:L_i]
        act_i = all_actions_arr[i][:L_i] if all_actions_arr is not None else None
        for name in base_axes_needed:
            base_axis_values[name][i] = AXIS_FUNCTIONS[name](obs_i, actions=act_i)

    # Stack into (N, K) metrics in the requested order (with composites).
    reward_names = []
    metric_cols = []
    for kind, payload in expanded:
        if kind == 'plain':
            reward_names.append(payload)
            metric_cols.append(base_axis_values[payload])
        else:
            sub = payload
            reward_names.append('composite(' + '+'.join(sub) + ')')
            metric_cols.append(sum(base_axis_values[s] for s in sub) / len(sub))
    metrics = np.stack(metric_cols, axis=-1)  # (N_rollouts + N_demos, K)

    def _gt_metrics(obs_arr, lengths_arr, actions_arr=None):
        """Oracle axis values (same axes/order as `metrics`) for an arbitrary demo
        set — used for the --value_metrics per-trajectory comparison on the eval set."""
        base = {name: np.zeros(len(obs_arr), dtype=np.float32) for name in base_axes_needed}
        for i in range(len(obs_arr)):
            L_i = int(lengths_arr[i]); oi = obs_arr[i][:L_i]
            ai = actions_arr[i][:L_i] if actions_arr is not None else None
            for name in base_axes_needed:
                base[name][i] = AXIS_FUNCTIONS[name](oi, actions=ai)
        cols = [base[p] if kind == 'plain' else sum(base[s] for s in p) / len(p)
                for kind, p in expanded]
        return np.stack(cols, axis=-1)

    # Convenience pulls used downstream by the histogram plot.
    def _compute_or_default(name):
        if name in base_axis_values:
            return base_axis_values[name]
        vals = np.zeros(len(all_obs), dtype=np.float32)
        for i in range(len(all_obs)):
            L_i = int(all_lengths[i])
            obs_i = all_obs[i][:L_i]
            act_i = all_actions_arr[i][:L_i] if all_actions_arr is not None else None
            vals[i] = AXIS_FUNCTIONS[name](obs_i, actions=act_i)
        return vals

    all_success = _compute_or_default('success')
    all_speed = _compute_or_default('speed_reward')
    all_smoothness = _compute_or_default('smoothness')
    print(f"  Success rate: {all_success.mean():.3f}  "
          f"Mean speed: {all_speed.mean():.3f}  Mean smooth: {all_smoothness.mean():.3f}")

    num_rewards = metrics.shape[1]
    print(f"Training reward model on {len(all_obs)} episodes ({n_rollouts} rollouts + {n_demos} demos), num_rewards={num_rewards}")

    # Log ground truth metric distributions (rollouts vs demos)
    rollout_metrics = metrics[:n_rollouts]
    demo_metrics = metrics[n_rollouts:]
    fig, axes = plt.subplots(1, num_rewards, figsize=(4 * num_rewards, 3), squeeze=False)
    for k, name in enumerate(reward_names):
        ax = axes[0, k]
        ax.hist(rollout_metrics[:, k], bins=30, alpha=0.6, label='rollouts', edgecolor='black')
        ax.hist(demo_metrics[:, k], bins=30, alpha=0.6, label='demos', edgecolor='black')
        ax.set_title(f'{name} (ground truth)')
        ax.set_xlabel('value')
        ax.set_ylabel('count')
        ax.legend(fontsize=8)
    fig.suptitle('Ground Truth Metric Distributions (Rollouts + Demos)')
    fig.tight_layout()
    wandb.log({'reward_model/gt_distributions': wandb.Image(fig)})
    plt.close(fig)

    # Create train/val datasets (preferences across rollouts AND demos).
    # val_ratio is the --val_ratio CLI option (fraction held out for the
    # in-band "seen" val set).
    if load_prefs is not None:
        # Train on the EXACT pairs generate_preferences.py produced (the same
        # pairs the Qwen model gets). idx_a/idx_b index into all_obs in
        # demo_0..N order, so run with --rollout_data none so all_obs == demos.
        _p = np.load(load_prefs, allow_pickle=True)
        _ia, _ib, _lab = _p['idx_a'], _p['idx_b'], _p['labels']
        _nv = min(max(int(len(_ia) * val_ratio), 1), len(_ia) - 1)

        # Val = first _nv pairs (FIXED). Train = the pairs after it. For a
        # data-efficiency sweep, --max_train_prefs caps the train slice to the
        # next N — a prefix, so subsets are nested and val is identical across N.
        _train_sl = slice(_nv, None)
        if max_train_prefs is not None:
            _train_sl = slice(_nv, _nv + max_train_prefs)

        def _saved_ds(sl, sd):
            _ds = PreferencePairDataset(all_obs, all_lengths, metrics,
                                        max_seq_len=max_seq_len, stride=stride, n_pairs=1, seed=sd)
            _ds.idx_a, _ds.idx_b, _ds.labels = _ia[sl], _ib[sl], _lab[sl]
            return _ds

        train_dataset = _saved_ds(_train_sl, 42)
        val_dataset = _saved_ds(slice(0, _nv), 123)
        _n_train = len(_ia[_train_sl])
        print(f"  [load_prefs] {len(_ia)} saved pairs from {load_prefs}; "
              f"heldout={_nv} (fixed), train={_n_train}"
              + (f" [capped at --max_train_prefs {max_train_prefs}]"
                 if max_train_prefs is not None else " [all remaining]"))
        if max_train_prefs is not None and _n_train < max_train_prefs:
            print(f"  [load_prefs] WARNING: only {_n_train} train pairs available "
                  f"(< requested {max_train_prefs}); using all of them.")
    else:
        # Fresh-sampled pairs (no --load_prefs): compute how many train/val
        # pairs to draw. These counts only apply to this branch.
        N_total = len(all_obs)
        all_unique_pairs = N_total * (N_total - 1) // 2
        effective_n_pairs = n_pairs if n_pairs is not None else all_unique_pairs
        n_val_pairs = max(int(effective_n_pairs * val_ratio), min(100, effective_n_pairs))
        # Ensure we don't allocate all pairs to validation
        n_val_pairs = min(n_val_pairs, int(effective_n_pairs * 0.5))
        n_val_pairs = max(n_val_pairs, 1)
        n_train_pairs = effective_n_pairs - n_val_pairs
        print(f"  Preference pairs: {effective_n_pairs} total ({n_train_pairs} train, {n_val_pairs} val)"
              + (f" [all unique pairs]" if n_pairs is None else f" [specified]"))
        train_dataset = PreferencePairDataset(
            all_obs, all_lengths, metrics,
            max_seq_len=max_seq_len, stride=stride, n_pairs=n_train_pairs, seed=42)
        val_dataset = PreferencePairDataset(
            all_obs, all_lengths, metrics,
            max_seq_len=max_seq_len, stride=stride, n_pairs=n_val_pairs, seed=123)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Optional out-of-band ("unseen") eval set — a SEPARATE demo file + pairs.npz
    # (e.g. from --middle collection). Scored by the same model each epoch so we
    # can compare in-band val ("seen" range) vs. this middle-range win rate.
    eval_dataloader = None
    _ev_obs = _ev_len = _ev_gt = None   # eval demo set for --value_metrics
    if eval_prefs is not None and eval_demo_hdf5 is None:
        raise click.UsageError("--eval_prefs requires --eval_demo_hdf5")
    if eval_demo_hdf5 is not None:
        _ev_eps = load_demo_obs(eval_demo_hdf5, obs_keys)
        if _ev_eps[0].shape[-1] != obs_dim:
            raise ValueError(f"eval demos obs_dim {_ev_eps[0].shape[-1]} != train obs_dim {obs_dim} "
                             f"(check --obs_keys / that the eval set was collected the same way)")
        _ev_len = np.array([len(e) for e in _ev_eps], dtype=np.int32)
        _ev_obs = np.zeros((len(_ev_eps), int(_ev_len.max()), obs_dim), dtype=np.float32)
        for i, e in enumerate(_ev_eps):
            _ev_obs[i, :len(e)] = e
        if eval_prefs is not None:
            _pe = np.load(eval_prefs, allow_pickle=True)
            eval_ds = PreferencePairDataset(_ev_obs, _ev_len,
                                            np.zeros((len(_ev_eps), num_rewards), dtype=np.float32),
                                            max_seq_len=max_seq_len, stride=stride, n_pairs=1, seed=7)
            eval_ds.idx_a, eval_ds.idx_b, eval_ds.labels = _pe['idx_a'], _pe['idx_b'], _pe['labels']
            eval_dataloader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=2)
            print(f"  [eval_prefs] out-of-band eval: {len(_pe['idx_a'])} pairs over "
                  f"{len(_ev_eps)} demos from {eval_prefs}")
        if value_metrics:
            # Eval GT must be computed with the same inputs as train GT: load the
            # eval demos' actions so action-dependent axes (smoothness) don't get
            # their actions-None default on the test side only.
            _ev_act = None
            with h5py.File(eval_demo_hdf5, 'r') as f:
                _dg = f['data']
                if 'actions' in _dg['demo_0']:
                    _acts = [_dg[f'demo_{i}']['actions'][:].astype(np.float32)
                             for i in range(len(_ev_eps))]
                    _amax = max(_ev_obs.shape[1], max(len(a) for a in _acts))
                    _ev_act = np.zeros((len(_ev_eps), _amax, _acts[0].shape[-1]), dtype=np.float32)
                    for i, a in enumerate(_acts):
                        _ev_act[i, :len(a)] = a
            _ev_needs_act = sorted(base_axes_needed & ACTION_DEPENDENT_AXES)
            if _ev_needs_act and _ev_act is None:
                raise ValueError(
                    f"--value_metrics: axes {_ev_needs_act} require actions, but "
                    f"{eval_demo_hdf5} has no 'actions' dataset — eval GT would be "
                    f"silently wrong. Regenerate the eval demos.hdf5 with actions.")
            _ev_gt = _gt_metrics(_ev_obs, _ev_len, _ev_act)   # oracle axis values per eval trajectory
            print(f"  [value_metrics] test set = {len(_ev_obs)} eval demos "
                  f"(GT computed{', with actions' if _ev_act is not None else ''})")

    # Create model
    model = StateRewardModel(
        obs_dim=obs_dim,
        num_rewards=num_rewards,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        ffn_dim=512,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Train
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_acc = np.zeros(num_rewards)
        n_batches = 0

        for batch in dataloader:
            obs_a = batch['obs_a'].to(device)
            obs_b = batch['obs_b'].to(device)
            mask_a = batch['mask_a'].to(device)
            mask_b = batch['mask_b'].to(device)
            labels = batch['labels'].to(device)

            rewards_a = model(obs_a, mask_a)
            rewards_b = model(obs_b, mask_b)

            loss, acc = bradley_terry_loss(rewards_a, rewards_b, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc.cpu().numpy()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_acc = total_acc / n_batches

        # Validation
        model.eval()
        val_total_loss = 0.0
        val_total_acc = np.zeros(num_rewards)
        val_n_batches = 0
        with torch.no_grad():
            for batch in val_dataloader:
                obs_a = batch['obs_a'].to(device)
                obs_b = batch['obs_b'].to(device)
                mask_a = batch['mask_a'].to(device)
                mask_b = batch['mask_b'].to(device)
                labels = batch['labels'].to(device)

                rewards_a = model(obs_a, mask_a)
                rewards_b = model(obs_b, mask_b)
                loss_val, acc_val = bradley_terry_loss(rewards_a, rewards_b, labels)

                val_total_loss += loss_val.item()
                val_total_acc += acc_val.cpu().numpy()
                val_n_batches += 1

        val_avg_loss = val_total_loss / val_n_batches
        val_avg_acc = val_total_acc / val_n_batches

        # Out-of-band ("unseen" middle range) win rate, if an eval set was given.
        # Same forward + BT-accuracy as the val loop above, on the separate set.
        oos_avg_acc = None
        if eval_dataloader is not None:
            oos_total_acc = np.zeros(num_rewards)
            oos_n_batches = 0
            with torch.no_grad():
                for batch in eval_dataloader:
                    ra = model(batch['obs_a'].to(device), batch['mask_a'].to(device))
                    rb = model(batch['obs_b'].to(device), batch['mask_b'].to(device))
                    _, acc_oos = bradley_terry_loss(ra, rb, batch['labels'].to(device))
                    oos_total_acc += acc_oos.cpu().numpy()
                    oos_n_batches += 1
            oos_avg_acc = oos_total_acc / max(oos_n_batches, 1)

        # Log to wandb every epoch
        log_dict = {
            'reward_model/train_loss': avg_loss,
            'reward_model/train_acc_mean': avg_acc.mean(),
            'reward_model/train_heldout_loss': val_avg_loss,
            'reward_model/train_heldout_acc_mean': val_avg_acc.mean(),
            'reward_model/epoch': epoch + 1,
        }
        if oos_avg_acc is not None:
            log_dict['reward_model/val_acc_mean'] = oos_avg_acc.mean()
        for k, name in enumerate(reward_names):
            log_dict[f'reward_model/train_acc_{name}'] = avg_acc[k]
            log_dict[f'reward_model/train_heldout_acc_{name}'] = val_avg_acc[k]
            if oos_avg_acc is not None:
                log_dict[f'reward_model/val_acc_{name}'] = oos_avg_acc[k]
        # Log predicted score distributions every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == epochs:
            pred_scores = score_episodes(model, all_obs, all_lengths, max_seq_len, device, stride=stride)
            fig, axes = plt.subplots(1, num_rewards, figsize=(4 * num_rewards, 3), squeeze=False)
            for k, name in enumerate(reward_names):
                ax = axes[0, k]
                ax.hist(pred_scores[:, k], bins=30, alpha=0.7, edgecolor='black')
                ax.set_title(f'{name} (predicted)')
                ax.set_xlabel('score')
                ax.set_ylabel('count')
            fig.suptitle(f'Predicted Score Distributions (epoch {epoch+1})')
            fig.tight_layout()
            log_dict['reward_model/pred_distributions'] = wandb.Image(fig)
            plt.close(fig)

            # ---- per-trajectory reward-VALUE fidelity (predicted vs oracle GT) ----
            if value_metrics:
                # train predicted = pred_scores (just computed); train GT = metrics.
                vsets = [("train", pred_scores, metrics)]
                if _ev_gt is not None:
                    pred_te = score_episodes(model, _ev_obs, _ev_len, max_seq_len, device, stride=stride)
                    vsets.append(("val", pred_te, _ev_gt))
                # SHARED z-score stats pooled across all sets (per axis), so test
                # values sit on the same scale as train (interpolation vs collapse).
                pool_p = np.concatenate([s[1] for s in vsets], axis=0)
                pool_g = np.concatenate([s[2] for s in vsets], axis=0)
                p_mu, p_sd = pool_p.mean(0), pool_p.std(0) + 1e-8
                g_mu, g_sd = pool_g.mean(0), pool_g.std(0) + 1e-8
                for sname, pred, gt in vsets:
                    pz = (pred - p_mu) / p_sd #standardization as done in fpl
                    gz = (gt - g_mu) / g_sd
                    figv, axv = plt.subplots(1, num_rewards, figsize=(4 * num_rewards, 4), squeeze=False)
                    for k, name in enumerate(reward_names):
                        rho = _spearman(pred[:, k], gt[:, k])   # rank-based → normalization-invariant
                        mse = float(np.mean((pz[:, k] - gz[:, k]) ** 2))  # shared-scale value error
                        log_dict[f'reward_value/{sname}_spearman_{name}'] = rho
                        log_dict[f'reward_value/{sname}_mse_{name}'] = mse
                        ax = axv[0, k]
                        ax.scatter(gz[:, k], pz[:, k], s=10, alpha=0.5, edgecolor='none')
                        lo = float(min(gz[:, k].min(), pz[:, k].min()))
                        hi = float(max(gz[:, k].max(), pz[:, k].max()))
                        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.6)  # y=x reference
                        ax.set_title(f'{name}  ρ={rho:.2f}  mse={mse:.2f}')
                        ax.set_xlabel('GT (shared-norm)'); ax.set_ylabel('predicted (shared-norm)')
                    figv.suptitle(f'{sname}: predicted vs GT reward (epoch {epoch + 1})')
                    figv.tight_layout()
                    log_dict[f'reward_value/scatter_{sname}'] = wandb.Image(figv)
                    plt.close(figv)

        wandb.log(log_dict, step=epoch + 1)

        acc_str = ', '.join(f'{avg_acc[k]:.3f}' for k in range(num_rewards))
        val_acc_str = ', '.join(f'{val_avg_acc[k]:.3f}' for k in range(num_rewards))
        oos_str = ('  val_acc=[' + ', '.join(f'{oos_avg_acc[k]:.3f}' for k in range(num_rewards)) + ']') \
            if oos_avg_acc is not None else ''
        print(f"Epoch {epoch+1}/{epochs}  train_loss={avg_loss:.4f}  train_acc=[{acc_str}]  heldout_loss={val_avg_loss:.4f}  heldout_acc=[{val_acc_str}]{oos_str}")

    # Save model
    ckpt_path = os.path.join(output_dir, 'reward_model.pt')
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved reward model to {ckpt_path}")

    # Score rollouts and demos separately
    model.eval()
    if has_rollouts:
        rollout_scores = score_episodes(model, rollout_obs, rollout_lengths, max_seq_len, device, stride=stride)
    else:
        rollout_scores = np.zeros((0, num_rewards), dtype=np.float32)
    demo_scores = score_episodes(model, demo_obs_padded, demo_lengths, max_seq_len, device, stride=stride)

    # Normalize scores to [-1, 1] using min/max
    all_scores = np.concatenate([rollout_scores, demo_scores], axis=0)
    score_min = all_scores.min(axis=0)  # (K,)
    score_max = all_scores.max(axis=0)  # (K,)
    score_range = score_max - score_min
    score_range[score_range < 1e-8] = 1.0  # avoid division by zero

    rollout_z = 2.0 * (rollout_scores - score_min) / score_range - 1.0 if has_rollouts else np.zeros((0, num_rewards), dtype=np.float32)
    demo_z = 2.0 * (demo_scores - score_min) / score_range - 1.0

    # Save scores
    scores = {
        'score_min': score_min.tolist(),
        'score_max': score_max.tolist(),
        'reward_names': reward_names,
        'rollout_scores_raw': rollout_scores.tolist(),
        'rollout_scores_zscore': rollout_z.tolist(),
        'demo_scores_raw': demo_scores.tolist(),
        'demo_scores_zscore': demo_z.tolist(),
        'n_rollouts': len(rollout_scores),
        'n_demos': len(demo_scores),
    }
    scores_path = os.path.join(output_dir, 'scores.json')
    with open(scores_path, 'w') as f:
        json.dump(scores, f, indent=2)

    print(f"\nScoring complete:")
    print(f"  Reward dims: {reward_names}")
    print(f"  Score min: {score_min}")
    print(f"  Score max: {score_max}")
    if has_rollouts:
        print(f"  Rollout normalized range: [{rollout_z.min(axis=0)}, {rollout_z.max(axis=0)}]")
    print(f"  Demo normalized range:    [{demo_z.min(axis=0)}, {demo_z.max(axis=0)}]")
    print(f"  Saved scores to {scores_path}")

    # Log scoring summary to wandb
    for k, name in enumerate(reward_names):
        wandb.summary[f'scoring/score_min_{name}'] = float(score_min[k])
        wandb.summary[f'scoring/score_max_{name}'] = float(score_max[k])
        if has_rollouts:
            wandb.summary[f'scoring/rollout_norm_min_{name}'] = float(rollout_z[:, k].min())
            wandb.summary[f'scoring/rollout_norm_max_{name}'] = float(rollout_z[:, k].max())
        if len(demo_scores) > 0:
            wandb.summary[f'scoring/demo_norm_min_{name}'] = float(demo_z[:, k].min())
            wandb.summary[f'scoring/demo_norm_max_{name}'] = float(demo_z[:, k].max())
            wandb.summary[f'scoring/demo_norm_mean_{name}'] = float(demo_z[:, k].mean())
    wandb.summary['scoring/n_rollouts'] = len(rollout_scores)
    wandb.summary['scoring/n_demos'] = len(demo_scores)
    wandb.finish()


def _spearman(a, b):
    """Spearman rank correlation. Uses scipy rankdata (tie-aware) if available,
    else numpy ordinal ranks. Returns 0.0 for degenerate (constant) inputs."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    try:
        from scipy.stats import rankdata
        ra, rb = rankdata(a), rankdata(b)
    except Exception:
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def score_episodes(model, obs, episode_lengths, max_seq_len, device, batch_size=64, stride=1):
    """Score episodes with the reward model. Returns (N, K) array.

    Uses the same stride as training so saved scores match what the model saw.
    """
    N = len(obs)
    all_scores = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_obs = []
        batch_masks = []

        for i in range(start, end):
            idx_steps = stride_indices(int(episode_lengths[i]), max_seq_len, stride)
            padded = np.zeros((max_seq_len, obs.shape[-1]), dtype=np.float32)
            mask = np.ones(max_seq_len, dtype=bool)
            padded[:len(idx_steps)] = obs[i, idx_steps]
            mask[:len(idx_steps)] = False
            batch_obs.append(padded)
            batch_masks.append(mask)

        batch_obs = torch.from_numpy(np.stack(batch_obs)).to(device)
        batch_masks = torch.from_numpy(np.stack(batch_masks)).to(device)

        with torch.no_grad():
            scores = model(batch_obs, batch_masks)
        all_scores.append(scores.cpu().numpy())

    return np.concatenate(all_scores, axis=0)


if __name__ == '__main__':
    main()
