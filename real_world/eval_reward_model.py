"""Evaluate a trained reward-model checkpoint on arbitrary episodes.

Thin driver over the training stack — no metric logic lives here:
  pairs/labels:  dataset.build_pairs_from_episodes (any registered oracle axes)
  datasets:      PreferenceDataset / OpenPreferenceDataset (same expansion)
  win rate:      train_reward_model.evaluate
  spearman/mse:  train_reward_model.run_value_metrics (incl. scatter PNGs;
                 z-scale pooled across all sets in this invocation)

Runs locally (forward passes only).

Example — horizontal-wipe model scored on the vertical task:
    python eval_reward_model.py --ckpt ckpts/best_val.pt \
        --episodes ../diffusion_policy/shared_data_vertical_wipe/episodes.hdf5 \
        --reward_axes circularity_vertical,wiped_frac --value_metrics
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (PreferenceDataset, OpenPreferenceDataset,
                     build_pairs_from_episodes)
from qwen_model import QwenRewardModel
from train_reward_model import evaluate, run_value_metrics


def build_model(ckpt, device):
    saved = ckpt["args"]
    model_type = saved["model"]
    if not model_type.startswith("qwen"):
        sys.exit(f"eval_reward_model currently supports qwen* checkpoints, got '{model_type}'")
    is_open = model_type in ("qwen_open", "qwen_open_discounted", "qwen_open_cum")
    model = QwenRewardModel(
        num_preferences=1 if is_open else len(saved.get("reward_axes", "").split(",")),
        model_name=saved.get("qwen_model_name", "Qwen/Qwen3-VL-4B-Instruct"),
        # honor how the checkpoint was actually trained (unlike infer.py's
        # qwen_lora-only check, this also loads LoRA'd open models)
        use_lora=saved.get("use_lora", False) or model_type == "qwen_lora",
        lora_r=saved.get("lora_r", 64),
        lora_alpha=saved.get("lora_alpha", 16),
        reward_sigmoid=saved.get("reward_sigmoid", False),
        gradient_checkpointing=False,
        discounted=model_type in ("qwen_discounted", "qwen_open_discounted"),
        open_cum=model_type == "qwen_open_cum",
    ).to(device)
    model.load_checkpoint_state_dict(ckpt["model"])
    model.eval()
    return model, saved, is_open


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", action="append", required=True,
                   help="episodes.hdf5 to evaluate on (repeatable)")
    p.add_argument("--z_ref_episodes", action="append", default=[],
                   help="episodes.hdf5 included ONLY in the value-metrics z-score "
                        "pool (no win-rate pass) — e.g. pass the spread set here "
                        "while evaluating middle, to match training's train+val pooling")
    p.add_argument("--reward_axes", required=True,
                   help="comma-separated oracle axes for pair labels (prompts via AXIS_PROMPT)")
    p.add_argument("--n_pairs", type=int, default=300)
    p.add_argument("--pair_seed", type=int, default=42)
    p.add_argument("--pair_n_episodes", type=int, default=None)
    p.add_argument("--stride", type=int, default=None, help="default: checkpoint's value")
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=16,
                   help="inference micro-batch (throughput only; metrics identical)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--value_metrics", action="store_true",
                   help="also run the trainer's run_value_metrics (spearman/mse/scatters)")
    p.add_argument("--out_dir", default="eval_out",
                   help="where scatter PNGs and results.json land")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, saved, is_open = build_model(ckpt, device)
    stride = args.stride or saved.get("stride", 20)
    seq_len = args.seq_len or saved.get("seq_len", 20)
    img = args.img_size or saved.get("img_size", 128)
    step = int(ckpt.get("step", 0))
    print(f"[eval] ckpt={args.ckpt} (model={saved['model']}, trained axes={saved.get('reward_axes')}, "
          f"step={step}) | frames: stride={stride} seq_len={seq_len} img={img}")

    os.makedirs(args.out_dir, exist_ok=True)
    results = {"ckpt": args.ckpt, "ckpt_step": step, "reward_axes": args.reward_axes,
               "n_pairs": args.n_pairs, "pair_seed": args.pair_seed, "sets": {}}

    named_sets, phrases = [], None
    for ep_path in args.episodes:
        samples, phrases = build_pairs_from_episodes(
            ep_path, args.reward_axes, args.n_pairs, args.pair_seed,
            n_episodes=args.pair_n_episodes)
        base = PreferenceDataset([], preference_keys=phrases, stride=stride,
                                 seq_len=seq_len, img_size=(img, img),
                                 training=False, preload=False,
                                 action_chunk_size=0, only_large=False)
        base.samples.extend(samples)
        ds = OpenPreferenceDataset(base, phrases, skip_equal=True) if is_open else base
        name = os.path.basename(os.path.dirname(os.path.abspath(ep_path)))
        named_sets.append((name, ds))

        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        num_prefs = 1 if is_open else len(phrases)
        from tqdm import tqdm
        loss, acc, axis_acc = evaluate(model, tqdm(loader, desc=f"win-rate {name}"),
                                       device, 0.0, num_prefs)
        per_axis = axis_acc if axis_acc else dict(zip(phrases, acc.tolist()))
        results["sets"][name] = {
            "episodes": ep_path,
            "loss": float(loss),
            "acc_mean": float(np.mean(list(per_axis.values()))),
            "acc": {k: float(v) for k, v in per_axis.items()},
            "n_pair_samples": len(ds),
        }
        print(f"\n[eval] {name}: win-rate mean {results['sets'][name]['acc_mean']:.3f} "
              f"(loss {loss:.4f}, {len(ds)} pair-samples)")
        for k, v in per_axis.items():
            print(f"    {k}: {float(v):.3f}")

    # z-reference sets: same pair/dataset construction, but they only join the
    # value-metrics pool (and its per-set report) — no win-rate pass.
    for ep_path in args.z_ref_episodes:
        samples, phrases = build_pairs_from_episodes(
            ep_path, args.reward_axes, args.n_pairs, args.pair_seed,
            n_episodes=args.pair_n_episodes)
        base = PreferenceDataset([], preference_keys=phrases, stride=stride,
                                 seq_len=seq_len, img_size=(img, img),
                                 training=False, preload=False,
                                 action_chunk_size=0, only_large=False)
        base.samples.extend(samples)
        ds = OpenPreferenceDataset(base, phrases, skip_equal=True) if is_open else base
        name = "zref_" + os.path.basename(os.path.dirname(os.path.abspath(ep_path)))
        named_sets.append((name, ds))
        results["sets"][name] = {"episodes": ep_path, "z_reference_only": True}

    if args.value_metrics:
        vm = run_value_metrics(model, named_sets, device, phrases,
                               args.batch_size, is_open,
                               out_dir=args.out_dir, step=step, use_wandb=False)
        results["reward_value"] = {k: float(v) for k, v in vm.items()
                                   if isinstance(v, (int, float, np.floating))}

    out_json = os.path.join(args.out_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] wrote {out_json} (+ scatter PNGs in {args.out_dir}/)")


if __name__ == "__main__":
    main()
