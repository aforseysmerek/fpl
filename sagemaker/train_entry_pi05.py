"""SageMaker container entrypoint for the pi0.5 reward-conditioned finetune.

Thin driver over the authors' pipeline: sets up SageMaker paths/env, then
execs openpi's scripts/train.py exactly as scripts/infer_then_train_pi05.sh
does (plus --checkpoint-base-dir into the live-synced dir).

Hyperparameters (read from SM_HP_* env — CLI values that start with '-'
break argparse, see the qwen entry):
  repo_id           LeRobot dataset repo id (resolved under the data channel)
  exp_name          experiment name (checkpoint subdir + wandb run name)
  config            train config name   [default: pi05_droid_finetune]
  num_train_steps   [default: 30000 — paper value, overrides the config's 60k]
  extra_args        optional extra CLI args passed through to train.py
"""
import os
import shlex
import subprocess
import sys

OPENPI_DIR = "/opt/ml/code/openpi"
DATA_DIR = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
CKPT_DIR = "/opt/ml/checkpoints"
# Optional warm-start channel (launch_pi05.py --init-ckpt-s3). Absent -> the
# config's default weight loader (gs:// base model) is used untouched.
INIT_CKPT_DIR = os.environ.get("SM_CHANNEL_INIT_CKPT", "/opt/ml/input/data/init_ckpt")


def main() -> int:
    repo_id = os.environ.get("SM_HP_REPO_ID")
    exp_name = os.environ.get("SM_HP_EXP_NAME")
    config = os.environ.get("SM_HP_CONFIG", "pi05_droid_finetune")
    num_train_steps = os.environ.get("SM_HP_NUM_TRAIN_STEPS", "30000")
    wandb_project = os.environ.get("SM_HP_WANDB_PROJECT", "fpl_pi05")
    extra_args = os.environ.get("SM_HP_EXTRA_ARGS", "")
    assert repo_id and exp_name, "repo_id and exp_name hyperparameters are required"

    env = os.environ.copy()
    # LeRobot resolves repo_id under HF_LEROBOT_HOME -> the mounted S3 channel.
    env["HF_LEROBOT_HOME"] = DATA_DIR
    # openpi asset cache (base checkpoint + norm stats from gs://) -> local disk.
    env.setdefault("OPENPI_DATA_HOME", "/tmp/openpi_cache")
    # Authors' launch script sets this for JAX memory behavior.
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

    os.makedirs(CKPT_DIR, exist_ok=True)
    cmd = [
        "/opt/venv/bin/python", "scripts/train.py", config,
        f"--exp-name={exp_name}",
        f"--data.repo_id={repo_id}",
        f"--num-train-steps={num_train_steps}",
        f"--project-name={wandb_project}",
        f"--checkpoint-base-dir={CKPT_DIR}",
        "--keep-train-state-only-latest",   # authors' flag: prune old optimizer states
        "--overwrite",                       # fresh checkpoint dir per job
    ] + shlex.split(extra_args)

    # Warm start: if the launcher mounted a checkpoint channel, initialize
    # weights from it instead of the config's gs:// base model. The channel
    # holds the CONTENTS of a step's params dir (or a params/ subdir if the
    # step dir itself was mounted).
    if os.path.isdir(INIT_CKPT_DIR) and os.listdir(INIT_CKPT_DIR):
        params_path = os.path.join(INIT_CKPT_DIR, "params")
        if not os.path.isdir(params_path):
            params_path = INIT_CKPT_DIR
        cmd.append(f"--weight-loader.params-path={params_path}")
        print(f"[pi05_entry] warm start: weight loader <- {params_path}", flush=True)

    print(f"[pi05_entry] HF_LEROBOT_HOME={DATA_DIR}", flush=True)
    print(f"[pi05_entry] exec: {' '.join(cmd)}", flush=True)
    rc = subprocess.Popen(cmd, cwd=OPENPI_DIR, env=env).wait()
    print(f"[pi05_entry] train.py exited rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
