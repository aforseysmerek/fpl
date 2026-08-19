"""SageMaker container entrypoint for real_world/train_reward_model.py.

SageMaker passes hyperparameters as CLI flags. We accept ONE of:
  --hello 1            sanity job: print env, touch outputs, exit (no data needed)
  --train_args "..."   single run; arg string handed to the trainer
  --sweep_json "[...]" JSON list of arg strings; runs in waves, one per GPU

Placeholders usable inside arg strings:
  {data} -> the S3 input channel mount (preferences dir root)
  {out}  -> /opt/ml/checkpoints (live-synced to checkpoint_s3_uri)
  {i}    -> sweep index (sweep mode only)

If --preferences_dir / --out_dir are absent from an arg string they are
appended automatically ({data} and {out}[/run{i}] respectively).
--nproc N (single-run mode) launches the trainer under torchrun for DDP —
the trainer self-initializes when LOCAL_RANK is set.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys

CODE_DIR = "/opt/ml/code/real_world"
TRAINER = "train_reward_model.py"
DATA_DIR = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
CKPT_DIR = "/opt/ml/checkpoints"
MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")


def substitute(argstr: str, i: int | None = None) -> str:
    argstr = argstr.replace("{data}", DATA_DIR).replace("{out}", CKPT_DIR)
    if i is not None:
        argstr = argstr.replace("{i}", str(i))
    return argstr


def build_cmd(argstr: str, nproc: int = 1, run_suffix: str = "") -> list[str]:
    argv = shlex.split(argstr)
    if "--preferences_dir" not in argv:
        argv += ["--preferences_dir", DATA_DIR]
    if "--out_dir" not in argv:
        argv += ["--out_dir", CKPT_DIR + run_suffix]
    if nproc > 1:
        return ["torchrun", "--standalone", f"--nproc_per_node={nproc}", TRAINER] + argv
    return [sys.executable, TRAINER] + argv


def launch(cmd: list[str], gpu: int | None = None) -> subprocess.Popen:
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[train_entry] launch (gpu={gpu}): {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd, cwd=CODE_DIR, env=env)


def hello() -> int:
    import torch

    print("[hello] container OK")
    print(f"[hello] python={sys.version.split()[0]} torch={torch.__version__} "
          f"gpus={torch.cuda.device_count()}")
    print(f"[hello] data channel {DATA_DIR} exists={os.path.isdir(DATA_DIR)}")
    if os.path.isdir(DATA_DIR):
        print(f"[hello] first entries: {sorted(os.listdir(DATA_DIR))[:5]}")
    for d in (CKPT_DIR, MODEL_DIR):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "hello.txt"), "w") as f:
            f.write("hello from train_entry\n")
    print(f"[hello] wrote markers to {CKPT_DIR} and {MODEL_DIR}; done.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hello", default=None)
    p.add_argument("--train_args", default=None)
    p.add_argument("--sweep_json", default=None)
    p.add_argument("--nproc", type=int, default=1)
    # SageMaker passes hyperparameters as CLI flags, but argparse rejects a
    # value that starts with '-' (train_args always does: "--model ...").
    # The toolkit also exports each hyperparameter as SM_HP_<NAME> - prefer
    # those; the CLI path remains for local/manual runs.
    if any(k in os.environ for k in ("SM_HP_TRAIN_ARGS", "SM_HP_SWEEP_JSON", "SM_HP_HELLO")):
        args = argparse.Namespace(
            hello=os.environ.get("SM_HP_HELLO"),
            train_args=os.environ.get("SM_HP_TRAIN_ARGS"),
            sweep_json=os.environ.get("SM_HP_SWEEP_JSON"),
            nproc=int(os.environ.get("SM_HP_NPROC", "1")),
        )
        print("[train_entry] using SM_HP_* hyperparameters from environment", flush=True)
    else:
        args, unknown = p.parse_known_args()
        if unknown:
            print(f"[train_entry] ignoring extra args: {unknown}", flush=True)

    os.makedirs(CKPT_DIR, exist_ok=True)
    if args.hello:
        return hello()

    if args.sweep_json:
        import torch

        specs = json.loads(args.sweep_json)
        assert isinstance(specs, list) and specs, "--sweep_json must be a non-empty JSON list"
        ngpu = max(torch.cuda.device_count(), 1)
        print(f"[train_entry] sweep: {len(specs)} runs on {ngpu} GPUs (waves of {ngpu})", flush=True)
        failures = []
        for wave_start in range(0, len(specs), ngpu):
            wave = specs[wave_start:wave_start + ngpu]
            procs = []
            for k, spec in enumerate(wave):
                i = wave_start + k
                cmd = build_cmd(substitute(spec, i), run_suffix=f"/run{i:02d}")
                procs.append((i, launch(cmd, gpu=k)))
            for i, proc in procs:
                rc = proc.wait()
                print(f"[train_entry] run{i:02d} exited rc={rc}", flush=True)
                if rc != 0:
                    failures.append(i)
        if failures:
            print(f"[train_entry] FAILED runs: {failures}", flush=True)
            return 1
        return 0

    assert args.train_args, "provide --train_args, --sweep_json, or --hello"
    cmd = build_cmd(substitute(args.train_args), nproc=args.nproc)
    rc = launch(cmd).wait()
    print(f"[train_entry] trainer exited rc={rc}", flush=True)
    # Leave a marker so the end-of-job model.tar.gz is never empty.
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "run_info.json"), "w") as f:
        json.dump({"returncode": rc, "checkpoints": "see checkpoint_s3_uri"}, f)
    return rc


if __name__ == "__main__":
    sys.exit(main())
