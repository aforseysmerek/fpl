"""Build the training image and submit a Qwen reward-model job to a SageMaker
Batch queue (TRI pattern, adapted from TRI-ML/batch_test).

Examples (from repo root; see sagemaker/README.md for the full runbook):

  # end-to-end pipeline test (no data needed)
  python sagemaker/launch_sagemaker.py --hello

  # single training run
  python sagemaker/launch_sagemaker.py \
      --data-s3 s3://<bucket>/fpl/preferences/wipe_v1/ \
      --train-args "--task auto --use_lora --preload --epochs 50"

  # 8 configs in parallel, one per GPU (see sweep_example.txt)
  python sagemaker/launch_sagemaker.py --data-s3 s3://... --sweep-file sweep.txt

  # test the image locally on your own GPU first
  python sagemaker/launch_sagemaker.py --local \
      --data-s3 file:///abs/path/to/preferences \
      --train-args "--task auto --use_lora --epochs 1"

Required config (flag or env var):
  --role-arn  / SAGEMAKER_ARN     SageMaker execution role in the target account
  --s3-bucket / S3_REMOTE_SYNC    s3://bucket[/prefix] for outputs + code staging
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import boto3

NAME = "fpl-qwen-reward"
REPO_ROOT = Path(__file__).resolve().parent.parent
SM_DIR = Path(__file__).resolve().parent

INSTANCE_MAPPER = {
    "p4de": "ml.p4de.24xlarge",  # 8x A100 80GB
    "p5": "ml.p5.48xlarge",      # 8x H100 80GB
}
DEFAULT_QUEUE = "fss-tri-cam-robotics-p5-48xlarge-us-west-2"
DEFAULT_ROLE = "arn:aws:iam::141701954645:role/CAM-Robotics-Sagemaker-role-us-west-2"


def run_command(command: str):
    print(f"=> {command}")
    subprocess.run(command, shell=True, check=True)


def get_account(profile: str, region: str) -> str:
    out = subprocess.run(
        ["aws", "--profile", profile, "--region", region, "sts",
         "get-caller-identity", "--query", "Account", "--output", "text"],
        capture_output=True, text=True,
    )
    if out.returncode != 0 or not out.stdout.strip().isdigit():
        sys.exit(f"Cannot resolve AWS account (try: aws sso login --profile {profile})\n"
                 f"{out.stderr.strip()}")
    return out.stdout.strip()


def build_and_push_image(user: str, profile: str, region: str, account: str,
                         skip_build: bool) -> str:
    # Keep the dot: ECR write perms are scoped per-user to repository/<sso-username>*,
    # so the repo name must start with "first.last".
    algorithm_name = f"{user}-{NAME}"
    registry = f"{account}.dkr.ecr.{region}.amazonaws.com"
    fullname = f"{registry}/{algorithm_name}:latest"
    if skip_build:
        print(f"--skip-build: reusing {fullname}")
        return fullname

    login = (f"aws ecr get-login-password --region {region} --profile {profile} | "
             f"docker login --username AWS --password-stdin")
    commands = [
        # DLC registry (base image) + our registry. Login to the HOSTNAME, not
        # the full image URI (past bug: full-URI logins broke credential save).
        f"{login} 763104351884.dkr.ecr.{region}.amazonaws.com",
        f"{login} {registry}",
        (f"cd {REPO_ROOT} && docker build --progress=plain -f sagemaker/Dockerfile "
         f"--build-arg AWS_REGION={region} -t {algorithm_name} ."),
        f"docker tag {algorithm_name} {fullname}",
        (f"aws --region {region} --profile {profile} ecr describe-repositories "
         f"--repository-names {algorithm_name} --no-cli-pager || "
         f"aws --region {region} --profile {profile} ecr create-repository "
         f"--repository-name {algorithm_name} --no-cli-pager"),
    ]
    run_command("\n".join(f"{c} || exit 1" for c in commands))
    run_command(f"docker push {fullname}")
    time.sleep(5)  # let ECR settle before SageMaker pulls
    return fullname


def load_secrets(env: dict) -> dict:
    secrets = SM_DIR / "secrets.env"
    if secrets.exists():
        for line in secrets.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("\"'")
    else:
        print("WARNING: sagemaker/secrets.env missing -> no WANDB_API_KEY; "
              "add --no_wandb to --train-args or create it (see secrets.env.example)")
    return env


def sanitize(name: str) -> str:
    clean = "".join(c if c.isalnum() or c == "-" else "-" for c in name).strip("-")
    while "--" in clean:
        clean = clean.replace("--", "-")
    return clean or "job"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", default="alexandra.forsey-smerek")
    p.add_argument("--profile", default=os.environ.get("AWS_PROFILE", "sagemaker"))
    p.add_argument("--region", default="us-west-2")
    p.add_argument("--role-arn", default=os.environ.get("SAGEMAKER_ARN", DEFAULT_ROLE),
                   help="SageMaker execution role ARN (env: SAGEMAKER_ARN)")
    p.add_argument("--s3-bucket", default=os.environ.get("S3_REMOTE_SYNC"),
                   help="s3://bucket[/prefix] for outputs (env: S3_REMOTE_SYNC)")
    p.add_argument("--data-s3", default=None,
                   help="s3:// URI of the preferences dir (or file:// with --local)")

    # what to run (choose one)
    p.add_argument("--train-args", default=None,
                   help="arg string for train_reward_model.py; {data}/{out} placeholders ok")
    p.add_argument("--sweep-file", default=None,
                   help="text file, one train-arg string per line (# and blanks skipped)")
    p.add_argument("--hello", action="store_true", help="submit a pipeline sanity job")
    p.add_argument("--nproc", type=int, default=1,
                   help="GPUs for DDP (single-run mode; trainer supports torchrun)")

    # queue / instance
    p.add_argument("--queue", default=DEFAULT_QUEUE)
    p.add_argument("--instance-type", default="p5", choices=list(INSTANCE_MAPPER))
    p.add_argument("--priority", type=int, default=10, help="1 (low) .. 9999 (high)")
    p.add_argument("--share-identifier", default="default")
    p.add_argument("--max-run-days", type=float, default=2.0)
    p.add_argument("--project-tag", default="MM:PJ-0077",
                   help="tri.project cost tag - CONFIRM the right code for your team")

    # workflow
    p.add_argument("--skip-build", action="store_true", help="reuse last pushed image")
    p.add_argument("--image", default=None, help="use this exact image URI (skips build)")
    p.add_argument("--local", action="store_true", help="run via SageMaker local mode")
    p.add_argument("--dry-run", action="store_true", help="print config, submit nothing")
    args = p.parse_args()

    modes = [bool(args.train_args), bool(args.sweep_file), args.hello]
    if sum(modes) != 1:
        sys.exit("Choose exactly one of --train-args / --sweep-file / --hello")
    if not args.hello and not args.data_s3:
        sys.exit("--data-s3 is required for training jobs (s3://... or file:// with --local)")
    if not args.s3_bucket:
        sys.exit("Missing output bucket: pass --s3-bucket or set S3_REMOTE_SYNC")

    inst = INSTANCE_MAPPER[args.instance_type]
    inst_token = inst.replace("ml.", "").replace(".", "-")  # e.g. p5-48xlarge
    if not args.local and inst_token not in args.queue:
        print(f"WARNING: queue '{args.queue}' does not mention '{inst_token}' - "
              f"instance/queue mismatch will fail at submit time")

    account = get_account(args.profile, args.region)
    print(f"account={account} profile={args.profile} region={args.region}")
    role_account = args.role_arn.split(":")[4]
    if role_account != account:
        print(f"WARNING: role ARN account ({role_account}) != session account ({account})")

    # hyperparameters -> CLI flags for train_entry.py
    hyperparameters = {}
    if args.hello:
        hyperparameters["hello"] = "1"
    elif args.sweep_file:
        lines = [ln.strip() for ln in Path(args.sweep_file).read_text().splitlines()]
        specs = [ln for ln in lines if ln and not ln.startswith("#")]
        if not specs:
            sys.exit(f"{args.sweep_file} contains no runs")
        hyperparameters["sweep_json"] = json.dumps(specs)
        print(f"sweep: {len(specs)} runs")
    else:
        hyperparameters["train_args"] = args.train_args
        if args.nproc > 1:
            hyperparameters["nproc"] = str(args.nproc)

    base_job_name = sanitize(f"{args.user.replace('.', '-')}-{NAME}")
    job_name = f"{base_job_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"[:63].rstrip("-")

    bucket = args.s3_bucket.rstrip("/")
    output_root = f"{bucket}/fpl/{args.user}/{NAME}"
    checkpoint_s3 = f"{output_root}/{job_name}/checkpoints"

    environment = load_secrets({
        "SM_USE_RESERVED_CAPACITY": "1",
        "NCCL_DEBUG": "WARN",
        "FI_EFA_FORK_SAFE": "1",
        "HF_HOME": "/tmp/hf",  # model download lands on the big local volume
    })

    max_run = int(args.max_run_days * 24 * 60 * 60)

    if args.dry_run:
        print("\n--- DRY RUN ---")
        print(json.dumps({
            "job_name": job_name, "queue": args.queue, "instance": inst,
            "image": args.image or f"{account}.dkr.ecr.{args.region}.amazonaws.com/"
                                   f"{args.user}-{NAME}:latest",
            "role": args.role_arn, "data": args.data_s3,
            "checkpoint_s3_uri": checkpoint_s3, "output_path": output_root,
            "hyperparameters": hyperparameters,
            "env_keys": sorted(environment.keys()),
            "tags": {"tri.project": args.project_tag,
                     "tri.owner.email": f"{args.user}@tri.global"},
        }, indent=2))
        return

    image = args.image or build_and_push_image(
        args.user, args.profile, args.region, account, args.skip_build)

    # Late imports: slow, and not needed for --dry-run.
    import sagemaker
    from sagemaker.inputs import TrainingInput
    from sagemaker.pytorch import PyTorch

    boto3.setup_default_session(profile_name=args.profile, region_name=args.region)
    if args.local:
        from sagemaker.local import LocalSession
        sagemaker_session = LocalSession()
        sagemaker_session.config = {"local": {"local_code": True}}
    else:
        sagemaker_session = sagemaker.Session(
            boto3.session.Session(profile_name=args.profile, region_name=args.region))

    estimator = PyTorch(
        entry_point=str(SM_DIR / "train_entry.py"),
        sagemaker_session=sagemaker_session,
        base_job_name=base_job_name,
        hyperparameters=hyperparameters,
        role=args.role_arn,
        image_uri=image,
        instance_count=1,
        instance_type="local_gpu" if args.local else inst,
        environment=environment,
        max_run=max_run,
        input_mode="File",  # h5py random access; FastFile only helps streaming formats
        output_path=output_root,
        code_location=output_root,
        checkpoint_s3_uri=None if args.local else checkpoint_s3,
        checkpoint_local_path=None if args.local else "/opt/ml/checkpoints",
        # NOTE: no `distribution=` - train_entry.py owns process layout
        # (torchrun for --nproc DDP, one-proc-per-GPU for sweeps).
        tags=[
            {"Key": "tri.project", "Value": args.project_tag},
            {"Key": "tri.owner.email", "Value": f"{args.user}@tri.global"},
        ],
    )

    inputs = {"training": TrainingInput(s3_data=args.data_s3)} if args.data_s3 else None

    if args.local:
        print("Running in SageMaker local mode...")
        estimator.fit(inputs, job_name=job_name)
        return

    from sagemaker.aws_batch.training_queue import TrainingQueue as Queue
    queue = Queue(queue_name=args.queue)
    print(f"Submitting {job_name} to {args.queue} (priority {args.priority})")
    queued = queue.map(
        estimator,
        inputs=[inputs],
        job_names=[job_name],
        priority=args.priority,
        share_identifier=args.share_identifier,
        timeout={"attemptDurationSeconds": max_run},
        # One attempt only: the queue's default policy re-ran a deterministically
        # crashing job ~21 times (~15 min of 8xH100 per attempt).
        retry_config={"attempts": 1},
    )
    print(f"Queued: {queued}")
    print("\nMonitor with:")
    print(f"  batchy ls {args.queue}")
    print(f"  sagey logs {job_name} --follow")
    print(f"  aws s3 ls {checkpoint_s3}/ --recursive   # live checkpoints")


if __name__ == "__main__":
    main()
