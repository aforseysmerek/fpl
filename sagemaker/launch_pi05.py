"""Build the pi0.5 finetune image and submit a reward-conditioned policy
training job to the SageMaker Batch queue.

Sibling of launch_sagemaker.py (Qwen reward training) — same account/queue
plumbing, its own image (Dockerfile.pi05) and hyperparameter surface.

Example:
    python sagemaker/launch_pi05.py \
        --repo-id alex/table_wipe_spread_n200_std_1dp \
        --exp-name table_wipe_spread_n200_pi05

Required config (flag or env var):
  --role-arn / SAGEMAKER_ARN    execution role (defaults to CAM-Robotics role)
  --s3-bucket / S3_REMOTE_SYNC  s3://bucket/prefix for outputs + code staging
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

NAME = "fpl-pi05"
REPO_ROOT = Path(__file__).resolve().parent.parent
SM_DIR = Path(__file__).resolve().parent

INSTANCE_MAPPER = {"p5": "ml.p5.48xlarge"}
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
    algorithm_name = f"{user}-{NAME}"
    registry = f"{account}.dkr.ecr.{region}.amazonaws.com"
    fullname = f"{registry}/{algorithm_name}:latest"
    if skip_build:
        print(f"--skip-build: reusing {fullname}")
        return fullname

    login = (f"aws ecr get-login-password --region {region} --profile {profile} | "
             f"docker login --username AWS --password-stdin")
    commands = [
        f"{login} {registry}",
        (f"cd {REPO_ROOT} && docker build --progress=plain -f sagemaker/Dockerfile.pi05 "
         f"-t {algorithm_name} ."),
        f"docker tag {algorithm_name} {fullname}",
        (f"aws --region {region} --profile {profile} ecr describe-repositories "
         f"--repository-names {algorithm_name} --no-cli-pager || "
         f"aws --region {region} --profile {profile} ecr create-repository "
         f"--repository-name {algorithm_name} --no-cli-pager"),
    ]
    run_command("\n".join(f"{c} || exit 1" for c in commands))
    run_command(f"docker push {fullname}")
    time.sleep(5)
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
        print("WARNING: sagemaker/secrets.env missing -> no WANDB_API_KEY (openpi "
              "logs to wandb by default)")
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
    p.add_argument("--role-arn", default=os.environ.get("SAGEMAKER_ARN", DEFAULT_ROLE))
    p.add_argument("--s3-bucket", default=os.environ.get("S3_REMOTE_SYNC"))

    # what to train
    p.add_argument("--repo-id", required=True,
                   help="LeRobot dataset repo id (must exist under --data-s3)")
    p.add_argument("--exp-name", required=True,
                   help="experiment name (checkpoint subdir + wandb run name)")
    p.add_argument("--config", default="pi05_droid_finetune")
    p.add_argument("--num-train-steps", type=int, default=30000,
                   help="paper value; overrides the vendored config's 60k")
    p.add_argument("--extra-args", default="",
                   help="extra CLI args passed through to openpi train.py")
    p.add_argument("--wandb-project", default="fpl_pi05",
                   help="wandb project name (openpi's default is 'openpi')")
    p.add_argument("--data-s3", default=None,
                   help="S3 prefix of the LeRobot root (default: <s3-bucket>/lerobot/)")
    p.add_argument("--init-ckpt-s3", default=None,
                   help="Warm start: S3 URI of a trained checkpoint's params dir (e.g. "
                        ".../checkpoints/pi05_droid_finetune/<exp>/29999/params — point at "
                        "params/, not the step dir, to avoid pulling the 37GB train_state). "
                        "Mounted as the init_ckpt channel; the entry then overrides openpi's "
                        "--weight-loader.params-path so training initializes from it instead "
                        "of the config's gs:// base model.")

    # queue / instance
    p.add_argument("--queue", default=DEFAULT_QUEUE)
    p.add_argument("--instance-type", default="p5", choices=list(INSTANCE_MAPPER))
    p.add_argument("--priority", type=int, default=400)
    p.add_argument("--share-identifier", default="default")
    p.add_argument("--max-run-days", type=float, default=2.0)
    p.add_argument("--project-tag", default="MM:PJ-0077")

    # workflow
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--image", default=None, help="use this exact image URI (skips build)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.s3_bucket:
        sys.exit("Missing output bucket: pass --s3-bucket or set S3_REMOTE_SYNC")
    bucket = args.s3_bucket.rstrip("/")
    data_s3 = (args.data_s3 or f"{bucket}/lerobot/").rstrip("/") + "/"

    account = get_account(args.profile, args.region)
    print(f"account={account} profile={args.profile} region={args.region}")

    hyperparameters = {
        "repo_id": args.repo_id,
        "exp_name": args.exp_name,
        "config": args.config,
        "num_train_steps": str(args.num_train_steps),
        "wandb_project": args.wandb_project,
    }
    if args.extra_args:
        hyperparameters["extra_args"] = args.extra_args

    base_job_name = sanitize(f"{args.user.replace('.', '-')}-{NAME}")
    job_name = f"{base_job_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"[:63].rstrip("-")

    output_root = f"{bucket}/fpl/{args.user}/{NAME}"
    checkpoint_s3 = f"{output_root}/{job_name}/checkpoints"

    environment = load_secrets({
        "SM_USE_RESERVED_CAPACITY": "1",
        "NCCL_DEBUG": "WARN",
        "FI_EFA_FORK_SAFE": "1",
    })

    max_run = int(args.max_run_days * 24 * 60 * 60)

    if args.dry_run:
        print("\n--- DRY RUN ---")
        print(json.dumps({
            "job_name": job_name, "queue": args.queue,
            "instance": INSTANCE_MAPPER[args.instance_type],
            "image": args.image or f"{account}.dkr.ecr.{args.region}.amazonaws.com/"
                                   f"{args.user}-{NAME}:latest",
            "role": args.role_arn, "data": data_s3,
            "init_ckpt": args.init_ckpt_s3,
            "checkpoint_s3_uri": checkpoint_s3, "output_path": output_root,
            "hyperparameters": hyperparameters,
            "env_keys": sorted(environment.keys()),
            "tags": {"tri.project": args.project_tag,
                     "tri.owner.email": f"{args.user}@tri.global"},
        }, indent=2))
        return

    image = args.image or build_and_push_image(
        args.user, args.profile, args.region, account, args.skip_build)

    import sagemaker
    from sagemaker.estimator import Estimator
    from sagemaker.inputs import TrainingInput
    from sagemaker.aws_batch.training_queue import TrainingQueue as Queue

    boto3.setup_default_session(profile_name=args.profile, region_name=args.region)
    sagemaker_session = sagemaker.Session(
        boto3.session.Session(profile_name=args.profile, region_name=args.region))

    # Generic Estimator (not PyTorch): fully custom image, entrypoint baked in
    # via SAGEMAKER_PROGRAM + the sagemaker-training toolkit inside the image.
    estimator = Estimator(
        image_uri=image,
        sagemaker_session=sagemaker_session,
        base_job_name=base_job_name,
        hyperparameters=hyperparameters,
        role=args.role_arn,
        instance_count=1,
        instance_type=INSTANCE_MAPPER[args.instance_type],
        environment=environment,
        max_run=max_run,
        input_mode="File",
        output_path=output_root,
        checkpoint_s3_uri=checkpoint_s3,
        checkpoint_local_path="/opt/ml/checkpoints",
        tags=[
            {"Key": "tri.project", "Value": args.project_tag},
            {"Key": "tri.owner.email", "Value": f"{args.user}@tri.global"},
        ],
    )

    inputs = {"training": TrainingInput(s3_data=data_s3)}
    if args.init_ckpt_s3:
        inputs["init_ckpt"] = TrainingInput(s3_data=args.init_ckpt_s3.rstrip("/") + "/")
        print(f"Warm start: init_ckpt channel <- {args.init_ckpt_s3}")

    queue = Queue(queue_name=args.queue)
    print(f"Submitting {job_name} to {args.queue} (priority {args.priority})")
    queued = queue.map(
        estimator,
        inputs=[inputs],
        job_names=[job_name],
        priority=args.priority,
        share_identifier=args.share_identifier,
        timeout={"attemptDurationSeconds": max_run},
        retry_config={"attempts": 1},
    )
    print(f"Queued: {queued}")
    print("\nMonitor with:")
    print(f"  batchy ls {args.queue}")
    print(f"  aws s3 ls {checkpoint_s3}/ --recursive   # live checkpoints")


if __name__ == "__main__":
    main()
