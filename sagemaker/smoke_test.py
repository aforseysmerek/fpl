"""SageMaker launch smoke test for the FPL Qwen reward pipeline.

Verifies prerequisites without building an image or submitting a job.

Usage:
    python sagemaker/smoke_test.py                      # uses env/defaults
    python sagemaker/smoke_test.py --role-arn arn:...   # once IE tells you the role
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

SM_DIR = Path(__file__).resolve().parent
EXPECTED_ACCOUNT = "141701954645"  # shared-compute (tri-cam-robotics queue lives here)
DEFAULT_QUEUE = "fss-tri-cam-robotics-p5-48xlarge-us-west-2"
DEFAULT_ROLE = "arn:aws:iam::141701954645:role/CAM-Robotics-Sagemaker-role-us-west-2"

_OK = "\033[32m OK\033[0m"
_WARN = "\033[33mWARN\033[0m"
_FAIL = "\033[31mFAIL\033[0m"


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _report(status, label, detail=None):
    print(f"  [{status}] {label}")
    if detail:
        print(f"        {detail}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", default=os.environ.get("AWS_PROFILE", "sagemaker"))
    p.add_argument("--region", default="us-west-2")
    p.add_argument("--queue", default=DEFAULT_QUEUE)
    p.add_argument("--role-arn", default=os.environ.get("SAGEMAKER_ARN", DEFAULT_ROLE))
    p.add_argument("--s3-bucket", default=os.environ.get("S3_REMOTE_SYNC"))
    p.add_argument("--expect-account", default=EXPECTED_ACCOUNT)
    args = p.parse_args()

    print(f"FPL SageMaker smoke test (profile={args.profile}, region={args.region})\n")
    passed = True

    # 1. SSO session valid + right account
    r = _run(["aws", "--profile", args.profile, "--region", args.region, "sts",
              "get-caller-identity", "--query", "Account", "--output", "text"])
    account = r.stdout.strip()
    if r.returncode != 0:
        _report(_FAIL, "AWS auth", f"hint: aws sso login --profile {args.profile}")
        print(f"        {r.stderr.strip()[:200]}")
        print("\nAuth failed - remaining checks would all fail; fix this first.")
        sys.exit(1)
    if account == args.expect_account:
        _report(_OK, f"AWS auth -> account {account} (shared-compute)")
    else:
        _report(_WARN, f"AWS auth -> account {account}, expected {args.expect_account}",
                "wrong profile? the tri-cam-robotics queue lives in shared-compute")

    # 2. Docker daemon
    r = _run(["docker", "info"])
    ok = r.returncode == 0
    _report(_OK if ok else _FAIL, "Docker daemon running",
            None if ok else "hint: sudo systemctl start docker")
    passed &= ok

    # 3. ECR login (registry hostname, not image URI; catch broken cred helper)
    registry = f"{account}.dkr.ecr.{args.region}.amazonaws.com"
    token = _run(["aws", "ecr", "get-login-password", "--region", args.region,
                  "--profile", args.profile])
    if token.returncode == 0:
        login = _run(["docker", "login", "--username", "AWS", "--password-stdin", registry],
                     input=token.stdout)
        cred_err = "error storing credentials" in login.stderr
        ok = login.returncode == 0 and not cred_err
        _report(_OK if ok else _FAIL, f"ECR login ({registry})",
                None if ok else (login.stderr.strip() or login.stdout.strip())[:200])
        if cred_err:
            print("        hint: broken credential helper - remove 'credsStore' "
                  "from ~/.docker/config.json")
        passed &= ok
    else:
        _report(_FAIL, "ECR login", token.stderr.strip()[:200])
        passed = False

    # 4. sagemaker SDK new enough for Batch queues (launch-side dependency)
    try:
        import boto3  # noqa: F401
        import sagemaker
        from sagemaker.aws_batch.training_queue import TrainingQueue  # noqa: F401
        _report(_OK, f"sagemaker SDK {sagemaker.__version__} (+aws_batch) importable")
    except ImportError as e:
        _report(_FAIL, "sagemaker SDK with aws_batch support",
                f"hint: pip install -U 'sagemaker>=2.245' boto3  ({e})")
        passed = False

    # 5. secrets.env
    secrets = SM_DIR / "secrets.env"
    if secrets.exists():
        keys = [ln.split("=")[0].strip() for ln in secrets.read_text().splitlines()
                if "=" in ln and not ln.strip().startswith("#")]
        if "WANDB_API_KEY" in keys:
            _report(_OK, "secrets.env with WANDB_API_KEY")
        else:
            _report(_WARN, "secrets.env exists but no WANDB_API_KEY",
                    "trainer will need --no_wandb in --train-args")
    else:
        _report(_WARN, "sagemaker/secrets.env missing",
                "cp sagemaker/secrets.env.example sagemaker/secrets.env  # then add key")

    # 6. execution role (known unknown until IE answers -> warn, not fail)
    if args.role_arn:
        role_account = args.role_arn.split(":")[4] if args.role_arn.count(":") >= 5 else "?"
        if role_account == account:
            _report(_OK, f"execution role set ({args.role_arn.split('/')[-1]})")
        else:
            _report(_FAIL, f"role account {role_account} != session account {account}")
            passed = False
    else:
        _report(_WARN, "no execution role configured",
                "set SAGEMAKER_ARN or --role-arn (ask IE: SageMaker execution role "
                "for shared-compute)")

    # 7. queue visibility (the classic 'does not exist' = wrong account/no access)
    try:
        import boto3
        batch = boto3.session.Session(
            profile_name=args.profile, region_name=args.region).client("batch")
        qs = batch.describe_job_queues(jobQueues=[args.queue])["jobQueues"]
        if qs:
            q = qs[0]
            _report(_OK, f"queue {args.queue} ({q['state']}/{q['status']})")
        else:
            _report(_FAIL, f"queue {args.queue} not visible",
                    "wrong account, or your principal lacks access to this queue")
            passed = False
    except Exception as e:
        _report(_FAIL, f"queue check errored: {str(e)[:150]}")
        passed = False

    # 8. output bucket reachable
    # Buckets live in the manip-cluster account (cross-account perms are
    # pre-configured for jobs); try the launch profile first, then manip-cluster.
    if args.s3_bucket:
        bucket = args.s3_bucket.replace("s3://", "").split("/")[0]
        import boto3
        reached = None
        errors = []
        for prof in (args.profile, "manip-cluster"):
            try:
                s3 = boto3.session.Session(
                    profile_name=prof, region_name=args.region).client("s3")
                s3.head_bucket(Bucket=bucket)
                reached = prof
                break
            except Exception as e:
                errors.append(f"{prof}: {str(e)[:100]}")
        if reached:
            _report(_OK, f"s3 bucket reachable ({bucket}, via profile '{reached}')")
        else:
            _report(_FAIL, f"s3 bucket {bucket} unreachable", "; ".join(errors))
            passed = False
    else:
        _report(_WARN, "no output bucket configured",
                "set S3_REMOTE_SYNC or --s3-bucket (a manip-cluster bucket; "
                "see 'aws s3 ls --profile manip-cluster')")

    print()
    print("All hard checks passed." if passed else "One or more checks FAILED.")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
