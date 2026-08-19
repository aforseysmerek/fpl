# SageMaker training for the Qwen reward model

Submits `real_world/train_reward_model.py` to TRI's reserved GPU queues
(AWS Batch -> SageMaker, shared-compute account `141701954645`,
queue `fss-tri-cam-robotics-p5-48xlarge-us-west-2`, 8x H100 per node).

Files: `Dockerfile` + `requirements.txt` (the training image, deps pinned to the
local `qwen_rl` env), `train_entry.py` (runs inside the container),
`launch_sagemaker.py` (build + push + submit, run from your machine),
`smoke_test.py` (prereq checker), `sweep_example.txt`, `secrets.env.example`.

## Key config

- **Execution role** (launcher default): `arn:aws:iam::141701954645:role/CAM-Robotics-Sagemaker-role-us-west-2`
- **ECR**: push your own images to `141701954645.dkr.ecr.us-west-2.amazonaws.com/<first.last>-*`
  (write perms are scoped per-user to repos prefixed with your SSO username, dot included).
  If push 403s at the manifest step, ask IE for `ecr:PutImage`+`ecr:BatchGetImage` on
  `repository/<first.last>*` for the BatchOperator role.
- **Data**: cross-account S3 access from shared-compute requires whitelisting -> upload
  data into the account's designated bucket instead of pointing at old-account buckets.

## Buckets

All training data, artifacts and checkpoints go to S3 buckets in the
**manip-cluster account (682769330988)** — cross-account permissions from
shared-compute are pre-configured. Do NOT create buckets in shared-compute.
Find/choose a bucket with `aws s3 ls --profile manip-cluster`; request new
ones via #cam-robotics or #ie.

## Unknowns to fill in

1. Which **manip-cluster S3 bucket** to use (see above).
2. Correct **tri.project cost tag** + recommended job **priority** (launcher defaults:
   `MM:PJ-0077`, priority 10).

Then persist:
```bash
echo 'export S3_REMOTE_SYNC=s3://<BUCKET>' >> ~/.bashrc
```

> **Transition week:** the queue moves to shared-compute on 2026-07-24; until then it
> has limited capacity for testing only — run `--hello` / short runs, not sweeps.

## One-time setup

```bash
cp sagemaker/secrets.env.example sagemaker/secrets.env   # then add WANDB_API_KEY
pip install -U 'sagemaker>=2.245' boto3                   # in the env you launch from
```

## Every session

```bash
aws sso login --profile sagemaker    # SSO token expires every ~8-12h
```

## Data upload

The trainer reads a preferences dir (pair subdirs with `rollout_A.hdf5`,
`rollout_B.hdf5`, `preference.json`). Upload it once per dataset version:

```bash
# note: buckets live in manip-cluster, so upload with that profile
aws s3 sync /path/to/preferences/ s3://<BUCKET>/fpl/preferences/<name>_v1/ --profile manip-cluster
```

The whole prefix mounts at `{data}` in the container (input mode `File` =
full download to local NVMe before training; right choice for h5py).

## Test ladder (cheap -> real)

```bash
python sagemaker/smoke_test.py                      # 1. prereqs, no build/submit
python sagemaker/launch_sagemaker.py --hello --dry-run   # 2. print job config only
python sagemaker/launch_sagemaker.py --local \
    --data-s3 file:///abs/path/to/preferences \
    --train-args "--task auto --use_lora --epochs 1"     # 3. image on YOUR gpu
python sagemaker/launch_sagemaker.py --hello        # 4. real queue, trivial job
```

Step 4 proves image pull + PassRole + S3 output + logs end-to-end; after it
succeeds, real runs only change the payload.

## Real runs

```bash
# single run (1 GPU used; the other 7 idle - prefer sweep/DDP on p5)
python sagemaker/launch_sagemaker.py \
    --data-s3 s3://<BUCKET>/fpl/preferences/wipe_v1/ \
    --train-args "--task auto --use_lora --preload --epochs 50 --wandb_run_name wipe_lora"

# one run, 8-way DDP (trainer supports torchrun)
python sagemaker/launch_sagemaker.py --data-s3 s3://... \
    --train-args "--task auto --use_lora --preload --epochs 50" --nproc 8

# N configs in parallel, one per GPU (best use of the node)
python sagemaker/launch_sagemaker.py --data-s3 s3://... --sweep-file my_sweep.txt
```

Notes: `--skip-build` reuses the last pushed image (rebuild only when code/deps
change). `{data}`/`{out}` placeholders work inside `--train-args`. Trainer args
reference: `python real_world/train_reward_model.py --help`.

## Outputs

- **Live checkpoints**: `s3://<BUCKET>/fpl/<user>/fpl-qwen-reward/<job>/checkpoints/`
  (`/opt/ml/checkpoints` syncs continuously - crash-safe)
- **End-of-job**: `.../ <job>/output/model.tar.gz` (marker only; checkpoints are canonical)
- **Metrics**: wandb project `reward_learning` (trainer default)

## Monitoring

```bash
batchy ls fss-tri-cam-robotics-p5-48xlarge-us-west-2
sagey logs <job-name> --follow
sagetui / batchtui
```

## Troubleshooting

| Error | Cause / fix |
|---|---|
| `Token has expired` | `aws sso login --profile sagemaker` |
| queue `does not exist` | wrong account (profile must hit 141701954645) or no queue access |
| `not authorized to perform: iam:PassRole` | your principal can't pass the execution role -> ask IE |
| ECR push denied / repo create denied | BatchOperator role may lack ECR write -> ask IE |
| job stuck RUNNABLE | queue busy (check `batchy ls`) or reserved capacity exhausted |
| `CUDA out of memory` | LoRA run should fit easily in 80GB; check `--batch_size`, `--seq_len` |
| HF download fails in job | container lacks internet egress -> stage weights in S3 (ask before building this) |
| region mismatch errors | profile region must be us-west-2 (`~/.aws/config`) |
