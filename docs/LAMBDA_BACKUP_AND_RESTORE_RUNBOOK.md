# Lambda Backup and Restore Runbook

This runbook captures the exact backup/restore flow for ephemeral Lambda GPU sessions.

## Current backup snapshot

- Bucket: `counting-experiments-backup`
- Region: `us-west-2`
- Prefix: `lambda-backups/20260509_092929`
- Scope included:
  - `qwen-image-finetune/outputs/` (excluding `outputs/counting_lora/cache/`)
  - `qwen-image-finetune/*.log`
- Scope excluded:
  - `qwen-image-finetune/outputs/counting_lora/cache/`
  - `qwen-image-finetune/data/counting/` local parquet mirror

## What was saved

- Checkpoints/evals/plots/TensorBoard events:
  - `s3://counting-experiments-backup/lambda-backups/20260509_092929/outputs/counting_lora/counting_qwen_image_edit/`
- Logs:
  - `s3://counting-experiments-backup/lambda-backups/20260509_092929/logs/`

## Verify backup exists

```bash
aws s3 ls s3://counting-experiments-backup/lambda-backups/20260509_092929/outputs/counting_lora/counting_qwen_image_edit/ --recursive --summarize
aws s3 ls s3://counting-experiments-backup/lambda-backups/20260509_092929/logs/
```

## Next GPU setup

### 1) Clone repos and prepare environment

```bash
mkdir -p /home/ubuntu/sanjana-fs-us-south-2
cd /home/ubuntu/sanjana-fs-us-south-2

# Replace with your actual remotes
git clone <qwen-image-finetune-remote>
git clone <count-data-gen-remote>

cd qwen-image-finetune
```

### 2) Restore outputs and logs from S3

Use the helper script:

```bash
bash scripts/restore_next_gpu.sh \
  --backup-prefix lambda-backups/20260509_092929 \
  --repo-root /home/ubuntu/sanjana-fs-us-south-2/qwen-image-finetune \
  --bucket counting-experiments-backup \
  --region us-west-2
```

If your environment is already set up, skip dependency installation:

```bash
bash scripts/restore_next_gpu.sh \
  --backup-prefix lambda-backups/20260509_092929 \
  --repo-root /home/ubuntu/sanjana-fs-us-south-2/qwen-image-finetune \
  --bucket counting-experiments-backup \
  --region us-west-2 \
  --skip-deps
```

### 3) Validate restored artifacts

```bash
cd /home/ubuntu/sanjana-fs-us-south-2/qwen-image-finetune
ls outputs/counting_lora/counting_qwen_image_edit/
ls outputs/counting_lora/counting_qwen_image_edit/v4/
ls *.log
```

### 4) Reopen TensorBoard for v0-v4 telemetry

```bash
tensorboard --logdir=/home/ubuntu/sanjana-fs-us-south-2/qwen-image-finetune/outputs/counting_lora/counting_qwen_image_edit
```

## Notes

- TensorBoard event files (`events.out.tfevents.*`) are artifacts, not source code. Keep them in S3/artifact storage, not git.
- Cache is intentionally excluded to reduce transfer size; rebuild it when you run training again.
