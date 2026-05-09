#!/usr/bin/env bash
set -euo pipefail

# Restore qwen-image-finetune artifacts from S3 backup on a fresh GPU.
#
# Usage:
#   bash scripts/restore_next_gpu.sh \
#     --backup-prefix lambda-backups/20260509_092929 \
#     --repo-root /home/ubuntu/sanjana-fs-us-south-2/qwen-image-finetune
#
# Optional:
#   --skip-deps        Skip pip installs (assume env already prepared)
#   --skip-outputs     Skip restoring outputs
#   --skip-logs        Skip restoring logs
#   --bucket           S3 bucket (default: counting-experiments-backup)
#   --region           AWS region (default: us-west-2)

BUCKET="counting-experiments-backup"
REGION="us-west-2"
BACKUP_PREFIX=""
REPO_ROOT="/home/ubuntu/sanjana-fs-us-south-2/qwen-image-finetune"
SKIP_DEPS=false
SKIP_OUTPUTS=false
SKIP_LOGS=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backup-prefix)
      BACKUP_PREFIX="$2"
      shift 2
      ;;
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --bucket)
      BUCKET="$2"
      shift 2
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    --skip-deps)
      SKIP_DEPS=true
      shift
      ;;
    --skip-outputs)
      SKIP_OUTPUTS=true
      shift
      ;;
    --skip-logs)
      SKIP_LOGS=true
      shift
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$BACKUP_PREFIX" ]]; then
  echo "Error: --backup-prefix is required"
  exit 1
fi

echo "======================================"
echo "Restore from S3 backup"
echo "======================================"
echo "Bucket:       $BUCKET"
echo "Region:       $REGION"
echo "Backup:       $BACKUP_PREFIX"
echo "Repo root:    $REPO_ROOT"
echo "Skip deps:    $SKIP_DEPS"
echo "Skip outputs: $SKIP_OUTPUTS"
echo "Skip logs:    $SKIP_LOGS"
echo "======================================"

mkdir -p "$REPO_ROOT"
cd "$REPO_ROOT"

if ! $SKIP_DEPS; then
  echo
  echo "[1/3] Installing Python dependencies"
  if [[ -f requirements.txt ]]; then
    pip install -r requirements.txt
  fi
  if [[ -f pyproject.toml ]]; then
    pip install -e .
  fi
fi

if ! $SKIP_OUTPUTS; then
  echo
  echo "[2/3] Restoring outputs (non-cache backup)"
  mkdir -p "$REPO_ROOT/outputs"
  aws s3 sync \
    "s3://${BUCKET}/${BACKUP_PREFIX}/outputs/" \
    "$REPO_ROOT/outputs/" \
    --region "$REGION" \
    --no-progress
fi

if ! $SKIP_LOGS; then
  echo
  echo "[3/3] Restoring logs"
  aws s3 cp \
    "s3://${BUCKET}/${BACKUP_PREFIX}/logs/" \
    "$REPO_ROOT/" \
    --recursive \
    --region "$REGION" \
    --no-progress
fi

echo
echo "Restore complete."
echo
echo "Suggested checks:"
echo "  ls outputs/counting_lora/counting_qwen_image_edit/"
echo "  ls outputs/counting_lora/counting_qwen_image_edit/v4/"
echo "  tensorboard --logdir=$REPO_ROOT/outputs/counting_lora/counting_qwen_image_edit"
