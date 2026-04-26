"""
Download the counting dataset from S3.

Downloads parquet files from s3://counting-dataset/count-dataset-splits/{train,val,test}/data/
into a local directory that ImageDataset can read directly (native parquet support).

    data/counting/train/*.parquet
    data/counting/val/*.parquet
    data/counting/test/*.parquet

No conversion is needed — the training pipeline reads parquet files natively,
decoding PIL images on the fly during cache building and training.

Usage:
    python scripts/counting/download_data.py
    python scripts/counting/download_data.py --splits train val
    python scripts/counting/download_data.py --output-dir /mnt/data/counting
"""

import argparse
import os
from pathlib import Path

import boto3

S3_BUCKET = "counting-dataset"
S3_REGION = "us-west-2"
SPLITS = {
    "train": "count-dataset-splits-filtered/train/data",
    "val": "count-dataset-splits-filtered/val/data",
    "test": "count-dataset-splits-filtered/test/data",
}


def download_split(bucket: str, prefix: str, local_dir: str, region: str) -> list[str]:
    """Download all parquet files from an S3 prefix, skipping already-cached files."""
    s3 = boto3.client("s3", region_name=region)
    os.makedirs(local_dir, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    local_files = []
    downloaded = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".parquet"):
                continue
            filename = os.path.basename(key)
            local_path = os.path.join(local_dir, filename)
            if os.path.exists(local_path) and os.path.getsize(local_path) == obj["Size"]:
                print(f"  Cached {filename}")
            else:
                size_mb = obj["Size"] / 1e6
                print(f"  Downloading {filename} ({size_mb:.1f} MB)...")
                s3.download_file(bucket, key, local_path)
                downloaded += 1
            local_files.append(local_path)

    return local_files


def main():
    parser = argparse.ArgumentParser(
        description="Download counting dataset parquet files from S3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <repo_root>/data/counting)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=list(SPLITS.keys()),
        default=list(SPLITS.keys()),
        help="Which splits to download",
    )
    parser.add_argument("--region", type=str, default=S3_REGION)
    parser.add_argument("--bucket", type=str, default=S3_BUCKET)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "data" / "counting"

    print("=" * 60)
    print("Counting Dataset: Download from S3")
    print("=" * 60)
    print(f"  S3 bucket:  s3://{args.bucket}/")
    print(f"  Region:     {args.region}")
    print(f"  Output dir: {output_dir}")
    print(f"  Splits:     {', '.join(args.splits)}")
    print()

    for split_name in args.splits:
        s3_prefix = SPLITS[split_name]
        split_dir = output_dir / split_name
        print(f"[{split_name}] s3://{args.bucket}/{s3_prefix}/ -> {split_dir}/")
        files = download_split(args.bucket, s3_prefix, str(split_dir), args.region)
        total_mb = sum(os.path.getsize(f) for f in files) / 1e6
        print(f"[{split_name}] {len(files)} parquet file(s), {total_mb:.0f} MB total")
        print()

    print("=" * 60)
    print("Done!")
    print()
    for split_name in args.splits:
        split_dir = output_dir / split_name
        if split_dir.exists():
            pq_files = list(split_dir.glob("*.parquet"))
            total_mb = sum(f.stat().st_size for f in pq_files) / 1e6
            print(f"  {split_name}/: {len(pq_files)} files, {total_mb:.0f} MB")
    print()
    print("Next step: run the training script")
    print("  bash scripts/counting/train_counting.sh")


if __name__ == "__main__":
    main()
