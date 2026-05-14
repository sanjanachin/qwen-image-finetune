"""
Download the counting dataset from S3.

By default, downloads parquet files from:
  s3://counting-dataset/count-dataset-splits-filtered/{train,val,test}/data/

With --only-good, downloads from the LLM-evaluated prefix and keeps only
rows where is_good=True:
  s3://counting-dataset/count-dataset-splits-filtered-evaluated-data/{train,val,test}/data/

In both cases files land in a local directory that ImageDataset can read directly
(native parquet support):

    data/counting/train/*.parquet
    data/counting/val/*.parquet
    data/counting/test/*.parquet

No conversion is needed — the training pipeline reads parquet files natively,
decoding PIL images on the fly during cache building and training.

Usage:
    # Full dataset (default)
    python scripts/counting/download_data.py

    # Quality-filtered: only rows where is_good=True
    python scripts/counting/download_data.py --only-good

    # Other options
    python scripts/counting/download_data.py --splits train val
    python scripts/counting/download_data.py --output-dir /mnt/data/counting

Cache note (--only-good):
    The embedding cache (outputs/counting_lora/cache/) is content-addressed via
    perceptual hash of image pixels and MD5 of the prompt string, so existing cache
    entries for quality-passing rows are fully reused — no cache rebuild is needed
    when toggling --only-good on data from the same generation run.
"""

import argparse
import os
import tempfile
from pathlib import Path

import boto3
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

S3_BUCKET = "counting-dataset"
S3_REGION = "us-west-2"
S3_PREFIX_FILTERED = "count-dataset-splits-filtered"
S3_PREFIX_EVALUATED = "count-dataset-splits-filtered-evaluated-data"


def _split_prefixes(base_prefix: str) -> dict[str, str]:
    return {split: f"{base_prefix}/{split}/data" for split in ("train", "val", "test")}


def _is_already_filtered(
    local_path: str,
    only_good: bool = False,
    count_filter: int | None = None,
) -> bool:
    """Return True if the local parquet already satisfies the requested filters.

    Reads only the columns needed for the check (is_good, count_added) so it
    stays fast even for large shards. Used as the skip-check when any filtering
    is active, replacing the byte-size comparison which breaks when the local file
    is smaller than the S3 object.

    Returns False when no filters are active (caller falls through to the normal
    byte-size check in that case).
    """
    if not only_good and count_filter is None:
        return False
    if not os.path.exists(local_path):
        return False
    try:
        cols = []
        if only_good:
            cols.append("is_good")
        if count_filter is not None:
            cols.append("count_added")
        table = pq.read_table(local_path, columns=cols)
        if len(table) == 0:
            return False
        if only_good:
            col = table.column("is_good")
            if col.null_count != 0 or not pc.all(col).as_py():
                return False
        if count_filter is not None:
            counts = table.column("count_added")
            if not pc.all(pc.equal(counts, count_filter)).as_py():
                return False
        return True
    except Exception:
        return False


def download_split(
    bucket: str,
    prefix: str,
    local_dir: str,
    region: str,
    only_good: bool = False,
    count_filter: int | None = None,
) -> tuple[list[str], int, int]:
    """Download all parquet files from an S3 prefix into local_dir.

    When only_good=True and/or count_filter is set, each shard is filtered before
    being written locally. Filters are composable: when both are active a row must
    satisfy is_good=True AND count_added==count_filter.

    The skip-check is content-based when any filter is active (verifies the relevant
    columns) rather than byte-size-based, so it works correctly across runs and when
    switching between filtered and unfiltered modes.

    Returns:
        (local_files, total_rows_kept, total_rows_source)
    """
    needs_filter = only_good or count_filter is not None

    s3 = boto3.client("s3", region_name=region)
    os.makedirs(local_dir, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    local_files: list[str] = []
    total_kept = 0
    total_source = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".parquet"):
                continue
            filename = os.path.basename(key)
            local_path = os.path.join(local_dir, filename)

            if needs_filter:
                if _is_already_filtered(local_path, only_good=only_good, count_filter=count_filter):
                    n_kept = pq.read_metadata(local_path).num_rows
                    print(f"  Cached {filename} ({n_kept} rows, filters already applied)")
                    total_kept += n_kept
                    local_files.append(local_path)
                    continue

                size_mb = obj["Size"] / 1e6
                print(f"  Downloading {filename} ({size_mb:.1f} MB) and filtering...")

                # Download to a temp file in the same directory so the write stays local
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tmp.parquet", dir=local_dir)
                os.close(tmp_fd)
                try:
                    s3.download_file(bucket, key, tmp_path)
                    table = pq.read_table(tmp_path)
                    n_source = len(table)
                    total_source += n_source

                    if only_good and "is_good" not in table.schema.names:
                        raise ValueError(
                            f"Parquet shard {key!r} has no 'is_good' column. "
                            "Ensure evaluate_split_quality.py has finished and "
                            "uploaded to the evaluated prefix before using --only-good."
                        )
                    if count_filter is not None and "count_added" not in table.schema.names:
                        raise ValueError(
                            f"Parquet shard {key!r} has no 'count_added' column. "
                            "This column is required for --count filtering."
                        )

                    # Build a composable boolean mask
                    mask = pa.array([True] * n_source)
                    if only_good:
                        mask = pc.and_(mask, pc.field("is_good") == True)  # noqa: E712
                    if count_filter is not None:
                        mask = pc.and_(mask, pc.field("count_added") == count_filter)
                    filtered = table.filter(mask)

                    n_kept = len(filtered)
                    total_kept += n_kept
                    pq.write_table(filtered, local_path)
                    print(
                        f"  {filename}: kept {n_kept}/{n_source} rows "
                        f"({100 * n_kept / n_source:.0f}% match filters)"
                    )
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            else:
                if os.path.exists(local_path) and os.path.getsize(local_path) == obj["Size"]:
                    print(f"  Cached {filename}")
                else:
                    size_mb = obj["Size"] / 1e6
                    print(f"  Downloading {filename} ({size_mb:.1f} MB)...")
                    s3.download_file(bucket, key, local_path)

            local_files.append(local_path)

    return local_files, total_kept, total_source


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
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
        help="Which splits to download",
    )
    parser.add_argument(
        "--only-good",
        action="store_true",
        default=False,
        help=(
            "Download only rows where is_good=True, sourced from the LLM-evaluated "
            f"S3 prefix ({S3_PREFIX_EVALUATED}). Requires evaluate_split_quality.py "
            "to have already run and uploaded results to that prefix. The embedding "
            "cache is content-addressed and does not need to be rebuilt when toggling "
            "this flag."
        ),
    )
    parser.add_argument(
        "--evaluated-prefix",
        type=str,
        default=S3_PREFIX_EVALUATED,
        help="S3 prefix for LLM-evaluated data (used with --only-good)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Keep only rows where count_added equals N. count_added is the "
            "SAM3-verified number of objects actually added and is the value "
            "substituted into the training prompt. Independent of --only-good; "
            "when both are set rows must satisfy both conditions. Example: --count 9"
        ),
    )
    parser.add_argument("--region", type=str, default=S3_REGION)
    parser.add_argument("--bucket", type=str, default=S3_BUCKET)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "data" / "counting"

    base_prefix = args.evaluated_prefix if args.only_good else S3_PREFIX_FILTERED
    splits = _split_prefixes(base_prefix)

    print("=" * 60)
    print("Counting Dataset: Download from S3")
    print("=" * 60)
    print(f"  S3 bucket:   s3://{args.bucket}/")
    print(f"  S3 prefix:   {base_prefix}")
    print(f"  Region:      {args.region}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Splits:      {', '.join(args.splits)}")
    print(f"  Only-good:   {'yes (is_good=True rows only)' if args.only_good else 'no (all rows)'}")
    print(f"  Count filter: {args.count if args.count is not None else 'none (all counts)'}")
    print()

    split_stats: dict[str, tuple[int, int]] = {}
    for split_name in args.splits:
        s3_prefix = splits[split_name]
        split_dir = output_dir / split_name
        print(f"[{split_name}] s3://{args.bucket}/{s3_prefix}/ -> {split_dir}/")
        files, kept, source = download_split(
            args.bucket,
            s3_prefix,
            str(split_dir),
            args.region,
            only_good=args.only_good,
            count_filter=args.count,
        )
        total_mb = sum(os.path.getsize(f) for f in files) / 1e6
        if (args.only_good or args.count is not None) and source > 0:
            print(
                f"[{split_name}] {len(files)} parquet file(s), {total_mb:.0f} MB — "
                f"{kept}/{source} rows kept ({100 * kept / source:.0f}% match filters)"
            )
        else:
            print(f"[{split_name}] {len(files)} parquet file(s), {total_mb:.0f} MB total")
        split_stats[split_name] = (kept, source)
        print()

    print("=" * 60)
    print("Done!")
    print()
    for split_name in args.splits:
        split_dir = output_dir / split_name
        if split_dir.exists():
            pq_files = list(split_dir.glob("*.parquet"))
            total_mb = sum(f.stat().st_size for f in pq_files) / 1e6
            kept, source = split_stats.get(split_name, (0, 0))
            if (args.only_good or args.count is not None) and source > 0:
                print(
                    f"  {split_name}/: {len(pq_files)} files, {total_mb:.0f} MB "
                    f"({kept}/{source} rows, {100 * kept / source:.0f}% match filters)"
                )
            else:
                print(f"  {split_name}/: {len(pq_files)} files, {total_mb:.0f} MB")
    print()
    print("Next step: run the training script")
    print("  bash scripts/counting/train_counting.sh")


if __name__ == "__main__":
    main()
