"""
Download all train examples with a specified count for visual auditing of the
SAM3 labels.

Each parquet file is downloaded once into a temp dir, scanned cheaply with
column projection on a single integer column, optionally subsampled, then the
images for matching rows are extracted from the same local cache.

Output layout:

    <output-dir>/count_<N>/<id>/
        original.png   - input image (before edit)
        edited.png     - target image (after edit)
        metadata.json  - all non-image columns + source parquet info

Usage examples:

    # All train entries where count_added == 18
    python scripts/counting/inspect_train_counts.py 18

    # Random 30-entry sample of count_added == 18, reproducible
    python scripts/counting/inspect_train_counts.py 18 --max-entries 30 --seed 42

    # Filter by a different column (e.g. SAM3-detected count after edit)
    python scripts/counting/inspect_train_counts.py 18 --column actual_count_edited

    # Use the val split instead of train
    python scripts/counting/inspect_train_counts.py 18 \\
        --prefix count-dataset-splits-filtered/val/data
"""

import argparse
import io
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import boto3
import pyarrow.parquet as pq
from PIL import Image

S3_BUCKET = "counting-dataset"
S3_REGION = "us-west-2"
S3_PREFIX = "count-dataset-splits-filtered/train/data"

# Columns saved into metadata.json (everything except the image structs).
METADATA_COLUMNS = [
    "id",
    "object_name",
    "prompt_template",
    "prompt",
    "requested_count",
    "actual_count_edited",
    "actual_count_original",
    "count_added",
    "confidence_scores_edited",
    "confidence_scores_above_20_edited",
    "confidence_scores_original",
    "confidence_scores_above_20_original",
]

INT_FILTER_COLUMNS = [
    "count_added",
    "requested_count",
    "actual_count_edited",
    "actual_count_original",
]


def list_parquet_keys(s3, bucket: str, prefix: str) -> list[str]:
    """Enumerate every .parquet object under the given S3 prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                keys.append(obj["Key"])
    return sorted(keys)


def save_image_field(image_struct, out_path: Path) -> None:
    """Decode the parquet image struct {bytes, path} and save as PNG."""
    if isinstance(image_struct, dict):
        image_bytes = image_struct["bytes"]
    elif isinstance(image_struct, (bytes, bytearray)):
        image_bytes = bytes(image_struct)
    else:
        raise TypeError(f"Unsupported image cell type: {type(image_struct)}")
    Image.open(io.BytesIO(image_bytes)).convert("RGB").save(out_path)


def make_jsonable(v):
    """Convert pandas/numpy/pyarrow types into vanilla Python for json.dump."""
    if hasattr(v, "tolist"):  # numpy arrays / pandas series
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [make_jsonable(x) for x in v]
    if hasattr(v, "item"):  # numpy scalars
        return v.item()
    return v


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download train examples with a specified count for visual inspection. "
            "Useful for auditing SAM3 labelling quality at problematic counts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "count", type=int,
        help="Count value to filter on (matched against --column).",
    )
    parser.add_argument(
        "--column", default="count_added", choices=INT_FILTER_COLUMNS,
        help="Integer column to filter on.",
    )
    parser.add_argument(
        "--max-entries", type=int, default=None,
        help="Cap total entries (random subsample). Default: keep all matches.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for subsampling.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output base dir (default: <repo_root>/data/counting/inspection/).",
    )
    parser.add_argument("--bucket", default=S3_BUCKET)
    parser.add_argument("--region", default=S3_REGION)
    parser.add_argument(
        "--prefix", default=S3_PREFIX,
        help="S3 prefix under <bucket>/ containing the parquet files to scan.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    base_out = Path(args.output_dir) if args.output_dir else (
        repo_root / "data" / "counting" / "inspection"
    )
    out_dir = base_out / f"count_{args.count}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Counting Dataset: Inspection Download")
    print("=" * 60)
    print(f"  S3 source:  s3://{args.bucket}/{args.prefix}/")
    print(f"  Filter:     {args.column} == {args.count}")
    print(f"  Output:     {out_dir}/")
    if args.max_entries is not None:
        print(f"  Max:        {args.max_entries} entries (random, seed={args.seed})")
    print()

    s3 = boto3.client("s3", region_name=args.region)

    parquet_keys = list_parquet_keys(s3, args.bucket, args.prefix)
    if not parquet_keys:
        sys.exit(f"Error: no .parquet objects under s3://{args.bucket}/{args.prefix}/")
    print(f"[1/2] Scanning {len(parquet_keys)} parquet file(s)...")

    # Cache parquet files in a single tempdir so phase 2 reuses phase-1 downloads.
    with tempfile.TemporaryDirectory() as tmpdir:
        matching: list[tuple[str, int]] = []  # (key, row_idx)
        total_rows = 0
        for ki, key in enumerate(parquet_keys, 1):
            local = os.path.join(tmpdir, os.path.basename(key))
            s3.download_file(args.bucket, key, local)
            col = pq.read_table(local, columns=[args.column]).column(args.column).to_pylist()
            hits = [(key, i) for i, v in enumerate(col) if v == args.count]
            matching.extend(hits)
            total_rows += len(col)
            print(
                f"      [{ki}/{len(parquet_keys)}] {os.path.basename(key)}: "
                f"{len(col)} rows, {len(hits)} match"
            )

        pct = 100.0 * len(matching) / max(total_rows, 1)
        print()
        print(
            f"      Found {len(matching)} matching entries across "
            f"{total_rows} total rows ({pct:.2f}%)"
        )

        if not matching:
            print(f"\nNo entries with {args.column}={args.count}. Nothing to save.")
            return

        if args.max_entries is not None and args.max_entries < len(matching):
            rng = random.Random(args.seed)
            matching = rng.sample(matching, args.max_entries)
            print(f"      Subsampled to {len(matching)} entries (seed={args.seed})")

        # Group by parquet so each file is loaded once for image extraction.
        by_key: dict[str, list[int]] = {}
        for key, row_idx in matching:
            by_key.setdefault(key, []).append(row_idx)

        print(
            f"\n[2/2] Extracting {len(matching)} entries from "
            f"{len(by_key)} parquet file(s)..."
        )
        saved = 0
        for key in sorted(by_key.keys()):
            rows = by_key[key]
            local = os.path.join(tmpdir, os.path.basename(key))
            df = pq.read_table(local).to_pandas()
            for row_idx in rows:
                row = df.iloc[row_idx]
                entry_id = str(row["id"])
                entry_dir = out_dir / entry_id
                entry_dir.mkdir(exist_ok=True)

                save_image_field(row["original_image"], entry_dir / "original.png")
                save_image_field(row["edited_image"], entry_dir / "edited.png")

                meta = {c: make_jsonable(row[c]) for c in METADATA_COLUMNS if c in df.columns}
                meta["_source_parquet"] = f"s3://{args.bucket}/{key}"
                meta["_source_row_idx"] = int(row_idx)
                with open(entry_dir / "metadata.json", "w") as f:
                    json.dump(meta, f, indent=2)
                saved += 1
            print(f"      {os.path.basename(key)}: saved {len(rows)} entries")

    print()
    print("=" * 60)
    print(f"Done. Saved {saved} entries to {out_dir}/")
    print("=" * 60)
    print()
    print("Each entry directory contains:")
    print("  original.png   - input image (before edit)")
    print("  edited.png     - target image (after edit)")
    print("  metadata.json  - prompt, all count fields, SAM3 confidence scores")


if __name__ == "__main__":
    main()
