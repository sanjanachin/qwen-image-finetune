"""
Evaluate counting accuracy across training checkpoints to track how counting
ability evolves over training.

Loads checkpoints at a configurable step interval, runs a reproducible subset
of the test set (default: 5 samples per count, 90 total) through each, counts
objects with SAM3, and produces:

    1. sweep_results.json  -- all numerical results
    2. 7 PNG plots:
       - overall_accuracy.png      (Accuracy + Within-1 Accuracy)
       - overall_mae.png
       - overall_medae.png
       - overall_mean_signed_error.png
       - per_count_mae.png         (heatmap)
       - per_count_medae.png       (heatmap)
       - per_count_mean_signed_error.png  (heatmap, diverging)

The base model (no LoRA) is evaluated first and plotted as step 0.

Usage:
    python scripts/counting/eval_checkpoint_sweep.py \\
        --run-dir outputs/counting_lora/counting_qwen_image_edit/v2 \\
        --step-interval 5000 \\
        --samples-per-count 5
"""

import argparse
import gc
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO_ROOT / "scripts" / "counting"))
from eval_counting import (
    SAM3Counter,
    compute_metrics,
    compute_per_count_metrics,
    load_pipeline,
    run_inference,
)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def discover_checkpoints(run_dir: Path, step_interval: int) -> list[tuple[int, Path]]:
    """Find checkpoint directories whose step number is divisible by step_interval.

    Returns list of (step, path) sorted by step ascending.
    """
    pattern = re.compile(r"checkpoint-\d+-(\d+)$")
    results = []
    for d in run_dir.iterdir():
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m is None:
            continue
        step = int(m.group(1))
        lora_file = d / "pytorch_lora_weights.safetensors"
        if not lora_file.exists():
            continue
        if step % step_interval == 0:
            results.append((step, d))
    results.sort(key=lambda x: x[0])
    return results


# ---------------------------------------------------------------------------
# Subset sampling
# ---------------------------------------------------------------------------

def subsample_test_set(
    samples: list[dict[str, Any]],
    samples_per_count: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select a fixed, reproducible subset with samples_per_count entries per
    ground-truth count value."""
    rng = np.random.default_rng(seed)
    by_count: dict[int, list[dict]] = defaultdict(list)
    for s in samples:
        by_count[s["count_added"]].append(s)

    subset = []
    for count in sorted(by_count):
        pool = by_count[count]
        n = min(samples_per_count, len(pool))
        indices = rng.choice(len(pool), size=n, replace=False)
        for i in sorted(indices):
            subset.append(pool[i])
    return subset


# ---------------------------------------------------------------------------
# Single-checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    pipeline,
    samples: list[dict],
    counter: SAM3Counter,
    inf_cfg: dict,
    seed: int | None,
    save_dir: Path | None = None,
) -> tuple[list[int], list[int]]:
    """Run inference + SAM3 counting, return (ground_truths, predictions).

    If save_dir is provided, saves each generated image and a per-sample
    metadata JSON file into that directory.
    """
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    generator = (
        torch.Generator(device="cuda").manual_seed(seed)
        if seed is not None
        else None
    )

    ground_truths: list[int] = []
    predictions: list[int] = []

    for idx, sample in enumerate(tqdm(samples, desc="eval", leave=False)):
        original_image: Image.Image = sample["original_image"]
        if not isinstance(original_image, Image.Image):
            original_image = Image.open(original_image).convert("RGB")
        else:
            original_image = original_image.convert("RGB")

        if generator is not None and seed is not None:
            generator.manual_seed(seed + idx)

        gen_img = run_inference(
            pipeline,
            image=original_image,
            prompt=sample["prompt"],
            negative_prompt=inf_cfg["negative_prompt"],
            num_inference_steps=inf_cfg["num_inference_steps"],
            true_cfg_scale=inf_cfg["true_cfg_scale"],
            generator=generator,
        )

        detected = counter.count(gen_img, sample["object_name"])
        ground_truths.append(sample["count_added"])
        predictions.append(detected)

        if save_dir is not None:
            gen_img.save(save_dir / f"sample_{idx:03d}.png")
            meta = {
                "idx": idx,
                "prompt": sample["prompt"],
                "object_name": sample["object_name"],
                "ground_truth": sample["count_added"],
                "detected": detected,
                "error": abs(sample["count_added"] - detected),
            }
            with open(save_dir / f"sample_{idx:03d}_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

    return ground_truths, predictions


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_overall_line(
    steps: list[int],
    values_dict: dict[str, list[float]],
    title: str,
    ylabel: str,
    save_path: Path,
    y_ref_line: float | None = None,
):
    """Line chart for one or more overall metrics over training steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, vals in values_dict.items():
        ax.plot(steps, vals, marker="o", markersize=4, label=label)
    if y_ref_line is not None:
        ax.axhline(y=y_ref_line, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(values_dict) > 1:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path.name}")


def _plot_heatmap(
    steps: list[int],
    counts: list[int],
    matrix: np.ndarray,
    title: str,
    save_path: Path,
    diverging: bool = False,
):
    """Heatmap with x=step, y=count, color=metric value."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(10, len(steps) * 0.8), 8))

    if diverging:
        vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
        im = ax.imshow(
            matrix, aspect="auto", cmap="RdBu_r",
            vmin=-vmax, vmax=vmax, origin="lower",
        )
    else:
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", origin="lower")

    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels([str(c) for c in counts])
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Requested Count")
    ax.set_title(title)

    for i in range(len(counts)):
        for j in range(len(steps)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path.name}")


def generate_plots(results: dict, output_dir: Path):
    """Generate all 7 plots from the sweep results dict."""
    checkpoints = results["checkpoints"]
    steps = sorted(int(s) for s in checkpoints)
    step_keys = [str(s) for s in steps]

    print("\nGenerating plots...")

    # --- Overall line charts ---
    def _get_overall(metric: str) -> list[float]:
        return [checkpoints[sk]["overall"][metric] for sk in step_keys]

    # 1. Accuracy + Within-1 Accuracy
    _plot_overall_line(
        steps,
        {
            "Accuracy": _get_overall("accuracy"),
            "Within-1 Accuracy": _get_overall("within_1_accuracy"),
        },
        title="Counting Accuracy over Training",
        ylabel="Accuracy (%)",
        save_path=output_dir / "overall_accuracy.png",
    )

    # 2. MAE
    _plot_overall_line(
        steps,
        {"MAE": _get_overall("mae")},
        title="Mean Absolute Error over Training",
        ylabel="MAE",
        save_path=output_dir / "overall_mae.png",
    )

    # 3. MedAE
    _plot_overall_line(
        steps,
        {"MedAE": _get_overall("medae")},
        title="Median Absolute Error over Training",
        ylabel="MedAE",
        save_path=output_dir / "overall_medae.png",
    )

    # 4. Mean Signed Error
    _plot_overall_line(
        steps,
        {"Mean Signed Error": _get_overall("mean_signed_error")},
        title="Mean Signed Error over Training (+ = overcounting)",
        ylabel="Mean Signed Error",
        save_path=output_dir / "overall_mean_signed_error.png",
        y_ref_line=0.0,
    )

    # --- Per-count heatmaps ---
    all_counts = set()
    for sk in step_keys:
        all_counts.update(int(c) for c in checkpoints[sk]["per_count"])
    counts = sorted(all_counts)

    def _build_matrix(metric: str) -> np.ndarray:
        mat = np.full((len(counts), len(steps)), np.nan)
        for j, sk in enumerate(step_keys):
            pc = checkpoints[sk]["per_count"]
            for i, c in enumerate(counts):
                if str(c) in pc:
                    mat[i, j] = pc[str(c)][metric]
        return mat

    # 5. Per-count MAE
    _plot_heatmap(
        steps, counts, _build_matrix("mae"),
        title="Per-Count MAE over Training",
        save_path=output_dir / "per_count_mae.png",
    )

    # 6. Per-count MedAE
    _plot_heatmap(
        steps, counts, _build_matrix("medae"),
        title="Per-Count MedAE over Training",
        save_path=output_dir / "per_count_medae.png",
    )

    # 7. Per-count Mean Signed Error (diverging)
    _plot_heatmap(
        steps, counts, _build_matrix("mean_signed_error"),
        title="Per-Count Mean Signed Error (blue=under, red=over)",
        save_path=output_dir / "per_count_mean_signed_error.png",
        diverging=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sweep checkpoints and evaluate counting accuracy over training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir", required=True,
        help="Training run directory containing checkpoint-* subdirectories",
    )
    parser.add_argument(
        "--step-interval", type=int, default=5000,
        help="Only evaluate checkpoints whose step is divisible by this value",
    )
    parser.add_argument(
        "--samples-per-count", type=int, default=5,
        help="Number of test samples per ground-truth count value",
    )
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "counting_eval.yaml"),
        help="Path to evaluation config YAML",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for plots and JSON (default: <run-dir>/sweep_results/)",
    )
    parser.add_argument(
        "--save-images", action="store_true",
        help="Save all generated images and per-sample metadata for each checkpoint",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        sys.exit(f"Error: run directory not found: {run_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "sweep_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    inf_cfg = cfg["inference"]
    seed = inf_cfg.get("seed")
    model_path = cfg["model"]["pretrained_model_name_or_path"]

    # ------------------------------------------------------------------
    # Discover checkpoints
    # ------------------------------------------------------------------
    checkpoints = discover_checkpoints(run_dir, args.step_interval)
    if not checkpoints:
        sys.exit(f"No checkpoints found in {run_dir} with step interval {args.step_interval}")

    print(f"Found {len(checkpoints)} checkpoints: "
          f"steps {checkpoints[0][0]} to {checkpoints[-1][0]}")

    # ------------------------------------------------------------------
    # Load test data and subsample
    # ------------------------------------------------------------------
    parquet_dir = cfg["data"]["test_parquet_dir"]
    if not Path(parquet_dir).is_absolute():
        parquet_dir = str(REPO_ROOT / parquet_dir)

    print("Loading test data...")
    ds = load_dataset("parquet", data_files=f"{parquet_dir}/*.parquet", split="train")
    all_samples = []
    for row in ds:
        all_samples.append({
            "original_image": row["original_image"],
            "prompt": row["prompt"],
            "object_name": row["object_name"],
            "count_added": row["count_added"],
        })

    samples = subsample_test_set(all_samples, args.samples_per_count, seed=42)
    counts_in_subset = sorted(set(s["count_added"] for s in samples))
    print(f"Subsampled {len(samples)} entries "
          f"({args.samples_per_count}/count, counts {counts_in_subset[0]}-{counts_in_subset[-1]})")

    # ------------------------------------------------------------------
    # Load existing results for resume
    # ------------------------------------------------------------------
    results_path = output_dir / "sweep_results.json"
    results: dict[str, Any] = {
        "config": {
            "run_dir": str(run_dir),
            "step_interval": args.step_interval,
            "samples_per_count": args.samples_per_count,
            "eval_config": args.config,
            "num_samples": len(samples),
            "counts": counts_in_subset,
        },
        "checkpoints": {},
    }

    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print(f"Resumed from {results_path} "
              f"({len(results['checkpoints'])} checkpoints already evaluated)")

    # ------------------------------------------------------------------
    # Initialize SAM3
    # ------------------------------------------------------------------
    sam3_cfg = cfg["sam3"]
    sam3_path = sam3_cfg["path"]
    if not Path(sam3_path).is_absolute():
        sam3_path = str(REPO_ROOT / sam3_path)

    counter = SAM3Counter(
        sam3_path=sam3_path,
        minimum_confidence_threshold=sam3_cfg["minimum_confidence_threshold"],
        confidence_threshold=sam3_cfg["confidence_threshold"],
        nms_iou_threshold=sam3_cfg["nms_iou_threshold"],
    )

    # ------------------------------------------------------------------
    # Load pipeline once
    # ------------------------------------------------------------------
    pipe = load_pipeline(model_path)

    # ------------------------------------------------------------------
    # Evaluate base model (step 0)
    # ------------------------------------------------------------------
    if "0" not in results["checkpoints"]:
        print("\n=== Step 0: Base model (no LoRA) ===")
        img_dir = (output_dir / "images" / "step_0") if args.save_images else None
        gt, pred = evaluate_checkpoint(pipe, samples, counter, inf_cfg, seed, save_dir=img_dir)
        overall = compute_metrics(gt, pred)
        per_count = compute_per_count_metrics(gt, pred)
        results["checkpoints"]["0"] = {
            "overall": overall,
            "per_count": {str(k): v for k, v in per_count.items()},
        }
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Accuracy={overall['accuracy']:.1f}%  MAE={overall['mae']:.2f}  "
              f"MSE={overall['mean_signed_error']:.2f}")
    else:
        print("\nStep 0 (base model) already evaluated, skipping.")

    # ------------------------------------------------------------------
    # Evaluate each checkpoint
    # ------------------------------------------------------------------
    for step, ckpt_path in checkpoints:
        step_key = str(step)
        if step_key in results["checkpoints"]:
            print(f"\nStep {step} already evaluated, skipping.")
            continue

        print(f"\n=== Step {step}: {ckpt_path.name} ===")
        lora_file = str(ckpt_path / "pytorch_lora_weights.safetensors")
        pipe.load_lora_weights(lora_file)

        img_dir = (output_dir / "images" / f"step_{step}") if args.save_images else None
        gt, pred = evaluate_checkpoint(pipe, samples, counter, inf_cfg, seed, save_dir=img_dir)

        pipe.unload_lora_weights()

        overall = compute_metrics(gt, pred)
        per_count = compute_per_count_metrics(gt, pred)

        results["checkpoints"][step_key] = {
            "overall": overall,
            "per_count": {str(k): v for k, v in per_count.items()},
        }

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"  Accuracy={overall['accuracy']:.1f}%  MAE={overall['mae']:.2f}  "
              f"MSE={overall['mean_signed_error']:.2f}")

    # ------------------------------------------------------------------
    # Cleanup GPU
    # ------------------------------------------------------------------
    del pipe
    counter.unload()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    generate_plots(results, output_dir)

    print(f"\nAll results saved to: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
