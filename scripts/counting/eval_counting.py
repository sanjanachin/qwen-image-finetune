"""
Evaluate counting accuracy: base Qwen-Image-Edit vs a fine-tuned LoRA checkpoint.

For every test sample the script generates an edited image with each model,
counts objects in the output using SAM3, and compares the detected count to
the ground-truth count from the dataset prompt.

Metrics (reported overall and per ground-truth count):
    - Accuracy (exact match)
    - Within-1 Accuracy (|error| <= 1)
    - Mean Absolute Error (MAE)
    - Median Absolute Error (MedAE)
    - Mean Signed Error (positive = over-counting)
    - RMSE

Usage:
    python scripts/counting/eval_counting.py \\
        --checkpoint outputs/counting_lora/counting_qwen_image_edit/v0/checkpoint-0-5000

    # Save per-sample images and metadata:
    python scripts/counting/eval_counting.py \\
        --checkpoint outputs/counting_lora/counting_qwen_image_edit/v0/checkpoint-0-5000 \\
        --save-results

    # Custom config:
    python scripts/counting/eval_counting.py \\
        --checkpoint /path/to/checkpoint \\
        --config configs/counting_eval.yaml
"""

import argparse
import gc
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_eval_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# SAM3 counting (mirrors count-data-gen stage 5 two-threshold design)
# ---------------------------------------------------------------------------

class SAM3Counter:
    """Count objects in images using SAM3.

    Uses the same two-threshold approach as count-data-gen/src/utils/sam3_client.py:
    a low ``minimum_confidence_threshold`` is passed to Sam3Processor so that a
    wide set of candidate detections is returned; a higher
    ``confidence_threshold`` is then applied as a post-filter to determine the
    authoritative count.  Mask-IoU NMS is optionally applied in between.
    """

    def __init__(
        self,
        sam3_path: str = "../sam3",
        minimum_confidence_threshold: float = 0.2,
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.5,
        device: str = "cuda",
    ):
        self.sam3_path = Path(sam3_path).resolve()
        self.minimum_confidence_threshold = minimum_confidence_threshold
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = device
        self.model = None
        self.processor = None
        self._autocast = None

        self._setup_path()

    def _setup_path(self):
        try:
            import sam3  # noqa: F401
        except ImportError:
            if self.sam3_path.exists() and str(self.sam3_path) not in sys.path:
                sys.path.insert(0, str(self.sam3_path))

    def _ensure_loaded(self):
        if self.model is not None:
            return

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        import sam3
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        bpe_path = str(Path(sam3.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz")
        self.model = build_sam3_image_model(bpe_path=bpe_path, device=self.device)
        self.processor = Sam3Processor(
            self.model,
            device=self.device,
            confidence_threshold=self.minimum_confidence_threshold,
        )
        self._autocast = torch.autocast("cuda", dtype=torch.bfloat16)
        print(f"[SAM3] Loaded on {self.device}  "
              f"(processor thresh={self.minimum_confidence_threshold}, "
              f"post-filter thresh={self.confidence_threshold}, "
              f"NMS IoU={self.nms_iou_threshold})")

    def _apply_nms(self, inference_state: dict):
        masks = inference_state.get("masks", [])
        scores = inference_state.get("scores", None)

        if self.nms_iou_threshold <= 0 or scores is None or len(masks) < 2:
            return masks, scores

        from sam3.perflib.nms import nms_masks

        keep = nms_masks(
            pred_probs=scores,
            pred_masks=masks.squeeze(1).float(),
            prob_threshold=0.0,
            iou_threshold=self.nms_iou_threshold,
        )
        return masks[keep], scores[keep]

    def count(self, image: Image.Image, object_name: str, max_retries: int = 3) -> int:
        """Return the number of ``object_name`` instances detected in ``image``."""
        last_err = None
        for attempt in range(max_retries):
            try:
                self._ensure_loaded()
                with self._autocast:
                    state = self.processor.set_image(image)
                    self.processor.reset_all_prompts(state)
                    state = self.processor.set_text_prompt(state=state, prompt=object_name)

                masks, scores = self._apply_nms(state)

                if scores is not None:
                    above = [s for s in scores.cpu().tolist() if s > self.confidence_threshold]
                    return len(above)
                return len(masks) if masks is not None else 0

            except Exception as e:
                last_err = e
                is_oom = "out of memory" in str(e).lower()
                print(f"[SAM3] Attempt {attempt + 1}/{max_retries} "
                      f"{'OOM' if is_oom else 'ERROR'}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                if attempt < max_retries - 1:
                    time.sleep(5 if is_oom else 2)

        print(f"[SAM3] All {max_retries} attempts failed for '{object_name}'")
        raise last_err  # type: ignore[misc]

    def unload(self):
        """Release GPU memory held by the SAM3 model."""
        del self.model, self.processor, self._autocast
        self.model = self.processor = self._autocast = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Qwen-Image-Edit inference helpers
# ---------------------------------------------------------------------------

def load_pipeline(model_path: str, lora_path: str | None = None):
    """Load QwenImageEditPipeline in bf16 on CUDA, optionally with LoRA."""
    from diffusers import QwenImageEditPipeline

    print(f"[Model] Loading pipeline: {model_path}")
    pipe = QwenImageEditPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    if lora_path is not None:
        print(f"[Model] Loading LoRA weights: {lora_path}")
        pipe.load_lora_weights(lora_path)

    print("[Model] Pipeline ready")
    return pipe


def run_inference(
    pipeline,
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    true_cfg_scale: float,
    generator: torch.Generator | None = None,
) -> Image.Image:
    height, width = image.size[1], image.size[0]
    height = (height // 16) * 16
    width = (width // 16) * 16

    output = pipeline(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        true_cfg_scale=true_cfg_scale,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        generator=generator,
    )
    return output.images[0]


def free_pipeline(pipeline):
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(ground_truth: list[int], predicted: list[int]) -> dict[str, float]:
    gt = np.array(ground_truth, dtype=float)
    pred = np.array(predicted, dtype=float)
    errors = np.abs(gt - pred)
    signed = pred - gt

    return {
        "accuracy": float(np.mean(gt == pred) * 100.0),
        "within_1_accuracy": float(np.mean(errors <= 1) * 100.0),
        "mae": float(np.mean(errors)),
        "medae": float(np.median(errors)),
        "mean_signed_error": float(np.mean(signed)),
        "rmse": float(np.sqrt(np.mean(signed ** 2))),
    }


def compute_per_count_metrics(
    ground_truth: list[int],
    predicted: list[int],
) -> dict[int, dict[str, float]]:
    by_count: dict[int, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for gt, pred in zip(ground_truth, predicted):
        by_count[gt][0].append(gt)
        by_count[gt][1].append(pred)

    return {
        count: compute_metrics(gts, preds)
        for count, (gts, preds) in sorted(by_count.items())
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "accuracy": ("Accuracy (exact match)", "%", "pp"),
    "within_1_accuracy": ("Within-1 Accuracy", "%", "pp"),
    "mae": ("Mean Absolute Error", "", ""),
    "medae": ("Median Absolute Error", "", ""),
    "mean_signed_error": ("Mean Signed Error", "", ""),
    "rmse": ("RMSE", "", ""),
}


def _fmt_val(key: str, val: float) -> str:
    suffix = METRIC_LABELS[key][1]
    if suffix == "%":
        return f"{val:.1f}%"
    return f"{val:.2f}"


def _fmt_delta(key: str, delta: float) -> str:
    unit = METRIC_LABELS[key][2]
    if unit == "pp":
        return f"{delta:+.1f} pp"
    return f"{delta:+.2f}"


def print_overall_report(
    base: dict[str, float],
    ft: dict[str, float],
    num_samples: int,
    checkpoint: str,
    count_range: tuple[int, int],
):
    w = 80
    print()
    print("=" * w)
    print("Counting Evaluation: Base Model vs Fine-tuned")
    print("=" * w)
    print(f"Test Set: {num_samples} images  |  Counts: {count_range[0]}-{count_range[1]}  "
          f"|  Checkpoint: {checkpoint}")
    print()

    header = f"{'Metric':<30} {'Base Model':>12}  {'Fine-tuned':>12}  {'Delta':>12}"
    print(header)
    print("-" * w)

    for key in METRIC_LABELS:
        label = METRIC_LABELS[key][0]
        bv, fv = base[key], ft[key]
        delta = fv - bv
        print(f"{label:<30} {_fmt_val(key, bv):>12}  {_fmt_val(key, fv):>12}  {_fmt_delta(key, delta):>12}")

    print("=" * w)


def print_per_count_report(
    base_pc: dict[int, dict[str, float]],
    ft_pc: dict[int, dict[str, float]],
):
    counts = sorted(set(base_pc) | set(ft_pc))
    if not counts:
        return

    print()
    print("Per-Count Results")
    header = (f"{'Count':>5}  {'N':>3}  "
              f"{'Base Acc':>8}  {'FT Acc':>8}  {'D Acc':>7}  "
              f"{'Base MAE':>8}  {'FT MAE':>8}  {'D MAE':>7}  "
              f"{'Base MedAE':>10}  {'FT MedAE':>10}  {'D MedAE':>8}")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for c in counts:
        bm = base_pc.get(c, {})
        fm = ft_pc.get(c, {})
        n = 12  # balanced dataset

        ba, fa = bm.get("accuracy", 0), fm.get("accuracy", 0)
        bmae, fmae = bm.get("mae", 0), fm.get("mae", 0)
        bmed, fmed = bm.get("medae", 0), fm.get("medae", 0)
        print(f"{c:>5}  {n:>3}  "
              f"{ba:>7.1f}%  {fa:>7.1f}%  {fa - ba:>+6.1f}  "
              f"{bmae:>8.2f}  {fmae:>8.2f}  {fmae - bmae:>+7.2f}  "
              f"{bmed:>10.1f}  {fmed:>10.1f}  {fmed - bmed:>+8.1f}")

    print("-" * len(header))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_data(parquet_dir: str) -> list[dict[str, Any]]:
    """Load test samples from parquet files.

    Returns a list of dicts with keys:
        original_image, prompt, object_name, count_added
    """
    ds = load_dataset("parquet", data_files=f"{parquet_dir}/*.parquet", split="train")
    samples = []
    for row in ds:
        samples.append({
            "original_image": row["original_image"],
            "prompt": row["prompt"],
            "object_name": row["object_name"],
            "count_added": row["count_added"],
        })
    return samples


def extract_count_from_prompt(prompt: str) -> int | None:
    """Parse the requested count from prompts like 'add 14 bicyclists on the bridge'."""
    m = re.search(r"add (\d+) ", prompt, re.IGNORECASE)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_model(
    pipeline,
    samples: list[dict],
    sam3_counter: SAM3Counter,
    cfg: dict,
    label: str,
    save_dir: Path | None = None,
) -> tuple[list[int], list[int], list[Image.Image]]:
    """Run inference + SAM3 counting for all samples with one model.

    Returns (ground_truths, predictions, generated_images).
    """
    inf_cfg = cfg["inference"]
    neg_prompt = inf_cfg["negative_prompt"]
    steps = inf_cfg["num_inference_steps"]
    cfg_scale = inf_cfg["true_cfg_scale"]
    seed = inf_cfg.get("seed")

    generator = torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None

    ground_truths: list[int] = []
    predictions: list[int] = []
    images_out: list[Image.Image] = []

    for idx, sample in enumerate(tqdm(samples, desc=f"{label} inference")):
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
            negative_prompt=neg_prompt,
            num_inference_steps=steps,
            true_cfg_scale=cfg_scale,
            generator=generator,
        )

        detected = sam3_counter.count(gen_img, sample["object_name"])
        gt = sample["count_added"]

        ground_truths.append(gt)
        predictions.append(detected)
        images_out.append(gen_img)

        if save_dir is not None:
            sd = save_dir / f"sample_{idx:05d}"
            sd.mkdir(parents=True, exist_ok=True)
            gen_img.save(sd / f"{label}_output.png")
            if idx == 0 or not (save_dir / f"sample_{idx:05d}" / "control.png").exists():
                original_image.save(sd / "control.png")
            meta = {
                "idx": idx,
                "prompt": sample["prompt"],
                "object_name": sample["object_name"],
                "ground_truth": gt,
                f"{label}_detected": detected,
                f"{label}_error": abs(gt - detected),
            }
            with open(sd / f"{label}_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

    return ground_truths, predictions, images_out


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate counting accuracy: base Qwen-Image-Edit vs fine-tuned LoRA checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to LoRA checkpoint directory (must contain pytorch_lora_weights.safetensors)",
    )
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "counting_eval.yaml"),
        help="Path to evaluation config YAML",
    )
    parser.add_argument("--save-results", action="store_true", help="Save per-sample images and metadata")
    parser.add_argument("--results-dir", default=None, help="Output directory (default: auto under checkpoint)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    ckpt = Path(args.checkpoint)
    lora_file = ckpt / "pytorch_lora_weights.safetensors"
    if not lora_file.exists():
        sys.exit(f"Error: LoRA weights not found at {lora_file}")

    cfg = load_eval_config(args.config)

    parquet_dir = cfg["data"]["test_parquet_dir"]
    if not Path(parquet_dir).is_absolute():
        parquet_dir = str(REPO_ROOT / parquet_dir)
    if not Path(parquet_dir).exists():
        sys.exit(f"Error: Test parquet directory not found at {parquet_dir}")

    model_path = cfg["model"]["pretrained_model_name_or_path"]

    # ------------------------------------------------------------------
    # Prepare results directory
    # ------------------------------------------------------------------
    save_dir: Path | None = None
    if args.save_results:
        if args.results_dir:
            save_dir = Path(args.results_dir)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = ckpt / f"eval_results_{ts}"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {save_dir}")

    # ------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------
    print("Loading test data...")
    samples = load_test_data(parquet_dir)
    gt_counts = [s["count_added"] for s in samples]
    count_min, count_max = min(gt_counts), max(gt_counts)
    print(f"Loaded {len(samples)} test samples  (counts {count_min}-{count_max})")

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
    # Evaluate BASE model
    # ------------------------------------------------------------------
    print("\n--- Evaluating BASE model (no LoRA) ---")
    pipe_base = load_pipeline(model_path)
    base_gt, base_pred, _ = evaluate_model(
        pipe_base, samples, counter, cfg, label="base", save_dir=save_dir,
    )
    free_pipeline(pipe_base)
    print("Base model evaluation complete.\n")

    # ------------------------------------------------------------------
    # Evaluate FINE-TUNED model
    # ------------------------------------------------------------------
    print("--- Evaluating FINE-TUNED model ---")
    pipe_ft = load_pipeline(model_path, lora_path=str(lora_file))
    ft_gt, ft_pred, _ = evaluate_model(
        pipe_ft, samples, counter, cfg, label="finetuned", save_dir=save_dir,
    )
    free_pipeline(pipe_ft)
    print("Fine-tuned model evaluation complete.\n")

    # ------------------------------------------------------------------
    # Unload SAM3
    # ------------------------------------------------------------------
    counter.unload()

    # ------------------------------------------------------------------
    # Compute and print metrics
    # ------------------------------------------------------------------
    base_metrics = compute_metrics(base_gt, base_pred)
    ft_metrics = compute_metrics(ft_gt, ft_pred)
    base_pc = compute_per_count_metrics(base_gt, base_pred)
    ft_pc = compute_per_count_metrics(ft_gt, ft_pred)

    print_overall_report(base_metrics, ft_metrics, len(samples), str(ckpt), (count_min, count_max))
    print_per_count_report(base_pc, ft_pc)

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    results_payload = {
        "checkpoint": str(ckpt),
        "config": args.config,
        "num_samples": len(samples),
        "count_range": [count_min, count_max],
        "timestamp": datetime.now().isoformat(),
        "base_metrics": base_metrics,
        "finetuned_metrics": ft_metrics,
        "base_per_count": {str(k): v for k, v in base_pc.items()},
        "finetuned_per_count": {str(k): v for k, v in ft_pc.items()},
        "per_sample": [
            {
                "idx": i,
                "prompt": samples[i]["prompt"],
                "object_name": samples[i]["object_name"],
                "ground_truth": base_gt[i],
                "base_detected": base_pred[i],
                "finetuned_detected": ft_pred[i],
                "base_error": abs(base_gt[i] - base_pred[i]),
                "finetuned_error": abs(ft_gt[i] - ft_pred[i]),
            }
            for i in range(len(samples))
        ],
    }

    if save_dir is not None:
        out_json = save_dir / "results.json"
        with open(out_json, "w") as f:
            json.dump(results_payload, f, indent=2)
        print(f"\nFull results saved to: {out_json}")
    else:
        # Always save a minimal results JSON next to the checkpoint
        out_json = ckpt / "eval_results.json"
        with open(out_json, "w") as f:
            json.dump(results_payload, f, indent=2)
        print(f"\nResults JSON saved to: {out_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
