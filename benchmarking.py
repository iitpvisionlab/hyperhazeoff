#!/usr/bin/env python3
"""
Evaluate dehazing model quality using standard image metrics.

Supports multiple evaluation modes:
- default: hierarchical clean vs dehazed
- baseline: hazed vs clean within benchmark
- clean_pairs: clean vs clean within sets
- flat: prefix matching between flat directories
"""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from piq import DISTS, LPIPS
from utils.metrics.chromaticity_difference import ChromaticityDifference
from utils.metrics.metric import psnr_torch, sam_torch, ssim_torch


# Global metric configuration
RGB_METRIC_FUNCTIONS: List[Tuple[callable, str]] = [
    (psnr_torch, "psnr"),
    (ssim_torch, "ssim"),
    (sam_torch, "sam"),
    (DISTS(), "dists"),
    (LPIPS(), "lpips"),
    (ChromaticityDifference("prolab"), "chromdiff"),
]

HYPERSPECTRAL_METRIC_FUNCTIONS: List[Tuple[callable, str]] = [
    (psnr_torch, "psnr"),
    (ssim_torch, "ssim"),
    (sam_torch, "sam"),
]


def get_metric_functions(file_format: str) -> List[Tuple[callable, str]]:
    """Get appropriate metric functions based on file format.

    Args:
        file_format: Image file format ('png' or 'npy').

    Returns:
        List of (metric_function, metric_name) tuples.
    """
    if file_format == "npy":
        return HYPERSPECTRAL_METRIC_FUNCTIONS
    return RGB_METRIC_FUNCTIONS


def load_image_file(file_path: Path, normalize_to_max: bool = False) -> np.ndarray:
    """Load and normalize image file to float32 array in [0, 1].

    Args:
        file_path: Path to image file (.png or .npy).
        normalize_to_max: If True, normalize numpy files by their maximum value.

    Returns:
        Normalized image array.

    Raises:
        ValueError: If unsupported file format.
    """
    if file_path.suffix.lower() == ".png":
        image = np.array(Image.open(file_path)).astype(np.float32) / 255.0
    elif file_path.suffix.lower() == ".npy":
        image = np.load(file_path).astype(np.float32)
        if normalize_to_max:
            max_value = image.max()
            if max_value > 0:
                image /= max_value
        image = np.clip(image, 0, 1)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    return image


def calculate_image_metrics(
    clean_path: Path,
    dehazed_path: Path,
    device: torch.device,
    metric_functions: List[Tuple[callable, str]],
) -> Dict[str, float]:
    """Calculate quality metrics between clean and dehazed images.

    Args:
        clean_path: Path to clean reference image.
        dehazed_path: Path to dehazed output image.
        device: Computing device (CPU or CUDA).
        metric_functions: List of (metric_function, metric_name) tuples.

    Returns:
        Dictionary mapping metric names to computed values.
    """
    clean_image = (
        torch.from_numpy(load_image_file(clean_path, normalize_to_max=False))
        .permute(2, 0, 1)
        .to(device)
    )
    dehazed_image = (
        torch.from_numpy(load_image_file(dehazed_path, normalize_to_max=False))
        .permute(2, 0, 1)
        .to(device)
    )

    metrics = {}
    for metric_function, metric_name in metric_functions:
        metric_value = metric_function(clean_image.unsqueeze(0), dehazed_image.unsqueeze(0))
        metrics[metric_name] = round(metric_value.mean().item(), 4)
    return metrics


def find_image_pairs(
    benchmark_root: Path, dehazed_root: Path, file_format: str
) -> List[Tuple[Path, Path, Path]]:
    """Find all matching clean-dehazed image pairs in hierarchical dataset.

    Args:
        benchmark_root: Root directory containing clean benchmark images.
        dehazed_root: Root directory containing dehazed results.
        file_format: Image file format ('png' or 'npy').

    Returns:
        List of (set_directory, clean_file, dehazed_file) tuples.
    """
    image_pairs = []
    for set_directory in sorted(benchmark_root.iterdir()):
        if not set_directory.is_dir():
            continue

        dehazed_files = list((dehazed_root / set_directory.name).glob(f"*_dehazed.{file_format}"))
        clean_files = sorted(set_directory.glob(f"*_clean*.{file_format}"))

        if not dehazed_files or not clean_files:
            continue

        dehazed_file = dehazed_files[0]
        for clean_file in clean_files:
            image_pairs.append((set_directory, clean_file, dehazed_file))
    return image_pairs


def find_baseline_pairs(benchmark_root: Path, file_format: str) -> List[Tuple[Path, Path, Path]]:
    """Baseline mode: compare hazed vs clean images within benchmark directory.

    Args:
        benchmark_root: Root directory containing both hazed and clean images.
        file_format: Image file format.

    Returns:
        List of (set_directory, clean_file, hazed_file) tuples.
    """
    image_pairs = []
    for set_directory in sorted(benchmark_root.iterdir()):
        if not set_directory.is_dir():
            continue

        hazed_files = list(set_directory.glob(f"*_hazed*.{file_format}"))
        clean_files = sorted(set_directory.glob(f"*_clean*.{file_format}"))

        if not hazed_files or not clean_files:
            continue

        hazed_file = hazed_files[0]
        for clean_file in clean_files:
            image_pairs.append((set_directory, clean_file, hazed_file))
    return image_pairs


def find_clean_pairs(benchmark_root: Path, file_format: str) -> List[Tuple[Path, Path, Path]]:
    """Clean-pairs mode: compare clean vs clean images within the same set.

    Args:
        benchmark_root: Root directory containing clean images.
        file_format: Image file format.

    Returns:
        List of (set_directory, clean_a, clean_b) tuples for unique pairs.
    """
    image_pairs = []
    for set_directory in sorted(benchmark_root.iterdir()):
        if not set_directory.is_dir():
            continue

        clean_files = sorted(set_directory.glob(f"*_clean*.{file_format}"))
        if len(clean_files) < 2:
            continue

        for i in range(len(clean_files)):
            for j in range(i + 1, len(clean_files)):
                image_pairs.append((set_directory, clean_files[i], clean_files[j]))
    return image_pairs


def find_flat_pairs(clean_dir: Path, dehazed_dir: Path, file_format: str) -> List[Tuple[Path, Path, Path]]:
    """Flat mode: match files from two flat directories by prefix.

    Args:
        clean_dir: Directory containing clean files (0001.npy, 0002.npy, etc.).
        dehazed_dir: Directory containing dehazed files ({clean_name}_{suffix}.{file_format}).
        file_format: Image file format.

    Returns:
        List of (parent_dir, clean_file, dehazed_file) tuples.
    """
    image_pairs = []
    clean_files = sorted(clean_dir.glob(f"*.{file_format}"))

    for clean_file in clean_files:
        clean_stem = clean_file.stem
        pattern = f"{clean_stem}_*.{file_format}"
        dehazed_files = list(dehazed_dir.glob(pattern))

        for dehazed_file in dehazed_files:
            image_pairs.append((clean_dir, clean_file, dehazed_file))
    return image_pairs


def compute_all_metrics(
    image_pairs: List[Tuple[Path, Path, Path]],
    device: torch.device,
    metric_functions: List[Tuple[callable, str]],
) -> Tuple[List[Dict], Dict[str, float], Dict[str, int]]:
    """Compute metrics for all image pairs and aggregate statistics.

    Args:
        image_pairs: List of (set_dir, clean_file, dehazed_file) tuples.
        device: Computing device.
        metric_functions: List of metric functions to compute.

    Returns:
        Tuple of (detailed_results, metric_totals, metric_counts).
    """
    detailed_results = []
    metric_totals = defaultdict(float)
    metric_sample_counts = defaultdict(int)

    for set_directory, clean_file, dehazed_file in tqdm(image_pairs, desc="Computing metrics"):
        metrics = calculate_image_metrics(clean_file, dehazed_file, device, metric_functions)

        result_row = {
            "set_name": set_directory.name,
            "clean_file": clean_file.name,
            "dehazed_file": dehazed_file.name,
            **metrics,
        }
        detailed_results.append(result_row)

        for metric_name, metric_value in metrics.items():
            metric_totals[metric_name] += metric_value
            metric_sample_counts[metric_name] += 1

    return detailed_results, dict(metric_totals), dict(metric_sample_counts)


def write_detailed_metrics_csv(
    results: List[Dict],
    output_path: Path,
    metric_functions: List[Tuple[callable, str]],
) -> None:
    """Write detailed metrics for all image pairs to CSV file.

    Args:
        results: List of metric dictionaries for each image pair.
        output_path: Path where CSV file will be saved.
        metric_functions: List of metric functions.
    """
    if not results:
        print("No results to save")
        return

    column_names = ["set_name", "clean_file", "dehazed_file"] + [
        name for _, name in metric_functions
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=column_names, delimiter=";")
        writer.writeheader()
        writer.writerows(results)
    print(f"Detailed metrics saved to {output_path}")


def write_aggregated_metrics_json(
    model_name: str,
    run_date: str,
    results: List[Dict],
    metric_totals: Dict[str, float],
    metric_counts: Dict[str, int],
    output_path: Path,
    file_format: str,
) -> Dict:
    """Calculate and save aggregated metrics to JSON file.

    Args:
        model_name: Name of the evaluated model.
        run_date: Date of evaluation run.
        results: All detailed results.
        metric_totals: Sum of each metric across all pairs.
        metric_counts: Number of samples for each metric.
        output_path: Path where JSON file will be saved.
        file_format: Image file format used.

    Returns:
        Aggregated metrics dictionary.
    """
    aggregated_metrics = {
        metric_name: round(metric_totals[metric_name] / metric_counts[metric_name], 4)
        for metric_name in metric_totals
    }

    summary = {
        "model_name": model_name,
        "evaluation_date": run_date,
        "file_format": file_format,
        "total_image_pairs": len(results),
        "mean_metrics": aggregated_metrics,
    }

    output_path.write_text(json.dumps(summary, indent=2))
    print(f"Aggregated metrics saved to {output_path}")
    return summary


def display_evaluation_summary(summary: Dict) -> None:
    """Print evaluation summary to console.

    Args:
        summary: Dictionary containing aggregated metrics and metadata.
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Summary for {summary['model_name']}")
    print(f"{'='*60}")
    print(f"File format: {summary['file_format']}")
    print(f"Total image pairs evaluated: {summary['total_image_pairs']}")
    print("\nMean Metrics:")
    for metric_name, mean_value in summary["mean_metrics"].items():
        print(f"  {metric_name:>12}: {mean_value:.4f}")
    print(f"{'='*60}\n")


def parse_command_line_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate dehazing model quality using standard image metrics"
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        help="Directory containing clean benchmark images (hierarchical mode)",
    )
    parser.add_argument(
        "--dehazed-dir",
        type=Path,
        help="Directory containing dehazed model outputs",
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        help="Directory containing clean files (flat mode)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model being evaluated",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./data") / "metrics",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Computing device for metric calculations",
    )
    parser.add_argument(
        "--format-data",
        default="npy",
        choices=["png", "npy"],
        help="Image file format",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="default",
        choices=["default", "baseline", "clean_pairs", "flat"],
        help=(
            "Evaluation mode: 'default' (hierarchical clean vs dehazed), "
            "'baseline' (hazed vs clean), 'clean_pairs' (clean vs clean), "
            "or 'flat' (flat directories with prefix matching)"
        ),
    )
    return parser.parse_args()


def run_evaluation(
    benchmark_dir: Optional[Path],
    dehazed_dir: Optional[Path],
    clean_dir: Optional[Path],
    model_name: str,
    output_dir: Path,
    device: str,
    file_format: str,
    mode: str,
) -> None:
    """Run complete evaluation pipeline.

    Args:
        benchmark_dir: Benchmark directory (required for most modes).
        dehazed_dir: Dehazed results directory.
        clean_dir: Clean directory (flat mode).
        model_name: Model name for output files.
        output_dir: Output directory base.
        device: Compute device.
        file_format: Image format.
        mode: Evaluation mode.
    """
    evaluation_date = datetime.now().strftime("%Y%m%d")
    output_root = output_dir / evaluation_date / model_name
    output_root.mkdir(parents=True, exist_ok=True)

    compute_device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    metric_functions = get_metric_functions(file_format)

    # Select appropriate pair-finding function
    if mode == "flat":
        if clean_dir is None or dehazed_dir is None:
            raise ValueError("Flat mode requires both --clean-dir and --dehazed-dir")
        print("\nFlat mode: matching files by prefix from separate directories.\n")
        image_pairs = find_flat_pairs(clean_dir, dehazed_dir, file_format)
    elif mode == "baseline":
        if benchmark_dir is None:
            raise ValueError("Baseline mode requires --benchmark-dir")
        print("\nBaseline mode: comparing hazed vs clean inside benchmark-dir.\n")
        image_pairs = find_baseline_pairs(benchmark_dir, file_format)
    elif mode == "clean_pairs":
        if benchmark_dir is None:
            raise ValueError("Clean-pairs mode requires --benchmark-dir")
        print("\nClean-pairs mode: comparing clean vs clean within each scene.\n")
        image_pairs = find_clean_pairs(benchmark_dir, file_format)
    else:  # default mode
        if benchmark_dir is None or dehazed_dir is None:
            raise ValueError("Default mode requires both --benchmark-dir and --dehazed-dir")
        image_pairs = find_image_pairs(benchmark_dir, dehazed_dir, file_format)

    detailed_results, metric_totals, metric_counts = compute_all_metrics(
        image_pairs, compute_device, metric_functions
    )

    write_detailed_metrics_csv(
        detailed_results,
        output_root / f"{model_name}_detailed_metrics.csv",
        metric_functions,
    )

    summary = write_aggregated_metrics_json(
        model_name,
        evaluation_date,
        detailed_results,
        metric_totals,
        metric_counts,
        output_root / f"{model_name}_summary.json",
        file_format,
    )

    display_evaluation_summary(summary)


if __name__ == "__main__":
    args = parse_command_line_arguments()
    run_evaluation(
        benchmark_dir=args.benchmark_dir,
        dehazed_dir=args.dehazed_dir,
        clean_dir=args.clean_dir,
        model_name=args.model_name,
        output_dir=args.out_dir,
        device=args.device,
        file_format=args.format_data,
        mode=args.mode,
    )
