import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from utils.metrics.metric import *
from utils.data.visualization import CSSO, CSNC
from utils.metrics.chromaticity_difference import ChromaticityDifference
from piq import DISTS, LPIPS


RGB_METRIC_FUNCTIONS = [
    (psnr_torch, "psnr"),
    (ssim_torch, "ssim"),
    (sam_torch, "sam"),
    (DISTS(), "dists"),
    (LPIPS(), "lpips"),
    (ChromaticityDifference("prolab"), "chromdiff"),
]


HYPERSPECTRAL_METRIC_FUNCTIONS = [
    (psnr_torch, "psnr"),
    (ssim_torch, "ssim"),
    (sam_torch, "sam"),
]


def get_metric_functions(file_format: str) -> List[Tuple]:
    """
    Get appropriate metric functions based on file format.

    Parameters
    ----------
    file_format : str
        Image file format ('png' or 'npy').

    Returns
    -------
    List[Tuple]
        List of (metric_function, metric_name) tuples.
    """
    if file_format == "npy":
        return HYPERSPECTRAL_METRIC_FUNCTIONS
    else:
        return RGB_METRIC_FUNCTIONS


def load_image_file(file_path: Path, normalize_to_max: bool = False) -> np.ndarray:
    """
    Load and normalize image file to float32 array.

    Parameters
    ----------
    file_path : Path
        Path to image file (.png or .npy).
    normalize_to_max : bool
        If True, normalize numpy files by their maximum value.

    Returns
    -------
    np.ndarray
        Normalized image array with values in [0, 1].
    """
    if file_path.suffix.lower() == ".png":
        image = np.array(Image.open(file_path)).astype(np.float32) / 255.0
    elif file_path.suffix.lower() == ".npy":
        image = np.load(file_path).astype(np.float32)
        if normalize_to_max:
            max_value = image.max()
            if max_value > 0:
                image = image / max_value
        image = image.clip(0, 1)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    return image


def calculate_image_metrics(
    clean_path: Path,
    dehazed_path: Path,
    device: torch.device,
    metric_functions: List[Tuple],
) -> Dict[str, float]:
    """
    Calculate quality metrics between clean and dehazed images.

    Parameters
    ----------
    clean_path : Path
        Path to clean reference image.
    dehazed_path : Path
        Path to dehazed output image.
    device : torch.device
        Computing device (CPU or CUDA).
    metric_functions : List[Tuple]
        List of (metric_function, metric_name) tuples to compute.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping metric names to computed values.
    """

    clean_image = (
        torch.from_numpy(load_image_file(clean_path, normalize_to_max=True))
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
        metric_value = metric_function(
            clean_image.unsqueeze(0), dehazed_image.unsqueeze(0)
        )
        metrics[metric_name] = round(metric_value.mean().item(), 4)

    return metrics


def find_image_pairs(
    benchmark_root: Path, dehazed_root: Path, file_format: str
) -> List[Tuple[Path, Path, Path]]:
    """
    Find all matching clean-dehazed image pairs in dataset.

    Parameters
    ----------
    benchmark_root : Path
        Root directory containing clean benchmark images.
    dehazed_root : Path
        Root directory containing dehazed results.
    file_format : str
        Image file format ('png' or 'npy').

    Returns
    -------
    List[Tuple[Path, Path, Path]]
        List of (set_directory, clean_file, dehazed_file) tuples.
    """
    image_pairs = []

    for set_directory in sorted(benchmark_root.iterdir()):
        if not set_directory.is_dir():
            continue

        dehazed_files = list(
            (dehazed_root / set_directory.name).glob(f"*_dehazed.{file_format}")
        )
        clean_files = sorted(set_directory.glob(f"*_clean*.{file_format}"))

        if not dehazed_files or not clean_files:
            continue

        dehazed_file = dehazed_files[0]
        for clean_file in clean_files:
            image_pairs.append((set_directory, clean_file, dehazed_file))

    return image_pairs


def compute_all_metrics(
    image_pairs: List[Tuple[Path, Path, Path]],
    device: torch.device,
    metric_functions: List[Tuple],
) -> Tuple[List[Dict], Dict[str, float], Dict[str, int]]:
    """
    Compute metrics for all image pairs and aggregate statistics.

    Parameters
    ----------
    image_pairs : List[Tuple[Path, Path, Path]]
        List of (set_dir, clean_file, dehazed_file) tuples.
    device : torch.device
        Computing device.
    metric_functions : List[Tuple]
        List of (metric_function, metric_name) tuples to compute.

    Returns
    -------
    Tuple[List[Dict], Dict[str, float], Dict[str, int]]
        Tuple containing:
        - List of metric dictionaries for each pair
        - Sum of each metric across all pairs
        - Count of samples for each metric
    """
    detailed_results = []
    metric_totals = defaultdict(float)
    metric_sample_counts = defaultdict(int)

    for set_directory, clean_file, dehazed_file in tqdm(
        image_pairs, desc="Computing metrics"
    ):
        metrics = calculate_image_metrics(
            clean_file, dehazed_file, device, metric_functions
        )

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
    results: List[Dict], output_path: Path, metric_functions: List[Tuple]
) -> None:
    """
    Write detailed metrics for all image pairs to CSV file.

    Parameters
    ----------
    results : List[Dict]
        List of metric dictionaries for each image pair.
    output_path : Path
        Path where CSV file will be saved.
    metric_functions : List[Tuple]
        List of (metric_function, metric_name) tuples.
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
    """
    Calculate and save aggregated metrics to JSON file.

    Parameters
    ----------
    model_name : str
        Name of the evaluated model.
    run_date : str
        Date of evaluation run.
    results : List[Dict]
        All detailed results.
    metric_totals : Dict[str, float]
        Sum of each metric across all pairs.
    metric_counts : Dict[str, int]
        Number of samples for each metric.
    output_path : Path
        Path where JSON file will be saved.
    file_format : str
        Image file format used.

    Returns
    -------
    Dict
        Aggregated metrics dictionary.
    """
    aggregated_metrics = {
        metric_name: round(metric_totals[metric_name] / metric_counts[metric_name], 4)
        for metric_name in metric_totals.keys()
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
    """
    Print evaluation summary to console.

    Parameters
    ----------
    summary : Dict
        Dictionary containing aggregated metrics and metadata.
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Summary for {summary['model_name']}")
    print(f"{'='*60}")
    print(f"File format: {summary['file_format']}")
    print(f"Total image pairs evaluated: {summary['total_image_pairs']}")
    print(f"\nMean Metrics:")
    for metric_name, mean_value in summary["mean_metrics"].items():
        print(f"  {metric_name:>12}: {mean_value:.4f}")
    print(f"{'='*60}\n")


def parse_command_line_arguments() -> argparse.Namespace:
    """
    Parse and validate command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate dehazing model quality using standard image metrics"
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        required=True,
        help="Directory containing clean benchmark images",
    )
    parser.add_argument(
        "--dehazed-dir",
        type=Path,
        required=True,
        help="Directory containing dehazed model outputs",
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
        default=Path("./output") / "metrics",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Computing device for metric calculations",
    )
    parser.add_argument(
        "--format-data", default="npy", choices=["png", "npy"], help="Image file format"
    )
    return parser.parse_args()


def run_evaluation(
    benchmark_dir: Path,
    dehazed_dir: Path,
    model_name: str,
    output_dir: Path,
    device: str,
    file_format: str,
) -> None:
    """
    Run complete model evaluation pipeline.

    Parameters
    ----------
    benchmark_dir : Path
        Directory with clean reference images.
    dehazed_dir : Path
        Directory with dehazed results.
    model_name : str
        Name of evaluated model.
    output_dir : Path
        Output directory for results.
    device : str
        Computing device ('cpu' or 'cuda').
    file_format : str
        Image file format ('png' or 'npy').
    """
    evaluation_date = datetime.now().strftime("%Y%m%d")
    output_root = output_dir / evaluation_date / model_name
    output_root.mkdir(parents=True, exist_ok=True)

    compute_device = torch.device(
        device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    metric_functions = get_metric_functions(file_format)

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
        model_name=args.model_name,
        output_dir=args.out_dir,
        device=args.device,
        file_format=args.format_data,
    )
