#!/usr/bin/env python3
"""
Calculate average metrics for a specified subset and the remaining samples.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


def normalize_identifier(identifier: str) -> str:
    """Normalize identifiers to lowercase for matching."""
    return identifier.lower().strip()


def parse_subset_specification(subset_path: Path) -> Set[Tuple[str, str, str, str]]:
    """Parse subset specification CSV file."""
    subset_entries = set()

    with subset_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = (
                normalize_identifier(row["Set"]),
                normalize_identifier(row["Crop"]),
                normalize_identifier(row["Clean"]),
                normalize_identifier(row["Hazed-Dehazed"]),
            )
            subset_entries.add(entry)

    return subset_entries


def extract_flight_id(filename: str) -> str:
    """Extract flight line identifier from filename."""
    name_without_ext = filename.rsplit(".", 1)[0]
    flight_id = name_without_ext.split("_")[0]
    return normalize_identifier(flight_id)


def check_if_in_subset(
    row: Dict[str, str], subset_entries: Set[Tuple[str, str, str, str]]
) -> bool:
    """Check if a metrics row matches any entry in the subset."""
    set_name = normalize_identifier(row["set_name"])
    clean_flight = extract_flight_id(row["clean_file"])
    dehazed_flight = extract_flight_id(row["dehazed_file"])

    # Extract crop number from set_name (e.g., 's03c00' -> 'c00')
    if len(set_name) >= 3:
        crop_part = set_name[3:]
    else:
        crop_part = ""

    # Check against all subset entries
    for subset_set, subset_crop, subset_clean, subset_hazed in subset_entries:
        set_match = set_name == subset_set or set_name.startswith(subset_set)
        crop_match = crop_part == subset_crop
        clean_match = clean_flight == subset_clean
        hazed_match = dehazed_flight == subset_hazed

        if set_match and crop_match and clean_match and hazed_match:
            return True

    return False


def parse_metrics_csv(
    metrics_path: Path, subset_entries: Set[Tuple[str, str, str, str]]
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], List[str]]:
    """Parse metrics CSV and split into subset and remainder."""
    subset_metrics = defaultdict(list)
    remainder_metrics = defaultdict(list)
    metric_names = []

    with metrics_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")

        all_columns = reader.fieldnames
        metric_names = [
            col
            for col in all_columns
            if col not in ["set_name", "clean_file", "dehazed_file"]
        ]

        for row in reader:
            is_subset = check_if_in_subset(row, subset_entries)
            target_dict = subset_metrics if is_subset else remainder_metrics

            for metric_name in metric_names:
                try:
                    value = float(row[metric_name])
                    target_dict[metric_name].append(value)
                except (ValueError, KeyError):
                    continue

    return dict(subset_metrics), dict(remainder_metrics), metric_names


def calculate_averages(metrics_dict: Dict[str, List[float]]) -> Dict[str, float]:
    """Calculate average for each metric."""
    averages = {}
    for metric_name, values in metrics_dict.items():
        if values:
            averages[metric_name] = round(sum(values) / len(values), 4)
        else:
            averages[metric_name] = 0.0
    return averages


def print_results(
    subset_avg: Dict[str, float],
    remainder_avg: Dict[str, float],
    all_avg: Dict[str, float],
    subset_count: int,
    remainder_count: int,
    metric_names: List[str],
) -> None:
    """Print formatted results to console."""
    print(f"\n{'='*70}")
    print(f"{'METRICS SUMMARY':^70}")
    print(f"{'='*70}\n")

    print(f"{'Metric':<15} {'Subset':<15} {'Remainder':<15} {'All':<15}")
    print(
        f"{'':<15} ({subset_count} samples) ({remainder_count} samples) ({subset_count + remainder_count} samples)"
    )
    print(f"{'-'*70}")

    for metric_name in metric_names:
        print(
            f"{metric_name:<15} "
            f"{subset_avg.get(metric_name, 0):<15.4f} "
            f"{remainder_avg.get(metric_name, 0):<15.4f} "
            f"{all_avg.get(metric_name, 0):<15.4f}"
        )

    print(f"{'='*70}\n")


def save_results_json(
    output_path: Path,
    subset_avg: Dict[str, float],
    remainder_avg: Dict[str, float],
    all_avg: Dict[str, float],
    subset_count: int,
    remainder_count: int,
) -> None:
    """Save results to JSON file."""
    results = {
        "subset": {"sample_count": subset_count, "metrics": subset_avg},
        "remainder": {"sample_count": remainder_count, "metrics": remainder_avg},
        "all": {"sample_count": subset_count + remainder_count, "metrics": all_avg},
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Calculate average metrics for subset and remainder"
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        required=True,
        help="Path to detailed metrics CSV file",
    )
    parser.add_argument(
        "--subset-csv",
        type=Path,
        required=True,
        help="Path to subset specification CSV file",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to save results JSON (optional)",
    )

    args = parser.parse_args()

    if not args.metrics_csv.exists():
        print(f"Error: Metrics CSV file not found: {args.metrics_csv}")
        return

    if not args.subset_csv.exists():
        print(f"Error: Subset CSV file not found: {args.subset_csv}")
        return

    print(f"Loading subset specification from: {args.subset_csv}")
    subset_entries = parse_subset_specification(args.subset_csv)
    print(f"Found {len(subset_entries)} subset entries")

    print(f"\nLoading metrics from: {args.metrics_csv}")
    subset_metrics, remainder_metrics, metric_names = parse_metrics_csv(
        args.metrics_csv, subset_entries
    )

    subset_count = len(next(iter(subset_metrics.values()))) if subset_metrics else 0
    remainder_count = (
        len(next(iter(remainder_metrics.values()))) if remainder_metrics else 0
    )

    print(f"Subset samples: {subset_count}")
    print(f"Remainder samples: {remainder_count}")
    print(f"Total samples: {subset_count + remainder_count}")

    subset_avg = calculate_averages(subset_metrics)
    remainder_avg = calculate_averages(remainder_metrics)

    all_metrics = defaultdict(list)
    for metric_name in metric_names:
        all_metrics[metric_name].extend(subset_metrics.get(metric_name, []))
        all_metrics[metric_name].extend(remainder_metrics.get(metric_name, []))
    all_avg = calculate_averages(dict(all_metrics))

    print_results(
        subset_avg, remainder_avg, all_avg, subset_count, remainder_count, metric_names
    )

    if args.output_json:
        save_results_json(
            args.output_json,
            subset_avg,
            remainder_avg,
            all_avg,
            subset_count,
            remainder_count,
        )


if __name__ == "__main__":
    main()
