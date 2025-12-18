import argparse
import csv
import json
import os
from typing import Dict, List, Tuple, Callable, Optional


METRICS_INFO: Dict[str, bool] = {
    "psnr": True,
    "ssim": True,
    "sam": False,
    "dists": False,
    "lpips": False,
    "chromdiff": False,
}


class MetricsProcessor:
    """
    Processor for selecting and aggregating dehazing metrics per scene.

    Parameters
    ----------
    input_csv : str
        Path to the CSV file with full metrics.
    baseline_csv : str, optional
        Path to the baseline CSV; required only for ``baseline_aggregate`` mode.

    Notes
    -----
    All CSV files are expected to use ``;`` as a delimiter and to contain
    at least a ``set_name`` column.
    """

    def __init__(self, input_csv: str, baseline_csv: Optional[str] = None) -> None:
        self.input_csv = input_csv
        self.baseline_csv = baseline_csv
        self.base_dir = os.path.dirname(os.path.abspath(input_csv))

    # ---------- helpers ----------

    @staticmethod
    def _read_rows(path: str) -> List[Dict[str, str]]:
        """
        Read semicolon-delimited CSV into a list of dicts.

        Parameters
        ----------
        path : str
            Path to the input CSV file.

        Returns
        -------
        list of dict
            Each dict represents one row (column name -> string value).
        """
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            return list(reader)  # type: ignore[return-value]

    @staticmethod
    def _get_metrics(columns: List[str]) -> List[str]:
        """
        Filter supported metric names that are present in given columns.

        Parameters
        ----------
        columns : list of str
            List of column names.

        Returns
        -------
        list of str
            Metric names that are both in ``METRICS_INFO`` and in ``columns``.
        """
        return [m for m in METRICS_INFO if m in columns]

    @staticmethod
    def _group_by_set_name(
        rows: List[Dict[str, str]]
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Group rows by the ``set_name`` field.

        Parameters
        ----------
        rows : list of dict
            Input rows.

        Returns
        -------
        dict of str to list of dict
            Mapping from ``set_name`` to list of rows.
        """
        grouped: Dict[str, List[Dict[str, str]]] = {}
        for row in rows:
            key = row["set_name"]
            grouped.setdefault(key, []).append(row)
        return grouped

    @staticmethod
    def _safe_float(value: str) -> float:
        """
        Convert string to float, treating empty strings as zero.

        Parameters
        ----------
        value : str
            String representation of a number.

        Returns
        -------
        float
            Parsed float, or ``0.0`` if value is empty.
        """
        if value == "":
            return 0.0
        return float(value)

    def _save_rows_and_summary(
        self,
        rows: List[Dict[str, str]],
        metrics: List[str],
        csv_name: str,
        json_name: str,
    ) -> Tuple[List[Dict[str, str]], Dict[str, float], str, str]:
        """
        Save rows to CSV and compute a mean-summary for selected metrics.

        Parameters
        ----------
        rows : list of dict
            Rows to write.
        metrics : list of str
            Metric names to summarize.
        csv_name : str
            Name of the CSV file.
        json_name : str
            Name of the JSON file.

        Returns
        -------
        rows : list of dict
            Same list of rows that was passed in.
        summary : dict of str to float
            Mean value per metric across all rows, rounded to 4 decimals.
        output_csv : str
            Path to the written CSV.
        output_json : str
            Path to the written JSON.
        """
        if not rows:
            raise ValueError("No rows to save")

        fieldnames = list(rows[0].keys())
        output_csv = os.path.join(self.base_dir, csv_name)
        output_json = os.path.join(self.base_dir, json_name)

        # write CSV
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows(rows)

        # compute summary
        summary: Dict[str, float] = {}
        for m in metrics:
            vals = [self._safe_float(r[m]) for r in rows]
            summary[m] = round(sum(vals) / len(vals), 4)

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        return rows, summary, output_csv, output_json

    # ---------- modes ----------

    def ranked(self) -> Tuple[str, str]:
        """
        Run normalized-rank selection and print paths to outputs.

        Returns
        -------
        output_csv : str
            Path to the CSV with best rows per scene.
        output_json : str
            Path to the JSON with metric summary.
        """
        rows = self._read_rows(self.input_csv)
        if not rows:
            raise ValueError("Input CSV is empty")

        metrics = self._get_metrics(list(rows[0].keys()))
        grouped = self._group_by_set_name(rows)

        best_rows: List[Dict[str, str]] = []

        for set_name, group_rows in grouped.items():
            total_ranks: List[float] = [0.0] * len(group_rows)

            for metric in metrics:
                maximize = METRICS_INFO[metric]
                values = [self._safe_float(r[metric]) for r in group_rows]

                if maximize:
                    best_val = max(values)
                    diffs = [best_val - v for v in values]
                else:
                    best_val = min(values)
                    diffs = [v - best_val for v in values]

                max_diff = max(diffs)
                if max_diff == 0:
                    norm = [0.0] * len(diffs)
                else:
                    norm = [d / max_diff for d in diffs]

                for i, v in enumerate(norm):
                    total_ranks[i] += v

            best_idx = min(range(len(group_rows)), key=lambda i: total_ranks[i])
            best_rows.append(group_rows[best_idx])

        _, summary, output_csv, output_json = self._save_rows_and_summary(
            best_rows,
            metrics,
            csv_name="best_metrics_rank_grouped_normalized.csv",
            json_name="summary_metrics_rank_grouped_normalized.json",
        )

        print("Best ranked metrics saved to:", output_csv)
        print("Summary saved to:", output_json)
        return output_csv, output_json

    def aggregate_best(self) -> Tuple[str, str]:
        """
        Best-value selection: The score is computed against all available reference images for each dehazed image and each metric, and the best score is retained. This produces a single representative metric value per scene, although the selected reference may differ across metrics.
        Run max/min aggregation per scene and print paths to outputs.

        Returns
        -------
        output_csv : str
            Path to the CSV with aggregated metrics per scene.
        output_json : str
            Path to the JSON with metric summary.
        """
        rows = self._read_rows(self.input_csv)
        if not rows:
            raise ValueError("Input CSV is empty")

        metrics = self._get_metrics(list(rows[0].keys()))
        grouped = self._group_by_set_name(rows)

        aggregated_rows: List[Dict[str, str]] = []

        for set_name, group_rows in grouped.items():
            agg_row: Dict[str, str] = {"set_name": set_name}

            for metric in metrics:
                maximize = METRICS_INFO[metric]
                values = [self._safe_float(r[metric]) for r in group_rows]
                best_val = max(values) if maximize else min(values)
                agg_row[metric] = str(best_val)

            first = group_rows[0]
            agg_row.setdefault("clean_file", first.get("clean_file", ""))
            agg_row.setdefault("dehazed_file", first.get("dehazed_file", ""))

            aggregated_rows.append(agg_row)

        _, summary, output_csv, output_json = self._save_rows_and_summary(
            aggregated_rows,
            metrics,
            csv_name="best_metrics_aggregate.csv",
            json_name="summary_metrics_aggregate.json",
        )

        print("Best aggregate metrics saved to:", output_csv)
        print("Summary saved to:", output_json)
        return output_csv, output_json

    def baseline_aggregate(self) -> Tuple[str, str]:
        """
        Run aggregation only on pairs that are present in the baseline CSV.

        Returns
        -------
        output_csv : str
            Path to the CSV with aggregated baseline metrics.
        output_json : str
            Path to the JSON with summary statistics.

        Raises
        ------
        ValueError
            If ``baseline_csv`` was not provided during initialization.
        """
        if self.baseline_csv is None:
            raise ValueError("baseline_csv is required for baseline_aggregate mode")

        full_rows = self._read_rows(self.input_csv)
        baseline_rows = self._read_rows(self.baseline_csv)

        for row in baseline_rows:
            row["dehazed_file_normalized"] = row["dehazed_file"].replace(
                "_hazed", "_dehazed"
            )

        full_index: Dict[str, Dict[str, str]] = {}
        for row in full_rows:
            key = f"{row['set_name']}|{row['clean_file']}|{row['dehazed_file']}"
            full_index[key] = row

        filtered_rows: List[Dict[str, str]] = []
        for row in baseline_rows:
            key = f"{row['set_name']}|{row['clean_file']}|{row['dehazed_file_normalized']}"
            if key in full_index:
                filtered_rows.append(full_index[key])

        if not filtered_rows:
            raise ValueError("No rows matched baseline entries")

        metrics = self._get_metrics(list(filtered_rows[0].keys()))
        grouped = self._group_by_set_name(filtered_rows)

        aggregated_rows: List[Dict[str, str]] = []

        for set_name, group_rows in grouped.items():
            agg_row: Dict[str, str] = {"set_name": set_name}

            for metric in metrics:
                maximize = METRICS_INFO[metric]
                values = [self._safe_float(r[metric]) for r in group_rows]
                best_val = max(values) if maximize else min(values)
                agg_row[metric] = str(best_val)

            first = group_rows[0]
            agg_row["clean_file"] = first.get("clean_file", "")
            agg_row["dehazed_file"] = first.get("dehazed_file", "")

            aggregated_rows.append(agg_row)

        _, summary, output_csv, output_json = self._save_rows_and_summary(
            aggregated_rows,
            metrics,
            csv_name="baseline_aggregate_metrics.csv",
            json_name="baseline_aggregate_summary.json",
        )

        print("Baseline filtered aggregate metrics saved to:", output_csv)
        print("Summary saved to:", output_json)
        return output_csv, output_json

    # ---------- dispatcher ----------

    def run(self, mode: str) -> None:
        """
        Run processing in a selected mode.

        Parameters
        ----------
        mode : {'ranked', 'aggregate_best', 'baseline_aggregate'}
            Processing mode to execute.
        """
        dispatch: Dict[str, Callable[[], Tuple[str, str]]] = {
            "ranked": self.ranked,
            "best": self.aggregate_best,
            "baseline": self.baseline_aggregate,
        }
        dispatch[mode]()


def main() -> None:
    """
    Entry point for command-line interface.

    Parses arguments and runs the requested processing mode.
    """
    parser = argparse.ArgumentParser(description="Process best metrics per scene")
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV file with metrics",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="ranked",
        choices=["ranked", "best", "baseline"],
        help="Mode of processing",
    )
    parser.add_argument(
        "--baseline_csv",
        type=str,
        help="Path to baseline CSV, required for baseline_aggregate mode",
    )
    args = parser.parse_args()

    if args.mode == "baseline" and not args.baseline_csv:
        raise ValueError(
            "baseline_csv argument is required for baseline_aggregate mode"
        )

    processor = MetricsProcessor(
        input_csv=args.input_csv,
        baseline_csv=args.baseline_csv,
    )
    processor.run(args.mode)


if __name__ == "__main__":
    main()
