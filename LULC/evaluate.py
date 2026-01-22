from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _labels_from_one_hot(y: np.ndarray) -> np.ndarray:
    if y.ndim != 2:
        raise ValueError(f"Expected 2D one-hot array, got shape={y.shape}")
    return np.argmax(y, axis=1)


def eval_model_by_class(model, generator, label_names: Sequence[str]) -> pd.DataFrame:
    y_true: List[int] = []
    y_pred: List[int] = []

    for i in range(len(generator)):
        batch_x, batch_y = generator[i]
        probs = model.predict_on_batch(batch_x)
        y_true.extend(_labels_from_one_hot(np.asarray(batch_y)))
        y_pred.extend(np.argmax(np.asarray(probs), axis=1))

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    overall_acc = float(np.mean(y_true_arr == y_pred_arr))

    results: List[Dict[str, Any]] = []
    for class_idx, name in enumerate(label_names):
        mask = y_true_arr == class_idx
        count = int(np.sum(mask))
        acc = float(np.mean(y_pred_arr[mask] == class_idx)) if count else 0.0

        results.append(
            {
                "class": name,
                "label_count": count,
                "class_acc": acc,
                "overall_acc": overall_acc,
            }
        )

    return pd.DataFrame(results)


def dump_class_report_json(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    run_info: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Convert eval_model_by_class() DataFrame into a compact, stable JSON structure.
    """
    required = {"class", "label_count", "class_acc", "overall_acc"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Unexpected dataframe schema, missing columns: {sorted(missing)}"
        )

    overall_acc = float(df["overall_acc"].iloc[0]) if len(df) else 0.0
    total = int(df["label_count"].sum()) if len(df) else 0

    per_class: Dict[str, Any] = {}
    for _, row in df.iterrows():
        name = str(row["class"])
        per_class[name] = {
            "count": int(row["label_count"]),
            "acc": float(row["class_acc"]),
        }

    payload: Dict[str, Any] = {
        "summary": {
            "overall_acc": overall_acc,
            "total": total,
            "num_classes": int(len(df)),
        },
        "per_class": per_class,
    }
    if run_info:
        payload["run_info"] = run_info

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def aggregate_predictions(
    model,
    test_generator,
    labels: Sequence[str],
    output_json_path: str | Path,
    log_every_n_batches: int = 10,
) -> List[Dict[str, Any]]:
    """
    Run inference on a generator and save per-patch predictions to JSON.

    labels: index -> class name (must match model output order).
    """
    output_path = Path(output_json_path)
    num_classes = len(labels)
    if num_classes == 0:
        raise ValueError("labels must be non-empty")

    all_predictions: List[Dict[str, Any]] = []

    for batch_idx in range(len(test_generator)):
        batch_x, batch_y = test_generator[batch_idx]

        probs = np.asarray(model.predict_on_batch(batch_x), dtype=np.float32)
        pred_ids = np.argmax(probs, axis=1)
        conf = np.max(probs, axis=1)

        true_ids = np.argmax(np.asarray(batch_y), axis=1)

        # Безопасно работаем с последним неполным батчем
        n = min(len(pred_ids), len(true_ids))

        for j in range(n):
            true_id = int(true_ids[j])
            pred_id = int(pred_ids[j])

            result: Dict[str, Any] = {
                "batch_index": int(batch_idx),
                "in_batch_index": int(j),
                "predicted_class": labels[pred_id],
                "predicted_class_numeric": pred_id,
                "confidence": float(conf[j]),
                "true_class": labels[true_id],
                "true_class_numeric": true_id,
                "is_correct": pred_id == true_id,
                "probabilities": {
                    labels[k]: float(probs[j, k]) for k in range(num_classes)
                },
            }
            all_predictions.append(result)

        if log_every_n_batches > 0 and (batch_idx + 1) % log_every_n_batches == 0:
            logger.info("Processed batches: %d/%d", batch_idx + 1, len(test_generator))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)

    correct = sum(1 for r in all_predictions if r["is_correct"])
    acc = (correct / len(all_predictions)) if all_predictions else 0.0
    logger.info("Total samples: %d; accuracy: %.4f", len(all_predictions), acc)

    return all_predictions
