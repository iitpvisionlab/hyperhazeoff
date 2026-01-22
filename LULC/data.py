from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


EUROSAT_LABEL_MAPPING: Dict[str, int] = {
    "Crop": 0,
    "Forest": 1,
    "HerbaceousVegetation": 2,
    "Highway": 3,
    "Industrial": 4,
    "Residential": 5,
    "River": 6,
    "SeaLake": 7,
}


@dataclass(frozen=True)
class PatchInfo:
    image_path: Path
    coords: Tuple[int, int, int, int]  # x, y, width, height
    label_id: int
    source_json: Path


class FineTuningDataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        images_dir: str | Path,
        annotations_dir: str | Path,
        batch_size: int = 16,
        target_size: Tuple[int, int] = (64, 64),
        num_channels: int = 9,
        clean: bool = True,
        num_classes: int = 8,
        shuffle: bool = True,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)

        self.batch_size = int(batch_size)
        self.target_size = tuple(target_size)
        self.num_channels = int(num_channels)
        self.clean = bool(clean)
        self.num_classes = int(num_classes)
        self.shuffle = bool(shuffle)

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.num_channels <= 0:
            raise ValueError(f"num_channels must be > 0, got {self.num_channels}")

        self.patches: List[PatchInfo] = []
        self._collect_all_patches_data()

        if not self.patches:
            raise ValueError(f"Patches not found in directory {self.images_dir}")

        self.indices = np.arange(len(self.patches))
        self.on_epoch_end()

        logger.info(
            "Found %d patches in %s with %s",
            len(self.patches),
            self.images_dir.name,
            self.annotations_dir.name,
        )

    def __len__(self) -> int:
        # Include the last smaller batch. [web:26]
        return math.ceil(len(self.patches) / self.batch_size)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        start = index * self.batch_size
        stop = min((index + 1) * self.batch_size, len(self.patches))
        batch_indices = self.indices[start:stop]

        batch_images: List[np.ndarray] = []
        batch_labels: List[int] = []

        for i in batch_indices:
            patch_info = self.patches[int(i)]
            img = self._load_image(patch_info.image_path)

            x, y, w, h = patch_info.coords
            patch = img[y : y + h, x : x + w, : self.num_channels]

            if patch.shape[:2] != self.target_size:
                raise ValueError(
                    f"Patch has shape {patch.shape[:2]} but target_size is {self.target_size} "
                    f"(source={patch_info.source_json}, image={patch_info.image_path})"
                )

            batch_images.append(self.normalize_band(patch))
            batch_labels.append(patch_info.label_id)

        x_batch = np.asarray(batch_images, dtype=np.float32)
        y_batch = tf.keras.utils.to_categorical(batch_labels, self.num_classes)
        return x_batch, y_batch

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _collect_all_patches_data(self) -> None:
        for json_path in self.annotations_dir.rglob("*.json"):
            if self.clean and "hazed" in json_path.name:
                continue
            if not self.clean and "clean" in json_path.name:
                continue

            rel = json_path.relative_to(self.annotations_dir)
            image_path = (self.images_dir / rel).with_suffix(".npy")

            # Backward-compat: try dehazed if hazed path not found
            if not image_path.exists():
                image_path = Path(str(image_path).replace("hazed", "dehazed"))

            if not image_path.exists():
                logger.warning("Image not found for annotation %s", json_path)
                continue

            self._extract_patches_data_from_annotation(json_path, image_path)

    def _extract_patches_data_from_annotation(
        self, json_path: Path, image_path: Path
    ) -> None:
        with json_path.open("r", encoding="utf-8") as f:
            annotation = json.load(f)

        img = self._load_image(image_path)
        h_img, w_img = img.shape[0], img.shape[1]

        for obj in annotation.get("objects", []):
            if obj.get("type") != "rect":
                continue

            coords = obj.get("data", [])
            tags: Sequence[str] = obj.get("tags", [])

            if len(coords) < 4:
                logger.warning("Invalid coordinate format in %s: %s", json_path, coords)
                continue

            x, y, w, h = map(int, coords[:4])

            if x < 0 or y < 0 or x + w > w_img or y + h > h_img:
                logger.warning(
                    "Coords outside image boundaries (%s): x=%d y=%d w=%d h=%d",
                    json_path,
                    x,
                    y,
                    w,
                    h,
                )
                continue

            raw_label = tags[0] if tags else "Unknown"
            if raw_label not in EUROSAT_LABEL_MAPPING:
                logger.warning("Unknown label '%s' in %s", raw_label, json_path)
                continue

            self.patches.append(
                PatchInfo(
                    image_path=image_path,
                    coords=(x, y, w, h),
                    label_id=EUROSAT_LABEL_MAPPING[raw_label],
                    source_json=json_path,
                )
            )

    @staticmethod
    def normalize_band(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        # Min-max scaling with eps to avoid division-by-zero. [web:21]
        x_min = np.min(x)
        x_max = np.max(x)
        denom = max(float(x_max - x_min), eps)
        return (x - x_min) / denom

    @staticmethod
    @lru_cache(maxsize=128)
    def _load_image(path: Path) -> np.ndarray:
        img = np.load(path)

        # If channels-first (13, H, W) -> channels-last (H, W, 13)
        if img.ndim == 3 and img.shape[0] == 13:
            img = np.moveaxis(img, 0, 2)

        return img
