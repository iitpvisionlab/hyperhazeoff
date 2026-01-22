from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent

BAND_IDS: Tuple[str, ...] = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    # "B10",
    "B11",
    "B12",
)


def load_srf_platform(meta_sentinel_json: Path, platform: str) -> Dict[str, dict]:
    with meta_sentinel_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for obj in data:
        if obj.get("platform") == platform:
            return obj["bands"]

    raise KeyError(f"Platform {platform} not found in {meta_sentinel_json}")


def load_markup_keys(markup_root: Path) -> Set[str]:
    """
    Return a set of keys like:
      "S01C00/f160621t01p00r17_clean"
    """
    keys: Set[str] = set()
    for p in markup_root.rglob("*.json"):
        rel = p.relative_to(markup_root)
        keys.add(f"{rel.parts[0]}/{p.stem}")
    return keys


def resolve_npy_for_markup(scene_dir: Path, stem: str) -> Optional[Path]:
    p_hazed = scene_dir / f"{stem}.npy"
    if p_hazed.is_file():
        return p_hazed

    if stem.endswith("_hazed"):
        alt_stem = stem.removesuffix("_hazed") + "_dehazed"
        p_dehazed = scene_dir / f"{alt_stem}.npy"
        if p_dehazed.is_file():
            return p_dehazed

    return None


def _ensure_hwc(cube: np.ndarray) -> np.ndarray:
    """Ensure cube is (H, W, C). Accepts (C, H, W) as a common alternative."""
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube, got shape={cube.shape}")

    # Heuristic: if first axis looks like channels.
    if cube.shape[0] < min(cube.shape[1], cube.shape[2]):
        return np.moveaxis(cube, 0, 2)

    return cube


def _infer_wavelengths_file(image_path: Path) -> Path:
    """
    Best-effort inference: find the first .npy under image_path that is a 1D array.
    Prefer passing meta_wavelengths explicitly in production code.
    """
    candidates = list(image_path.glob("*.npy")) + list(image_path.glob("*/*.npy"))
    for p in candidates:
        arr = np.load(p, allow_pickle=False)
        if arr.ndim == 1:
            return p

    raise FileNotFoundError(
        "Could not infer a wavelengths file (expected a 1D .npy). Pass meta_wavelengths explicitly."
    )


def main(
    image_path: Path,
    save_to: Path,
    meta_sentinel: Path,
    meta_wavelengths: Path | None = None,
    platform: str = "S2A",
    markup_root: Path | None = None,
) -> None:
    if meta_wavelengths is None:
        meta_wavelengths = _infer_wavelengths_file(image_path)

    wavelengths = np.load(meta_wavelengths, allow_pickle=False)
    if wavelengths.ndim != 1:
        raise ValueError("wavelengths must be a 1D array")

    save_to.mkdir(parents=True, exist_ok=True)

    srf_bands = load_srf_platform(meta_sentinel, platform)

    weights: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    out_band_ids: List[str] = []
    denoms: List[float] = []

    for band_id in BAND_IDS:
        band = srf_bands.get(band_id)
        if band is None:
            continue

        b_waves = np.asarray(band["wl_nm"], dtype=np.float64)
        b_vals = np.asarray(band["value"], dtype=np.float64)
        if b_waves.size < 2:
            continue

        order = np.argsort(b_waves)
        b_waves = b_waves[order]
        b_vals = b_vals[order]

        band_mask = (wavelengths >= b_waves[0]) & (wavelengths <= b_waves[-1])
        if not np.any(band_mask):
            continue

        m_waves = wavelengths[band_mask]
        srf = np.interp(m_waves, b_waves, b_vals)

        dw = np.empty_like(m_waves)
        if dw.size == 1:
            dw[0] = 1.0
        else:
            dw[:-1] = m_waves[1:] - m_waves[:-1]
            dw[-1] = dw[-2]

        sxdw = (srf * dw).astype(np.float32)
        denom = float(np.sum(sxdw))
        if denom <= 0:
            continue

        masks.append(band_mask)
        weights.append(sxdw)
        denoms.append(denom)
        out_band_ids.append(band_id)

    if not weights:
        raise RuntimeError(
            f"No bands computed. Check platform='{platform}' and SRF JSON."
        )

    def process_cube(p_npy: Path, out_dir: Path) -> None:
        cube = _ensure_hwc(np.load(p_npy, allow_pickle=False))
        h, w, _ = cube.shape
        out = np.zeros((h, w, len(out_band_ids)), dtype=cube.dtype)

        for i, (wgt, mask, denom) in enumerate(zip(weights, masks, denoms)):
            out[..., i] = np.sum(cube[..., mask] * wgt, axis=-1) / denom

        np.save(out_dir / p_npy.name, out)

    if markup_root is not None:
        for key in sorted(load_markup_keys(markup_root)):
            scene, stem = key.split("/", 1)
            p_npy = resolve_npy_for_markup(image_path / scene, stem)
            if p_npy is None:
                continue

            out_dir = save_to / scene
            out_dir.mkdir(parents=True, exist_ok=True)
            process_cube(p_npy, out_dir)
    else:
        for p_npy in image_path.glob("*/*.npy"):
            out_dir = save_to / p_npy.parent.name
            out_dir.mkdir(parents=True, exist_ok=True)
            process_cube(p_npy, out_dir)

    (save_to / "bands_order.txt").write_text("\n".join(out_band_ids), encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("image_path", type=Path)
    parser.add_argument("save_to", type=Path)
    parser.add_argument("markup_root", type=Path, help="Path to LULC markup root")
    parser.add_argument("--platform", default="S2A", choices=["S2A", "S2B", "S2C"])
    args = parser.parse_args()

    def resolve_under_base(p: Path) -> Path:
        return (BASE / p).resolve() if not p.is_absolute() else p.resolve()

    main(
        image_path=resolve_under_base(args.image_path),
        save_to=resolve_under_base(args.save_to),
        meta_sentinel=(BASE / "meta" / "sentinel" / "s2_srf.json").resolve(),
        meta_wavelengths=(
            BASE / "meta" / "wls" / "wavelengths_realhyper.npy"
        ).resolve(),
        platform=args.platform,
        markup_root=resolve_under_base(args.markup_root),
    )
