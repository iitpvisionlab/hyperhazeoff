from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pathlib import Path
from typing import Any, Callable, Dict, OrderedDict, List, Type, Union, Optional

import yaml
from tqdm import tqdm

from models import *
from datasets.realhyperpdid import (
    RRHPDID,
    UnitToSigned,
    CustomCompose,
    MaxNorm,
    CustomTransform,
)

import argparse
from PIL import Image


MODEL_REGISTRY: Dict[str, Union[Type[torch.nn.Module], Callable]] = {
    "dcp": dcp,  # only for RGB
    "cadcp": cadcp,  # only for RGB
    "dehazeformer_b": dehazeformer_b,
    "dehazeformer_m": dehazeformer_m,
    "dehazeformer": DehazeFormer,
    "aacnet": AACNet,
    "hdmba": HDMba,
    "aidtransformer": AIDTransformer,
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = base.copy()
    for k, v in override.items():
        if v is not None:
            out[k] = v
    return out


def _load_ckpt(model: torch.nn.Module, ckpt: Path, device: torch.device) -> None:

    checkpoint = torch.load(ckpt, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise RuntimeError("Loaded checkpoint is not a dict or compatible format")

    clean_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        clean_key = key
        for prefix in ["module.", "model.", "backbone."]:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        clean_state_dict[clean_key] = value

    try:
        missing_keys, unexpected_keys = model.load_state_dict(
            clean_state_dict, strict=True
        )
        print("Weights loaded successfully with strict=True")
    except RuntimeError as e:
        print(f"Strict loading failed: {e}")
        print("Attempting non-strict loading...")
        missing_keys, unexpected_keys = model.load_state_dict(
            clean_state_dict, strict=False
        )
        critical_missing = [
            k
            for k in missing_keys
            if not k.endswith((".num_batches_tracked", ".running_mean", ".running_var"))
        ]
        if critical_missing:
            print(f"Critical weights missing: {critical_missing}")
            raise RuntimeError("Critical model weights are missing!")

    if missing_keys:
        print(f"Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)}")


def save_file(dehazed: np.ndarray, filename: Union[str, Path], format: str) -> None:
    if format == "png":
        dehazed = (
            dehazed.astype(np.uint8)
            if dehazed.dtype == np.uint8
            else (dehazed * 255).astype(np.uint8)
        )
        img = Image.fromarray(dehazed.astype(np.uint8))
        img.save(filename)
    else:
        np.save(filename, dehazed)


def classic_processing(haze: torch.Tensor, cfg: Dict[str, Any]) -> List[np.ndarray,]:

    haze_np = haze.cpu().squeeze(0).numpy() if torch.is_tensor(haze) else haze
    if cfg["arch"] == "dcp":
        pred = dcp(haze_np)
    elif cfg["arch"] == "cadcp":
        pred = cadcp(haze_np)
    return [pred]


def nn_processing(
    haze: torch.Tensor,
    model: Union[torch.nn.Module, Callable],
    cfg: Dict[str, Any],
    device,
) -> torch.Tensor:

    haze = haze.to(device, non_blocking=True)
    with torch.no_grad():
        pred = model(haze)

    if cfg["arch"].startswith("dehazeformer"):
        pred = torch.clamp(pred, -1.0, 1.0)
        pred = pred * 0.5 + 0.5

    pred = torch.clamp(pred, 0.0, 1.0)

    return pred


def run(cfg: Dict[str, Any]) -> None:
    """Run inference according to *cfg*."""
    transform: Optional[CustomTransform] = None
    device = torch.device(
        "cuda"
        if cfg["device"] == "auto" and torch.cuda.is_available()
        else cfg["device"]
    )

    arch_key = cfg["arch"]
    if arch_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown architecture '{arch_key}'.")

    if arch_key not in ("dcp", "cadcp"):
        channels = cfg["in_channels"]
        try:
            model = MODEL_REGISTRY[arch_key](in_channels=channels).to(device)
        except TypeError:
            model = MODEL_REGISTRY[arch_key]().to(device)

        if ckpt := cfg.get("ckpt"):
            print("Loading checkpoint …")
            _load_ckpt(model, Path(ckpt), device)
        if device == "cuda" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)  # type: ignore
        model.eval()

        transform = (
            CustomCompose([MaxNorm(), UnitToSigned()])
            if arch_key.startswith("dehazeformer")
            else MaxNorm()
        )

    # ─── transforms ───
    dataset = RRHPDID(cfg["data_root"], transform, format=cfg["format"])

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch"],
        num_workers=cfg["workers"],
        pin_memory=True,
        shuffle=False,
    )

    # ─── loop ───
    for g_idx, batch in enumerate(tqdm(loader, desc="Inference", unit="batch")):
        haze = batch
        src = dataset.hazy_files[g_idx]
        if cfg["arch"] in ("dcp", "cadcp"):
            pred = classic_processing(haze, cfg)
            cubes = pred
        else:

            haze = haze.permute(0, 3, 1, 2).float()
            pred = nn_processing(haze, model, cfg, device)
            cubes = pred.cpu().permute(0, 2, 3, 1).numpy()

        for l_idx, cube in enumerate(cubes):
            idx = g_idx * cfg["batch"] + l_idx
            src_list = src if isinstance(src, (tuple, list)) else [src]
            fname = src_list[0].name.replace("_hazed", "_dehazed")

            out_path: Path = (
                Path(cfg["out_dir"]) / cfg["arch"] / src_list[0].parent.name
            )
            out_path.mkdir(parents=True, exist_ok=True)
            save_file(cube, out_path / fname, format=cfg["format"])


from pathlib import Path
from typing import Any, Dict


def parse_cli() -> Dict[str, Any]:
    p = argparse.ArgumentParser("RRHPDID inference (YAML config)")


    p.add_argument("--config", type=Path, required=True, help="YAML config file")


    p.add_argument("--data_root", type=Path, help="Root dataset directory")
    p.add_argument("--out_dir", type=Path, help="Output directory")
    p.add_argument(
        "--arch", choices=list(MODEL_REGISTRY.keys()), help="Model architecture"
    )
    p.add_argument("--ckpt", type=Path, help="Checkpoint path")
    p.add_argument("--batch", type=int, help="Batch size")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Device")
    p.add_argument("--workers", type=int, help="Number of data loader workers")
    p.add_argument("--format", type=str, help="Input data format (e.g. png)")
    p.add_argument(
        "--in_channels",
        type=np.uint16,
        help="Spectral channels for Hyperspectral Images",
    )

    args = vars(p.parse_args())

    cfg_base = _load_yaml(args.pop("config"))
    cfg = _merge(cfg_base, {k: v for k, v in args.items() if v is not None})

    for field in ("data_root", "out_dir"):
        if field not in cfg or cfg[field] is None:
            raise ValueError(f"Missing required field `{field}` in YAML or CLI")

    return cfg


if __name__ == "__main__":
    run(parse_cli())
