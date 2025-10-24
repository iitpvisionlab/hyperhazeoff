# realhyperpdid.py

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path

from typing import Callable, Union, List, Optional, Sequence, Tuple


class CustomTransform(nn.Module):
    """Base class: input is ``haze`` or ``(haze, clear)``.

    Sub‑classes must implement :pymeth:`forward` and keep the same signature.
    """

    def forward(self, haze: torch.Tensor):  # noqa: D401
        raise NotImplementedError


class CustomCompose(CustomTransform):
    """Compose a list of :class:`CustomTransform` objects.

    Works exactly like ``torchvision.transforms.Compose`` but supports the pair
    API. Each transform in the list receives the output of the previous one.
    """

    def __init__(self, transforms: List[CustomTransform]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, haze: torch.Tensor):
        for t in self.transforms:
            haze = t(haze)
        return haze


class MaxNorm(CustomTransform):
    """Scale every cube to the ``[0, 1]`` range by its own max value."""

    def forward(self, haze: torch.Tensor):
        eps = 1e-8
        haze = haze / (haze.max() + eps)
        return haze


class UnitToSigned(CustomTransform):
    """Convert ``[0, 1]`` tensors to ``[‑1, 1]`` (works pair‑wise)."""

    def forward(self, haze: torch.Tensor):
        haze = haze * 2 - 1
        return haze


class SignedToUnit(CustomTransform):
    """Convert ``[-1, 1]`` tensors to ``[0, 1]`` (works pair‑wise)."""

    def forward(self, haze: torch.Tensor):
        haze = haze * 0.5 + 0.5
        return haze


class RRHPDID(Dataset):
    """Dataset that yields hazy cubes from the RRealHyperPDID dataset.

    Parameters
    ----------
    root : str or pathlib.Path

    transform : Callable or None
        Applied to the loaded arrays **after** conversion to ``torch.Tensor``

    format : str
        Hazed sample data format for Hyperspectral Images .npy, for RGB .png
    """

    def __init__(
        self,
        root: Union[str, Path],
        transform: Union[Callable, None] = None,
        format: str = "npy",
    ) -> None:
        self.root = Path(root)
        self.transform = transform

        self.hazy_files: list[Path] = sorted(self.root.rglob(f"*_hazed.{format}"))
        if not self.hazy_files:
            raise FileNotFoundError(
                f"No '*_hazed.{format}' files found under " f"{self.root}"
            )

    # @staticmethod
    # def _np_to_tensor(arr: np.ndarray) -> torch.Tensor:
    #     return torch.from_numpy(arr.astype(np.float32))

    @staticmethod
    def _reader(path: Path) -> np.ndarray:
        loaders = {
            ".png": lambda p: np.array(Image.open(p)).astype(np.uint8),
            ".npy": lambda p: np.load(p).astype(np.float32),
        }
        return loaders[path.suffix.lower()](path)

    def __len__(self) -> int:
        return len(self.hazy_files)

    def __getitem__(self, idx: int):
        haze_path = self.hazy_files[idx]
        haze = self._reader(haze_path)
        if self.transform is not None:
            haze = self.transform(haze)
        return haze


if __name__ == "__main__":
    dataset = RRHPDID(
        "/workspace/output/cli-samara/dehazing/data/source/RRealHyperPDID/CSNC",
        format="png",
    )

    hsi = next(iter(dataset))
    print(hsi.shape, hsi.min(), hsi.max())
