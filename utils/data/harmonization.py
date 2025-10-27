from __future__ import annotations


import argparse
from pathlib import Path
from typing import Sequence, Tuple, Union, Literal, Any


import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from tqdm import tqdm


def harmonize_cube(
    cube: NDArray[np.floating],
    wl_src_nm: NDArray[np.floating],
    wl_tgt_nm: NDArray[np.floating],
    kind: Literal["linear", "nearest"] = "linear",
    clip_negative: bool = True,
) -> NDArray[np.floating]:
    """
    Resample spectral bands of a hyperspectral cube to target wavelengths.


    Parameters
    ----------
    cube : ndarray, shape (H, W, B_src)
        Input hyperspectral cube with B_src bands.
    wl_src_nm : ndarray, shape (B_src,)
        Source wavelengths in nanometers.
    wl_tgt_nm : ndarray, shape (B_tgt,)
        Target wavelengths in nanometers.
    kind : {'linear', 'nearest'}, default 'linear'
        Interpolation method passed to SciPy interp1d.
    clip_negative : bool, default True
        If True, negative interpolated values are set to zero.


    Returns
    -------
    ndarray, shape (H, W, B_tgt)
        Resampled hyperspectral cube with B_tgt bands.


    Raises
    ------
    ValueError
        If the number of bands in cube does not match length of wl_src_nm.
    """
    if cube.shape[-1] != wl_src_nm.size:
        raise ValueError(
            f"cube has {cube.shape[-1]} bands but wl_src_nm has {wl_src_nm.size} entries"
        )

    h, w, bsrc = cube.shape
    flat = cube.reshape(-1, bsrc)
    interpolator = interp1d(
        wl_src_nm,
        flat,
        axis=1,
        kind=kind,
        bounds_error=False,
        fill_value="extrapolate",
    )
    res_flat = interpolator(wl_tgt_nm)
    res = res_flat.reshape(h, w, wl_tgt_nm.size)

    if clip_negative:
        np.maximum(res, 0, out=res)

    return res.astype(cube.dtype, copy=False)


def drop_bands(
    cube: NDArray[Any],
    bands_to_drop: Union[Sequence[int], set[int]],
    wavelengths_nm: NDArray[Any] | None = None,
) -> Union[NDArray[Any], Tuple[NDArray[Any], NDArray[Any]]]:
    """
    Remove specified spectral bands from a hyperspectral cube (and wavelengths).


    Parameters
    ----------
    cube : ndarray, shape (H, W, B)
        Input hyperspectral cube with B bands.
    bands_to_drop : sequence of int or set of int
        Zero-based indices of bands to remove.
    wavelengths_nm : ndarray of shape (B,) or None, optional
        If provided, returns updated wavelengths array with dropped indices removed.


    Returns
    -------
    cube_out : ndarray, shape (H, W, B_out)
        Hyperspectral cube after removing specified bands.
    wavelengths_out : ndarray, shape (B_out,), optional
        Filtered wavelengths if wavelengths_nm was given.
    """
    total_bands = cube.shape[-1]
    keep_indices = np.setdiff1d(np.arange(total_bands), list(bands_to_drop))
    cube = cube[..., keep_indices]

    if wavelengths_nm is None:
        return cube

    wavelengths_out = wavelengths_nm[keep_indices]
    return cube, wavelengths_out


def save_wavelengths(
    wl_nm: NDArray[Any],
    path: Path,
) -> None:
    """
    Save wavelength array to a .npy file, creating directories if needed.


    Parameters
    ----------
    wl_nm : ndarray
        Wavelength values to save.
    path : pathlib.Path
        Output path for the .npy file (extension added if missing).


    Returns
    -------
    None
    """
    output_path = path.with_suffix(".npy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, wl_nm.astype(np.float32))


# ----------------------------------------------------------------------
# CLI for patch interpolation
# ----------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for hyperspectral patch interpolation.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Interpolate raw hyperspectral patches"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory with raw .npy patches (e.g., HSI/)",
    )
    parser.add_argument(
        "--wl-src",
        type=Path,
        required=True,
        help="Path to source wavelengths .npy file",
    )
    parser.add_argument(
        "--wl-tgt",
        type=Path,
        required=True,
        help="Path to target wavelengths .npy file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save interpolated patches",
    )
    parser.add_argument(
        "--patch-ids",
        nargs="*",
        default=[],
        help="List of patch directories to process (e.g., S01C00 S02C06). If empty, process all.",
    )

    return parser.parse_args()


def process_samples(
    input_dir: Path,
    output_dir: Path,
    wl_src: NDArray[np.floating],
    wl_tgt: NDArray[np.floating],
    patch_ids: list[str] | None = None,
) -> None:
    """
    Process hyperspectral patches by interpolating them to target wavelengths.

    Parameters
    ----------
    input_dir : Path
        Directory containing input patch subdirectories.
    output_dir : Path
        Directory where interpolated patches will be saved.
    wl_src : ndarray
        Source wavelengths array.
    wl_tgt : ndarray
        Target wavelengths array.
    patch_ids : list of str, optional
        List of specific patch IDs to process. If None or empty, processes all patches.

    Returns
    -------
    None
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for patch_dir in tqdm(sorted(input_dir.iterdir())):
        if not patch_dir.is_dir():
            continue

        patch_id = patch_dir.name

        # Skip if specific patch IDs were requested and this isn't one of them
        if patch_ids and patch_id not in patch_ids:
            continue

        # Create corresponding output directory
        out_patch = output_dir / patch_id
        out_patch.mkdir(parents=True, exist_ok=True)

        # Process all .npy files in this patch directory
        for file in patch_dir.glob("*.npy"):
            cube = np.load(file)
            cube_i = harmonize_cube(cube, wl_src, wl_tgt)
            out_file = out_patch / file.name
            np.save(out_file, cube_i)


def main() -> None:
    """
    Main entry point for the CLI application.
    """
    args = parse_arguments()

    # Load wavelength arrays
    wl_src = np.load(args.wl_src)
    wl_tgt = np.load(args.wl_tgt)

    # Process patches
    process_samples(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        wl_src=wl_src,
        wl_tgt=wl_tgt,
        patch_ids=args.patch_ids if args.patch_ids else None,
    )


if __name__ == "__main__":
    main()
