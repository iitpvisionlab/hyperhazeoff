from __future__ import annotations


import argparse
from pathlib import Path
from typing import Literal


import numpy as np
import numpy.typing as npt
from PIL import Image
from tqdm import tqdm


import scipy.io as spio
from scipy.interpolate import PchipInterpolator
from bisect import bisect


PERCENTILE = 99.25


def hsi_to_pseudocolor(
    hsi: npt.NDArray[np.floating], wls: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """
    Create a pseudocolor RGB quicklook using nearest bands to 610/550/462 nm.


    Parameters
    ----------
    hsi : ndarray, shape (H, W, C)
        Hyperspectral cube.
    wls : ndarray, shape (C,)
        Wavelength centers (nm) corresponding to the bands in `hsi`.


    Returns
    -------
    rgb_quicklook : ndarray, shape (H, W, 3)
        Pseudocolor image with values in [0, 1].


    Notes
    -----
    - Red channel uses the band nearest to 646.7 nm, green to 547.6 nm, blue to
      449.1 nm.
    - Percentile-based normalization is applied.
    """

    def find_nearest_idx(array, value):
        return np.abs(array - value).argmin()

    idx_r = find_nearest_idx(wls, 646.7)
    idx_g = find_nearest_idx(wls, 547.6)
    idx_b = find_nearest_idx(wls, 449.1)

    rgb_quicklook = np.stack(
        [hsi[..., idx_r], hsi[..., idx_g], hsi[..., idx_b]], axis=-1
    ).astype(np.float32)

    return rgb_quicklook


def hsi_to_color(
    wY: npt.NDArray,
    HSI: npt.NDArray,
    ydim: int,
    xdim: int,
    d: Literal[50, 55, 65, 75] = 65,
    threshold: float = 0.002,
):
    """
    Convert hyperspectral image to sRGB using color matching functions.

    Parameters
    ----------
    wY : ndarray
        Wavelengths in nm.
    HSI : ndarray
        HSI as a (#pixels x #bands) matrix.
    ydim : int
        Y dimension of image.
    xdim : int
        X dimension of image.
    d : {50, 55, 65, 75}, default 65
        Determines the illuminant used.
    threshold : float, default 0.002
        Threshold value for contrast enhancement.

    Returns
    -------
    ndarray, shape (xdim, ydim, 3)
        RGB image with values in [0, 1].

    Notes
    -----
    If you use this method, please cite:
    M. Magnusson, J. Sigurdsson, S. E. Armansson, M. O. Ulfarsson,
    H. Deborah and J. R. Sveinsson,
    "Creating RGB Images from Hyperspectral Images Using a Color Matching Function",
    IEEE International Geoscience and Remote Sensing Symposium, Virtual Symposium, 2020
    """
    # Load reference illuminant
    D = spio.loadmat("./utils/data/D_illuminants.mat")

    w = D["wxyz"][:, 0]
    x = D["wxyz"][:, 1]
    y = D["wxyz"][:, 2]
    z = D["wxyz"][:, 3]

    D = D["D"]

    i = {50: 2, 55: 3, 65: 1, 75: 4}

    wI = D[:, 0]
    I = D[:, i[d]]

    # Interpolate to image wavelengths
    I = PchipInterpolator(wI, I, extrapolate=True)(wY)
    x = PchipInterpolator(w, x, extrapolate=True)(wY)
    y = PchipInterpolator(w, y, extrapolate=True)(wY)
    z = PchipInterpolator(w, z, extrapolate=True)(wY)

    # Truncate at 780nm
    i = bisect(wY, 780)
    HSI = HSI[:, 0:i] / HSI.max()
    wY = wY[:i]
    I = I[:i]
    x = x[:i]
    y = y[:i]
    z = z[:i]

    # Compute k
    k = 1 / np.trapz(y * I, wY)

    # Compute X,Y & Z for image
    X = k * np.trapz(HSI @ np.diag(I * x), wY, axis=1)
    Z = k * np.trapz(HSI @ np.diag(I * z), wY, axis=1)
    Y = k * np.trapz(HSI @ np.diag(I * y), wY, axis=1)

    XYZ = np.array([X, Y, Z])

    # Convert to RGB
    M = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    )
    sRGB = M @ XYZ

    # Gamma correction
    gamma_map = sRGB > 0.0031308
    sRGB[gamma_map] = 1.055 * np.power(sRGB[gamma_map], (1.0 / 2.4)) - 0.055
    sRGB[np.invert(gamma_map)] = 12.92 * sRGB[np.invert(gamma_map)]
    # Note: RL, GL or BL values less than 0 or greater than 1 are clipped to 0 and 1.
    sRGB[sRGB > 1] = 1
    sRGB[sRGB < 0] = 0

    if threshold:
        for idx in range(3):
            y = sRGB[idx, :]
            a, b = np.histogram(y, 100)
            b = b[:-1] + np.diff(b) / 2
            a = np.cumsum(a) / np.sum(a)
            th = b[0]
            i = a < threshold
            if i.any():
                th = b[i][-1]
            y = y - th
            y[y < 0] = 0

            a, b = np.histogram(y, 100)
            b = b[:-1] + np.diff(b) / 2
            a = np.cumsum(a) / np.sum(a)
            i = a > 1 - threshold
            th = b[i][0]
            y[y > th] = th
            y = y / th
            sRGB[idx, :] = y

    R = np.reshape(sRGB[0, :], [xdim, ydim])
    G = np.reshape(sRGB[1, :], [xdim, ydim])
    B = np.reshape(sRGB[2, :], [xdim, ydim])

    return np.transpose(np.array([R, G, B]), [1, 2, 0])


def CSNC(hsi: np.ndarray, wls: npt.NDArray) -> np.ndarray:
    """
    Create RGB visualization using Color Space Nearest Channel method.

    Parameters
    ----------
    hsi : ndarray, shape (H, W, C)
        Hyperspectral cube.
    wls : ndarray, shape (C,)
        Wavelengths in nm.

    Returns
    -------
    ndarray, shape (H, W, 3)
        RGB image with values in [0, 1].
    """
    rgb_quicklook = hsi_to_pseudocolor(hsi, wls)
    p_max = np.percentile(rgb_quicklook, PERCENTILE).astype(np.float32)
    rgb_quicklook /= p_max
    return rgb_quicklook.clip(0, 1)


def CSSO(hsi: np.ndarray, wls: npt.NDArray) -> np.ndarray:
    """
    Create RGB visualization using Color Space Spectral Optimization method.

    Parameters
    ----------
    hsi : ndarray, shape (H, W, C)
        Hyperspectral cube.
    wls : ndarray, shape (C,)
        Wavelengths in nm.

    Returns
    -------
    ndarray, shape (H, W, 3)
        sRGB image with values in [0, 1].
    """
    xdim, ydim, zdim = hsi.shape
    wl = np.squeeze(wls).tolist()
    hsi_flat = np.reshape(hsi, [-1, zdim]) / hsi.max()
    illuminant = 65
    threshold = 0.002
    srgb = hsi_to_color(wl, hsi_flat, xdim, ydim, illuminant, threshold)
    return srgb.clip(0, 1)


# ----------------------------------------------------------------------
# CLI for batch visualization
# ----------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for hyperspectral visualization.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate RGB visualizations from hyperspectral images"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory with hyperspectral .npy files",
    )
    parser.add_argument(
        "--wl", type=Path, required=True, help="Path to wavelengths .npy file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save RGB visualizations",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["csnc", "csso", "both"],
        default="both",
        help="Visualization method: 'csnc' (pseudocolor), 'csso' (spectral), or 'both'",
    )
    parser.add_argument(
        "--patch-ids",
        nargs="*",
        default=[],
        help="List of patch directories to process (e.g., S01C00 S02C06). If empty, process all.",
    )

    return parser.parse_args()


def process_visualization(
    input_dir: Path,
    output_dir: Path,
    wls: npt.NDArray,
    method: str = "both",
    patch_ids: list[str] | None = None,
) -> None:
    """
    Process hyperspectral images and generate RGB visualizations.

    Parameters
    ----------
    input_dir : Path
        Directory containing input HSI subdirectories.
    output_dir : Path
        Directory where visualizations will be saved.
    wls : ndarray
        Wavelengths array.
    method : str, default "both"
        Visualization method: 'csnc', 'csso', or 'both'.
    patch_ids : list of str, optional
        List of specific patch IDs to process. If None or empty, processes all patches.

    Returns
    -------
    None
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each patch directory
    for patch_dir in tqdm(sorted(input_dir.iterdir()), desc="Processing patches"):
        if not patch_dir.is_dir():
            continue

        patch_id = patch_dir.name

        # Skip if specific patch IDs were requested and this isn't one of them
        if patch_ids and patch_id not in patch_ids:
            continue

        # Process all .npy files in this patch directory
        for hsi_file in patch_dir.glob("*.npy"):
            # Load HSI cube
            hsi = np.load(hsi_file)

            # Generate visualizations based on method
            if method in ["csnc", "both"]:
                # Create CSNC method directory structure
                out_csnc = output_dir / "CSNC" / patch_id
                out_csnc.mkdir(parents=True, exist_ok=True)

                rgb = CSNC(hsi, wls)
                rgb_img = Image.fromarray((rgb * 255).astype(np.uint8))
                out_file = out_csnc / f"{hsi_file.stem}.png"
                rgb_img.save(out_file)

            if method in ["csso", "both"]:
                # Create CSSO method directory structure
                out_csso = output_dir / "CSSO" / patch_id
                out_csso.mkdir(parents=True, exist_ok=True)

                srgb = CSSO(hsi, wls)
                srgb_img = Image.fromarray((srgb * 255).astype(np.uint8))
                out_file = out_csso / f"{hsi_file.stem}.png"
                srgb_img.save(out_file)


def main() -> None:
    """
    Main entry point for the CLI application.
    """
    args = parse_arguments()

    # Load wavelengths
    wls = np.load(args.wl)

    # Process visualizations
    process_visualization(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        wls=wls,
        method=args.method,
        patch_ids=args.patch_ids if args.patch_ids else None,
    )


if __name__ == "__main__":
    main()
