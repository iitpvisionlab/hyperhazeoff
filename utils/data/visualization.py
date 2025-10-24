from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

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
    gain : ndarray, shape (C,)
        Per-band gain coefficients to apply.
    class_name : str
        If 'clean', apply DOS before gain; in all cases scale to the 98th
        percentile per channel.

    Returns
    -------
    rgb_quicklook : ndarray, shape (H, W, 3)
        Pseudocolor image with values in [0, 1].

    Notes
    -----
    - Red channel uses the band nearest to 610 nm, green to 550 nm, blue to
      462 nm.
    - Gain is applied per selected channel, followed by percentile-based
      normalization.
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
    # wY: wavelengths in nm
    # Y : HSI as a (#pixels x #bands) matrix,
    # dims: x & y dimension of image
    # d: 50, 55, 65, 75, determines the illuminant used, if in doubt use d65
    # thresholdRGB : True if thesholding should be done to increase contrast

    # If you use this method, please cite the following paper:
    #  M. Magnusson, J. Sigurdsson, S. E. Armansson, M. O. Ulfarsson,
    #  H. Deborah and J. R. Sveinsson,
    #  "Creating RGB Images from Hyperspectral Images Using a Color Matching Function",
    #  IEEE International Geoscience and Remote Sensing Symposium, Virtual Symposium, 2020

    #  @INPROCEEDINGS{hsi2rgb,
    #  author={M. {Magnusson} and J. {Sigurdsson} and S. E. {Armansson}
    #  and M. O. {Ulfarsson} and H. {Deborah} and J. R. {Sveinsson}},
    #  booktitle={IEEE International Geoscience and Remote Sensing Symposium},
    #  title={Creating {RGB} Images from Hyperspectral Images using a Color Matching Function},
    #  year={2020}, volume={}, number={}, pages={}}

    # Paper is available at
    # https://www.researchgate.net/profile/Jakob_Sigurdsson

    """
    # Load reference illuminant
    D = spio.loadmat("./D_illuminants.mat")

    w = D["wxyz"][:, 0]
    x = D["wxyz"][:, 1]
    y = D["wxyz"][:, 2]
    z = D["wxyz"][:, 3]

    D = D["D"]

    i = {50: 2, 55: 3, 65: 1, 75: 4}

    wI = D[:, 0]
    I = D[:, i[d]]

    # Interpolate to image wavelengths
    I = PchipInterpolator(wI, I, extrapolate=True)(
        wY
    )  # interp1(wI,I,wY,'pchip','extrap')';
    x = PchipInterpolator(w, x, extrapolate=True)(
        wY
    )  # interp1(w,x,wY,'pchip','extrap')';
    y = PchipInterpolator(w, y, extrapolate=True)(
        wY
    )  # interp1(w,y,wY,'pchip','extrap')';
    z = PchipInterpolator(w, z, extrapolate=True)(
        wY
    )  # interp1(w,z,wY,'pchip','extrap')';

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

    R = np.reshape(sRGB[0, :], [ydim, xdim])
    G = np.reshape(sRGB[1, :], [ydim, xdim])
    B = np.reshape(sRGB[2, :], [ydim, xdim])

    return np.transpose(np.array([R, G, B]), [1, 2, 0])


def CSNC(hsi: np.ndarray, wls: npt.NDArray):

    rgb_quicklook = hsi_to_pseudocolor(hsi, wls)
    p_max = np.percentile(rgb_quicklook, PERCENTILE).astype(np.float32)
    rgb_quicklook /= p_max
    return rgb_quicklook.clip(0, 1)


def CSSO(hsi: np.ndarray, wls: npt.NDArray):

    xdim, ydim, zdim = hsi.shape
    wl = np.squeeze(wls).tolist()
    hsi = np.reshape(hsi, [-1, zdim]) / hsi.max()
    illuminant = 65
    # Do minor thresholding
    threshold = 0.002
    srgb = hsi_to_color(wl, hsi, xdim, ydim, illuminant, threshold)
    return srgb.clip(0, 1)


if __name__ == "__main__":

    """
    Usage Example:

    Specify the path [hsi_file_path] to the HSI file for which visualization is required
    and the corresponding wavelengths [hsi_wls_path].

    """

    hsi_file_path = ""
    hsi_wls_path = ""

    hsi = np.load(hsi_file_path)
    wls = np.load(hsi_wls_path)

    rgb = CSNC(hsi, wls)
    srgb = CSSO(hsi, wls)

    from PIL import Image

    rgb = Image.fromarray((rgb * 255).astype(np.uint8))
    srgb = Image.fromarray((srgb * 255).astype(np.uint8))

    rgb.save("rgb.png")
    srgb.save("srgb.png")
