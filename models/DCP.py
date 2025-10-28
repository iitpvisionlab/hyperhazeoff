import numpy as np

from cv2.ximgproc import guidedFilter
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2hsv
import numpy as np


def get_dark_value(x, y, I, dx=7, dy=7):
    """Get minimal value through all the channels in considered window for one pixel by its coords

    Args:
        x, y (int) : pixel coordinates
        I (numpy ndarray): image, shape (x_size, y_size, 3)
        dx, dy (int, optional): window size. Defaults to (7, 7).

    Returns:
        int: dark value
    """
    r = np.min(I[x - dx : x + dx + 1, y - dy : y + dy + 1])
    return r


def get_dark_channel(I, dx, dy):
    x_size, y_size = I.shape[:2]
    I_dc = np.zeros_like(I)
    pad_I = np.pad(I, ((dx, dx), (dy, dy), (0, 0)), mode="reflect")
    for x in range(dx, x_size + dx):
        for y in range(dy, y_size + dy):
            I_dc[x - dx, y - dy] = np.min(
                pad_I[x - dx : x + dx + 1, y - dy : y + dy + 1]
            )
    return I_dc[..., 0]


def window_min(I, dx, dy):
    """Window minimum filter for an image

    Args:
        I (numpy ndarray): image, shape (x_size, y_size)
        dx, dy (int): window size.

    Returns:
        numpy ndarray: image (x_size, y_size)
    """
    x_size, y_size = I.shape
    I_dc = np.zeros_like(I)
    pad_I = np.pad(I, ((dx, dx), (dy, dy)), mode="edge")
    for x in range(dx, x_size + dx):
        for y in range(dy, y_size + dy):
            I_dc[x - dx, y - dy] = get_dark_value(x, y, pad_I, dx, dy)

    return I_dc


def zhu_depth_estim(I, dx=7, dy=7, r=30, eps=0.01, gf_on=True):
    """Atmospheric Light Estimation Based Remote Sensing Image Dehazing by
    Z. Zhu et. al.: [https://www.mdpi.com/2072-4292/13/13/2432/htm](https://www.mdpi.com/2072-4292/13/13/2432/htm)
    Depth map estimation

    Args:
        I (numpy ndarray): image, shape (x_size, y_size, 3)
        dx, dy (int, optional): window size. Defaults to (7, 7).
        r (int, optional): guided filter radius. Defaults to 30.
        eps (float, optional): reg param for guided filter. Defaults to 0.01.
        gf_on (bool, optional): apply guided filter or not. Defaults to True.

    Returns:
        numpy ndarray: depth map (x_size, y_size)
    """
    import matplotlib.pyplot as plt

    I_hsv = rgb2hsv(I)
    w0 = 0.121779
    w1 = 0.959710
    w2 = -0.780245
    sigma = 0.041337

    d = (w0 + w1 * I_hsv[:, :, 2] + w2 * I_hsv[:, :, 1]).astype(np.float32)
    d = gaussian_filter(d, sigma)
    d = window_min(d, dx, dy)

    if gf_on:
        d = guidedFilter(I, d, r, eps)

    return d


def zhu_depth(
    I,
    dx=7,
    dy=7,
    t0=0.01,
    r=30,
    eps=0.01,
    d_quantile=0.999,
    gf_on=True,
    beta=1.0,
):
    """Atmospheric Light Estimation Based Remote Sensing Image Dehazing by
    Z. Zhu et. al.: [https://www.mdpi.com/2072-4292/13/13/2432/htm](https://www.mdpi.com/2072-4292/13/13/2432/htm)
    Dehazing algorithm

    Args:
        I (numpy ndarray): image, shape (x_size, y_size, 3)
        dx, dy (int, optional): window size. Defaults to (7, 7).
        t0 (float, optional): reg param for transmission map. Defaults to 0.1.
        r (int, optional): guided filter radius. Defaults to 30.
        eps (float, optional): reg param for guided filter. Defaults to 0.01.
        d_quantile (float, optional): quantile for veil color estimation. Defaults to 0.999.
        gf_on (bool, optional): apply guided filter or not. Defaults to True.
        beta (float, optional): absorption coefficient. Defaults to 1.2

    Returns:
        numpy ndarray: depth map (x_size, y_size)
    """

    d = zhu_depth_estim(I, dx, dy, r, eps, True)

    q_d = np.quantile(d, d_quantile)
    I_intens = I.sum(axis=2)

    I_max = I_intens[d >= q_d].max()
    a_rgb = I[(d >= q_d) & (I_intens == I_max)].mean(axis=0)
    # print("Zhu depth map цвет дымки:", a_rgb)

    A = a_rgb * np.ones(I.shape)

    t = np.exp(-beta * d)
    t = np.dstack((t, t, t))

    J = (I - A) / (np.minimum(np.maximum(t, 0.1), 0.9)) + A
    J = np.clip(J, 0, 255).astype(np.uint8)
    return J


def zhu_dcp(
    I,
    dx=7,
    dy=7,
    k=0.95,
    t0=0.01,
    r=30,
    eps=0.01,
    d_quantile=0.999,
    gf_on=True,
    a_mean=True,
):
    """Atmospheric Light Estimation Based Remote Sensing Image Dehazing by
    Z. Zhu et. al.: [https://www.mdpi.com/2072-4292/13/13/2432/htm](https://www.mdpi.com/2072-4292/13/13/2432/htm)
    and Dark Channel Prior dehazing algorithm

    Args:
        I (numpy ndarray): image, shape (x_size, y_size, 3)
        dx, dy (int, optional): window size. Defaults to (7, 7).
        k (float, optional): reg param for dehazed image. Defaults to 0.95.
        t0 (float, optional): reg param for transmission map. Defaults to 0.1.
        r (int, optional): guided filter radius. Defaults to 30.
        eps (float, optional): reg param for guided filter. Defaults to 0.01.
        d_quantile (float, optional): quantile for veil color estimation. Defaults to 0.999.
        gf_on (bool, optional): apply guided filter or not. Defaults to True.
        a_mean (bool, optional): average to estimate veil color or not . Defaults to True.

    Returns:
        numpy ndarray: depth map (x_size, y_size)
    """

    d = zhu_depth_estim(I, dx, dy, r, eps, gf_on)

    q_d = np.quantile(d, d_quantile)
    I_intens = I.sum(axis=2)

    if a_mean:  # усреднение по пикселям
        a_rgb = (I[d >= q_d]).mean(axis=0)
    else:  # самый яркий пиксель
        I_max = I_intens[d >= q_d].max()
        a_rgb = I[(d >= q_d) & (I_intens == I_max)].mean(axis=0)
    A = a_rgb * np.ones(I.shape)
    # print(f"Zhu: mean={a_mean}, gf={gf_on} цвет дымки:", a_rgb)

    I_a = I / A
    V = get_dark_channel(I_a, dx, dy).astype(np.float32)
    t = 1 - k * V
    t = guidedFilter(I, t, r, eps)
    t = np.dstack((t, t, t))

    J = (I - A) / (np.minimum(np.maximum(t, 0.1), 0.9)) + A
    J = np.clip(J, 0, 255).astype(np.uint8)
    return J


def idcp(I, dx=7, dy=7, k=0.95, t0=0.1, r=30, eps=0.01, dc_quantile=0.999):
    """Dark Channel Prior algorithm using guided filter
    K. He: [http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.672.3815&rep=rep1&type=pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.672.3815&rep=rep1&type=pdf)

    Args:
        I (numpy ndarray): image, shape (x_size, y_size, 3)
        dx, dy (int, optional): window size. Defaults to (7, 7).
        k (float, optional): reg param for dehazed image. Defaults to 0.95.
        t0 (float, optional): reg param for transmission map. Defaults to 0.1.
        r (int, optional): guided filter radius. Defaults to 30.
        eps (float, optional): reg param for guided filter. Defaults to 0.01.
        dc_quantile (float, optional): quantile for veil color estimation. Defaults to 0.999.

    Returns:
        numpy ndarray: dehazed image
    """
    # get veil color
    I_dark = get_dark_channel(I, dx, dy)
    q_I_dark = np.quantile(I_dark, dc_quantile)
    I_intens = I.sum(axis=2)
    q_intens = np.quantile(I_intens[I_dark >= q_I_dark], 0.9)
    a_rgb = (I[(I_dark >= q_I_dark) & (I_intens >= q_intens)]).mean(axis=0)
    A = a_rgb * np.ones(I.shape)
    # print("IDCP Цвет дымки:", a_rgb)

    # get and filter transmisson map
    I_a = I / A
    V = get_dark_channel(I_a, dx, dy).astype(np.float32)
    t = 1 - k * V
    t = guidedFilter(I, t, r, eps)
    t = np.dstack((t, t, t))

    # final dehazed image
    J = (I - A) / (np.minimum(np.maximum(t, 0.1), 0.9)) + A
    J = np.clip(J, 0, 255).astype(np.uint8)
    return J


# if __name__ == "__main__":
#     dehazed_cadcp = zhu_dcp(haze_img, gf_on=True, a_mean=True)
#     dehazed_dcp = idcp(haze_img)
