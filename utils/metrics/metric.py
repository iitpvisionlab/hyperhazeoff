from torch import Tensor
import torch
import torch.nn.functional as F
from typing import Literal, Optional, Tuple

__all__ = [
    "ssim_torch",
    "uqi_torch",
    "sam_torch",
    "mse_torch",
    "rmse_torch",
    "psnr_torch",
    "nsam_torch",
    "dssim_torch",
]


def _validate_and_prepare_input(
    img1: torch.Tensor, img2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Validates and prepares input tensors, returning both in (B, C, H, W) format.

    Parameters
    ----------
    img1 : torch.Tensor
        First input tensor.
    img2 : torch.Tensor
        Second input tensor.

    Returns
    -------
    tuple of torch.Tensor
        Tuple of prepared tensors (img1, img2).

    Raises
    ------
    TypeError
        If inputs are not torch.Tensor.
    ValueError
        If input shapes do not match or dimensions are not 3 or 4.
    """
    if not isinstance(img1, torch.Tensor) or not isinstance(img2, torch.Tensor):
        raise TypeError(
            f"Inputs must be torch.Tensor, got {type(img1)} and {type(img2)}"
        )
    if img1.shape != img2.shape:
        raise ValueError(f"Input shapes must match, got {img1.shape} and {img2.shape}")
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    if img1.dim() != 4:
        raise ValueError(f"Inputs must be 3D or 4D, got {img1.dim()} dims")
    if img1.numel() == 0 or img2.numel() == 0:
        raise ValueError("Input tensors must not be empty")

    # Check range [0, 1]

    img1_min, img1_max = img1.min().item(), img1.max().item()
    img2_min, img2_max = img2.min().item(), img2.max().item()
    assert (
        img1_min >= 0 and img1_max <= 1
    ), f"img1 values must be in range [0, 1], got min={img1_min:.6f}, max={img1_max:.6f}"

    assert (
        img2_min >= 0 and img2_max <= 1
    ), f"img2 values must be in range [0, 1], got min={img2_min:.6f}, max={img2_max:.6f}"
    return img1.to(torch.float), img2.to(torch.float)


def _apply_reduction(
    tensor: torch.Tensor, reduction: Literal["mean", "sum", "none"]
) -> torch.Tensor:
    """
    Applies reduction operation to a tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.
    reduction : {'mean', 'sum', 'none'}
        Reduction method.

    Returns
    -------
    torch.Tensor
        Reduced tensor.

    Raises
    ------
    ValueError
        If reduction type is not recognized.
    """
    if reduction == "mean":
        return tensor.mean()
    if reduction == "sum":
        return tensor.sum()
    if reduction == "none":
        return tensor
    raise ValueError(f"Unknown reduction: {reduction}")


def _create_gaussian_kernel_torch(
    kernel_size: int,
    sigma: float,
    channels: int = 3,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Creates a 2D Gaussian kernel for convolution.

    Parameters
    ----------
    kernel_size : int
        Size of the Gaussian kernel (height and width).
    sigma : float
        Standard deviation of the Gaussian.
    channels : int, optional
        Number of channels to expand the kernel for (default is 3).
    device : torch.device or None, optional
        Device on which to create the kernel (default is None).

    Returns
    -------
    torch.Tensor
        Gaussian kernel tensor of shape (channels, 1, kernel_size, kernel_size).

    Raises
    ------
    ValueError
        If kernel_size or sigma is not positive.
    """
    if kernel_size <= 0 or sigma <= 0:
        raise ValueError("kernel_size and sigma must be positive")
    start = (1 - kernel_size) / 2
    end = (1 + kernel_size) / 2
    kernel_1d = torch.arange(start, end, step=1, dtype=torch.float)
    kernel_1d = torch.exp(-torch.pow(kernel_1d / sigma, 2) / 2)
    kernel_1d = (kernel_1d / kernel_1d.sum()).unsqueeze(dim=0)

    kernel_2d = torch.matmul(kernel_1d.t(), kernel_1d)
    kernel_2d = kernel_2d.expand(channels, 1, kernel_size, kernel_size).contiguous()
    return kernel_2d.to(device)


def ssim_map_torch(
    x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor, kernel_size: int
) -> torch.Tensor:
    """
    Computes the SSIM map between two images.

    Parameters
    ----------
    x : torch.Tensor
        First image tensor of shape (B, C, H, W).
    y : torch.Tensor
        Second image tensor of shape (B, C, H, W).
    kernel : torch.Tensor
        Gaussian kernel tensor of shape (C, 1, kernel_size, kernel_size).
    kernel_size : int
        Size of the Gaussian kernel.

    Returns
    -------
    torch.Tensor
        SSIM map tensor of shape (B, C, H, W).
    """
    pad = kernel_size // 2
    # x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    # y = F.pad(y, (pad, pad, pad, pad), mode='reflect')
    ux = F.conv2d(x, kernel, padding=kernel_size // 2, groups=x.shape[1])
    uy = F.conv2d(y, kernel, padding=kernel_size // 2, groups=y.shape[1])
    uxx = F.conv2d(x * x, kernel, padding=kernel_size // 2, groups=x.shape[1])
    uyy = F.conv2d(y * y, kernel, padding=kernel_size // 2, groups=y.shape[1])
    uxy = F.conv2d(x * y, kernel, padding=kernel_size // 2, groups=x.shape[1])
    vx = torch.clamp(uxx - ux * ux, min=0.0)
    vy = torch.clamp(uyy - uy * uy, min=0.0)
    vxy = uxy - ux * uy
    c1 = 0.01**2
    c2 = 0.03**2
    eps = torch.finfo(vx.dtype).eps
    numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
    denominator = (ux**2 + uy**2 + c1) * (vx + vy + c2) + eps
    ssim = numerator / denominator
    return ssim


def ssim_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_size: int = 11,
    sigma: float = 1.5,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    Computes the SSIM (Structural Similarity Index) between two images.

    Parameters
    ----------
    x : torch.Tensor
        First image tensor of shape (B, C, H, W) or (C, H, W).
    y : torch.Tensor
        Second image tensor of shape (B, C, H, W) or (C, H, W).
    kernel_size : int, optional
        Size of the Gaussian kernel (default is 11).
    sigma : float, optional
        Standard deviation of the Gaussian kernel (default is 1.5).
    reduction : {'mean', 'sum', 'none'}, optional
        Reduction method: 'mean' or 'sum' returns a scalar, 'none' returns a map (default is 'mean').

    Returns
    -------
    torch.Tensor
        SSIM scalar or map, depending on reduction.
    """
    x, y = _validate_and_prepare_input(x, y)
    # print(x.shape)
    channels = x.shape[1]
    device = x.device
    kernel = _create_gaussian_kernel_torch(kernel_size, sigma, channels, device=device)
    ssim_map_val = ssim_map_torch(x, y, kernel, kernel_size)
    return _apply_reduction(ssim_map_val, reduction)


def uqi_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    Computes the Universal Quality Index (UQI) map between two images.

    Parameters
    ----------
    x : torch.Tensor
        First image tensor of shape (B, C, H, W) or (C, H, W).
    y : torch.Tensor
        Second image tensor of shape (B, C, H, W) or (C, H, W).
    window_size : int, optional
        Size of the Gaussian window (default is 11).
    reduction : {'mean', 'sum', 'none'}, optional
        Reduction method: 'mean' or 'sum' returns a scalar, 'none' returns a map (default is 'mean').

    Returns
    -------
    torch.Tensor
        UQI scalar or map, depending on reduction.
    """
    x, y = _validate_and_prepare_input(x, y)
    _, channels, _, _ = x.shape
    kernel = _create_gaussian_kernel_torch(
        window_size, sigma=1.5, channels=channels, device=x.device
    )
    pad = (window_size - 1) // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    y = F.pad(y, (pad, pad, pad, pad), mode="reflect")
    mu_x = F.conv2d(x, kernel, groups=channels)
    mu_y = F.conv2d(y, kernel, groups=channels)
    mu_x_sq = F.conv2d(x * x, kernel, groups=channels)
    mu_y_sq = F.conv2d(y * y, kernel, groups=channels)
    mu_xy = F.conv2d(x * y, kernel, groups=channels)
    sigma_x_sq = torch.clamp(mu_x_sq - mu_x.pow(2), min=0.0)
    sigma_y_sq = torch.clamp(mu_y_sq - mu_y.pow(2), min=0.0)
    sigma_xy = mu_xy - mu_x * mu_y
    numerator = 4 * mu_x * mu_y * sigma_xy
    denominator = (mu_x.pow(2) + mu_y.pow(2)) * (sigma_x_sq + sigma_y_sq)
    eps = torch.finfo((sigma_x_sq + sigma_y_sq).dtype).eps
    uqi_map = numerator / (denominator + eps)
    uqi_map = uqi_map[..., pad:-pad, pad:-pad]
    return _apply_reduction(uqi_map, reduction)


def sam_torch(
    x: torch.Tensor, y: torch.Tensor, reduction: Literal["mean", "sum", "none"] = "mean"
) -> torch.Tensor:
    """
    Computes the Spectral Angle Mapper (SAM) between two tensors.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, C, H, W) or (C, H, W).
    y : torch.Tensor
        Target tensor of same shape as input.
    reduction : {'mean', 'sum', 'none'}, optional
        Reduction method: 'mean' or 'sum' returns a scalar, 'none' returns a map (default is 'mean').

    Returns
    -------
    torch.Tensor
        SAM scalar or map, depending on reduction.
    """
    x, y = _validate_and_prepare_input(x, y)
    if torch.equal(x, y):
        sam_score = torch.zeros_like(x[:, 0, :, :])
    else:
        dot = (x * y).sum(dim=1)
        norm_x = x.norm(dim=1)
        norm_y = y.norm(dim=1)
        denominator = norm_x * norm_y + 1e-12
        sam_score = torch.clamp(dot / denominator, -1.0, 1.0).acos()
    return _apply_reduction(sam_score, reduction)


def mse_torch(x: torch.Tensor, y: torch.Tensor) -> Tensor:
    """
    Computes the Mean Squared Error (MSE) between two tensors.

    Parameters
    ----------
    x : torch.Tensor
        First input tensor.
    y : torch.Tensor
        Second input tensor.

    Returns
    -------
    torch.Tensor
        MSE value.
    """
    x, y = _validate_and_prepare_input(x, y)
    diff = x - y
    mse_map = torch.mean(diff**2)
    return mse_map


def rmse_torch(x: torch.Tensor, y: torch.Tensor) -> Tensor:
    """
    Computes the Root Mean Squared Error (RMSE) between two tensors.

    Parameters
    ----------
    x : torch.Tensor
        First input tensor.
    y : torch.Tensor
        Second input tensor.

    Returns
    -------
    torch.Tensor
        RMSE value.
    """
    x, y = _validate_and_prepare_input(x, y)
    mse_val = mse_torch(x, y)
    rmse_val = torch.sqrt(mse_val)
    return rmse_val


def inverse_rmse_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the inverse Root Mean Squared Error (1 - RMSE) between two tensors.

    Parameters
    ----------
    x : torch.Tensor
        First input tensor.
    y : torch.Tensor
        Second input tensor.
    reduction : {'mean', 'sum', 'none'}, optional
        Reduction method.

    Returns
    -------
    torch.Tensor
        Inverse RMSE value according to reduction.
    """
    rmse = rmse_torch(x, y)
    return 1 - rmse


def psnr_torch(
    x: torch.Tensor, y: torch.Tensor, reduction: Literal["mean", "sum", "none"] = "mean"
) -> torch.Tensor:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between two tensors.

    Parameters
    ----------
    x : torch.Tensor
        First input tensor.
    y : torch.Tensor
        Second input tensor.
    reduction : {'mean', 'sum', 'none'}, optional
        Reduction method: 'mean' or 'sum' returns a scalar, 'none' returns a map (default is 'mean').

    Returns
    -------
    torch.Tensor
        PSNR scalar or map, depending on reduction.

    Raises
    ------
    ZeroDivisionError
        If RMSE is zero (identical images).
    """
    x, y = _validate_and_prepare_input(x, y)
    rmse = rmse_torch(x, y)
    max_value = 1.0
    if torch.any(rmse == 0):
        raise ZeroDivisionError("RMSE is zero, PSNR is infinite (identical images).")
    psnr_val = 20 * torch.log10(max_value / rmse)
    return psnr_val


def nsam_torch(
    x: torch.Tensor, y: torch.Tensor, reduction: Literal["mean", "sum", "none"] = "mean"
) -> torch.Tensor:
    """
    Computes the Normalized Spectral Angle Mapper (NSAM) between two tensors.

    Parameters
    ----------
    x : torch.Tensor
        First input tensor.
    y : torch.Tensor
        Second input tensor.
    reduction : {'mean', 'sum', 'none'}, optional
        Reduction method: 'mean' or 'sum' returns a scalar, 'none' returns a map (default is 'mean').

    Returns
    -------
    torch.Tensor
        NSAM scalar or map, depending on reduction.
    """
    sam = sam_torch(x, y, reduction="none")
    nsam = sam / torch.pi
    return _apply_reduction(nsam, reduction)


def dssim_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_size: int = 11,
    sigma: float = 1.5,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    Computes the DSSIM (Structural Dissimilarity Index) between two images.

    DSSIM is defined as (1 - SSIM) / 2.

    Parameters
    ----------
    x : torch.Tensor
        First image tensor of shape (B, C, H, W) or (C, H, W).
    y : torch.Tensor
        Second image tensor of shape (B, C, H, W) or (C, H, W).
    kernel_size : int, optional
        Size of the Gaussian kernel (default is 11).
    sigma : float, optional
        Standard deviation of the Gaussian kernel (default is 1.5).
    reduction : {'mean', 'sum', 'none'}, optional
        Reduction method: 'mean' or 'sum' returns a scalar, 'none' returns a map (default is 'mean').

    Returns
    -------
    torch.Tensor
        DSSIM scalar or map, depending on reduction.
    """
    ssim_map = ssim_torch(x, y, kernel_size=kernel_size, sigma=sigma, reduction="none")
    dssim_map = (1 - ssim_map) / 2
    return _apply_reduction(dssim_map, reduction)


if __name__ == "__main__":
    # Select device: use CUDA if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Example: create two random batches of images (B, C, H, W)
    B, C, H, W = 4, 182, 256, 256  # Batch of 4 RGB images
    img1 = torch.rand(B, C, H, W, device=device)
    img2 = torch.rand(B, C, H, W, device=device)

    # SSIM
    ssim_map = ssim_torch(img1, img2, reduction="none")
    print(f"SSIM map shape: {ssim_map.shape}")
    ssim_mean = ssim_torch(img1, img2, reduction="mean")
    print(f"SSIM (mean over batch): {ssim_mean.item():.4f}")

    # UQI
    uqi_map = uqi_torch(img1, img2, reduction="none")
    print(f"UQI map shape: {uqi_map.shape}")
    uqi_mean = uqi_torch(img1, img2, reduction="mean")
    print(f"UQI (mean over batch): {uqi_mean.item():.4f}")

    # SAM
    sam_map = sam_torch(img1, img2, reduction="none")
    print(f"SAM map shape: {sam_map.shape}")
    sam_mean = sam_torch(img1, img2, reduction="mean")
    print(f"SAM (mean over batch): {sam_mean.item():.4f}")

    # MSE
    mse_map = mse_torch(img1, img2)
    print(f"MSE map shape: {mse_map.shape}")
    mse_mean = mse_torch(img1, img2)
    print(f"MSE (mean over batch): {mse_mean.item():.6f}")

    # RMSE
    rmse_map = rmse_torch(img1, img2)
    print(f"RMSE map shape: {rmse_map.shape}")
    rmse_mean = rmse_torch(img1, img2)
    print(f"RMSE (mean over batch): {rmse_mean.item():.6f}")

    # PSNR
    try:
        psnr_map = psnr_torch(img1, img2, reduction="none")
        print(f"PSNR map shape: {psnr_map.shape}")
        psnr_mean = psnr_torch(img1, img2, reduction="mean")
        print(f"PSNR (mean over batch): {psnr_mean.item():.2f} dB")
    except ZeroDivisionError:
        print("PSNR: Infinite (identical images)")

    # NSAM
    nsam_map = nsam_torch(img1, img2, reduction="none")
    print(f"NSAM map shape: {nsam_map.shape}")
    nsam_mean = nsam_torch(img1, img2, reduction="mean")
    print(f"NSAM (mean over batch): {nsam_mean.item():.4f}")

    # DSSIM
    dssim_map = dssim_torch(img1, img2, reduction="none")
    print(f"DSSIM map shape: {dssim_map.shape}")
    dssim_mean = dssim_torch(img1, img2, reduction="mean")
    print(f"DSSIM (mean over batch): {dssim_mean.item():.4f}")
