"""
SSIM & PSNR Utility Module
===========================

This module provides implementations of:

- Structural Similarity Index (SSIM)
- Peak Signal-to-Noise Ratio (PSNR)

Typical use cases:
    - Image Restoration (Dehazing, Denoising)
    - Super-Resolution
    - Image-to-Image Translation
    - GAN Evaluation

Expected tensor format:
    Shape: (B, C, H, W)
    Value range: [0, 1]
"""

from math import exp
import math
import numpy as np
import torch
import torch.nn.functional as F


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """
    Creates a normalized 1D Gaussian kernel.

    Args:
        window_size (int): Size of the Gaussian window.
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: 1D tensor of shape (window_size,)
                      normalized to sum to 1.
    """
    gauss = torch.tensor(
        [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
         for x in range(window_size)]
    )

    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> torch.Tensor:
    """
    Creates a 2D Gaussian window expanded for grouped convolution.

    The resulting tensor can be used in F.conv2d with:
        groups=channel

    Args:
        window_size (int): Kernel size (typically 11).
        channel (int): Number of image channels.

    Returns:
        torch.Tensor:
            Shape: (channel, 1, window_size, window_size)
    """
    _1D_window = gaussian(window_size, sigma=1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float()
    _2D_window = _2D_window.unsqueeze(0).unsqueeze(0)

    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()

    return window


def _ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
    size_average: bool = True
) -> torch.Tensor:
    """
    Internal SSIM computation.

    Computes local statistics (mean, variance, covariance)
    using Gaussian-weighted convolution.

    Args:
        img1 (Tensor): First image (B, C, H, W)
        img2 (Tensor): Second image (B, C, H, W)
        window (Tensor): Gaussian kernel
        window_size (int): Kernel size
        channel (int): Number of channels
        size_average (bool): If True returns scalar mean,
                             otherwise returns batch-wise values.

    Returns:
        torch.Tensor: SSIM score
    """

    # Local means
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Local variances
    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel)
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel)
        - mu2_sq
    )

    # Local covariance
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    # Stability constants (standard values)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # SSIM formula
    ssim_map = (
        (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ) / (
        (mu1_sq + mu2_sq + C1) *
        (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        # Return per-image SSIM
        return ssim_map.mean(dim=(1, 2, 3))


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True
) -> torch.Tensor:
    """
    Computes the Structural Similarity Index (SSIM).

    SSIM measures perceptual similarity between two images
    based on luminance, contrast, and structure.

    Args:
        img1 (Tensor): Predicted image (B, C, H, W)
        img2 (Tensor): Ground-truth image (B, C, H, W)
        window_size (int, optional): Gaussian kernel size. Default: 11.
        size_average (bool, optional):
            If True returns scalar mean.
            If False returns per-image SSIM values.

    Returns:
        torch.Tensor:
            SSIM score in range [-1, 1]
            1 = perfect similarity

    Notes:
        - Input images must be in range [0, 1].
        - Images must have identical shape.
        - Function is differentiable.
    """

    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)

    _, channel, _, _ = img1.size()

    window = create_window(window_size, channel)
    window = window.to(img1.device).type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR).

    PSNR is based on Mean Squared Error (MSE) and is commonly
    used for image restoration evaluation.

    Args:
        pred (Tensor): Predicted image (B, C, H, W)
        gt (Tensor): Ground-truth image (B, C, H, W)

    Returns:
        float: PSNR value in decibels (dB)

    Typical interpretation:
        < 20 dB  -> Poor
        ~30 dB   -> Acceptable
        35+ dB   -> Good
        40+ dB   -> Very good

    Notes:
        - Not differentiable (uses NumPy).
        - Should be used only for evaluation.
        - Assumes input range [0, 1].
    """

    pred = pred.clamp(0, 1).cpu().numpy()
    gt = gt.clamp(0, 1).cpu().numpy()

    mse = np.mean((pred - gt) ** 2)

    if mse == 0:
        return 100.0

    rmse = math.sqrt(mse)

    return 20 * math.log10(1.0 / rmse)
