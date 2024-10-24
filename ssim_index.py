import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d


def ssim_index(img1, img2, K=[0.01, 0.03], window=None, L=255):
    """
    Compute the Structural Similarity (SSIM) index between two images.

    Args:
        img1 (ndarray): First image.
        img2 (ndarray): Second image.
        K (list, optional): Constants in the SSIM formula (default: [0.01, 0.03]).
        window (ndarray, optional): Local window for statistics (default is Gaussian).
        L (int, optional): Dynamic range of the pixel values (default: 255).

    Returns:
        mssim (float): Mean SSIM index value between the two images.
        ssim_map (ndarray): SSIM index map of the test image.
    """

    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")

    M, N = img1.shape

    # Set default window if not provided
    if window is None:
        window = gaussian_filter(np.ones((11, 11)), 1.5)

    window = window / np.sum(window)

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2

    # Apply the window using convolution
    mu1 = convolve2d(img1, window, mode='valid')
    mu2 = convolve2d(img2, window, mode='valid')

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = convolve2d(img1 ** 2, window, mode='valid') - mu1_sq
    sigma2_sq = convolve2d(img2 ** 2, window, mode='valid') - mu2_sq
    sigma12 = convolve2d(img1 * img2, window, mode='valid') - mu1_mu2

    # Calculate SSIM map
    if C1 > 0 and C2 > 0:
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    else:
        numerator1 = 2 * mu1_mu2 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2

        ssim_map = np.ones(mu1.shape)
        valid = (denominator1 * denominator2 > 0)
        ssim_map[valid] = (numerator1[valid] * numerator2[valid]) / (denominator1[valid] * denominator2[valid])

        ssim_map[(denominator1 != 0) & (denominator2 == 0)] = numerator1[(denominator1 != 0) & (denominator2 == 0)] / \
                                                              denominator1[(denominator1 != 0) & (denominator2 == 0)]

    mssim = np.mean(ssim_map)

    return mssim, ssim_map

# Example usage:
# img1 and img2 are assumed to be numpy arrays with pixel values in the range [0, 255]
# mssim, ssim_map = ssim_index(img1, img2)
