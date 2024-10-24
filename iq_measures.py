import numpy as np
from skimage.util import img_as_float  # Corrected import
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import laplace  # Corrected import for laplacian filter
import os


def read_yuv_file(file_path, width, height):
    """
    Reads a YUV 4:2:0 image from a file and returns the Y (luminance) channel as a NumPy array.

    :param file_path: Path to the YUV file
    :param width: Width of the video frame
    :param height: Height of the video frame
    :return: NumPy array containing the Y (luminance) channel
    """
    with open(file_path, 'rb') as file:
        # Read Y component (luminance)
        y_size = width * height
        y = np.frombuffer(file.read(y_size), dtype=np.uint8).reshape((height, width))

    # Convert to float for image quality calculations
    return img_as_float(y)


def iq_measures(A, B, width, height, disp=False):
    """
    Computes image quality metrics between two YUV images.

    :param A: Path to the first YUV file (reference)
    :param B: Path to the second YUV file (comparison)
    :param width: Width of the YUV frame
    :param height: Height of the YUV frame
    :param disp: If True, display the results in the console
    :return: Tuple of image quality metrics
    """
    # Ensure both files exist
    if not os.path.exists(A) or not os.path.exists(B):
        raise FileNotFoundError("One or both YUV files do not exist.")

    # Load YUV files and extract the Y (luminance) channel
    A = read_yuv_file(A, width, height)
    B = read_yuv_file(B, width, height)

    # Image dimensions
    x, y = A.shape
    R = A - B
    Pk = np.sum(A ** 2)
    MSE = np.sum(R ** 2) / (x * y)  # MSE

    if disp:
        print(f"MSE (Mean Squared Error) = {MSE}")

    # PSNR
    if MSE > 0:
        PSNR = 10 * np.log10(1.0 / MSE)
    else:
        PSNR = float('inf')

    if disp:
        print(f"PSNR (Peak Signal / Noise Ratio) = {PSNR} dB")

    # Initialize additional metrics
    AD = SC = NK = MD = LMSE = NAE = PQS = 0

    if disp:
        # AD - Average Difference
        AD = np.sum(R) / (x * y)
        print(f"AD (Average Difference) = {AD}")

        # SC - Structural Content
        Bs = np.sum(B ** 2)
        if Bs == 0:
            SC = float('inf')
        else:
            SC = Pk / Bs
        print(f"SC (Structural Content) = {SC}")

        # NK - Normalized Cross-Correlation
        NK = np.sum(A * B) / Pk
        print(f"NK (Normalized Cross-Correlation) = {NK}")

        # MD - Maximum Difference
        MD = np.max(np.abs(R))
        print(f"MD (Maximum Difference) = {MD}")

        # LMSE - Laplacian Mean Squared Error
        OP = 4 * laplace(A)
        LMSE = np.sum((OP - 4 * laplace(B)) ** 2) / np.sum(OP ** 2)
        print(f"LMSE (Laplacian Mean Squared Error) = {LMSE}")

        # NAE - Normalized Absolute Error
        NAE = np.sum(np.abs(R)) / np.sum(np.abs(A))
        print(f"NAE (Normalized Absolute Error) = {NAE}")

    return MSE, PSNR, AD, SC, NK, MD, LMSE, NAE, PQS


# Example Usage
A = r'D:\Major P\YUV\bus_qcif.YUV'  # Replace with the actual YUV file path
B = r'D:\Major P\YUV\bus_qcif.YUV'  # Replace with the actual YUV file path
width, height = 176, 144  # Example dimensions for QCIF resolution

MSE, PSNR, AD, SC, NK, MD, LMSE, NAE, PQS = iq_measures(A, B, width, height, disp=True)
def mse_psnr(A, B):
    """
    Computes the Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR) between two images.

    :param A: NumPy array representing the reference image (Y channel of YUV)
    :param B: NumPy array representing the comparison image (Y channel of YUV)
    :return: Tuple containing the MSE and PSNR
    """
    # Ensure both images have the same shape
    assert A.shape == B.shape, "Input images must have the same dimensions"

    # Compute Mean Squared Error (MSE)
    mse = np.mean((A - B) ** 2)

    # Compute Peak Signal-to-Noise Ratio (PSNR)
    if mse == 0:
        psnr = float('inf')  # If the images are identical, PSNR is infinite
    else:
        psnr = 10 * np.log10(1.0 / mse)

    return mse, psnr
