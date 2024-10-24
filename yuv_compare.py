import numpy as np
from yuv_import import yuv_import
from seq_frames import seq_frames
from iq_measures import mse_psnr
def yuv_compare(yuvfile1, yuvfile2, dims, frames=None, yuvformat='YUV420_8'):
    """
    Compares two YUV sequences by computing PSNR and MSE for Y, U, and V components.

    Args:
        yuvfile1 (str): First YUV sequence file.
        yuvfile2 (str): Second YUV sequence file.
        dims (tuple): Frame dimensions (width, height).
        frames (int, optional): Number of frames to compare. Defaults to all frames.
        yuvformat (str, optional): YUV format, default is 'YUV420_8'.

    Returns:
        PSNRY, PSNRU, PSNRV (ndarray): PSNR values for Y, U, and V components.
        MSEY, MSEU, MSEV (ndarray): MSE values for Y, U, and V components.
    """

    if frames is None:
        frames = np.inf

    # Get the number of frames in both YUV sequences
    numfrm1 = seq_frames(yuvfile1, dims, yuvformat)
    numfrm2 = seq_frames(yuvfile2, dims, yuvformat)
    numfrm = min([numfrm1, numfrm2, frames])

    # Initialize arrays for PSNR and MSE values
    PSNRY = np.zeros(numfrm)
    PSNRU = np.zeros(numfrm)
    PSNRV = np.zeros(numfrm)
    MSEY = np.zeros(numfrm)
    MSEU = np.zeros(numfrm)
    MSEV = np.zeros(numfrm)

    # Compare the frames
    for i in range(numfrm):
        # Import Y, U, V components from both sequences
        Y1, U1, V1 = yuv_import(yuvfile1, dims, 1, i, yuvformat)
        Y2, U2, V2 = yuv_import(yuvfile2, dims, 1, i, yuvformat)

        # Calculate MSE and PSNR for each component
        MSEY[i], PSNRY[i] = mse_psnr(Y1[0], Y2[0])
        MSEU[i], PSNRU[i] = mse_psnr(U1[0], U2[0])
        MSEV[i], PSNRV[i] = mse_psnr(V1[0], V2[0])

    return PSNRY, PSNRU, PSNRV, MSEY, MSEU, MSEV

# Example usage:
# PSNRY, PSNRU, PSNRV, MSEY, MSEU, MSEV = yuv_compare('compressed.yuv', 'original.yuv', (352, 288))
