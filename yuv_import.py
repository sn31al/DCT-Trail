import numpy as np


def yuv_import(filename, dims, numfrm, startfrm=0, yuvformat='YUV420_8'):
    """
    Imports YUV frames from a file.

    Args:
        filename (str): Path to the YUV file.
        dims (tuple): Dimensions of the frame (width, height).
        numfrm (int): Number of frames to read.
        startfrm (int): Frame number to start reading from (0-based index).
        yuvformat (str): YUV format. Supported formats are 'YUV420_8' (default),
                         'YUV420_16', and 'YUV444_8'.

    Returns:
        Y (list): List of Y components for each frame.
        U (list): List of U components for each frame.
        V (list): List of V components for each frame.
    """

    # Define bit depth and sampling based on format
    inprec = np.uint8  # Default is 8-bit
    sampl = 420  # Default is YUV 4:2:0
    if yuvformat == 'YUV420_16':
        inprec = np.uint16  # 16-bit precision for YUV420_16
    elif yuvformat == 'YUV444_8':
        sampl = 444  # YUV 4:4:4 format

    # Set U and V dimensions based on sampling
    if sampl == 420:
        dimsUV = (dims[0] // 2, dims[1] // 2)
    else:
        dimsUV = dims

    # Frame element size
    Yd_size = dims[0] * dims[1]
    UVd_size = dimsUV[0] * dimsUV[1]
    frelem = Yd_size + 2 * UVd_size

    Y = []
    U = []
    V = []

    # Open the YUV file
    try:
        with open(filename, 'rb') as f:
            # Seek to the start frame
            f.seek(startfrm * frelem)

            # Read the specified number of frames
            for _ in range(numfrm):
                # Read Y component
                Yd = np.frombuffer(f.read(Yd_size), dtype=inprec).reshape((dims[1], dims[0]))
                Y.append(Yd.T)  # Transpose to match the original format

                # Read U component
                UVd_U = np.frombuffer(f.read(UVd_size), dtype=inprec).reshape((dimsUV[1], dimsUV[0]))
                U.append(UVd_U.T)

                # Read V component
                UVd_V = np.frombuffer(f.read(UVd_size), dtype=inprec).reshape((dimsUV[1], dimsUV[0]))
                V.append(UVd_V.T)


    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} does not exist!")

    return Y, U, V

# Example usage:
# Y, U, V = yuv_import('video.yuv', (352, 288), 2, 0, 'YUV420_8')
