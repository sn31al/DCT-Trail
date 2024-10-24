import os


def seq_frames(filename, dims, yuvformat='YUV420_8'):
    """
    Returns the number of frames in a YUV sequence file.

    Args:
        filename (str): Path to the YUV sequence file.
        dims (tuple): Dimensions of the frame (width, height).
        yuvformat (str): YUV format (optional, default='YUV420_8').
                         Supported formats are 'YUV444_8' and 'YUV420_8'.
                         Default is 'YUV420_8'.

    Returns:
        int: Number of frames in the YUV file.

    Raises:
        ValueError: If the YUV format is unsupported.
        FileNotFoundError: If the file cannot be opened.
    """
    Ysiz = dims[0] * dims[1]

    if yuvformat == 'YUV420_8':
        UVsiz = Ysiz // 4
        frelem = Ysiz + 2 * UVsiz
    elif yuvformat == 'YUV444_8':
        frelem = 3 * Ysiz
    else:
        raise ValueError(f"Format '{yuvformat}' not supported or unknown!")

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Cannot open file: {filename}")

    # Calculate file size and number of frames
    with open(filename, 'rb') as f:
        f.seek(0, os.SEEK_END)  # Move the cursor to the end of the file
        yuvbytes = f.tell()  # Get the current position (end of the file)

    frames = yuvbytes // frelem  # Calculate number of frames

    return frames

# Example usage:
# frames = seq_frames('football.yuv', (352, 288), 'YUV420_8')
