import numpy as np


def yuv_export(Y, U, V, filename, numfrm, mode='w'):
    """
    Exports YUV sequence to a file.

    Parameters:
    Y, U, V : list of numpy arrays
        Cell arrays of Y, U, and V components (one per frame).
    filename : str
        Name of the file where the YUV sequence will be saved.
    numfrm : int
        Number of frames to write.
    mode : str, optional
        File write mode: 'a' for append or 'w' for write (default is 'w').

    Example:
    yuv_export(Y, U, V, 'seq_test.yuv', 2)
    """
    if mode not in ['a', 'w']:
        raise ValueError("Mode must be either 'a' (append) or 'w' (write)")

    # Open the file in the appropriate mode ('ab' for binary append, 'wb' for binary write)
    with open(filename, mode + 'b') as f:
        for i in range(numfrm):
            # Write Y component
            Yd = Y[i].T  # Transpose back if needed
            f.write(Yd.astype(np.uint8).tobytes())

            # Write U component
            UVd = U[i].T  # Transpose back if needed
            f.write(UVd.astype(np.uint8).tobytes())

            # Write V component
            UVd = V[i].T  # Transpose back if needed
            f.write(UVd.astype(np.uint8).tobytes())
