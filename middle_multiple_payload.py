import numpy as np


def middle_multiple_payload(A):
    em_cap = 0

    # Reshape the matrix row-wise vector (M = reshape(M',[],1) in MATLAB)
    M = A.T.flatten()

    # Loop over diagonal offsets (from 4 to -4)
    for i in range(4, -5, -1):
        r = np.diag(A, i)  # Get diagonal with offset i
        r = np.flip(r)  # Reverse the order of the diagonal

        # Find indices of non-zero elements
        ind = np.nonzero(r)[0]

        # Count the number of ceaseless zeros at the start of the reversed diagonal
        if len(ind) == 0:
            z = len(r)
        else:
            z = ind[0]  # Number of leading zeros in the reversed diagonal

        # Update embedding capacity if zeros exceed or equal half the size of the diagonal
        if z >= len(r) // 2:
            em_cap += 2

    return em_cap
