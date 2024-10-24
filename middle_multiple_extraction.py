import numpy as np


def middle_multiple_extraction(A):
    E = []

    x = A.shape
    n = x[0] * x[1]
    M = A.copy()

    # Reshape the matrix M row-wise vector
    M = M.T.flatten()

    # Loop over diagonal offsets (from 4 to -4)
    for i in range(4, -5, -1):
        r = np.diag(A, i)  # Get diagonal with offset i
        r = np.flip(r)  # Reverse the order of the diagonal

        # Find indices of non-zero elements
        ind = np.nonzero(r)[0]

        # Count of leading zeros in the reversed diagonal
        if len(ind) == 0:
            z = r.size
        else:
            z = ind[0]  # Number of ceaseless zeros

        # Position of rij in the matrix A at ceil(size(r, 1) / 2)
        mid_index = (r.size + 1) // 2 - 1

        if i < 0:
            m = r.size + abs(i) + 1 - mid_index - 1
            l = r.size + 1 - mid_index - 1
        else:
            m = r.size + 1 - mid_index - 1
            l = r.size + abs(i) + 1 - mid_index - 1

        # Check if value matches one of the embedding values (-1, 0, 1, 2)
        if z >= mid_index - 1 and A[m, l] in [-1, 0, 1, 2]:

            # Extract bits based on the value at A[m, l]
            if A[m, l] == 0:
                A[m, l] = 0
                E.extend([0, 0])
            elif A[m, l] == 1:
                A[m, l] = 0
                E.extend([0, 1])
            elif A[m, l] == -1:
                A[m, l] = 0
                E.extend([1, 0])
            elif A[m, l] == 2:
                A[m, l] = 0
                E.extend([1, 1])

        # Adjust A[m, l] for embedding if needed
        if z == mid_index - 1 and A[m, l] != 0:
            if A[m, l] > 0:
                A[m, l] = (A[m, l] / abs(A[m, l])) * (abs(A[m, l]) - 2)
            else:
                A[m, l] = (A[m, l] / abs(A[m, l])) * (abs(A[m, l]) - 1)

    return A, E
