import numpy as np

# Initialize global variable `k`
k = 0


def middle_multiple_embedding(A, S):
    global k

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

        # Position of rij in the matrix aij at ceil(size(r, 1) / 2)
        mid_index = (r.size + 1) // 2 - 1

        if i < 0:
            m = r.size + abs(i) + 1 - mid_index - 1
            l = r.size + 1 - mid_index - 1
        else:
            m = r.size + 1 - mid_index - 1
            l = r.size + abs(i) + 1 - mid_index - 1

        # Check ambiguity
        if z == (mid_index - 1) and A[m, l] != 0:
            if A[m, l] > 0:
                A[m, l] = (A[m, l] / abs(A[m, l])) * (abs(A[m, l]) + 2)
            else:
                A[m, l] = (A[m, l] / abs(A[m, l])) * (abs(A[m, l]) + 1)

        # Embedding bits into the matrix based on the bit pairs
        if z >= mid_index:
            if S[k] == 0 and S[k + 1] == 0:
                A[m, l] = 0
            elif S[k] == 0 and S[k + 1] == 1:
                A[m, l] = 1
            elif S[k] == 1 and S[k + 1] == 0:
                A[m, l] = -1
            elif S[k] == 1 and S[k + 1] == 1:
                A[m, l] = 2

            k += 2  # Move forward by 2 in the sequence

    return A
