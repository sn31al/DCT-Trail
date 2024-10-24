import numpy as np
from scipy.fftpack import dct


def psnrhvsm(img1, img2, wstep=8):
    """
    Calculate PSNR-HVS-M and PSNR-HVS between two images.

    Parameters:
    img1: numpy array - first image
    img2: numpy array - second image
    wstep: int - step size for 8x8 window DCT calculations, default is 8

    Returns:
    p_hvs_m: PSNR-HVS-M value
    p_hvs: PSNR-HVS value
    """

    if img1.shape != img2.shape:
        return float('-inf'), float('-inf')

    LenY, LenX = img1.shape

    # CSF coefficients matrix
    CSFCof = np.array([
        [1.608443, 2.339554, 2.573509, 1.608443, 1.072295, 0.643377, 0.504610, 0.421887],
        [2.144591, 2.144591, 1.838221, 1.354478, 0.989811, 0.443708, 0.428918, 0.467911],
        [1.838221, 1.979622, 1.608443, 1.072295, 0.643377, 0.451493, 0.372972, 0.459555],
        [1.838221, 1.513829, 1.169777, 0.887417, 0.504610, 0.295806, 0.321689, 0.415082],
        [1.429727, 1.169777, 0.695543, 0.459555, 0.378457, 0.236102, 0.249855, 0.334222],
        [1.072295, 0.735288, 0.467911, 0.402111, 0.317717, 0.247453, 0.227744, 0.279729],
        [0.525206, 0.402111, 0.329937, 0.295806, 0.249855, 0.212687, 0.214459, 0.254803],
        [0.357432, 0.279729, 0.270896, 0.262603, 0.229778, 0.257351, 0.249855, 0.259950]
    ])

    # Masking coefficients matrix
    MaskCof = np.array([
        [0.390625, 0.826446, 1.000000, 0.390625, 0.173611, 0.062500, 0.038447, 0.026874],
        [0.694444, 0.694444, 0.510204, 0.277008, 0.147929, 0.029727, 0.027778, 0.033058],
        [0.510204, 0.591716, 0.390625, 0.173611, 0.062500, 0.030779, 0.021004, 0.031888],
        [0.510204, 0.346021, 0.206612, 0.118906, 0.038447, 0.013212, 0.015625, 0.026015],
        [0.308642, 0.206612, 0.073046, 0.031888, 0.021626, 0.008417, 0.009426, 0.016866],
        [0.173611, 0.081633, 0.033058, 0.024414, 0.015242, 0.009246, 0.007831, 0.011815],
        [0.041649, 0.024414, 0.016437, 0.013212, 0.009426, 0.006830, 0.006944, 0.009803],
        [0.019290, 0.011815, 0.011080, 0.010412, 0.007972, 0.010000, 0.009426, 0.010203]
    ])

    S1 = 0
    S2 = 0
    Num = 0

    for Y in range(0, LenY - 7, wstep):
        for X in range(0, LenX - 7, wstep):
            A = img1[Y:Y + 8, X:X + 8]
            B = img2[Y:Y + 8, X:X + 8]

            A_dct = dct(dct(A.T, norm='ortho').T, norm='ortho')
            B_dct = dct(dct(B.T, norm='ortho').T, norm='ortho')

            MaskA = maskeff(A, A_dct, MaskCof)
            MaskB = maskeff(B, B_dct, MaskCof)

            MaskA = max(MaskA, MaskB)

            for k in range(8):
                for l in range(8):
                    u = abs(A_dct[k, l] - B_dct[k, l])
                    S2 += (u * CSFCof[k, l]) ** 2  # PSNR-HVS

                    if k != 0 or l != 0:  # PSNR-HVS-M (See equation 3)
                        if u < MaskA / MaskCof[k, l]:
                            u = 0
                        else:
                            u -= MaskA / MaskCof[k, l]

                    S1 += (u * CSFCof[k, l]) ** 2  # PSNR-HVS-M
                    Num += 1

    if Num != 0:
        S1 /= Num
        S2 /= Num

        p_hvs_m = 10 * np.log10(255 ** 2 / S1) if S1 != 0 else 100000
        p_hvs = 10 * np.log10(255 ** 2 / S2) if S2 != 0 else 100000
    else:
        p_hvs_m, p_hvs = float('-inf'), float('-inf')

    return p_hvs_m, p_hvs


def maskeff(z, zdct, MaskCof):
    """
    Calculate the mask effectiveness value (Enorm) for the given block.

    Parameters:
    z: numpy array - original 8x8 block
    zdct: numpy array - DCT coefficients of the block
    MaskCof: numpy array - masking coefficients

    Returns:
    m: mask effectiveness
    """
    m = 0
    for k in range(8):
        for l in range(8):
            if k != 0 or l != 0:
                m += (zdct[k, l] ** 2) * MaskCof[k, l]

    pop = block_variance(z)
    if pop != 0:
        pop = (block_variance(z[:4, :4]) + block_variance(z[:4, 4:]) +
               block_variance(z[4:, :4]) + block_variance(z[4:, 4:])) / pop

    m = np.sqrt(m * pop) / 32
    return m


def block_variance(block):
    """
    Calculate the variance of the block.

    Parameters:
    block: numpy array - input block

    Returns:
    variance: variance of the block
    """
    return np.var(block) * len(block.flatten())
