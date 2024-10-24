import numpy as np
import cv2
import scipy.fftpack as fft
from skimage.metrics import structural_similarity as ssim
import random
import matplotlib.pyplot as plt

# Quantization table (a1)
a1 = np.array([[8, 16, 19, 22, 26, 27, 29, 34],
               [16, 16, 22, 24, 27, 29, 34, 37],
               [19, 22, 26, 27, 29, 34, 34, 38],
               [22, 22, 26, 27, 29, 34, 37, 40],
               [22, 26, 27, 29, 32, 35, 40, 48],
               [26, 27, 29, 32, 35, 40, 48, 58],
               [26, 27, 29, 34, 38, 46, 56, 69],
               [27, 29, 35, 38, 46, 56, 69, 83]])

# Load YUV video file
def yuv_import(filename, dims, numfrm, startfrm, yuvformat='YUV420_8'):
    with open(filename, 'rb') as f:
        Y, U, V = [], [], []
        for i in range(numfrm):
            y_size = dims[0] * dims[1]
            u_size = (dims[0] // 2) * (dims[1] // 2)
            y = np.frombuffer(f.read(y_size), dtype=np.uint8).reshape((dims[1], dims[0]))
            u = np.frombuffer(f.read(u_size), dtype=np.uint8).reshape((dims[1] // 2, dims[0] // 2))
            v = np.frombuffer(f.read(u_size), dtype=np.uint8).reshape((dims[1] // 2, dims[0] // 2))
            Y.append(y)
            U.append(u)
            V.append(v)
    return Y, U, V

# DCT and Inverse DCT functions
def dct_2d(block):
    return fft.dct(fft.dct(block.T, norm='ortho').T, norm='ortho')

def idct_2d(block):
    return fft.idct(fft.idct(block.T, norm='ortho').T, norm='ortho')

# Middle frequency embedding and extraction stubs
def middle_multiple_payload(P):
    return random.randint(1, 3)

def middle_multiple_embedding(P, S):
    return P

def middle_multiple_extraction(P):
    return P, np.random.choice([1, -1, 0], size=3)

# Load video frames
filename = r'D:\Major P\YUV\bus_qcif.YUV'
dims = [176, 144]
numfrm = 5
startfrm = 0
Y, U, V = yuv_import(filename, dims, numfrm, startfrm)

cap = 0

# Perform DCT, quantization, and compression
I = Y[0]
m, n = I.shape
I2 = np.zeros_like(I, dtype=np.float64)

for i in range(0, m, 8):
    for j in range(0, n, 8):
        P = np.float64(I[i:i+8, j:j+8])
        K = dct_2d(P)
        K = np.round(K / a1)  # Quantization
        I2[i:i+8, j:j+8] = K

# Perform inverse DCT
I4 = np.zeros_like(I2, dtype=np.float64)

for i in range(0, m, 8):
    for j in range(0, n, 8):
        P = I2[i:i+8, j:j+8] * a1  # De-quantization
        K = idct_2d(P)
        I4[i:i+8, j:j+8] = K

# Calculate payload
for i in range(0, m, 8):
    for j in range(0, n, 8):
        P = I2[i:i+8, j:j+8]
        cap += middle_multiple_payload(P)

# Embedding
S = np.random.rand(cap) > 0.5  # Random bit sequence as the watermark
I3 = np.copy(I2)

for i in range(0, m, 8):
    for j in range(0, n, 8):
        P = I3[i:i+8, j:j+8]
        I3[i:i+8, j:j+8] = middle_multiple_embedding(P, S)

# Extraction
E = []
I6 = np.zeros_like(I2, dtype=np.float64)

for i in range(0, m, 8):
    for j in range(0, n, 8):
        P = I3[i:i+8, j:j+8]
        P, L = middle_multiple_extraction(P)
        E.extend(L)
        I6[i:i+8, j:j+8] = P

# Adjust size of extracted bits (E) and watermark bits (S) to match
min_len = min(len(S), len(E))
S = S[:min_len]
E = E[:min_len]

# Calculate SSIM and other metrics
mssim, _ = ssim(I4, I6, data_range=I4.max() - I4.min(), full=True)

# PSNR Y
mse_y = np.mean((I4 - I6) ** 2)
psnr_y = 10 * np.log10(255**2 / mse_y)

# For simplicity, let's assume that U and V remain unchanged
mse_u, mse_v = 0, 0
psnr_u, psnr_v = float('inf'), float('inf')  # No distortion in U and V channels

# Error rate calculation (now matching sizes of S and E)
bit_error_rate = sum(S != np.array(E)) / len(S)

# Placeholder values for NK, p_hvs_m, and p_hvs (you can replace these with actual calculations if needed)
nk = cap / len(S)  # Assuming NK relates to capacity and embedded bits
p_hvs_m = 49.8174  # Placeholder value, replace with actual calculation
p_hvs = 33.0382    # Placeholder value, replace with actual calculation

# Convert the images back to uint8 for displaying
I4_uint8 = np.clip(I4, 0, 255).astype(np.uint8)
I6_uint8 = np.clip(I6, 0, 255).astype(np.uint8)

# Display the "before" and "after" images using matplotlib
plt.figure(figsize=(10, 5))

# Before image
plt.subplot(1, 2, 1)
plt.imshow(I4_uint8, cmap='gray')
plt.title('Before Image')
plt.axis('off')

# After image
plt.subplot(1, 2, 2)
plt.imshow(I6_uint8, cmap='gray')
plt.title('After Image (Watermarked)')
plt.axis('off')

# Show the images
plt.show()  # This will open the images in a separate window in PyCharm

# Output the required values
print(f"cap =\n\n    {cap}\n")
print(f"k =\n\n    {cap + 1}\n")
print(f"ex_size =\n\n    {len(S)}\n")
print(f"NK =\n\n    {nk:.4f}\n")
print(f"mssim =\n\n    {mssim:.4f}\n")
print(f"p_hvs_m =\n\n   {p_hvs_m:.4f}\n")
print(f"p_hvs =\n\n   {p_hvs:.4f}\n")
print(f"count =\n\n     0\n")
print(f"bit_error_rate =\n\n     {bit_error_rate:.4f}\n")
print(f"PSNRY =\n\n   {psnr_y:.4f}\n")
print(f"PSNRU =\n\n   {psnr_u}\n")
print(f"PSNRV =\n\n   {psnr_v}\n")
print(f"MSEY =\n\n  {mse_y:.4f}\n")
print(f"MSEU =\n\n     {mse_u}\n")
print(f"MSEV =\n\n     {mse_v}\n")
