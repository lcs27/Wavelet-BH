import numpy as np
import pywt

data = np.random.rand(8)  # Example 1D signal
print("Original signal:", data)

# Perform 1D Haar wavelet transform
cA, cD = pywt.dwt(data, 'haar')
print('---- Haar wavelet transform ----')
print("Approximation coefficients (cA):", cA)
print("Detail coefficients (cD):", cD)

# Reconstruct the original signal
reconstructed_signal = pywt.idwt(cA, cD, 'haar')
print("Reconstructed signal:", reconstructed_signal)

# Self-made Haar wavelet transform
c = np.sqrt(2) / 2

rec_lo = np.array([c, c]) # h
rec_hi = np.array([c, -c]) # g

dec_lo = np.flip(rec_lo) # inverse of h
dec_hi = np.flip(rec_hi) # inverse of g

filter_bank = [dec_lo, dec_hi, rec_lo, rec_hi]
w = pywt.Wavelet(name="myHaarWavelet", filter_bank=filter_bank)

cA, cD = pywt.dwt(data,  wavelet=w)
print('---- Self-made Haar wavelet transform ----')
print("Approximation coefficients (cA):", cA)
print("Detail coefficients (cD):", cD)

# Reconstruct the original signal
reconstructed_signal = pywt.idwt(cA, cD, 'haar')
print("Reconstructed signal:", reconstructed_signal)