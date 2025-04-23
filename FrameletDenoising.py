# -*- coding: utf-8 -*-
'''
Homework 5 code - Denoising using framelet transform
Author: LUO Chensheng
Time: 23 April 2025
'''
import numpy as np
import pywt
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

## Generate the signal
N = 512
t = np.linspace(0, 1, N)
fs = 1/(t[1] - t[0])
true_signal = np.cos(2*np.pi*5*t) + np.cos(2*np.pi*10*t) + np.cos(2*np.pi*15*t)
sigma = 0.5
print(sigma)
x = np.random.normal(0, sigma, N)+true_signal  # Add noise

## Plot the signal
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(t, x, label='Noisy signal')
ax.plot(t, true_signal, label='True signal',linewidth=2)
ax.set_title('Noisy signal')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.grid()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
ax.set_xlim(0,1)
ax.set_ylim(np.min(x), np.max(x))
fig.tight_layout()
fig.savefig('./result/signal.png',dpi=600)
plt.close()

## Preparing

# Calculate PSNR
def calculate_psnr(true_signal, denoised_signal):
    mse = np.mean((true_signal - denoised_signal) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(true_signal)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

## Denoising by Framelet
# Framelet decomposition
v2 = np.sqrt(2)
u1 = np.array([1/4,1/2,1/4])
u2 = np.array([-v2/4,0,v2/4])
u3 = np.array([-1/4,1/2,-1/4])

rev_u1 = np.flip(u1)
rev_u2 = np.flip(u2)
rev_u3 = np.flip(u3)

c1 = np.convolve(x, v2*rev_u1, mode='full')[1::2]
c2 = np.convolve(x, v2*rev_u2, mode='full')[1::2]
c3 = np.convolve(x, v2*rev_u3, mode='full')[1::2]

# Filter
T = sigma * np.sqrt(2*np.log(N)/np.log(2))/2
c1[np.abs(c1) < T] = 0
c2[np.abs(c2) < T] = 0
c3[np.abs(c3) < T] = 0

# Reconstruct 
c1_up = np.zeros(2 * len(c1)+1)
c1_up[1::2] = c1
c2_up = np.zeros(2 * len(c2)+1)
c2_up[1::2] = c2
c3_up = np.zeros(2 * len(c3)+1)
c3_up[1::2] = c3
x_de = np.convolve(v2*c1_up, u1, mode='full') + np.convolve(v2*c2_up, u2, mode='full') + np.convolve(v2*c3_up, u3, mode='full')
x_defr = x_de[2:-3]

# Plot
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(t, x_defr, label='Denoised by Framelet')
# ax.plot(t, x_dewv, label='Denoised by Framelet',linewidth=2)
ax.plot(t,true_signal, label='True signal',linewidth=2,linestyle='--')
ax.plot(t, x, label='Noisy signal',linestyle=':')
ax.set_title('Denoised by Framelet')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.grid()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
ax.set_xlim(0,1)
ax.set_ylim(np.min(x), np.max(x))
fig.tight_layout()
fig.savefig('./result/signal_denoised_fr.png',dpi=600)
plt.close()

# PSNR for Framelet
psnr_framelet = calculate_psnr(true_signal, x_defr)
print(f"PSNR (Framelet): {psnr_framelet:.2f} dB")

## Denoising by Wavelet
# Wavelet decomposition
cA, cD = pywt.dwt(x, 'sym2')
cA[np.abs(cA) < T] = 0
cD[np.abs(cD) < T] = 0
c3[np.abs(c3) < T] = 0
x_dewv = pywt.idwt(cA, cD, 'sym2')

# Plot
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(t, x_dewv, label='Denoised by Wavelet')
# ax.plot(t, x_dewv, label='Denoised by Framelet',linewidth=2)
ax.plot(t,true_signal, label='True signal',linewidth=2,linestyle='--')
ax.plot(t, x, label='Noisy signal',linestyle=':')
ax.set_title('Denoised by Wavelet')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.grid()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
ax.set_xlim(0,1)
ax.set_ylim(np.min(x), np.max(x))
fig.tight_layout()
fig.savefig('./result/signal_denoised_wv.png',dpi=600)
plt.close()

# PSNR for Wavelet
psnr_wavelet = calculate_psnr(true_signal, x_dewv)
print(f"PSNR (Wavelet): {psnr_wavelet:.2f} dB")