# -*- coding: utf-8 -*-
'''
Homework 4.1 code - Spectrogram, scalogram and their ridges
Author: LUO Chensheng
Time: 17 April 2025

Warning: This code utilize scipy.signal.ShortTimeFFT class, as the old scipy.signal.spectrogram is considered as legacy by official scipy documentation.
This class is only available in scipy version 1.12.0 or later.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
import scienceplots
plt.style.use('science')

## Generate the signal
N = 512
t = np.linspace(0, 1, N)
fs = 1/(t[1] - t[0])
alpha1, alpha2 = 40,15
beta1, beta2 = 0.69, 0.7  # To modify to   0.68, 0.72


x = np.cos(alpha1/(beta1 - t)) + np.cos(alpha2/(beta2 - t))

# x = np.zeros_like(t)
# for i in range(N):
#     if t[i] < beta1:
#         x[i] = np.cos(alpha1/(beta1 - t[i])) + np.cos(alpha2/(beta2 - t[i]))
#     elif t[i] >= beta1 and t[i] < beta2:
#         x[i] = np.cos(alpha2/(beta2 - t[i]))
#     else:
#         x[i] = 0

def freq1_theo(t):
    """Frequency of first component (before beta1)"""
    return alpha1/(2*np.pi*(beta1 - t)**2)
    # f = np.zeros_like(t)
    # for i in range(len(t)):
    #     if t[i] < beta1:
    #         f[i] = alpha1/(2*np.pi*(beta1 - t[i])**2)
    # return f

def freq2_theo(t):
    """Frequency of second component (between beta1 and beta2)"""
    return alpha2/(2*np.pi*(beta2 - t)**2)
    # f = np.zeros_like(t)
    # for i in range(len(t)):
    #     if t[i] < beta2:
    #         f[i] = alpha2/(2*np.pi*(beta2 - t[i])**2)
    # return f


## Plot the signal
fig,ax=plt.subplots(figsize=(6, 3))
ax.plot(t,x)
ax.axvline(beta1, color='r', linestyle='--', alpha=0.5, label=rf'$\beta_1 = {beta1}$')
ax.axvline(beta2, color='g', linestyle='--', alpha=0.5, label=rf'$\beta_2 = {beta1}$')
ax.set_title(rf'Signal $x(t) = \cos(\frac{{{alpha1}}}{{{beta1}-t}}) + \cos(\frac{{{alpha2}}}{{{beta2}-t}})$')
ax.set_xlabel(r'$t$ [s]')
ax.set_ylabel(r'$x$')
ax.set_xlim(0,1)
ax.set_ylim(-2,2)
fig.savefig('./result/signal.png',dpi=600)
plt.close()

## Spectrogram with Gaussian Window
# Prepare
window_len = 64
overlap = 62
nfft = 64
sigma = window_len/7
win = signal.windows.gaussian(window_len, sigma, sym=False)

# Compute spectrogram
stft = signal.ShortTimeFFT(win, hop=window_len - overlap, fs=fs, mfft=nfft, scale_to='magnitude', phase_shift=None)
Sx = stft.stft(x)  
Psx = np.abs(Sx)**2
Psx_db = 10 * np.log10(Psx + 1e-12)
t_spec = stft.t(N)
f = stft.f

# Plot spectrogram
fig,ax= plt.subplots(figsize=(6, 3))
im = ax.pcolormesh(t_spec, f, Psx_db, shading='gouraud', cmap='gray_r' ,vmax=np.max(Psx_db), vmin=np.max(Psx_db)-20)
fig.colorbar(im,label=r'$10 \log_{10} [P_sx(f,t)]$')
ax.set_ylabel(r'$f$ [Hz]')
ax.set_xlabel(r'$t$ [s]')
plt.title(rf'Spectrogram of $x(t) = \cos(\frac{{{alpha1}}}{{{beta1}-t}}) + \cos(\frac{{{alpha2}}}{{{beta2}-t}})$')
ax.plot(t,freq1_theo(t), 'r--', label=rf'${alpha1}/[2\pi({beta1}-t)^2]$')
ax.plot(t,freq2_theo(t), 'g--', label=rf'${alpha2}/[2\pi({beta2}-t)^2]$')
ax.legend()
fig.tight_layout()
ax.set_ylim(0,np.max(f))
fig.savefig('./result/spectrogram.png',dpi=600)
plt.close()
print("Spectrogram saved as spectrogram.png")

# Ridge Detection and plot
fig,ax= plt.subplots(figsize=(6, 3))
abs_min_height = 0.03 * np.max(Psx)
for i in range(Psx.shape[1]):
    peaks, properties = signal.find_peaks(Psx[:, i], height=abs_min_height)
    ax.scatter(t_spec[i]*np.ones_like(f[peaks]), f[peaks], color='k', s=1)
plt.title(rf'Spectrogram ridge of $x(t) = \cos(\frac{{{alpha1}}}{{{beta1}-t}}) + \cos(\frac{{{alpha2}}}{{{beta2}-t}})$')
ax.set_ylabel(r'$f$ [Hz]')
ax.set_xlabel(r'$t$ [s]')
ax.plot(t,freq1_theo(t), 'r--', label=rf'${alpha1}/[2\pi({beta1}-t)^2]$')
ax.plot(t,freq2_theo(t), 'g--', label=rf'${alpha2}/[2\pi({beta2}-t)^2]$')
ax.legend()
fig.tight_layout()
ax.set_ylim(0,np.max(f))
fig.savefig('./result/spectrogram_ridge.png',dpi=600)
plt.close()
print("Ridge points saved as spectrogram_ridge.png")

## Scalogram
# # Prepare
scales = 256/np.linspace(1, 128, 100)

# Compute scalogram
Wx, f = pywt.cwt(x, scales, 'cmor1.5-1.0', sampling_period=1/fs)
Pwx = np.abs(Wx)**2
Pwx_db = 10 * np.log10(Pwx + 1e-12)

# Plot scalogram
fig,ax= plt.subplots(figsize=(6, 3))
im = ax.pcolormesh(t, f, Pwx_db, shading='gouraud', cmap='gray_r' ,vmax=np.max(Pwx_db), vmin=np.max(Pwx_db)-20)
fig.colorbar(im,label=r'$10 \log_{10} [P_wx(f,t)]$')
ax.set_ylabel(r'$f$ [Hz]')
ax.set_xlabel(r'$t$ [s]')
plt.title(rf'Scalogram of $x(t) = \cos(\frac{{{alpha1}}}{{{beta1}-t}}) + \cos(\frac{{{alpha2}}}{{{beta2}-t}})$')
ax.plot(t,freq1_theo(t), 'r--', label=rf'${alpha1}/[2\pi({beta1}-t)^2]$')
ax.plot(t,freq2_theo(t), 'g--', label=rf'${alpha2}/[2\pi({beta2}-t)^2]$')
ax.legend()
fig.tight_layout()
ax.set_ylim(0,np.max(f))
fig.savefig('./result/scalogram.png',dpi=600)
plt.close()
print("Scalogram saved as scalogram.png")

# Ridge Detection and plot
fig,ax= plt.subplots(figsize=(6, 3))
abs_min_height = 0.07 * np.max(Pwx)
for i in range(Pwx.shape[1]):
    peaks, properties = signal.find_peaks(Pwx[:, i], height=abs_min_height)
    ax.scatter(t[i]*np.ones_like(f[peaks]), f[peaks], color='k', s=1)
plt.title(rf'Scalogram ridge of $x(t) = \cos(\frac{{{alpha1}}}{{{beta1}-t}}) + \cos(\frac{{{alpha2}}}{{{beta2}-t}})$')
ax.set_ylabel(r'$f$ [Hz]')
ax.set_xlabel(r'$t$ [s]')
ax.plot(t,freq1_theo(t), 'r--', label=rf'${alpha1}/[2\pi({beta1}-t)^2]$')
ax.plot(t,freq2_theo(t), 'g--', label=rf'${alpha2}/[2\pi({beta2}-t)^2]$')
ax.legend()
fig.tight_layout()
ax.set_ylim(0,np.max(f))
fig.savefig('./result/scalogram_ridge.png',dpi=600)
plt.close()
print("Ridge points saved as scalogram_ridge.png")
