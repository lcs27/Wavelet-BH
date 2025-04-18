# -*- coding: utf-8 -*-
'''
Homework 4.2 code - STFT, CWT and TFCWT
Author: LUO Chensheng
Time: 18 April 2025

Warning: This code utilize scipy.signal.ShortTimeFFT class, as the old scipy.signal.spectrogram is considered as legacy by official scipy documentation.
This class is only available in scipy version 1.12.0 or later.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
import scienceplots
plt.style.use('science')

## Signal initialization
N = 1024
t = np.linspace(0,1.4, N)
fs = 1/(t[1] - t[0])
alpha1, alpha2 = 3000, 500
beta1, beta2 =4,3
x = np.cos(alpha1/(beta1-t)) + np.cos(alpha2/(beta2-t))

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

# Plot the signal
fig,ax=plt.subplots(figsize=(6, 3))
ax.plot(t,x)
ax.set_title('Signal')
ax.set_xlabel(r'$t$ [s]')
ax.set_ylabel(r'$x$')
ax.set_xlim(0,1)
ax.set_ylim(-2,2)
fig.savefig('./result/signal.png',dpi=600)
plt.close()

## STFT
window_len = 512
overlap = 480
nfft = 512
sigma = window_len/8
win = signal.windows.gaussian(window_len, sigma, sym=False)
stft = signal.ShortTimeFFT(win, hop=window_len - overlap, fs=fs, mfft=nfft, scale_to='magnitude', phase_shift=None)
Sx = stft.stft(x)
t_spec = stft.t(N)
f1 = stft.f

## CWT
scales = 256/np.linspace(1, 32, 100)
Wx, f2 = pywt.cwt(x, scales, 'morl', sampling_period=1/fs)

## TFCWT
# TFCWT is a combination of STFT and CWT
tfcwt,f3 = pywt.cwt(x, scales, 'cmor1.5-1.0', sampling_period=1/fs)


## Final plot
fig,ax=plt.subplots(1,3,figsize=(8, 3))
im1 = ax[0].pcolormesh(t_spec, f1,10*np.log(np.abs(Sx)), shading='gouraud', cmap='jet')
ax[0].set_title('STFT')
ax[0].set_xlabel(r'$t$ [s]')
ax[0].set_ylabel(r'$f$ [Hz]')
ax[0].plot(t,freq1_theo(t), 'w--', label=rf'${alpha1}/[2\pi({beta1}-t)^2]$')
ax[0].plot(t,freq2_theo(t), 'w-.', label=rf'${alpha2}/[2\pi({beta2}-t)^2]$')
ax[0].set_ylim(0,np.max(f2))
ax[0].set_xlim(0,1.4)
im2 = ax[1].pcolormesh(t,f2,10*np.log(np.abs(Wx)), shading='gouraud', cmap='jet')
ax[1].set_title('CWT')
ax[1].set_xlabel(r'$t$ [s]')
ax[1].set_ylabel(r'$f$ [Hz]')
ax[1].plot(t,freq1_theo(t), 'w--', label=rf'${alpha1}/[2\pi({beta1}-t)^2]$')
ax[1].plot(t,freq2_theo(t), 'w-.', label=rf'${alpha2}/[2\pi({beta2}-t)^2]$')
ax[1].set_ylim(0,np.max(f2))
ax[1].set_xlim(0,1.4)
im3 = ax[2].pcolormesh(t,f3,10*np.log(np.abs(tfcwt)), shading='gouraud', cmap='jet')
ax[2].set_title('TFCWT')
ax[2].set_xlabel(r'$t$ [s]')
ax[2].set_ylabel(r'$f$ [Hz]')
fig.colorbar(im1, ax=ax[0])
fig.colorbar(im2, ax=ax[1])
fig.colorbar(im3, ax=ax[2])
ax[2].plot(t,freq1_theo(t), 'w--', label=rf'${alpha1}/[2\pi({beta1}-t)^2]$')
ax[2].plot(t,freq2_theo(t), 'w-.', label=rf'${alpha2}/[2\pi({beta2}-t)^2]$')
ax[2].set_ylim(0,np.max(f3))
ax[2].set_xlim(0,1.4)
fig.set_tight_layout(True)
fig.savefig('./result/combined_plot.png',dpi=600)
plt.close()
