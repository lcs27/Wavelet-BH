import numpy as np
import pywt
import matplotlib.pyplot as plt
import scienceplots
import h5py 
from scipy import signal
plt.style.use(['science','ieee'])

mm = 4096
f = h5py.File('./data/velgrad.h5', 'r')
theta = np.array(f['theta'][0,0:mm,0:mm]).transpose()
x = np.linspace(0,2*np.pi,mm,endpoint=False)
X,Y = np.meshgrid(x,x)
sampling_period = np.diff(x).mean()

pos = 100
oline = theta[:,pos]

# ## Spectrogram
# Prepare
window_len = 32
nfft = 512
sigma = window_len//4
win = signal.windows.gaussian(window_len, sigma, sym=True)

# Compute spectrogram
stft = signal.ShortTimeFFT(win, hop=sigma//2, fs=1/sampling_period,phase_shift=0)
Sx = stft.stft(oline)/np.sqrt(np.sqrt(np.pi*sigma**2))
Psx = np.abs(Sx)**2
x_spec = stft.t(mm)
freqs = stft.f
# Psx = Psx / np.max(Psx, axis=1, keepdims=True)
# Compute mean only for x_spec between 0 and 2*np.pi
mask = (x_spec >= 0) & (x_spec <= 2 * np.pi)
energy_per_wavenumber = np.mean(Psx[:, mask], axis=1) # Compute energy per wavenumber

print(x_spec)
print(freqs)
print(x_spec.shape,freqs.shape)

# Plot
# First figure: Spectrogram (magnitude)
Psx = Psx[:, mask] # Apply mask to Psx
Psx = Psx / np.max(Psx, axis=1, keepdims=True) # Normalize
fig, ax = plt.subplots(figsize=(4,2.5))
pcm = ax.pcolormesh(x_spec[mask], freqs, Psx,shading='auto')
# ax.set_yscale("log")
ax.set_xlim(0, 2 * np.pi)
ax.set_xlabel(r"$b$")
ax.set_ylabel(r"$k$")
ax.set_title("Wave-number-normalized spectrogram \n" + r"$P_{S} \theta_x(k,b) / \max_b(P_{S} \theta_x(k,b))$")
fig.colorbar(pcm, ax=ax)
plt.tight_layout()
fig.savefig('./result/turbulence/theta_spectro.jpg')
plt.close(fig)

# Second figure: Spectrogram (phase)
fig2, ax2 = plt.subplots(figsize=(4,2.5))
pcm2 = ax2.pcolormesh(x_spec, freqs, np.angle(Sx), cmap='hsv')
# ax2.set_yscale("log")
ax2.set_xlabel(r"$b$")
ax2.set_ylabel(r"$k$")
ax2.set_xlim(0, 2 * np.pi)
ax2.set_title("STFT Phase \n" + r"$\angle STFT_{\theta_x}(k,b)$")
fig2.colorbar(pcm2, ax=ax2)
plt.tight_layout()
fig2.savefig('./result/turbulence/theta_spectro_phase.jpg')
plt.close(fig2)

# Plot energy per wavenumber
fig, ax = plt.subplots(figsize=(2,2.5))
ax.plot(freqs, energy_per_wavenumber)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$k$")
ax.set_title("Spectrogram energy per wavenumber\n" + r"$\sum_b P_{S} \theta_x(k,b)$")
plt.tight_layout()
fig.savefig('./result/turbulence/theta_spectro_energy.jpg')
plt.close(fig)

# ## Scalogram 

widths = np.geomspace(2,mm//8, num=100) # geometric space for scales
cwtmatr, freqs_cwt = pywt.cwt(oline, widths, 'cmor0.5-1.0', sampling_period=sampling_period) # 'morl'
cwtmatrabs = np.abs(cwtmatr) / np.max(np.abs(cwtmatr), axis=1, keepdims=True) # absolute value of complex result and normalization
# Compute energy per wavenumber for the scalogram
energy_per_wavenumber_cwt = np.mean(np.abs(cwtmatr)**2, axis=1)

# Plot
# First figure: Scalogram (magnitude)
fig1, ax1 = plt.subplots(figsize=(4,2.5))
pcm1 = ax1.pcolormesh(x, freqs_cwt, cwtmatrabs)
ax1.set_yscale("log")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$k$")
ax1.set_title("Wave-number-normalized scalogram \n" + r"$P_{W} \theta_x(k,b) / \max_b(P_{S} \theta_x(k,b))$")
fig1.colorbar(pcm1, ax=ax1)
plt.tight_layout()
fig1.savefig('./result/turbulence/theta_cwt.jpg')
plt.close(fig1)

# Second figure: Phase
fig2, ax2 = plt.subplots(figsize=(4,2.5))
pcm2 = ax2.pcolormesh(x, freqs_cwt, np.angle(cwtmatr), cmap='hsv')
ax2.set_yscale("log")
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$k$")
ax2.set_title("CWT Phase \n" + r"$\angle CWT_{\theta_x}(k,b)$")
fig2.colorbar(pcm2, ax=ax2)
plt.tight_layout()
fig2.savefig('./result/turbulence/theta_cwt_phase.jpg')
plt.close(fig2)

# Plot energy per wavenumber (scalogram)
fig, ax = plt.subplots(figsize=(2,2.5))
ax.plot(freqs_cwt, energy_per_wavenumber_cwt)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$k$")
ax.set_ylabel("Energy")
ax.set_title("Scalogram energy per wavenumber\n" + r"$\sum_b P_{W} \theta_x(k,b)$")
plt.tight_layout()
fig.savefig('./result/turbulence/theta_cwt_energy.jpg')
plt.close(fig)


## Comparison with FFT
offt = np.fft.rfft(oline)/np.sqrt(mm)
freq = np.fft.rfftfreq(mm,d = sampling_period)

cwtspec = np.mean(np.abs(cwtmatr)**2,axis=1)

fig, ax = plt.subplots()
ax.loglog(freq,np.abs(offt)**2,label='FFT')
ax.plot(freqs_cwt, energy_per_wavenumber_cwt,label='CWT')
ax.plot(freqs, energy_per_wavenumber,label='STFT')
ax.legend()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$k$")
ax.set_ylabel("Energy")
plt.tight_layout()
fig.savefig('./result/turbulence/theta_fft.jpg')