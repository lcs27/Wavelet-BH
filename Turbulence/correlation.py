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
omega = np.array(f['omega'][0,0:mm,0:mm]).transpose()
x = np.linspace(0,2*np.pi,mm,endpoint=False)
X,Y = np.meshgrid(x,x)

pos = 100
thetaline = theta[:,pos]
omegaline = omega[:,pos]
fig,ax = plt.subplots()
ax.plot(x,thetaline,label=r'$\theta$')
ax.plot(x,omegaline,label=r'$\omega$')
ax.set_xlabel(r'$x$')
ax.autoscale(enable=True, axis='x', tight=True)
fig.savefig('./result/turbulence/theta_omega_line.jpg')

## Scalogram
sampling_period = np.diff(x).mean()
widths = np.geomspace(16,mm//4, num=100) # geometric space for scales
Sx_theta, freqs = pywt.cwt(thetaline, widths, 'cmor0.5-1.0', sampling_period=sampling_period)
Sx_omega, freqs = pywt.cwt(omegaline, widths, 'cmor0.5-1.0', sampling_period=sampling_period)

def cross_correlation(x, y):
    # Remove mean from x and y
    x = x
    y = y
    # Calculate numerator
    numerator = np.sum(x * y)
    # Calculate denominators
    denominator = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    correlation = numerator / denominator
    return correlation

# CWT scale correlation
cwt_theta = np.abs(Sx_theta)**2
cwt_omega = np.abs(Sx_omega)**2
correlation = np.zeros(cwt_theta.shape[0])
for f in range(cwt_theta.shape[0]):
    correlation[f] = cross_correlation(cwt_theta[f,:], cwt_omega[f,:])
cwt_theta = cwt_theta / np.max(cwt_theta, axis=1, keepdims=True)
cwt_omega = cwt_omega / np.max(cwt_omega, axis=1, keepdims=True)

# Plot correlation
fig3, ax3 = plt.subplots()
pcm3 = ax3.plot(freqs, correlation)
ax3.set_xscale("log")
ax3.set_xlabel(r"$k$")
ax3.set_ylabel(r"Correlation")
ax3.set_title(r"Wave-number correlation of $\theta$ and $\omega$ $\rho$(k)")
ax3.set_ylim(0,1)
fig3.tight_layout()
fig3.savefig('./result/turbulence/theta_omega_cwt.jpg')