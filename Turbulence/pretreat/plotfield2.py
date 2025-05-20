import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import h5py 
from scipy import signal
from ttictoc import tic,toc
plt.style.use(['science','ieee'])

mm = 4096
x = np.linspace(0,2*np.pi,mm,endpoint=False)
X,Y = np.meshgrid(x,x)
dx = x[1] - x[0]

f = h5py.File('./data/velgrad.h5', 'r')
theta = np.array(f['theta'][0,0:mm,0:mm]).transpose()
omega = np.array(f['omega'][0,0:mm,0:mm]).transpose()

f = h5py.File('./data/velgrad_noised.h5', 'r')
# n for noised
thetan = np.array(f['theta'][0,0:mm,0:mm]).transpose()
omegan = np.array(f['omega'][0,0:mm,0:mm]).transpose()


def calculate_psnr(true_signal, denoised_signal):
    mse = np.mean((true_signal - denoised_signal) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(true_signal)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

pos = 100
print(f'Position is y={x[pos]}')
oline = omega[:,pos]
olinen = omegan[:,pos]

fig,ax = plt.subplots(figsize=[4, 2])
ax.plot(x,oline)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\omega$')
ax.set_xticks(np.arange(0, 2*np.pi+0.01, np.pi/2))
ax.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
ax.autoscale(enable=True, axis='x', tight=True)
fig.savefig('./result/turbulence/omegaline.jpg')
plt.close()

fig,ax = plt.subplots(figsize=[4, 2])
ax.plot(x,olinen, linestyle='--', color='red', label='Noised')
ax.plot(x,oline, color='black', label='Original')
ax.legend()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\omega$')
ax.set_xticks(np.arange(0, 2*np.pi+0.01, np.pi/2))
ax.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
ax.autoscale(enable=True, axis='x', tight=True)
fig.savefig('./result/turbulence/noised_omegaline.jpg')
plt.close()

print('omega PSNR:',calculate_psnr(omega,omegan))


thetaline = theta[:,pos]
thetalinen = thetan[:,pos]

fig,ax = plt.subplots(figsize=[4, 2])
ax.plot(x,thetaline)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\theta$')
ax.autoscale(enable=True, axis='x', tight=True)
fig.savefig('./result/turbulence/thetaline.jpg')
plt.close()

fig,ax = plt.subplots(figsize=[4, 2])
ax.plot(x,thetalinen, linestyle='--', color='red', label='Noised')
ax.plot(x,thetaline, color='black', label='Original')
ax.legend()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\theta$')
ax.autoscale(enable=True, axis='x', tight=True)
fig.savefig('./result/turbulence/noised_thetaline.jpg')
plt.close()
print('theta PSNR:',calculate_psnr(theta,thetan))
