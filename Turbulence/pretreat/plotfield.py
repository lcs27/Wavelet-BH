import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import h5py 
from scipy import signal
from ttictoc import tic,toc
plt.style.use(['science','ieee'])

mm = 4096
f = h5py.File('./data/flowfield.h5', 'r')
u1 = np.array(f['u1'][0,0:mm,0:mm]).transpose()
u2 = np.array(f['u2'][0,0:mm,0:mm]).transpose()

x = np.linspace(0,2*np.pi,mm,endpoint=False)
X,Y = np.meshgrid(x,x)
dx = x[1] - x[0]

# Plot field

fig, ax = plt.subplots(figsize=[4, 3])
im = ax.pcolormesh(X, Y, u1, cmap='viridis', shading='auto')
fig.colorbar(im, ax=ax, label=r'$u$')
ax.set_title(r'$u$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
fig.tight_layout()
fig.savefig('./result/turbulence/u.jpg')
plt.close()
print('plot u done')

fig, ax = plt.subplots(figsize=[4, 3])
im = ax.pcolormesh(X, Y, u2, cmap='viridis', shading='auto')
fig.colorbar(im, ax=ax, label=r'$v$')
ax.set_title(r'$v$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
fig.tight_layout()
fig.savefig('./result/turbulence/v.jpg')
plt.close()
print('plot v done')


f = h5py.File('./data/flowfield_noised.h5', 'r')
u1 = np.array(f['u1'][0,0:mm,0:mm]).transpose()
u2 = np.array(f['u2'][0,0:mm,0:mm]).transpose()

fig, ax = plt.subplots(figsize=[4, 3])
im = ax.pcolormesh(X, Y, u1, cmap='viridis', shading='auto')
fig.colorbar(im, ax=ax, label=r'$u$')
ax.set_title(r'$u$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
fig.tight_layout()
fig.savefig('./result/turbulence/noised_u.jpg')
plt.close()
print('plot noised u done')

fig, ax = plt.subplots(figsize=[4, 3])
im = ax.pcolormesh(X, Y, u2, cmap='viridis', shading='auto')
fig.colorbar(im, ax=ax, label=r'$v$')
ax.set_title(r'$v$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
fig.tight_layout()
fig.savefig('./result/turbulence/noised_v.jpg')
plt.close()
print('plot noised v done')

f = h5py.File('./data/velgrad.h5', 'r')
theta = np.array(f['theta'][0,0:mm,0:mm]).transpose()
omega = np.array(f['omega'][0,0:mm,0:mm]).transpose()

fig, ax = plt.subplots(figsize=[4, 3])
im = ax.pcolormesh(X, Y, theta, cmap='viridis', shading='auto')
fig.colorbar(im, ax=ax, label=r'$\theta$')
ax.set_title(r'$\theta$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
fig.tight_layout()
fig.savefig('./result/turbulence/theta.jpg')
plt.close()
print('plot theta done')

fig, ax = plt.subplots(figsize=[4, 3])
im = ax.pcolormesh(X, Y, omega, cmap='viridis', shading='auto')
fig.colorbar(im, ax=ax, label=r'$\omega$')
ax.set_title(r'$\omega$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
fig.tight_layout()
fig.savefig('./result/turbulence/omega.jpg')
plt.close()
print('plot omega done')

f = h5py.File('./data/velgrad_noised.h5', 'r')
thetan = np.array(f['theta'][0,0:mm,0:mm]).transpose()
omegan = np.array(f['omega'][0,0:mm,0:mm]).transpose()

fig, ax = plt.subplots(figsize=[4, 3])
im = ax.pcolormesh(X, Y, thetan, cmap='viridis', shading='auto', vmin=theta.min(), vmax=theta.max())
fig.colorbar(im, ax=ax, label=r'$\theta$')
ax.set_title(r'$\theta$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
fig.tight_layout()
fig.savefig('./result/turbulence/noised_theta.jpg')
plt.close()
print('plot noised theta done')

fig, ax = plt.subplots(figsize=[4, 3])
im = ax.pcolormesh(X, Y, omegan, cmap='viridis', shading='auto', vmin=omega.min(), vmax=omega.max())
fig.colorbar(im, ax=ax, label=r'$\omega$')
ax.set_title(r'$\omega$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
fig.tight_layout()
fig.savefig('./result/turbulence/noised_omega.jpg')
plt.close()
print('plot noised omega done')
