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

# Add Gaussian noise to u1 and u2
noise_level = 0.2*2*np.pi/mm  # Adjust the noise level as needed
u1 += noise_level * np.random.normal(size=u1.shape)
u2 += noise_level * np.random.normal(size=u2.shape)

dx = x[1] - x[0]

# Plot field
fig, ax = plt.subplots(figsize=[4, 3])
im = ax.contourf(X, Y, u1, cmap='viridis')
fig.colorbar(im, ax=ax, label=r'$u$')
ax.set_title(r'$u$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
fig.tight_layout()
fig.savefig('./result/turbulence/noised_u.jpg')

# Plot u2
fig, ax = plt.subplots(figsize=[4, 3])
im = ax.contourf(X, Y, u2, cmap='viridis')
fig.colorbar(im, ax=ax, label=r'$v$')
ax.set_title(r'$v$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
fig.tight_layout()
fig.savefig('./result/turbulence/noised_v.jpg')

u1 = u1.transpose()
u2 = u2.transpose()

fn1 = h5py.File('./data/flowfield_noised.h5','w')
u1n1 = fn1.create_dataset("u1", (1,mm+1,mm+1), '<f8')
u2n1 = fn1.create_dataset("u2", (1,mm+1,mm+1), '<f8')
u3n1 = fn1.create_dataset("u3", (1,mm+1,mm+1), '<f8')
for i in range(0,mm):
    tic()
    for j in range(0,mm):
        u1n1[0,i,j]=u1[i,j]
        u2n1[0,i,j]=u2[i,j]
        u3n1[0,0,0]=0
    
    u1n1[0,i,mm]=u1[i,0]
    u2n1[0,i,mm]=u2[i,0]
    u3n1[0,i,mm]=0
    print(i,'/',mm,"remain",toc()*(mm-i),end='\r')

for j in range(0,mm):    
    u1n1[0,mm,j]=u1[0,j]
    u2n1[0,mm,j]=u2[0,j]
    u3n1[0,mm,j]=0
