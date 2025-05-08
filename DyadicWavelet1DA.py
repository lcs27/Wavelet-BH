import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','ieee'])

N=128
t = np.linspace(0,1,N, endpoint=False)
a = np.cos(2*np.pi*5*t) + np.cos(2*np.pi*10*t) + np.cos(2*np.pi*15*t)
sigma = 0.5
a = a + np.random.normal(0, sigma, N)
level = 4

## Self-made convolution
offset_l = 2
offset_r = 3
v2=np.sqrt(2)
h = np.array([0,0.125,0.375,0.375,0.125,0])*v2
g = np.array([0,0,1,-1,0,0])*v2
g_tilde = np.array([0.015625,0.109375,0.34375,-0.34375,-0.109375,-0.015625])*v2

low = np.zeros((level+1,N)) # low coefficients, [a0,a1,a2,a3,a4,a5,a6]
high = np.zeros((level+1,N)) # high coefficients, [0,d1,d2,d3,d4,d5,d6]

def expand(a, N):
    """
    Array expansion with zeros.
    
    Parameters:
        a: Original Numpy array
        N: Number of zeros to insert between elements
    
    Returns:
        Expanded NumPy array
    """
    if a.size == 0:
        return a
    
    # Create output array with correct size
    result = np.zeros(a.size + (a.size - 1) * N, dtype=a.dtype)
    
    # Place original elements at the right positions
    result[::N+1] = a
    return result

low[0,:] = a
for j in range(level):
    j2 = np.power(2,j)
    hj_rv = np.flip(expand(h,j2-1))
    gj_rv = np.flip(expand(g,j2-1))

    start_trim = j2 * offset_r
    end_trim = j2 * offset_l
    low[j+1,:] = np.convolve(low[j,:], hj_rv, mode='full')[start_trim:-end_trim]
    high[j+1,:] = np.convolve(low[j,:], gj_rv, mode='full')[start_trim:-end_trim]



# Plot low and high coefficients
fig,ax = plt.subplots(2,1,figsize=(4,4))
for j in range(level+1):
    ax[0].plot(low[j, :], label=f'low[{j}]')
    ax[1].plot(high[j, :], label=f'high[{j}]')

# Set titles and legends
ax[0].set_title('Low Coefficients')
ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
ax[1].set_title('High Coefficients')
ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
ax[0].set_xlim(0,N)
ax[1].set_xlim(0,N)

fig.tight_layout()
fig.savefig(f'./result/DyadicWavelet_coefficients.png')
plt.close()

recon = low[level,:]
for j in range(level,0,-1):
    j2 = np.power(2,j-1)
    hj_tilde = expand(h,j2-1)
    gj_tilde = expand(g_tilde,j2-1)

    start_trim = j2 * offset_l
    end_trim = j2 * offset_r
    recon = 0.5*(np.convolve(recon, hj_tilde, mode='full')[start_trim:-end_trim] + \
                 np.convolve(high[j,:], gj_tilde, mode='full')[start_trim:-end_trim])
    
fig,ax = plt.subplots(figsize=(4,2))
ax.plot(a, label=f'Original')
ax.plot(recon, label=f'Reconstruct')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
ax.set_xlim(0,N)
fig.tight_layout()
fig.savefig(f'./result/DyadicWavelet_Recon.png')
plt.close()
