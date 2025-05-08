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

offset = 3
v2=np.sqrt(2)
h = np.array([0,0,0.125,0.375,0.375,0.125,0])*v2
g = np.array([0,0,0,1,-1,0,0])*v2
g_tilde = np.array([0,0.015625,0.109375,0.34375,-0.34375,-0.109375,-0.015625])*v2

def DyWT(image,hrv,grv,offset,level=0):
    """
    Compute the 1D Dyadic Wavelet Transform (DyWT) for the given signal.

    This function calculates the approximation coefficients (`a`) and detail 
    coefficients (`d`) of the input signal using the provided reversed low-pass 
    filter (`hrv`) and reversed high-pass filter (`grv`). The transform is 
    performed at the specified decomposition level.

    Parameters:
    -----------
    image : numpy.ndarray
        Input 1D signal to be transformed.
    hrv : numpy.ndarray
        Reversed low-pass filter coefficients.
    grv : numpy.ndarray
        Reversed high-pass filter coefficients.
    offset : int
        Offset to align the filters correctly.
    level : int, optional
        Decomposition level of the wavelet transform (default is 0).

    Returns:
    --------
    a : numpy.ndarray
        Approximation coefficients obtained using the low-pass filter.
    d : numpy.ndarray
        Detail coefficients obtained using the high-pass filter.

    Notes:
    ------
    - Assumes periodic boundary conditions for the input signal.
    - Filters `hrv` and `grv` must satisfy wavelet transform properties 
      (e.g., orthogonality or biorthogonality), The length of the scaling 
      and wavelet filters must be `2 * offset + 1`.
    - Input signal length must be compatible with the decomposition level 
      to prevent aliasing.
    """
    N,=image.shape

    a = np.zeros_like(image)
    d = np.zeros_like(image)
    lenh = np.size(hrv)
    leng = np.size(grv)

    assert lenh == (2*offset+1)
    assert leng == (2*offset+1)

    j2 = np.power(2,level)
    for n in range(N):
        for k in range(lenh):
            k0 = k-offset
            a[n] += hrv[k]*image[(n-j2*k0)%N]
            d[n] += grv[k]*image[(n-j2*k0)%N]

    return a,d

low = np.zeros((level+1,N)) # low coefficients, [a0,a1,a2,a3,a4,a5,a6]
high = np.zeros((level+1,N)) # high coefficients, [0,d1,d2,d3,d4,d5,d6]

low[0,:] = a
h_rv = np.flip(h)
g_rv = np.flip(g)
for j in range(level):
    low[j+1,:], high[j+1,:] = DyWT(low[j,:],h_rv,g_rv,offset,level=j)

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
fig.savefig(f'./result/DyadicWavelet_coefficients_2.png')
plt.close()

# TODO: write reconstruct

def IDyWT(a,d,h,g,offset,level=0):
    """
    Perform the Inverse Dyadic Wavelet Transform (IDyWT) for 1D signals.

    Parameters:
    -----------
    a : numpy.ndarray
        The approximation coefficients array.
    d : numpy.ndarray
        The detail coefficients array.
    h : numpy.ndarray
        The scaling filter (low-pass filter).
    g : numpy.ndarray
        The wavelet filter (high-pass filter).
    offset : int
        The offset used to align the filters.
    level : int, optional
        The decomposition level (default is 0).

    Returns:
    --------
    numpy.ndarray
        The reconstructed signal after applying the inverse transform.

    Notes:
    ------
    - The function assumes periodic boundary conditions for the signal.
    - The length of the scaling and wavelet filters must be `2 * offset + 1`.
    - The input arrays `a` and `d` must have the same shape.
    """
    N,=a.shape
    N1,=d.shape
    assert N==N1
    image = np.zeros_like(a)
    lenh = np.size(h)
    leng = np.size(g)

    assert lenh == (2*offset+1)
    assert leng == (2*offset+1)

    j2 = np.power(2,level)
    for n in range(N):
        for k in range(lenh):
            k0 = k-offset
            image[n] += h[k]*a[(n-j2*k0)%N]
            image[n] += g[k]*d[(n-j2*k0)%N]

    image /= 2
    return image

recon = low[level,:]
for j in range(level,0,-1):
    recon = IDyWT(recon,high[j,:],h,g_tilde,offset=offset,level=(j-1))
    print(recon)
    
fig,ax = plt.subplots(figsize=(4,2))
ax.plot(a, label=f'Original')
ax.plot(recon, label=f'Reconstruct')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
ax.set_xlim(0,N)
fig.tight_layout()
fig.savefig(f'./result/DyadicWavelet_Recon_2.png')
plt.close()