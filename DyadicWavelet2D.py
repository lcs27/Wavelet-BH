import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pywt
import cv2
plt.style.use(['science','ieee'])

'''
This file reproduces Sun, Yan-Kui, and Wan Huang. "Separable 2D dyadic wavelet and its applications in contour detection of handwriting." 2007 International Conference on Wavelet Analysis and Pattern Recognition. Vol. 3. IEEE, 2007.
'''

data = cv2.imread("./data/pic.jpg")
data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY).astype(np.float32)


v2=np.sqrt(2)
offset = 4
h = np.array([0,0,0.0625,0.25,0.375,0.25,0.0625,0,0])*v2
g = np.array([-0.00008,-0.01643,-0.10872,-0.59261,0,0.59261,0.10872,0.01643,0.00008])*v2
q = np.array([0.00003,0.00727,0.03118,0.06623,0.79113,0.06623,0.03118,0.00727,0.00003])*v2

def DyWT2D(image,hrv,grv,qrv,offset,level=0):
    '''
    Dyadic Wavelet Transform 2D
    This function performs a 2D dyadic wavelet transform on the input image.
    It computes the approximation coefficients (a) and detail coefficients (d1, d2)
    using separable 2D wavelet filters (hrv, grv, qrv).

    Parameters:
    - image: 2D numpy array representing the input grayscale image.
    - hrv: 1D numpy array representing the reversed low-pass filter coefficients.
    - grv: 1D numpy array representing the reversed high-pass filter coefficients.
    - qrv: 1D numpy array representing the reversed quadrature filter coefficients.
    - offset: Integer offset for filter alignment.
    - level: Integer specifying the decomposition level (default is 0).

    Returns:
    - a: 2D numpy array of approximation coefficients.
    - d1: 2D numpy array of horizontal detail coefficients.
    - d2: 2D numpy array of vertical detail coefficients.
    '''
    N,M=image.shape

    a = np.zeros_like(image)
    d1 = np.zeros_like(image)
    d2 = np.zeros_like(image)
    lenh = np.size(hrv)
    leng = np.size(grv)
    lenq = np.size(qrv)
    assert lenh == 2*offset + 1
    assert leng == 2*offset + 1
    assert lenq == 2*offset + 1

    j2 = np.power(2,level)
    for n in range(N):
        for m in range(M):
            for k in range(lenh):
                for p in range(lenh):
                    k0 = k-offset
                    p0 = p-offset
                    a[n,m] += hrv[k]*hrv[p]*image[(n-j2*k0)%N,(m-j2*p0)%M]
                    d1[n,m] += grv[k]*qrv[p]*image[(n-j2*k0)%N,(m-j2*p0)%M]
                    d2[n,m] += qrv[k]*grv[p]*image[(n-j2*k0)%N,(m-j2*p0)%M]
    
    return a,d1,d2

hrv = np.flip(h)
grv = np.flip(g)
qrv = np.flip(q)
a,d1,d2 = DyWT2D(data,hrv,grv,qrv,offset,0)

fig, ax = plt.subplots(2, 2, figsize=(6, 6))
ax[0,0].imshow(data, cmap='gray')
ax[0,0].set_title('Original')
ax[1,0].imshow(a, cmap='gray')
ax[1,0].set_title('a')
ax[0,1].imshow(d1, cmap='gray')
ax[0,1].set_title('d1')
ax[1,1].imshow(d2, cmap='gray')
ax[1,1].set_title('d2')
fig.savefig('./result/2D_Dyadic.jpg')

def detect_edge(d1,d2,T):
    
    N1, M1 = d1.shape
    N2, M2 = d2.shape
    assert N1 == N2
    assert M1 == M2
    Mf = np.sqrt(d1**2 + d2**2)
    tanAf = d2/d1
    ff = np.zeros_like(Mf)

    for i in range(N1):
        for j in range(M1):
            if(Mf[i,j]>T):
                tanvalue = tanAf[i,j] 
                if(tanvalue<=(v2-1) and tanvalue>=(1-v2)):
                    if(Mf[i,j] > Mf[(i+1)%N1,j] and Mf[i,j] > Mf[(i-1)%N1,j]):
                        ff[i,j] = 1
                elif(tanvalue<=(v2+1) and tanvalue>=(v2-1)):
                    if(Mf[i,j] > Mf[(i+1)%N1,(j+1)%M1] and Mf[i,j] > Mf[(i-1)%N1,(j-1)%M1]):
                        ff[i,j] = 1
                elif(tanvalue>(v2+1) or  tanvalue<(-v2-1)):
                    if(Mf[i,j] > Mf[i,(j+1)%M1] and Mf[i,j] > Mf[i,(j-1)%M1]):
                        ff[i,j] = 1
                else:
                    if(Mf[i,j] > Mf[(i-1)%N1,(j+1)%M1] and Mf[i,j] > Mf[(i+1)%N1,(j-1)%M1]):
                        ff[i,j] = 1

    return ff

edge = detect_edge(d1,d2,2.5)
fig, ax = plt.subplots(1, 2, figsize=(6, 6))
ax[0].imshow(data, cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(edge, cmap='gray')
ax[1].set_title('Edge')
fig.savefig('./result/2D_Dyadic_Edge.jpg')

def IDyWT2D(a, d1, d2, h, g, q, offset, level=0):
    '''
    Inverse Dyadic Wavelet Transform 2D
    This function reconstructs the original image from the approximation (a)
    and detail coefficients (d1, d2) using separable 2D wavelet filters.

    Parameters:
    - a: 2D numpy array of approximation coefficients.
    - d1: 2D numpy array of horizontal detail coefficients.
    - d2: 2D numpy array of vertical detail coefficients.
    - h: 1D numpy array representing the low-pass filter coefficients.
    - g: 1D numpy array representing the high-pass filter coefficients.
    - q: 1D numpy array representing the quadrature filter coefficients.
    - offset: Integer offset for filter alignment.
    - level: Integer specifying the reconstruction level (default is 0).

    Returns:
    - image: 2D numpy array representing the reconstructed grayscale image.
    '''
    N, M = a.shape
    N1, M1 = d1.shape
    N2, M2 = d2.shape
    assert N == N1, N1 == N2
    assert M == M1, M1 == M2
    image = np.zeros((N, M))
    lenh = np.size(h)
    leng = np.size(g)
    lenq = np.size(q)
    assert lenh == 2 * offset + 1
    assert leng == 2 * offset + 1
    assert lenq == 2 * offset + 1

    j2 = np.power(2, level)
    for n in range(N):
        for m in range(M):
            for k in range(lenh):
                for p in range(lenh):
                    k0 = k - offset
                    p0 = p - offset
                    image[n, m] += h[k] * h[p] * a[(n - j2 * k0) % N, (m - j2 * p0) % M]
                    image[n, m] += g[k] * q[p] * d1[(n - j2 * k0) % N, (m - j2 * p0) % M]
                    image[n, m] += q[k] * g[p] * d2[(n - j2 * k0) % N, (m - j2 * p0) % M]

    image /= 4
    return image

# reconstructed = IDyWT2D(a, d1, d2, h, g, q, offset, 0)

# fig, ax = plt.subplots(1, 2, figsize=(6, 6))
# ax[0].imshow(data, cmap='gray')
# ax[0].set_title('Original')
# ax[1].imshow(reconstructed, cmap='gray')
# ax[1].set_title('Reconstructed')
# fig.savefig('./result/2D_Dyadic_Reconstructed.jpg')