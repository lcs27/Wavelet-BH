import numpy as np
import pywt
import matplotlib.pyplot as plt
import cv2

data = cv2.imread("./pic.jpg")
data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY).astype(np.float32)

# Perform 1D Haar wavelet transform
coeffs = pywt.dwt2(data, 'Haar',mode='periodization')
cA, (cH, cV, cD) = coeffs 
fig, ax = plt.subplots(3, 2, figsize=(20, 15))
ax[0,0].imshow(data, cmap='gray')
ax[0,0].set_title('Original')
ax[1,0].imshow(cA, cmap='gray')
ax[1,0].set_title('cA')
ax[1,1].imshow(cH, cmap='gray')
ax[1,1].set_title('cH')
ax[2,0].imshow(cV, cmap='gray')
ax[2,0].set_title('cV')
ax[2,1].imshow(cD, cmap='gray')
ax[2,1].set_title('cD')
ax[0,1].set_visible(False)
fig.savefig('./2D_transform.jpg')

# Reconstruct the original signal
# recons = pywt.idwt2(cA, (cH, cV, cD), 'haar')