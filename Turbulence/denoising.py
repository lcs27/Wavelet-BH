import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import h5py 
from scipy import signal
from ttictoc import tic,toc
import pywt
import os
import sys
plt.style.use(['science','ieee'])


# Tools:

def get_var(cD):
    coeffs = cD
    abs_coeffs = []
    for coeff in coeffs:
        abs_coeffs.append(abs(coeff))
    abs_coeffs.sort()
    pos = len(abs_coeffs) // 2
    var = abs_coeffs[pos] / 0.6745
    return var

def VisuShrink(details, sigma, mode='soft'):
    print(f'Method: VisuShrink with mode {mode} and sigma={sigma:.2f}')
    denoised_detail = [None]*len(details)
    for idx, coeff in enumerate(details):
        N = len(coeff)
        thre = sigma * np.sqrt(2*np.log(N))
        print(f'Level={idx}, Threshold={thre}')
        denoised_detail[idx] = pywt.threshold(coeff, thre, mode=mode)
    return denoised_detail

def NeighBlock(details, n, sigma):
    print(f'Method: NeighBlock with n = {n} and sigma={sigma:.2f}')
    res = []
    L0 = int(np.log2(n) // 2)
    L1 = max(1, L0 // 2)
    L = L0 + 2 * L1

    def nb_beta(sigma, L, detail):
        S2 = np.sum(detail ** 2)
        lmbd = 4.50524  # solution of lmbd - log(lmbd) = 3
        beta = (1 - lmbd * L * sigma**2 / S2)
        return max(0, beta)

    for d in details:
        d2 = d.copy()
        for start_b in range(0, len(d2), L0):
            end_b = min(len(d2), start_b + L0)
            start_B = start_b - L1
            end_B = start_B + L
            if start_B < 0:
                end_B -= start_B
                start_B = 0
            elif end_B > len(d2):
                start_B -= end_B - len(d2)
                end_B = len(d2)
            assert end_B - start_B == L
            d2[start_b:end_b] *= nb_beta(sigma, L, d2[start_B:end_B])
        res.append(d2)
    return res

def calculate_psnr(true_signal, denoised_signal):
    mse = np.mean((true_signal - denoised_signal) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(true_signal)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
if __name__ == "__main__":
    # Load data
    mm = 4096
    x = np.linspace(0,2*np.pi,mm,endpoint=False)
    X,Y = np.meshgrid(x,x)
    dx = x[1] - x[0]
    denoise_method = 3
    pos = 100
    field = 'theta'
    wav = 'db5'
    if(denoise_method == 1):
        fpos = field + '_' + wav + '_NB'
    elif(denoise_method == 2):
        fpos = field + '_' + wav + '_VSs'
    elif(denoise_method == 3):
        fpos = field + '_' + wav + '_VSh'
    else:
        raise NotImplementedError
    os.makedirs('./result/turbulence/denoising/'+fpos,exist_ok=True)
    sys.stdout = Logger(os.path.join('./result/turbulence/denoising', fpos+'_log.txt'))

    f = h5py.File('./data/velgrad.h5', 'r')
    theta = np.array(f['theta'][0,0:mm,0:mm]).transpose()
    omega = np.array(f['omega'][0,0:mm,0:mm]).transpose()

    f = h5py.File('./data/velgrad_noised.h5', 'r')
    # n for noised
    thetan = np.array(f['theta'][0,0:mm,0:mm]).transpose()
    omegan = np.array(f['omega'][0,0:mm,0:mm]).transpose()

    print(f'Position is y={x[pos]:.3f} with filed {field}')
    print(f'Wavelet: {wav}')
    
    if(field == 'theta'):
        ori_sig = theta[:,pos]
        noi_sig = thetan[:,pos]
    elif(field == 'omega'):
        ori_sig = omega[:,pos]
        noi_sig = omegan[:,pos]
    else:
        raise NotImplementedError

    level = 4
    coeff = pywt.wavedec(ori_sig, wav, level=level)
    avg = coeff[0]
    detail = coeff[1:]
    cA,cD = pywt.dwt(noi_sig, wav)
    sigma = np.sqrt(get_var(cD))
    coeffn = pywt.wavedec(noi_sig, wav, level=level)
    avgn = coeffn[0]
    detailn = coeffn[1:]

    if(denoise_method == 1):
        detaildn = NeighBlock(detailn, mm, sigma)
    elif(denoise_method == 2):
        detaildn = VisuShrink(detailn, sigma, 'soft')
    elif(denoise_method == 3):
        detaildn = VisuShrink(detailn, sigma, 'hard')
    else:
        raise NotImplementedError
    
    # Reconstruct the original signal
    denoi_sig = pywt.waverec([avgn]+detaildn, wav)

    # Plot 1: Comparison of signals
    fig1, ax1 = plt.subplots(figsize=(4, 2))
    ax1.plot(x, noi_sig, label='Noised', color='red', linestyle='--')
    ax1.plot(x, ori_sig, label='Original', color='black')
    ax1.plot(x, denoi_sig, label='Denoised', color='blue', linestyle='-.')
    ax1.set_title('Comparison of Signals')
    ax1.set_xlabel(r'$x$')
    ax1.set_xticks(np.arange(0, 2*np.pi+0.01, np.pi/2))
    ax1.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    ax1.legend()
    ax1.autoscale(axis="x", tight=True)
    fig1.tight_layout()
    fig1.savefig('./result/turbulence/denoising/'+fpos+'/denoised_signal.jpg')

    # Plot 2: Average coefficients
    fig2, ax2 = plt.subplots(figsize=(4, 2))
    ax2.plot(avgn, linestyle='--', color='red', label='Noised')
    ax2.plot(avg, color='black', label='Original')
    ax2.set_title('Average Coefficients')
    ax2.autoscale(axis="x", tight=True)
    fig2.tight_layout()
    fig2.savefig('./result/turbulence/denoising/'+fpos+'/denoised_avg_coeff.jpg')

    # Plot 3-6: Detail coefficients for each level
    for i in range(level):
        fig, ax = plt.subplots(figsize=(4, 2))
        label = f'Detail L{level - i}'
        ax.plot(detailn[i], linestyle='--', color='red', label='Noised')
        ax.plot(detaildn[i], linestyle='-.', color='blue', label='Denoised')
        ax.plot(detail[i], color='black', label='Original')
        ax.set_title(label)
        ax.autoscale(axis="x", tight=True)
        fig.tight_layout()
        fig.savefig('./result/turbulence/denoising/'+fpos+f'/denoised_detail_L{level-i}.jpg')

    psnr_noised = calculate_psnr(ori_sig, noi_sig)
    psnr_denoised = calculate_psnr(ori_sig,denoi_sig)

    print(f"PSNR (Original vs Noised): {psnr_noised:.2f} dB")
    print(f"PSNR (Original vs Denoised): {psnr_denoised:.2f} dB")

    print(f"Minimum: original: {min(ori_sig):.2f}, noised:  {min(noi_sig):.2f}, denoised: {min(denoi_sig):.2f}")
