import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import h5py 
from scipy import signal
from ttictoc import tic,toc
import pywt
import torch
import torch.nn as nn
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

class DenoiseCNN(nn.Module):
    def __init__(self,m,k):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, m, k, padding=1),
            nn.ReLU(),
            nn.Conv1d(m, m, k, padding=1),
            nn.ReLU(),
            nn.Conv1d(m, 1, k, padding=1)
        )
    def forward(self, x):
        return self.conv(x)

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
    pos = 100
    field = 'theta'
    wav = 'db5'
    fpos = field + '_' + wav + '_NN'
    os.makedirs('./result/turbulence/denoising/'+fpos,exist_ok=True)
    sys.stdout = Logger(os.path.join('./result/turbulence/denoising', fpos+'_log.txt'))

    print(f'wav:{wav}')


    f = h5py.File('./data/velgrad.h5', 'r')
    theta = np.array(f['theta'][0,0:mm,0:mm]).transpose()
    omega = np.array(f['omega'][0,0:mm,0:mm]).transpose()

    f = h5py.File('./data/velgrad_noised.h5', 'r')
    # n for noised
    thetan = np.array(f['theta'][0,0:mm,0:mm]).transpose()
    omegan = np.array(f['omega'][0,0:mm,0:mm]).transpose()

    # Prepare training data by varying pos for each detail level (cD1 to cD4)
    train_positions = np.random.choice(range(mm), size=300, replace=False)
    train_inputs = []
    train_targets = []
    means = []
    stds = []
    models = []
    losses = []
    optimizers = []
    epochs = 500
    m = 16  # number of channels for CNN
    k = 3 # number of kernel
    level = 4

    # Collect all detail coefficients and their noised versions
    detail_coeffs = []
    detail_coeffs_noised = []

    for p in train_positions:
        if(field == 'theta'):
            ori_sig = theta[:,p]
            noi_sig = thetan[:,p]
        elif(field == 'omega'):
            ori_sig = omega[:,p]
            noi_sig = omegan[:,p]
        else:
            raise NotImplementedError

        cA, cD4, cD3, cD2, cD1 = pywt.wavedec(ori_sig, wav, level=level)
        cAn, cD4n, cD3n, cD2n, cD1n = pywt.wavedec(noi_sig, wav, level=level)
        detail_coeffs.append([cD1, cD2, cD3, cD4])
        detail_coeffs_noised.append([cD1n, cD2n, cD3n, cD4n])
    
    print('Wavelet Decomposition finish')

    # Prepare data, models, and optimizers for each level
    for i in range(4):
        # Prepare training data for this level
        train_inputs_level = []
        train_targets_level = []
        for p in range(len(train_positions)):
            train_inputs_level.append(detail_coeffs_noised[p][i])
            train_targets_level.append(detail_coeffs[p][i])
        train_inputs_level = np.stack(train_inputs_level)
        train_targets_level = np.stack(train_targets_level)

        # Normalize
        mean = train_inputs_level.mean()
        std = train_inputs_level.std()
        means.append(mean)
        stds.append(std)
        train_inputs_norm = (train_inputs_level - mean) / std
        train_targets_norm = (train_targets_level - mean) / std

        # Convert to torch tensors
        train_inputs_tensor = torch.tensor(train_inputs_norm, dtype=torch.float32).unsqueeze(1)
        train_targets_tensor = torch.tensor(train_targets_norm, dtype=torch.float32).unsqueeze(1)

        # Model, loss, optimizer
        model = DenoiseCNN(m=m,k=k)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(train_inputs_tensor)
            loss = loss_fn(output, train_targets_tensor)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 5 == 0:
                print(f"Level cD{i+1} - Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

        models.append(model)

    # After training, denoise each detail level for the selected signal
    print(f'Position is y={x[pos]} with filed {field}')
    if(field == 'theta'):
        ori_sig = theta[:,pos]
        noi_sig = thetan[:,pos]
    elif(field == 'omega'):
        ori_sig = omega[:,pos]
        noi_sig = omegan[:,pos]
    cAn, cD4n, cD3n, cD2n, cD1n = pywt.wavedec(noi_sig, wav, level=level)

    detailn = [cD1n, cD2n, cD3n, cD4n]
    detaildn = []
    for i in range(4):
        models[i].eval()
        with torch.no_grad():
            inp = torch.tensor((detailn[i] - means[i]) / stds[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            denoi = models[i](inp).squeeze().numpy() * stds[i] + means[i]
            detaildn.append(denoi)

    cA, cD4, cD3, cD2, cD1 = pywt.wavedec(ori_sig, wav, level=level)
    detail = [cD4, cD3, cD2, cD1]
    detailn.reverse()
    detaildn.reverse()
    # Prepare avg coefficients for reconstruction
    avg = cA
    avgn = cAn

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
