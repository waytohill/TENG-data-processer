from tkinter import filedialog
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy
from scipy import signal
from scipy.optimize import leastsq
from scipy.ndimage import grey_closing, grey_opening
from scipy.signal import butter, filtfilt, iirnotch, oaconvolve, lfilter
import pywt
from scipy.fft import fft, fftfreq
import os


window_size = int(500)
windowsize = 20
window_length = 53
kvalue = 3
sample_rate = 500

def _open_file():
    filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

    data = pd.read_csv(filename)

    colnames = data.columns

    xdata = data[colnames[0]]
    ydata = data[colnames[1]]
    
    return xdata, ydata


# 滑动均值滤波
def moving_ave(data, windowsize):
    window = np.ones(int(windowsize))/float(windowsize)
    ret = np.convolve(data,window,'same')
    return ret

# sav-gol滤波
def savgolfil(data,windowlen,kvalue):
    ret = scipy.signal.savgol_filter(data,windowlen,kvalue)

    return ret

def medfilted(ydata, windowsize):
    ydata_baseline = signal.medfilt(ydata, int(windowsize)+1)

    return ydata_baseline

"""Baseline"""
# 形态学方法求解baseline
def morphology_baseline(signal, structure_size=100):
    structure = np.ones(structure_size)
    baseline = grey_closing(signal, structure=structure)

    return baseline
"""
# 自适应形态学滤波求解baseline
def morphological_baseline(signal, structure_size=500):
    kernel = np.ones(structure_size) / structure_size
    baseline = oaconvolve(signal, kernel, mode='same')
    return baseline

# 改进型小波变换求解baseline
def wavelet_baseline(signal,wavelet='sym8', level=6):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    baseline = pywt.waverec(coeffs, wavelet)

    return baseline
"""
"""denoise"""
# 工频陷波
def notch_filter(signal, fs=500, freq=50, Q=30):
    nyq = 0.5 * fs
    freq_normalized = freq / nyq
    b, a = iirnotch(freq_normalized, Q)

    return filtfilt(b, a, signal)



# 自适应小波阈值去噪
def advanced_wavelet_denoise(signal, wavelet='db8', level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thresholds = [sigma*np.sqrt(2*np.log(len(signal)))]*len(coeffs)
    thresholds[0] = 0
    coeffs = [pywt.threshold(c,t,mode='soft') for c,t in zip(coeffs, thresholds)]

    return pywt.waverec(coeffs, wavelet)

"""
# 小波去噪
def wavelet_denoise(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2*np.log(len(signal)))
    coeffs = [pywt.threshold(c, uthresh*0.6, mode='soft') for c in coeffs]

    return pywt.waverec(coeffs, wavelet)
# 脉冲保持滤波实现降噪
def pulse_preserving_filter(signal, alpha=0.8):

    return  filtfilt([alpha], [1, alpha-1], signal)
    #return lfilter([alpha], [1, alpha-1], signal)
"""
# FFT
def vibration_detect(ydata):
    
    # 去除直流分量->解决求解出的主频恒为0的问题
    ydata = ydata - np.mean(ydata)
    # fft结果
    fft_result = fft(ydata)
    # 幅值归一化
    fft_magnitude = np.abs(fft_result)/len(ydata)
    # 频率轴
    frequencies = fftfreq(len(ydata), d = 1/sample_rate)

    positive_freqs = frequencies[:len(frequencies) // 2]
    positive_magnitude = fft_magnitude[:len(fft_magnitude) // 2]

    # 检测主频率
    dominant_frequency = positive_freqs[np.argmax(positive_magnitude)]

    return dominant_frequency, positive_freqs, positive_magnitude

def setup_layout():
    fig = plt.figure(figsize=(12,8))
    gs = gridspec.GridSpec(3,2)

    return fig, gs

xdata, ydata = _open_file()

fig, gs = setup_layout()

# baseline
baseline_morphology = morphology_baseline(ydata)

# detrended ydata
ydata_morphology = -(ydata - baseline_morphology)

# denoise step 1
ydata_denoise1 = notch_filter(ydata_morphology)

# denoise step 2
ydata_denoise2 = advanced_wavelet_denoise(ydata_denoise1)[:len(xdata)]

ydata_final = moving_ave(savgolfil(ydata_denoise2,window_length, kvalue), windowsize)

dominant_frequency1, positive_freqs1, positive_magnitude1 = vibration_detect(np.array(ydata_denoise2, dtype = np.float64))
dominant_frequency2, positive_freqs2, positive_magnitude2 = vibration_detect(np.array(ydata_final, dtype = np.float64))

ax1 = fig.add_subplot(gs[0,0])
ax1.plot(xdata, ydata)
ax1.set_title('Origin')

ax11 = fig.add_subplot(gs[0,1])
ax11.plot(xdata, ydata_denoise2-np.mean(ydata_denoise2))
ax11.set_title("ydata_denoised2 with DC-off")

ax2 = fig.add_subplot(gs[1,0])
ax2.plot(xdata, ydata_denoise2)
ax2.set_title('Denoised curve with step 2')

ax3 = fig.add_subplot(gs[1,1])
ax3.plot(xdata, ydata_final)
ax3.set_title('Final denoised curve')

ax4 = fig.add_subplot(gs[2,0])
ax4.plot(positive_freqs1, positive_magnitude1, label= f"dominant frequency: {dominant_frequency1:.2f} Hz")
ax4.set_title('FFT of Step 2 denoised curve')

ax5 = fig.add_subplot(gs[2,1])
ax5.plot(positive_freqs2, positive_magnitude2, label= f"dominant frequency: {dominant_frequency2:.2f} Hz")
ax5.set_title('FFT of Final denoised curve')




plt.legend(loc='best')
plt.tight_layout()
plt.show()