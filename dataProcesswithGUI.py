import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy
from scipy import signal
from scipy.ndimage import grey_closing
from scipy.signal import iirnotch, filtfilt
from scipy.fft import fft, fftfreq
import pywt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os


window_length = 53
kvalue = 3
windowsize = 20
sample_rate = 500


def moving_ave(data, windowsize):
    window = np.ones(int(windowsize))/float(windowsize)
    return np.convolve(data, window, 'same')

def savgolfil(data, windowlen, kvalue):
    return scipy.signal.savgol_filter(data, windowlen, kvalue)

def morphology_baseline(signal, structure_size=100):
    structure = np.ones(structure_size)
    return grey_closing(signal, structure=structure)

def notch_filter(signal, fs=500, freq=50, Q=30):
    nyq = 0.5 * fs
    freq_normalized = freq / nyq
    b, a = iirnotch(freq_normalized, Q)
    return filtfilt(b, a, signal)

def advanced_wavelet_denoise(signal, wavelet='db8', level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thresholds = [sigma * np.sqrt(2 * np.log(len(signal)))] * len(coeffs)
    thresholds[0] = 0
    coeffs = [pywt.threshold(c, t, mode='soft') for c, t in zip(coeffs, thresholds)]
    return pywt.waverec(coeffs, wavelet)

def vibration_detect(ydata):
    ydata = ydata - np.mean(ydata)
    fft_result = fft(ydata)
    fft_magnitude = np.abs(fft_result) / len(ydata)
    frequencies = fftfreq(len(ydata), d=1 / sample_rate)
    positive_freqs = frequencies[:len(frequencies) // 2]
    positive_magnitude = fft_magnitude[:len(frequencies) // 2]
    dominant_frequency = positive_freqs[np.argmax(positive_magnitude)]
    return dominant_frequency, positive_freqs, positive_magnitude


def plot_all(xdata, ydata, filename):
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 2)

    baseline_morphology = morphology_baseline(ydata)
    ydata_morphology = -(ydata - baseline_morphology)
    ydata_denoise1 = notch_filter(ydata_morphology)
    ydata_denoise2 = advanced_wavelet_denoise(ydata_denoise1)[:len(xdata)]
    ydata_final = moving_ave(savgolfil(ydata_denoise2, window_length, kvalue), windowsize)

    dominant_frequency1, positive_freqs1, positive_magnitude1 = vibration_detect(np.array(ydata_denoise2))
    dominant_frequency2, positive_freqs2, positive_magnitude2 = vibration_detect(np.array(ydata_final))

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(xdata, ydata)
    ax1.set_title('Origin')

    ax11 = fig.add_subplot(gs[0, 1])
    ax11.plot(xdata, ydata_denoise2 - np.mean(ydata_denoise2))
    ax11.set_title("Denoised2 (DC removed)")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(xdata, ydata_denoise2)
    ax2.set_title('Denoised Step 2')

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(xdata, ydata_final)
    ax3.set_title('Final Denoised')

    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(positive_freqs1, positive_magnitude1, label=f"{dominant_frequency1:.2f} Hz")
    ax4.set_title('FFT of Step 2')
    ax4.legend()

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(positive_freqs2, positive_magnitude2, label=f"{dominant_frequency2:.2f} Hz")
    ax5.set_title('FFT of Final')
    ax5.legend()

    fig.tight_layout()
    return fig


def load_and_plot():
    filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not filename:
        return

    try:
        data = pd.read_csv(filename)
        xdata = data.iloc[:, 0]
        ydata = data.iloc[:, 1]

        fig = plot_all(xdata, ydata, filename)
        canvas.figure = fig
        canvas.draw()

        label_filename.config(text=f"Loaded: {os.path.basename(filename)}")
    except Exception as e:
        label_filename.config(text=f"Error: {str(e)}")


root = tk.Tk()
root.title("CSV Signal Viewer")

frame_top = tk.Frame(root)
frame_top.pack()

btn_load = tk.Button(frame_top, text="Open CSV and Plot", command=load_and_plot)
btn_load.pack(side=tk.LEFT, padx=10, pady=10)

label_filename = tk.Label(frame_top, text="No file loaded.")
label_filename.pack(side=tk.LEFT, padx=10)

frame_plot = tk.Frame(root)
frame_plot.pack()

fig_placeholder = plt.figure(figsize=(12, 8))
canvas = FigureCanvasTkAgg(fig_placeholder, master=frame_plot)
canvas.get_tk_widget().pack()

root.mainloop()
