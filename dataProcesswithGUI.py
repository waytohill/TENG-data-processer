import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy
from scipy import signal
from scipy.ndimage import grey_closing, grey_opening
from scipy.signal import iirnotch, filtfilt
from scipy.fft import fft, fftfreq
import pywt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
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

# 2-stage filter based on morphology
def morphology_baseline_removal(f0, B1_width, B2_width):

    f0 = np.asarray(f0).flatten()

    B1 = _get_struct_element(B1_width)
    f1 = _first_stage_filter(f0, B1)

    B2 = _get_struct_element(B2_width)
    f2 = _second_stage_filter(f1, B2)

    f_clean = f0 - f2
    return f_clean, f1, f2

def _get_struct_element(width):
    return np.ones((int(width),), dtype=np.uint8)

def _first_stage_filter(f0, B):
    open_close = grey_closing(grey_opening(f0, structure=B), structure=B)
    close_open = grey_opening(grey_closing(f0, structure=B), structure=B)
    return (open_close + close_open) / 2

def _second_stage_filter(f1, B):
    f1 = np.asarray(f1).flatten()
    open_close = grey_closing(grey_opening(f1, structure=B), structure=B)
    close_open = grey_opening(grey_closing(f1, structure=B), structure=B)
    return (open_close + close_open) / 2

def auto_select_widths(signal, min_pulse_width=20):
    peaks, _ = scipy.signal.find_peaks(signal, prominence=np.std(signal)/2)
    Wp = int(np.median(np.diff(peaks))) if len(peaks)>1 else min_pulse_width

    B1_width = int(0.8 * Wp)
    B2_width = int(1.2 * Wp)

    return max(5, B1_width), max(B2_width, B1_width + 10)


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
    gs = gridspec.GridSpec(4, 2)

    baseline_morphology = morphology_baseline(ydata)
    ydata_morphology = -(ydata - baseline_morphology)
    ydata_denoise1 = notch_filter(ydata_morphology)
    ydata_denoise2 = advanced_wavelet_denoise(ydata_denoise1)[:len(xdata)]
    ydata_final = moving_ave(savgolfil(ydata_denoise2, window_length, kvalue), windowsize)
    
    B1_width, B2_width = auto_select_widths(ydata)
    
    ydata_2stage_filt, f1, f2 = morphology_baseline_removal(ydata, B1_width, B2_width)
    
    ydata_2stage_filt_denoise1 = notch_filter(ydata_2stage_filt)
    ydata_2stage_filt_denoise2 = advanced_wavelet_denoise(ydata_2stage_filt_denoise1)[:len(xdata)]
    ydata_2stage_filt_denoise_final = moving_ave(savgolfil(ydata_2stage_filt_denoise2, window_length, kvalue), windowsize)




    
    dominant_frequency1, positive_freqs1, positive_magnitude1 = vibration_detect(np.array(ydata_final))
    dominant_frequency2, positive_freqs2, positive_magnitude2 = vibration_detect(np.array(ydata_2stage_filt_denoise_final))

    ax00 = fig.add_subplot(gs[0, 0])
    ax00.plot(xdata, ydata)
    ax00.set_title('Origin')

    ax10 = fig.add_subplot(gs[1, 0])
    ax10.plot(xdata, ydata_morphology)
    ax10.set_title("1-stage morphology")

    ax11 = fig.add_subplot(gs[1, 1])
    ax11.plot(xdata, ydata_2stage_filt)
    ax11.set_title('2-stage morphology')

    ax20 = fig.add_subplot(gs[2, 0])
    ax20.plot(xdata, ydata_final)
    ax20.set_title('Final of 1-stage')

    ax21 = fig.add_subplot(gs[2, 1])
    ax21.plot(xdata, ydata_2stage_filt_denoise_final)
    ax21.set_title('Final of 2-stage')

    ax30 = fig.add_subplot(gs[3, 0])
    ax30.plot(positive_freqs1, positive_magnitude1, label=f"{dominant_frequency1:.2f} Hz")
    ax30.set_title('FFT of 1-Stage Filt')
    ax30.legend()

    ax31 = fig.add_subplot(gs[3, 1])
    ax31.plot(positive_freqs2, positive_magnitude2, label=f"{dominant_frequency2:.2f} Hz")
    ax31.set_title('FFT of 2-Stage Filt')
    ax31.legend()

    fig.tight_layout()
    return fig

def bind_interaction(canvas):
    drag_data = {"x": 0, "y": 0, "is_dragging": False}

    def on_scroll(event):
        ax = event.inaxes
        if ax is None: return
        scale_factor = 1.2 if event.button == 'up' else 0.8
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata
        new_xlim = [xdata + (x - xdata) * scale_factor for x in xlim]
        new_ylim = [ydata + (y - ydata) * scale_factor for y in ylim]
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        canvas.draw()

    def on_press(event):
        if event.button == 3:
            drag_data["x"] = event.x
            drag_data["y"] = event.y
            drag_data["is_dragging"] = True

    def on_release(event):
        drag_data["is_dragging"] = False

    def on_motion(event):
        if drag_data["is_dragging"] and event.inaxes:
            dx = event.x - drag_data["x"]
            dy = event.y - drag_data["y"]
            ax = event.inaxes
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            dx_unit = dx * (xlim[1] - xlim[0]) / canvas.get_width()
            dy_unit = dy * (ylim[1] - ylim[0]) / canvas.get_height()
            ax.set_xlim(xlim[0] - dx_unit, xlim[1] - dx_unit)
            ax.set_ylim(ylim[0] + dy_unit, ylim[1] + dy_unit)
            drag_data["x"] = event.x
            drag_data["y"] = event.y
            canvas.draw()

    def on_double_click(event):
        if event.dblclick and event.inaxes:
            ax = event.inaxes
            ax.autoscale()
            canvas.draw()

    canvas.mpl_connect("scroll_event", on_scroll)
    canvas.mpl_connect("button_press_event", on_press)
    canvas.mpl_connect("button_release_event", on_release)
    canvas.mpl_connect("motion_notify_event", on_motion)
    canvas.mpl_connect("button_press_event", on_double_click)

def load_and_plot():
    global canvas, toolbar

    filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not filename:
        return

    try:
        data = pd.read_csv(filename)
        xdata = data.iloc[:, 0]
        ydata = data.iloc[:, 1]

        fig = plot_all(xdata, ydata, filename)

        # 移除旧图像并重新构建 canvas
        canvas.get_tk_widget().pack_forget()
        toolbar.pack_forget()

        canvas = FigureCanvasTkAgg(fig, master=frame_plot)
        toolbar = NavigationToolbar2Tk(canvas, frame_plot, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)

        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        bind_interaction(canvas)

        label_filename.config(text=f"Loaded: {os.path.basename(filename)}")
    except Exception as e:
        label_filename.config(text=f"Error: {str(e)}")

# GUI 初始化
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
toolbar = NavigationToolbar2Tk(canvas, frame_plot, pack_toolbar=False)
toolbar.update()
toolbar.pack(side=tk.TOP, fill=tk.X)

canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
canvas.draw()

bind_interaction(canvas)

root.mainloop()
