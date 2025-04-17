import numpy as np
import pandas as pd
import pywt
from tkinter import filedialog
import os
import tkinter as tk
from scipy.stats import kurtosis, skew, linregress
from scipy.signal import welch

import matplotlib.pyplot as plt

def openfile():
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename()
    root.destroy()
    return filename


def signal_stability_analysis(signal, fs, num_segments=5):
    """分析信号的长期稳定性
    参数:
    - signal: 输入信号
    - fs: 采样率
    - num_segments: 分段数量
    返回:
    - stability_metrics: 包含稳定性指标的字典
    """
    metrics = {}
    
    # 分段稳定性分析
    segment_length = len(signal) // num_segments
    means, stds = [], []
    
    for i in range(num_segments):
        seg = signal[i*segment_length : (i+1)*segment_length]
        means.append(np.mean(seg))
        stds.append(np.std(seg))
    
    metrics['segment_mean_std'] = np.std(means)  # 均值波动性
    metrics['segment_std_std'] = np.std(stds)    # 标准差波动性
    metrics['mean_cv'] = np.std(means)/np.mean(means)  # 均值变异系数
    
    # 趋势分析（使用线性回归）
    time = np.arange(len(signal)) / fs
    slope, _, _, _, _ = linregress(time, signal)
    metrics['trend_slope'] = slope  # 趋势斜率 (V/s)
    
    # 滑动窗口变异系数（窗口大小1秒）
    window_size = int(fs)
    if window_size > 0 and len(signal) > window_size:
        cv_values = []
        for i in range(0, len(signal)-window_size, window_size//2):
            window = signal[i:i+window_size]
            cv = np.std(window) / np.mean(window)
            cv_values.append(cv)
        metrics['cv_std'] = np.std(cv_values)  # CV波动性
    else:
        metrics['cv_std'] = np.nan
        
    return metrics


# 新增：时域分析函数
def time_domain_analysis(signal):
    """计算时域指标"""
    metrics = {}
    # 统计特征
    metrics['mean'] = np.mean(signal)
    metrics['std'] = np.std(signal)
    metrics['cv'] = metrics['std'] / metrics['mean']  # 变异系数
    # 基线漂移（使用高通滤波法）
    baseline = np.mean(signal)
    metrics['baseline_drift'] = np.sqrt(np.mean((signal - baseline)**2))  # RMS
    return metrics

# 新增：频域分析函数
def frequency_domain_analysis(signal, fs):
    """计算频域指标"""
    metrics = {}
    # 计算功率谱密度
    f, Pxx = welch(signal, fs, nperseg=1024)
    
    # 主频成分
    dominant_idx = np.argmax(Pxx)
    metrics['dominant_freq'] = f[dominant_idx]
    
    # 频带能量分布（假设关注0.1-10Hz）
    mask = (f >= 0.1) & (f <= 10)
    metrics['band_energy_ratio'] = np.trapz(Pxx[mask], f[mask]) / np.trapz(Pxx, f)
    
    # 总谐波失真（简化版）
    harmonics_mask = (f > metrics['dominant_freq']*1.1) & (f <= 10)
    metrics['thd'] = np.trapz(Pxx[harmonics_mask], f[harmonics_mask]) / np.trapz(Pxx, f)
    
    return metrics

def calculate_snr(signal, noise):
    """计算信噪比（SNR）"""
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)
    return snr_linear, snr_db

def wavelet_denoise(signal, wavelet='db4', level=1):
    """小波降噪"""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    denoised_signal = pywt.waverec(coeffs, wavelet)
    return denoised_signal[:len(signal)]

# 主程序
if __name__ == "__main__":
    # 读取数据
    csv_file_path = openfile()
    data = pd.read_csv(csv_file_path)
    
    
    time = data.iloc[:, 0].values
    original_signal = data.iloc[:, 1].values
    
    # 计算采样率
    #dt = np.mean(np.diff(time))
    #fs = 1 / dt if dt != 0 else 500  # 默认值
    fs = 500

    # 信号预处理
    pure_signal = wavelet_denoise(original_signal, wavelet='db4', level=3)
    noise = original_signal - pure_signal
    
     
    
    # 执行各项分析
    time_metrics = time_domain_analysis(pure_signal)
    freq_metrics = frequency_domain_analysis(pure_signal, fs)
    stability_metrics = signal_stability_analysis(pure_signal, fs, num_segments=5)
    snr_linear, snr_db = calculate_snr(pure_signal, noise)

    
    # 输出结果
    print("\n=== 时域分析 ===")
    print(f"均值: {time_metrics['mean']:.4f}")
    print(f"标准差: {time_metrics['std']:.4f}")
    print(f"变异系数: {time_metrics['cv']:.4f}")
    print(f"基线漂移 (RMS): {time_metrics['baseline_drift']:.4f}")
    
    print("\n=== 频域分析 ===")
    print(f"主频 (Hz): {freq_metrics['dominant_freq']:.2f}")
    print(f"有效频带能量占比: {freq_metrics['band_energy_ratio']*100:.2f}%")
    print(f"谐波失真指数: {freq_metrics['thd']*100:.2f}%")
    
    print("\n=== 其他指标 ===")
    print(f"信噪比 (dB): {snr_db:.2f}")
    print(f"峰度: {kurtosis(original_signal, fisher=True):.2f}")
    print(f"偏度: {skew(original_signal):.2f}")

    print("\n=== 信号稳定性分析 ===")
    print(f"分段均值波动性: {stability_metrics['segment_mean_std']:.4f} V")
    print(f"分段标准差波动性: {stability_metrics['segment_std_std']:.4f} V")
    print(f"长期趋势斜率: {stability_metrics['trend_slope']:.6f} V/s")
    print(f"滑动窗口CV波动性: {stability_metrics['cv_std']:.4f}")

    plt.figure(figsize=(12,6))
    plt.plot(pure_signal)
    plt.title("Denoised Signal with Stability Analysis")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude (V)")
    plt.show()