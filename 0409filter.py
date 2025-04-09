import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

# 读取数据
df = pd.read_csv('voltage.csv')
time = df['time'].values
raw_signal = df['voltage'].values
fs = 1 / (time[1] - time[0])  # 计算采样率（约500Hz）

# 滤波器设计函数
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandstop(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return b, a

# 分步滤波处理
def process_signal(signal, fs):
    # Step1: 去除基线漂移（高通0.5Hz）
    b, a = butter_bandpass(0.5, 40, fs, order=4)
    filtered = filtfilt(b, a, signal)
    
    # Step2: 消除工频干扰（带阻48-52Hz）
    b, a = butter_bandstop(48, 52, fs, order=2)
    filtered = filtfilt(b, a, filtered)
    
    # Step3: 平滑处理（可选移动平均）
    # filtered = np.convolve(filtered, np.ones(5)/5, mode='same')
    
    return filtered

# 应用滤波
processed_signal = process_signal(raw_signal, fs)

# 保存结果（示例）
df['processed'] = processed_signal
df.to_csv('processed_voltage.csv', index=False)