import numpy as np
from scipy import signal
import pandas as pd

# 读取数据
df = pd.read_csv("voltage.csv")
t = df['time'].values
v = df['voltage'].values

# 1. 基线校正
window_size = 500  # 对应1秒窗口（采样率500Hz）
poly_order = 3

baseline = []
for i in range(len(v)):
    start = max(0, i - window_size//2)
    end = min(len(v), i + window_size//2)
    coeffs = np.polyfit(t[start:end], v[start:end], poly_order)
    baseline.append(np.polyval(coeffs, t[i]))
v_corrected = v - baseline

# 2. 低通滤波
fs = 1/(t[1]-t[0])  # 计算采样率（约500Hz）
cutoff = 30  # 截止频率
sos = signal.butter(4, cutoff, 'lowpass', fs=fs, output='sos')
v_filtered = signal.sosfiltfilt(sos, v_corrected)

# 结果保存
#df_processed = pd.DataFrame({'time': t, 'voltage': v_filtered})
df['voltage'] = v_filtered
df.to_csv('processed_voltage2.csv', index=False)