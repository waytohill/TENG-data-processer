# TENG Data Processer

This is a Data Processer to process the data of TENG (Triboelectric Nanogenerator) and it is under construction.

数据处理流程：去除基线漂移->去噪->平滑滤波

### 去除基线漂移

```python
# dataprocessor.py
def morphology_baseline(signal, structure_size=100)
```

### 去噪

```python
# dataprocessor.py
# Step 1
def notch_filter(signal, fs=500, freq=50, Q=30)
# Step 2
def advanced_wavelet_denoise(signal, wavelet='db8', level=5)
```

### 平滑滤波

```python
# dataprocessor.py
# Step 1
def savgolfil(signal, windowlen, kvalue)
# Step 2
def moving_ave(signal, windowsize)
```



