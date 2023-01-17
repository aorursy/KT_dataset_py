import numpy as np
import pandas as pd
import seaborn as sns
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from scipy import signal

sns.set_style("whitegrid")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
signals = pq.read_table('../input/train.parquet', columns=[str(i) for i in range(999)]).to_pandas()
signals = signals.values.T.reshape((999//3, 3, 800000))
train_df = pd.read_csv('../input/metadata_train.csv')
train_df.head()
target = train_df['target'][::3]
target.value_counts()
def apply_convolution(sig, window):
    """Apply a simple same-size convolution with a given window size"""
    conv = np.repeat([0., 1., 0.], window)
    filtered = signal.convolve(sig, conv, mode='same') / window
    return filtered
plt.figure(figsize=(15, 10))
window = 10

for phase in range(3):
    sig = signals[0, phase, :]
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    convolved = apply_convolution(sig, window)
    plt.plot(convolved, label=f'Phase {phase} Convolved')

plt.legend()
plt.title(f"Applying convolutions - Window Size {window}", size=15)
plt.show()
plt.figure(figsize=(15, 10))
window = 100

for phase in range(3):
    sig = signals[0, phase, :]
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    convolved = apply_convolution(sig, window)
    plt.plot(convolved, label=f'Phase {phase} Convolved')

plt.legend()
plt.title(f"Applying convolutions - Window Size {window}", size=15)
plt.show()
plt.figure(figsize=(15, 10))
window = 1000

for phase in range(3):
    sig = signals[0, phase, :]
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    convolved = apply_convolution(sig, window)
    plt.plot(convolved, label=f'Phase {phase} Convolved')

plt.legend()
plt.title(f"Applying convolutions - Window Size {window}", size=15)
plt.show()
plt.figure(figsize=(15, 10))
window = 10

for phase in range(3):
    sig = signals[1, phase, :]
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    convolved = apply_convolution(sig, window)
    plt.plot(convolved, label=f'Phase {phase} Convolved')

plt.legend()
plt.title(f"Applying convolutions - Window Size {window}", size=15)
plt.show()
plt.figure(figsize=(15, 10))
window = 100

for phase in range(3):
    sig = signals[1, phase, :]
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    convolved = apply_convolution(sig, window)
    plt.plot(convolved, label=f'Phase {phase} Convolved')

plt.legend()
plt.title(f"Applying convolutions - Window Size {window}", size=15)
plt.show()
plt.figure(figsize=(15, 10))
window = 1000

for phase in range(3):
    sig = signals[1, phase, :]
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    convolved = apply_convolution(sig, window)
    plt.plot(convolved, label=f'Phase {phase} Convolved')

plt.legend()
plt.title(f"Applying convolutions - Window Size {window}", size=15)
plt.show()
%%time
smoothing = 0
plt.figure(figsize=(15, 10))

for phase in range(3):
    sig = signals[0, phase, :]
    filtered = signal.cspline1d(sig, smoothing)
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    plt.plot(filtered, label=f'Phase {phase} Filtered')

plt.legend()
plt.title(f"Applying Cubic Spline, Smoothing: {smoothing}", size=15)
plt.show()
smoothing = 1
plt.figure(figsize=(15, 10))

for phase in range(3):
    sig = signals[0, phase, :]
    filtered = signal.cspline1d(sig, smoothing)
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    plt.plot(filtered, label=f'Phase {phase} Filtered')

plt.legend()
plt.title(f"Applying Cubic Spline, Smoothing: {smoothing}", size=15)
plt.show()
smoothing = 10
plt.figure(figsize=(15, 10))

for phase in range(3):
    sig = signals[0, phase, :]
    filtered = signal.cspline1d(sig, smoothing)
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    plt.plot(filtered, label=f'Phase {phase} Filtered')

plt.legend()
plt.title(f"Applying Cubic Spline, Smoothing: {smoothing}", size=15)
plt.show()
smoothing = 0
plt.figure(figsize=(15, 10))

for phase in range(3):
    sig = signals[1, phase, :]
    filtered = signal.cspline1d(sig, smoothing)
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    plt.plot(filtered, label=f'Phase {phase} Filtered')

plt.legend()
plt.title(f"Applying Cubic Spline, Smoothing: {smoothing}", size=15)
plt.show()
smoothing = 1
plt.figure(figsize=(15, 10))

for phase in range(3):
    sig = signals[1, phase, :]
    filtered = signal.cspline1d(sig, smoothing)
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    plt.plot(filtered, label=f'Phase {phase} Filtered')

plt.legend()
plt.title(f"Applying Cubic Spline, Smoothing: {smoothing}", size=15)
plt.show()
smoothing = 10
plt.figure(figsize=(15, 10))

for phase in range(3):
    sig = signals[1, phase, :]
    filtered = signal.cspline1d(sig, smoothing)
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    plt.plot(filtered, label=f'Phase {phase} Filtered')

plt.legend()
plt.title(f"Applying Cubic Spline, Smoothing: {smoothing}", size=15)
plt.show()
%%time
# Start with negative target.
smoothing = 0
plt.figure(figsize=(15, 10))

for phase in range(3):
    sig = signals[1, phase, :]
    filtered = signal.qspline1d(sig, smoothing)
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    plt.plot(filtered, label=f'Phase {phase} Filtered')

plt.legend()
plt.title(f"Applying Quadratic Spline, Smoothing: {smoothing}", size=15)
plt.show()
smoothing = 0
plt.figure(figsize=(15, 10))

for phase in range(3):
    sig = signals[1, phase, :]
    filtered = signal.qspline1d(sig, smoothing)
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    plt.plot(filtered, label=f'Phase {phase} Filtered')

plt.legend()
plt.title(f"Applying Quadratic Spline, Smoothing: {smoothing}", size=15)
plt.show()
%%time
kernel_size = 1
plt.figure(figsize=(15, 10))

for phase in range(3):
    sig = signals[0, phase, :]
    filtered = signal.medfilt(sig, kernel_size)
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    plt.plot(filtered, label=f'Phase {phase} Filtered')

plt.legend()
plt.title(f"Applying Median Filters, Kernel Size: {kernel_size}", size=15)
plt.show()
kernel_size = 11
plt.figure(figsize=(15, 10))

for phase in range(3):
    sig = signals[0, phase, :]
    filtered = signal.medfilt(sig, kernel_size)
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    plt.plot(filtered, label=f'Phase {phase} Filtered')

plt.legend()
plt.title(f"Applying Median Filters, Kernel Size: {kernel_size}", size=15)
plt.show()
kernel_size = 51
plt.figure(figsize=(15, 10))

for phase in range(3):
    sig = signals[0, phase, :]
    filtered = signal.medfilt(sig, kernel_size)
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    plt.plot(filtered, label=f'Phase {phase} Filtered')

plt.legend()
plt.title(f"Applying Median Filters, Kernel Size: {kernel_size}", size=15)
plt.show()
kernel_size = 101
plt.figure(figsize=(15, 10))

for phase in range(3):
    sig = signals[0, phase, :]
    filtered = signal.medfilt(sig, kernel_size)
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    plt.plot(filtered, label=f'Phase {phase} Filtered')

plt.legend()
plt.title(f"Applying Median Filters, Kernel Size: {kernel_size}", size=15)
plt.show()
kernel_size = 1
plt.figure(figsize=(15, 10))

for phase in range(3):
    sig = signals[1, phase, :]
    filtered = signal.medfilt(sig, kernel_size)
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    plt.plot(filtered, label=f'Phase {phase} Filtered')

plt.legend()
plt.title(f"Applying Median Filters, Kernel Size: {kernel_size}", size=15)
plt.show()
kernel_size = 11
plt.figure(figsize=(15, 10))

for phase in range(3):
    sig = signals[1, phase, :]
    filtered = signal.medfilt(sig, kernel_size)
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    plt.plot(filtered, label=f'Phase {phase} Filtered')

plt.legend()
plt.title(f"Applying Median Filters, Kernel Size: {kernel_size}", size=15)
plt.show()
kernel_size = 51
plt.figure(figsize=(15, 10))

for phase in range(3):
    sig = signals[1, phase, :]
    filtered = signal.medfilt(sig, kernel_size)
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    plt.plot(filtered, label=f'Phase {phase} Filtered')

plt.legend()
plt.title(f"Applying Median Filters, Kernel Size: {kernel_size}", size=15)
plt.show()
kernel_size = 101
plt.figure(figsize=(15, 10))

for phase in range(3):
    sig = signals[1, phase, :]
    filtered = signal.medfilt(sig, kernel_size)
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    plt.plot(filtered, label=f'Phase {phase} Filtered')

plt.legend()
plt.title(f"Applying Median Filters, Kernel Size: {kernel_size}", size=15)
plt.show()
%%time
Wn = 0.50
plt.figure(figsize=(15, 10))

for phase in range(3):
    sig = signals[0, phase, :]
    
    b, a = signal.butter(3, Wn)
    filtered = signal.filtfilt(b, a, sig)
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    plt.plot(filtered, label=f'Phase {phase} Filtered')

plt.legend()
plt.title(f"Applying IIR Filtering with Butterworth, Wn: {Wn}", size=15)
plt.show()
Wn = 0.05
plt.figure(figsize=(15, 10))

for phase in range(3):
    sig = signals[0, phase, :]
    
    b, a = signal.butter(3, Wn)
    filtered = signal.filtfilt(b, a, sig)
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    plt.plot(filtered, label=f'Phase {phase} Filtered')

plt.legend()
plt.title(f"Applying IIR Filtering with Butterworth, Wn: {Wn}", size=15)
plt.show()
Wn = 0.01
plt.figure(figsize=(15, 10))

for phase in range(3):
    sig = signals[0, phase, :]
    
    b, a = signal.butter(3, Wn)
    filtered = signal.filtfilt(b, a, sig)
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    plt.plot(filtered, label=f'Phase {phase} Filtered')

plt.legend()
plt.title(f"Applying IIR Filtering with Butterworth, Wn: {Wn}", size=15)
plt.show()

%%time
Wn = 0.50
plt.figure(figsize=(15, 10))

for phase in range(3):
    sig = signals[1, phase, :]
    
    b, a = signal.butter(3, Wn)
    filtered = signal.filtfilt(b, a, sig)
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    plt.plot(filtered, label=f'Phase {phase} Filtered')

plt.legend()
plt.title(f"Applying IIR Filtering with Butterworth, Wn: {Wn}", size=15)
plt.show()
Wn = 0.05
plt.figure(figsize=(15, 10))

for phase in range(3):
    sig = signals[1, phase, :]
    
    b, a = signal.butter(3, Wn)
    filtered = signal.filtfilt(b, a, sig)
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    plt.plot(filtered, label=f'Phase {phase} Filtered')

plt.legend()
plt.title(f"Applying IIR Filtering with Butterworth, Wn: {Wn}", size=15)
plt.show()
Wn = 0.01
plt.figure(figsize=(15, 10))

for phase in range(3):
    sig = signals[1, phase, :]
    
    b, a = signal.butter(3, Wn)
    filtered = signal.filtfilt(b, a, sig)
    
    plt.plot(sig, label=f'Phase {phase} Raw')
    plt.plot(filtered, label=f'Phase {phase} Filtered')

plt.legend()
plt.title(f"Applying IIR Filtering with Butterworth, Wn: {Wn}", size=15)
plt.show()