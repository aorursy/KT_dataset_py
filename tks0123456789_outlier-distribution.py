import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_hdf('../input/train.h5')
y_mean_by_time = df.groupby('timestamp').y.mean()

y_mean_by_time.plot(figsize=(12, 6))
y_mean_by_time.rolling(window=10).std().plot(figsize=(12, 6))

plt.axhline(y=0.009, color='red')

plt.axvspan(0, 905, color='green', alpha=0.1)

plt.axvspan(906, 1505, color='red', alpha=0.1)

plt.text(330, 0.015, "Training", fontsize=20)

plt.text(1050, 0.015, "Validation", fontsize=20)