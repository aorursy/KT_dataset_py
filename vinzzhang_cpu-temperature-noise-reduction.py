import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter
cpu_data = pd.read_csv('../input/sysinfo.csv')
cpu_data.head()
cpu_data['timestamp'] = pd.to_datetime(cpu_data['timestamp'])
plt.figure(figsize=(12, 4))
plt.plot(cpu_data['timestamp'], cpu_data['temperature'], 'b.', alpha=0.5)
#Loess_smooth: 
loess_smoothed = lowess(cpu_data['temperature'], cpu_data['timestamp'], 
                        frac = 0.02)
plt.plot(cpu_data['timestamp'], loess_smoothed[:, 1], 'r-')
#Kalman_smooth
kalman_data = cpu_data[['temperature', 'cpu_percent']]
initial_state = kalman_data.iloc[0]
initial = np.array(initial_state)
observation_covariance = [[1 ** 2, 0], [0, 2 ** 2]]
transition_covariance = [[0.15 ** 2, 0], [0, 80 ** 2]]
initial_state_covariance = observation_covariance
transition_matrices = [[1, 0], [0.125, 1]]


kf = KalmanFilter(initial_state_covariance, 
                  observation_covariance,
                  transition_covariance,
                  transition_matrices
)
kalman_smoothed, _ = kf.smooth(kalman_data)
plt.figure(figsize=(12, 4))
plt.plot(cpu_data['timestamp'], kalman_smoothed[:, 0], 'g-')
plt.plot(cpu_data['timestamp'], cpu_data['temperature'], 'b.', alpha=0.5)

plt.show()

