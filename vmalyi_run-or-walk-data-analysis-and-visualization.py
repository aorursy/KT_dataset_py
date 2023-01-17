import warnings

warnings.filterwarnings('ignore')

import os

import shutil



import numpy as np

import pandas as pd



# Visualisation

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns
df = pd.read_csv('../input/dataset.csv')
df.info()
print('The dataset contains ' + str(df.shape[0]) + ' data samples and ' + str(df.shape[1]) + ' data columns')
df.isnull().sum()
df.describe()
print('Dataset contains ' + str(pd.value_counts(df['activity'].values)[0]) + ' "walk" data samples as well as ' + str(pd.value_counts(df['activity'].values)[1]) + ' "run" data samples')
print('The dataset contains ' + str(pd.value_counts(df['wrist'].values)[0]) + ' data samples collected on the left wrist as well as ' + str(pd.value_counts(df['wrist'].values)[1]) + ' data samples collected on the right wrist')
df.head(20)
# Wrist types

LEFT_WRIST = 0

RIGHT_WRIST = 1



# populate dataframe with 'walk' data only

df_walk_data = pd.DataFrame()

df_walk_data = df[(df.activity == 0)]



# populate dataframe with 'run' data only 

df_run_data = pd.DataFrame()

df_run_data = df[(df.activity == 1)]



walk_data_left_wrist_count = pd.value_counts(df_walk_data['wrist'].values, sort=False)[LEFT_WRIST]

walk_data_right_wrist_count = pd.value_counts(df_walk_data['wrist'].values, sort=False)[RIGHT_WRIST]



run_data_left_wrist_count = pd.value_counts(df_run_data['wrist'].values, sort=False)[LEFT_WRIST]

run_data_right_wrist_count = pd.value_counts(df_run_data['wrist'].values, sort=False)[RIGHT_WRIST]



print('Total number of "walk" data samples: ' + str(len(df_walk_data)))

print('    Number of left wrist samples: ' + str(walk_data_left_wrist_count))

print('    Number of right wrist samples: ' + str(walk_data_right_wrist_count))

print('Total number of "run" data samples: ' + str(len(df_run_data)))

print('    Number of left wrist samples: ' + str(run_data_left_wrist_count))

print('    Number of right wrist samples: ' + str(run_data_right_wrist_count))
SENSOR_DATA_COLUMNS = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z']



# populate dataframe with 'left' wrist only

df_left_wrist_data = pd.DataFrame()

df_left_wrist_data = df[df.wrist == 0]



# populate dataframe with 'right' wrist only

df_right_wrist_data = pd.DataFrame()

df_right_wrist_data = df[df.wrist == 1]
for c in SENSOR_DATA_COLUMNS:

    plt.figure(figsize=(10,5))

    plt.title("Sensor data distribution for both wrists")

    sns.distplot(df_left_wrist_data[c], label='left')

    sns.distplot(df_right_wrist_data[c], label='right')

    plt.legend()

    plt.show()