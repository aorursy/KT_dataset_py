from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/goldencheetah-metadata/metadata.csv')

dat.head()
df = dat.rename({'10m_critical_power': '10mcrit', '10s_critical_power': '10scrit'}, axis=1)
sns.regplot(x=df['10mcrit'], y=df['10scrit'])
sns.lmplot(x="10mcrit", y="10scrit", hue="slope", data=df);
sns.lmplot(x="10mcrit", y="10scrit", hue="altitude", data=df);