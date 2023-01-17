from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/professor-salaries/Salaries.csv')

dat.head()
df = dat.rename({'yrs.since.phd': 'yrssincephd', 'yrs.service': 'yrsservice'}, axis=1)
sns.regplot(x=df['yrssincephd'], y=df['yrsservice'])
sns.lmplot(x="yrssincephd", y="yrsservice", hue="rank", data=df);
sns.lmplot(x="yrssincephd", y="yrsservice", hue="sex", data=df);