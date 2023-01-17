# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import necessary libraries
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('/kaggle/input/cee-498-project12-earth-system-model/train.csv')
df.head()
df.shape
corr=df.corrwith(df.TSA).abs().sort_values(ascending=False).dropna()
corr[:20]
#Choose features that are empirically most related to temperature
df_features=df[["time","lat","lon","TSA","FSDS","FLDS","RAIN","TBOT","PBOT","QBOT","U10"]]
df_features.head()
start=df_features['time'].min()
end=df_features['time'].max()
count=np.count_nonzero(df_features['time'].unique())
print(f'The prediction is monthly mean from {start} to {end}, with {count} months.')
df_features.shape
df_features.info()
df_features.describe()
print(df_features[df_features.TSA == 0].time.unique())
print(df_features[df_features.TBOT == 0].time.unique())
df_features_dropped=df_features[df_features.TSA> 1] 
#0 instead of 1 is just to make sure that the machine does not mistakenly reconginze values that are very close but not exactly 0 as 0.
df_features_dropped.describe()
df_features_dropped.corr()
df_reindex = df_features_dropped.set_index(['time', 'lat', 'lon'])
df_reindex.head()
xr_full=df_reindex.to_xarray()
xr_full
xr_full['TSA'].shape
print(np.count_nonzero(df_features.lat.unique()),np.count_nonzero(df_features.lon.unique()))
corr_matrix = df_reindex.corr()
sns.heatmap(corr_matrix.abs(),cmap='viridis')
sns.pairplot(df_reindex)
variables=['TSA','FSDS','FLDS','RAIN','TBOT','PBOT','QBOT','U10']
fig,axes = plt.subplots(nrows=3, ncols=3,figsize=(24,16))
for i in range(len(variables)):
    xr_full[variables[i]][0].plot(ax=axes.flat[i],
                                  vmax=np.nanmax(xr_full[variables[i]][0]),
                                  vmin=np.nanmin(xr_full[variables[i]][0]))
fig.delaxes(axes[2][2])
plt.subplots_adjust(hspace=0.3)
variables=['TSA','FSDS','FLDS','RAIN','TBOT','PBOT','QBOT','U10']
fig,axes = plt.subplots(nrows=3, ncols=3,figsize=(24,16))
for i in range(len(variables)):
    xr_full[variables[i]][6].plot(ax=axes.flat[i],
                                  vmax=np.nanmax(xr_full[variables[i]][0]),
                                  vmin=np.nanmin(xr_full[variables[i]][0]))
fig.delaxes(axes[2][2])
plt.subplots_adjust(hspace=0.3)
fig,axes = plt.subplots(nrows=3, ncols=3,figsize=(24,16))
for i in range(len(variables)):
    plot_data = df_reindex[variables[i]].dropna()
    sns.boxplot(plot_data,ax=axes.flat[i])
fig.delaxes(axes[2][2])
plt.subplots_adjust(hspace=0.3)
fig,axes = plt.subplots(nrows=3, ncols=3,figsize=(24,16))
t=np.linspace(2015,2096,106)
for i in range(len(variables)):
    axes.flat[i].plot(t,df_reindex[variables[i]].groupby(by=['time']).mean())
    axes.flat[i].set_xlabel(variables[i])
fig.delaxes(axes[2][2])
plt.subplots_adjust(hspace=0.3)