# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/liverpool-ion-switching/train.csv")

df.head()
import gc

import time

import math

from numba import jit

from math import log, floor



import numpy as np

import pandas as pd

from pathlib import Path



import seaborn as sns

from matplotlib import colors

import matplotlib.pyplot as plt

from matplotlib.colors import Normalize



import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



import pywt

from statsmodels.robust import mad



import scipy

from scipy import signal

from scipy.signal import butter, deconvolve
plt.figure(figsize=(10,5))

plt.plot(df.time, df.signal)

plt.plot(df.time, df.open_channels,alpha=0.7)

plt.show()
train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')

plt.figure(figsize=(10, 5))

plt.plot(train['time'],train['signal'],color='r')

plt.title('signal data', fontsize=20)

plt.xlabel('time', fontsize=12)

plt.ylabel('signal', fontsize=12)

plt.show()
train.corr()
test.corr()
def add_batching_to_data(df : pd.DataFrame) -> pd.DataFrame :

    batches = df.shape[0] // 500000

    df['batch'] = 0

    for i in range(batches):

        idx = np.arange(i*500000, (i+1)*500000)

        df.loc[idx, 'batch'] = i + 1

    return df



def p5( x : pd.Series) -> pd.Series : return x.quantile(0.05)

def p95(x : pd.Series) -> pd.Series : return x.quantile(0.95)
train = add_batching_to_data(train)
train.groupby('batch')[['signal','open_channels']].agg(['min', 'max', 'median', p5, p95])
train.groupby('open_channels')[['signal','batch']].agg(['min', 'max', 'median', p5, p95])
train.groupby(['batch','open_channels'])[['signal']].agg(['min', 'max', 'median', p5, p95])
partial = train.iloc[::250, :]

partial.signal = np.round(partial.signal.values, 2)

partial['shifted_signal'] = (partial.signal.values + 10) ** 2

fig = px.scatter(partial, x='signal', y='open_channels', color='open_channels',size='shifted_signal',  title='Signal vs Channels')

fig.show()
fig = make_subplots(rows=5, cols=2,  subplot_titles=[f'Batch no {i+1}' for i in range(10)])

i = 1

for row in range(1, 6):

    for col in range(1, 3):

        data = train[train.batch==i]['open_channels'].value_counts(sort=False).values

        fig.add_trace(go.Bar(x=list(range(11)), y=data), row=row, col=col)       

        i += 1

fig.update_layout(width=800, height=1500, title_text="Target for each batch", showlegend=False)