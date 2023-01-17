# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

import plotly.figure_factory as ff

import warnings

from scipy import stats



warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel("/kaggle/input/Assignment.xlsx")

del df['Unnamed: 0']

df.columns = df.iloc[0]

df = df.drop(df.index[0])

df = df.reset_index()

del df['index']
df['datetime'] = pd.to_datetime(df['datetime'])

df = df.set_index(df['datetime'])
# percentage of data loss if I drop the null values

str((1-df.dropna().shape[0]/df.shape[0])*100)+"%"
df = df.infer_objects()

df = df.fillna(df.mean())

del df['datetime']
df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
# Add histogram data

x1 = df['t1'].as_matrix().astype(float)

x2 = df['t2'].as_matrix().astype(float)

x3 = df['t3'].as_matrix().astype(float)

x4 = df['t4'].as_matrix().astype(float)



# Group data together

hist_data = [x1, x2, x3, x4]



group_labels = ['Refrigerator 1', 'Refrigerator 2', 'Refrigerator 3', 'Refrigerator 4']



# Create distplot with custom bin_size

fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)

fig.show()
px.box(df,y='t1',points="all")
px.box(df,y='t2',points="all")
px.box(df,y='t3',points="all")
px.box(df,y='t4',points="all")