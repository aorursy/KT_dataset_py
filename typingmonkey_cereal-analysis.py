# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/cereal.csv', index_col=0)
df.head()
numeric_vars = df.mean().index[:-4] # Select only relevant composition variables
df_std = df[numeric_vars].copy()
df_std[numeric_vars] = (df_std[numeric_vars] - df_std[numeric_vars].mean()) / df_std[numeric_vars].std()
df_std.head()
corr_mat = np.corrcoef(df_std[numeric_vars].T)
u, s, v = np.linalg.svd(corr_mat)
factor_loadings = u * np.sqrt(s)
factor_scores = np.dot(df_std[numeric_vars].values, factor_loadings)
factor_scores.shape
[(s[:(i+1)].sum() / s.sum()).round(3) for i in range(len(s))]
data = []
for t in df['mfr'].unique():
    # Create a trace
    data.append(go.Scatter(
        x = factor_scores[df['mfr'] == t, 0],
        y = factor_scores[df['mfr'] == t, 1],
        text=df.loc[df['mfr'] == t].index,
        mode = 'markers',
        name=t))

layout = go.Layout(
    title='Factor scores',
    xaxis=dict(
        title='F1',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='F2',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),)
# Plot and embed in ipython notebook!
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='factor-scores')
