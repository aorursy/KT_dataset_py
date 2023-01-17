import os

import math



import numpy as np

import pandas as pd

from sklearn import metrics

import plotly.graph_objs as go

import plotly.express as px
df = pd.read_csv("../input/liverpool-ion-switching/train.csv")

train = df.copy()



n_groups = df.shape[0] // 50000

df["group"] = 0

for i in range(n_groups):

    ids = np.arange(i*50000, (i+1)*50000)

    df.loc[ids,"group"] = i



for i in range(n_groups):

    sub = df[df.group == i]

    signals = sub.signal.values

    imax, imin = math.floor(np.max(signals)), math.ceil(np.min(signals))

    signals = (signals - np.min(signals))/(np.max(signals) - np.min(signals))

    signals = signals*(imax-imin)

    df.loc[sub.index,"pred_open_channels"] = np.array(signals,np.int)



y_true = df.open_channels.values

y_pred = df.pred_open_channels.values
%%time

cm = metrics.confusion_matrix(y_true, y_pred, normalize='true')
fig = px.imshow(cm)

fig.show()
%time report = metrics.classification_report(y_true, y_pred)
print(report)
%%time

lwk = metrics.cohen_kappa_score(y_true, y_pred, weights='linear')

qwk = metrics.cohen_kappa_score(y_true, y_pred, weights='quadratic')\



print("Linear Weighted Kappa Score:", lwk)

print("Quadratic Weighted Kappa Score:", qwk)
true_bins = np.bincount(y_true)

pred_bins = np.bincount(y_pred.astype(int))[:10]
fig = go.Figure([

    go.Bar(y=true_bins, name='True Labels'),

    go.Bar(y=pred_bins, name='Pred Labels')

])



fig.show()