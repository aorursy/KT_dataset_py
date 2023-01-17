import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

%matplotlib inline

import plotly.plotly as py

import plotly.graph_objs as go
df = pd.read_csv('../input/en.yusufali.csv', dtype=object)
df.head(3)
df.tail(3)
df.describe()
cnt_srs = df['Surah'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('Surah', fontsize=12)

plt.ylabel('Number of Ayats', fontsize=12)

plt.title('No. of Ayats per Surah', fontsize=18)

plt.show()
for col in ['Surah', 'Ayah']:

    df[col] = pd.to_numeric(df[col])



def idx(i, j):

    df['index'] = df.index

    return int(df.loc[(df['Surah']==i) & (df['Ayah']==j), 'index'])



cut_points = [-1, idx(2,141), idx(2,252), idx(3,92), idx(4,23), idx(4,147), idx(5,81), idx(6,110), idx(7,87), idx(8,40),

             idx(9,92), idx(11,5), idx(12,52), idx(14,52), idx(16,128), idx(18,74), idx(20,135), idx(22,78), idx(25,20),

             idx(27,55), idx(29,45), idx(33,30), idx(36,27), idx(39,31), idx(41,46), idx(45,37), idx(51,30), idx(57,29),

             idx(66,12), idx(77,50), idx(114,6)]

label_names = [str(i) for i in range(1, len(cut_points))]



if 'Para' not in df.columns:

    df.insert(2, 'Para', pd.cut(df.index,cut_points,labels=label_names))

df.drop('index', axis=1, inplace=True)

df['Para'] = pd.to_numeric(df['Para'])

df.tail(9)
cnt_srs = df['Para'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('Para', fontsize=12)

plt.ylabel('Number of Ayats', fontsize=12)

plt.title('No. of Ayats per Para', fontsize=18)

plt.show()