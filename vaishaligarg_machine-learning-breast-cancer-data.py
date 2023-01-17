import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv("../input/data.csv")

df.shape #(569,33)

df.head()
df['diagnosis'].value_counts()
df.drop('id',axis=1,inplace=True)

df.drop('Unnamed: 32',axis=1,inplace=True)
#Map diagnosis columns to mask the data

df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
sns.heatmap(df.corr());
dfM=df[df['diagnosis'] ==1]

dfB=df[df['diagnosis'] ==0]



col_list=list(df.columns[1:])
fig, axes = plt.subplots(nrows=15, ncols=2, figsize=(8,35))

axes = axes.ravel()

for idx,ax in enumerate(axes):

    ax.figure

    binwidth= (max(df[col_list[idx]]) - min(df[col_list[idx]]))/50

    bins = np.arange(min(df[col_list[idx]]), max(dfM[col_list[idx]]) + binwidth, binwidth)

    ax.hist([dfM[col_list[idx]],dfB[col_list[idx]]], bins = bins, alpha = 0.6,stacked=True,label=['M','B'],color=['red','green'])

    ax.legend()

    ax.set_title(col_list[idx])

plt.tight_layout()

plt.show()
x = df[1:]

from sklearn.decomposition import PCA

pca = PCA(n_components=3)

pca.fit(x)

print(pca.explained_variance_ratio_)