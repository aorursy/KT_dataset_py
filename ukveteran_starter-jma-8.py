import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from matplotlib import cm

sns.set_style('ticks')

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
KD= pd.read_csv("../input/all_kaggle_datasets.csv")

KD.head()
KD.head().T
corr_mat = KD.corr(method='pearson')

plt.figure(figsize=(20,10))

sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')
plt.matshow(KD.corr())

plt.colorbar()

plt.show()
_dfs = []

for datasetId, category_row in KD.set_index("datasetId")["categories"].iteritems():

    category_dict = eval(category_row)

    categories = category_dict["categories"]

    _df = pd.DataFrame(categories)

    _df["datasetId"] = datasetId 

    if category_dict["type"] != "dataset":

        print(category_dict["type"])

    _dfs.append(_df)

categories_df = pd.concat(_dfs, sort=False)
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

categories_df.groupby("name")["totalCount"].sum().nlargest(10).plot(kind='bar', ax=ax)

ax.set_ylabel("Total count")

ax.set_xlabel("Categories")
plt.matshow(categories_df .corr())

plt.colorbar()

plt.show()
corr_mat = categories_df.corr(method='pearson')

plt.figure(figsize=(20,10))

sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')