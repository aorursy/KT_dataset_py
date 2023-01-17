# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.datasets import load_breast_cancer

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

import os

import seaborn as sns



# Any results you write to the current directory are saved as output.
dataset = load_breast_cancer()

# description of dataset

print(dataset.DESCR)
# show the keys of dataset

print(dataset.keys())

print(dataset.feature_names)

# show the shape of dataset 

display(dataset.data.shape)

# define the classification  

display(dataset.target_names)
# Convert matriz in dataframe using pandas to select features

df_features = pd.DataFrame(dataset.data,columns=dataset.feature_names)

df_features
#Convert matriz in dataframe with target values using pandas

df_target = pd.DataFrame(dataset.target,columns=["cancer"])

df_target

# concat both dataframes

df = pd.concat([df_features, df_target],axis=1)

df.head()
# identify a correlation beetwen the features

df_cor = df.corr()

df_cor
# plot the correlation beetwen the features

fig, ax = plt.subplots(figsize=(12,10))

sns.heatmap(df_cor,annot=True)
## Other datasets to explore

from sklearn.datasets import fetch_20newsgroups

fetch_20train = fetch_20newsgroups(subset='train')

fetch_20train.keys()
print(fetch_20train.DESCR)