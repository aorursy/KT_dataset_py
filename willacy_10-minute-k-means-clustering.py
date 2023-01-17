# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/big-five-personality-test/IPIP-FFM-data-8Nov2018/data-final.csv", delimiter='\t')
df.head()
df.shape
cols = []

for i in ["EXT","EST","AGR","CSN","OPN"]:

    for j in range(10):

        cols.append(i+str(j+1))
df2 = df.loc[:,cols]

df2 = df2.dropna()
scaler = StandardScaler()

scaled_data = scaler.fit_transform(df2)



kmeans = KMeans(n_clusters=5)



y = kmeans.fit_predict(scaled_data)



df2['Cluster'] = y


sns.countplot(x='Cluster', data=df2)