# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/movies.csv')
df.columns
df.shape
df = df.drop_duplicates()
df.shape
df.head()
df = df.drop(columns=["id", "imdb_id", "homepage", "overview", "tagline"])
df.columns
collist = list()

for col in df.columns:

    if df[df[col].isnull()].shape[0]>0 and df[df[col].isnull()].shape[0]<=100:

        collist.append(col)
collist
df = df.drop(columns=collist)
df.columns
for col in df.columns:

    print(df[col].dropna().shape[0])
df.shape
for col in df.columns:

    df = df[df[col]!=0]
df.shape
df.dtypes
df.release_date = pd.to_datetime(df.release_date)
df2 = pd.DataFrame(df.release_year.value_counts())
df2 = df2.reset_index()
df2.head()
plt.bar(df2["index"],df2["release_year"])

plt.show()
df3 = pd.DataFrame(df.director.value_counts())
df3 = df3.reset_index()
df3.head(5)
df.sort_values(by="runtime", ascending=False, inplace=True)
df4 = df.head(100)
df4[df4.vote_average >= 5.0].shape[0]*100/df4.shape[0]