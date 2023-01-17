# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as mplt

import seaborn as sn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



train = pd.read_csv("../input/train_2016.csv",

                    parse_dates=["transactiondate"],

                   date_parser=lambda x: pd.datetime.strptime(x, "%Y-%m-%d"))



# Any results you write to the current directory are saved as output.
train.head()
train.info()
props = pd.read_csv("../input/properties_2016.csv")
props.columns
df = pd.merge(train, props, how="left", on="parcelid")
df.shape
df = df.sort_values(by="parcelid")
df["year"] = df.transactiondate.map(lambda x: x.year)

df["year_month"] = df.transactiondate.map(lambda x: x.year * 100 + x.month)
df.parcelid.value_counts().sort_values(ascending=False).shape
split_date = pd.to_datetime('20161015', format="%Y%m%d")

df.transactiondate.min(), df.transactiondate.max()
ulim = np.percentile(df.logerror, 99)

llim = np.percentile(df.logerror, 1)

df["logerror"].ix[df.logerror > ulim] = ulim

df["logerror"].ix[df.logerror < llim] = llim

sn.boxplot(df.logerror)
cnt = df.year_month.value_counts()

sn.barplot(cnt.index, cnt.values, color="g")
df.parcelid.value_counts().reset_index()["parcelid"].value_counts()
mn = df.groupby("year_month")["logerror"].mean()

sn.pointplot(mn.index, mn.values)
sn.pointplot()