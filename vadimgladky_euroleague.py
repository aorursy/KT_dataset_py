# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt



from sklearn.cluster import KMeans



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
erl = pd.read_csv('/kaggle/input/euroleague-basketball-results-20032019/euroleague_dset_csv_03_20_upd.csv')
erl['DATE'] = pd.to_datetime(erl['DATE'], format='%d/%m/%Y',infer_datetime_format=True)
erl.head()
erl.describe()
erl['HT'] = erl['HT'].astype(str)

erl['AT'] = erl['AT'].astype(str)
erl["Month"] = erl["DATE"].dt.strftime('%m')

erl["Year"] = erl["DATE"].dt.strftime('%Y')
scores = pd.DataFrame(erl.groupby(["FTT"]).FTT.count().sort_values(ascending=False))
scores.rename(columns={'FTT':'COUNT'}, inplace=True)
scores.reset_index(inplace=True)
scores
plt.figure(figsize=(50, 10))

sns.barplot(x="FTT", y="COUNT", data=scores)
erl[erl.HT.str.contains('Barcelona') | erl.AT.str.contains('Barcelona')]
pan = erl.loc[(erl.HT=='Panathinaikos') & (erl.Year=="2018")]
pan.describe()
pan
series_check = pd.DataFrame(pan)
series_check.reset_index(inplace=True)
series_check
series_check.loc[series_check["WINNER"] == series_check["WINNER"].shift(1),'RESULT'] = 'WIN'

series_check.loc[series_check["WINNER"] != series_check["WINNER"].shift(1),'RESULT'] = 'LOSE'

series_check.loc[series_check["HT"] == series_check["WINNER"],'RESULT'] = 'SERIE'
series_check
sns.lmplot(x="FTT",y="GAPT",hue="RESULT", data=series_check)
sns.barplot(x="RESULT",y="GAPT",data=series_check)
a = pd.DataFrame(erl["P1T"]/erl["P1T"])

a.reset_index(inplace=True)
b = pd.DataFrame(erl["P2T"]/erl["P1T"])

b.reset_index(inplace=True)
c = pd.merge(a,b)
c.rename(columns={0:'P2_cf'}, inplace=True)
list(c.columns)
c
c.groupby([c.P2_cf>1]).index.count()
c.plot.line(x="index",y="P2_cf", figsize=(200,20))
x = pd.DataFrame(erl.groupby("Q1A").Q1A.count())
x