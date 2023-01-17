### Import des librairies utiles

import csv

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from __future__ import print_function

import warnings

warnings.filterwarnings("ignore")





df = pd.read_csv("../input/AllMoviesDetailsCleaned.csv", encoding='utf-8-sig', sep=";", 

                 engine="python",parse_dates=["release_date"])

print(df.columns)

df = df[["id","title","budget","genres","popularity","revenue","release_date"]] 

df = df.dropna(axis=0,how="any")

df = df[df.genres.str.contains("Crime|Romance")]

print(df.shape)

df.head()
df["popularity"] = pd.to_numeric(df['popularity'], errors='coerce')

df['release_date']= pd.to_datetime(df.release_date)

df['year']= df.release_date.dt.year

df['month']= df.release_date.dt.month

df.head()
df["type"] = pd.np.where(df.genres.str.contains("Crime"),"Crime",

             pd.np.where(df.genres.str.contains("Romance"),"Romance","Other"))



df.head()
df = df[(df.year < 2018) ]
data = pd.DataFrame(df.groupby(["year","type"],sort=True)['popularity'].mean()).reset_index()

# data.head()

df1 = df[df.revenue !=0]

df1["revenue"] = df1["revenue"]/1000000

# df1.head()

data1= pd.DataFrame(df1.groupby(["year","type"],sort=True)['revenue'].mean()).reset_index()

# data1

df2= df[df.budget !=0] 

df2["budget"] = df2["budget"]/1000000 

data2=pd.DataFrame(df2.groupby(["year","type"],sort=True)['budget'].mean()).reset_index()

# data2

data3= pd.DataFrame(df.groupby(["year","type"],sort=True)['id'].count()).reset_index()

# data3
import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

fig, ax= plt.subplots()

fig.set_size_inches(18, 7)



ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

sns.pointplot(x=data3["year"], y=data3["id"],hue=data3["type"], data=data3,

                join=True, palette={"Crime": "#2ecc71", "Romance": "#34495e"},ax=ax) 

ax.set(xlabel='year', ylabel='Count',title="Count By year")





fig, ax1 = plt.subplots()

fig.set_size_inches(18, 7)



ax1.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

sns.pointplot(x=data1["year"], y=data1["revenue"],hue=data1["type"], 

               palette={"Crime": "#2ecc71", "Romance": "#34495e"},data=data1,join=True,ax=ax1)

ax1.set(xlabel='year', ylabel='Avearage revenue',title="Average revenue By year")





fig, ax2 = plt.subplots()

fig.set_size_inches(18, 7)

ax2.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

sns.pointplot(x=data2["year"], y=data2["budget"],hue=data2["type"],

               palette={"Crime": "#2ecc71", "Romance": "#34495e"},data=data2,join=True,ax=ax2)

ax2.set(xlabel='year', ylabel='Avearage budget',title="Average budget By year")





fig, ax0 = plt.subplots()

fig.set_size_inches(18, 7)

ax0.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

sns.pointplot(x=data["year"], y=data["popularity"],hue=data["type"],

               palette={"Crime": "#2ecc71", "Romance": "#34495e"}, data=data,join=True,ax=ax0)

ax0.set(xlabel='year', ylabel='Avearage popularity',title="Average popularity By year")
