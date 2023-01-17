# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
## Libraries

import datetime

from matplotlib import pyplot as plt

import seaborn as sns

sns.set()



import folium
df = pd.read_csv("/kaggle/input/earthquakes-in-japan/Japan earthquakes 2001 - 2018.csv", header=0)
df.head()
# Data size

df.shape
# Data info

df.info()
# Basic statics values

df.describe()
# Null data

df.isnull().sum()
# create a datetime columns, and extract date factors.

df["datetime"] = pd.to_datetime(df["time"])



df["year"] = df["datetime"].dt.year

df["month"] = df["datetime"].dt.month

df["day"] = df["datetime"].dt.day

df["hour"] = df["datetime"].dt.hour
# Whole data

plt.figure(figsize=(15,6))

sns.countplot(df["mag"], color="red", alpha=0.5)

plt.xlabel("magnitude")

plt.ylabel("count")

plt.title("histgram of magnitude")
# Per year

plt.figure(figsize=(15,10))

sns.kdeplot(df.query("year==2001")["mag"], color="black")

sns.kdeplot(df.query("year==2002")["mag"], color="grey")

sns.kdeplot(df.query("year==2003")["mag"], color="brown")

sns.kdeplot(df.query("year==2004")["mag"], color="coral")

sns.kdeplot(df.query("year==2005")["mag"], color="orangered")

sns.kdeplot(df.query("year==2006")["mag"], color="sienna")

sns.kdeplot(df.query("year==2007")["mag"], color="olive")

sns.kdeplot(df.query("year==2008")["mag"], color="yellow")

sns.kdeplot(df.query("year==2009")["mag"], color="greenyellow")

sns.kdeplot(df.query("year==2010")["mag"], color="lime")

sns.kdeplot(df.query("year==2011")["mag"], color="red")

sns.kdeplot(df.query("year==2012")["mag"], color="aqua")

sns.kdeplot(df.query("year==2013")["mag"], color="darkcyan")

sns.kdeplot(df.query("year==2014")["mag"], color="navy")

sns.kdeplot(df.query("year==2015")["mag"], color="plum")

sns.kdeplot(df.query("year==2016")["mag"], color="magenta")

sns.kdeplot(df.query("year==2017")["mag"], color="pink")

sns.kdeplot(df.query("year==2018")["mag"], color="crimson")

plt.legend(["2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018"], facecolor="white")

plt.xlabel("magnitude")

plt.ylabel("count")

plt.title("Kde plot by magnitude of earthquake per year")
df_year_count = pd.DataFrame(data=df.groupby("year").place.count()).reset_index()

median = np.median(df_year_count["place"])



# Visualization by barplot

plt.figure(figsize=(10,6))

sns.barplot(df_year_count["year"], df_year_count["place"], color="blue", alpha=0.5)

plt.hlines(y=median, xmin=-1, xmax=19, color="red", linewidth=2)

plt.xlim([-0.5,18])

plt.title("Count of earthquake per year \n line is median:{}".format(median.round(0)))
# Mapping visualization in 2011

map = folium.Map(location=[37, 138],tiles='Stamen Terrain', zoom_start=5)



mapping = pd.DataFrame({

    'place':df.query("year==2011")["place"].values,

    'data':df.query("year==2011")["mag"].values,

    'lat':df.query("year==2011")["latitude"].values,

    'lon':df.query("year==2011")["longitude"].values

})



for i, r in mapping.iterrows():

    folium.CircleMarker(location=[r["lat"],r["lon"]], radius=r["data"]/20, popup=r["place"], color='red').add_to(map)



map
# Mapping visualization in 2010

map = folium.Map(location=[37, 138],tiles='Stamen Terrain', zoom_start=5)



mapping = pd.DataFrame({

    'place':df.query("year==2010")["place"].values,

    'data':df.query("year==2010")["mag"].values,

    'lat':df.query("year==2010")["latitude"].values,

    'lon':df.query("year==2010")["longitude"].values

})



for i, r in mapping.iterrows():

    folium.CircleMarker(location=[r["lat"],r["lon"]], radius=r["data"]/20, popup=r["place"], color='blue').add_to(map)



map
# Mapping visualization in 2009

map = folium.Map(location=[37, 138],tiles='Stamen Terrain', zoom_start=5)



mapping = pd.DataFrame({

    'place':df.query("year==2009")["place"].values,

    'data':df.query("year==2009")["mag"].values,

    'lat':df.query("year==2009")["latitude"].values,

    'lon':df.query("year==2009")["longitude"].values

})



for i, r in mapping.iterrows():

    folium.CircleMarker(location=[r["lat"],r["lon"]], radius=r["data"]/20, popup=r["place"], color='green').add_to(map)



map
# Per year

plt.figure(figsize=(10,6))

sns.distplot(df["depth"], color="green")

plt.xlabel("depth")

plt.ylabel("count")

plt.title("Distribution of earthquake depth")
# Mapping visualization

map = folium.Map(location=[37, 138],tiles='Stamen Terrain', zoom_start=5)



mapping_under = pd.DataFrame({

    'place':df.query("depth <= 100")["place"].values,

    'data':df.query("depth <= 100")["mag"].values,

    'lat':df.query("depth <= 100")["latitude"].values,

    'lon':df.query("depth <= 100")["longitude"].values

})



mapping_over = pd.DataFrame({

    'place':df.query("depth > 100")["place"].values,

    'data':df.query("depth > 100")["mag"].values,

    'lat':df.query("depth > 100")["latitude"].values,

    'lon':df.query("depth > 100")["longitude"].values

})



for i, r in mapping_under.iterrows():

    folium.CircleMarker(location=[r["lat"],r["lon"]], radius=r["data"]/20, popup=r["place"], color='blue').add_to(map)

    

for i, r in mapping_over.iterrows():

    folium.CircleMarker(location=[r["lat"],r["lon"]], radius=r["data"]/20, popup=r["place"], color='red').add_to(map)

    

map
plt.figure(figsize=(10,6))

sns.distplot(df.query("depth>100")["mag"], color="red")

sns.distplot(df.query("depth<=100")["mag"], color="blue")

plt.xlabel("magnitude")

plt.ylabel("count")

plt.title("Distribution of magnitude")

plt.legend(["depth>100", "depth<=100"], facecolor="white")
# Target area (35 < lat < 40) & (140 < lon 145), in 2009~2012

df_area = df.query("(latitude>35 & latitude<40) & (longitude>140 & longitude<145) & (year==2009 | year==2010 | year==2011 | year==2012)")



# Makling the dataframe

df_area_month = pd.DataFrame(data=df_area.groupby(["year","month"]).place.count()).reset_index()

df_area_month["year/month"] = df_area_month["year"]*100 + df_area_month["month"]

df_area_month["year/month"] = [str(i) for i in df_area_month["year/month"]]



# Visualization

plt.figure(figsize=(12,6))

plt.plot(df_area_month["year/month"], df_area_month["place"], color="blue")

plt.ylabel("count")

plt.xlabel("year/month")

plt.xticks(rotation=90)