# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

%matplotlib inline

from shapely.geometry import Point, Polygon

import geopandas as gpd



 #Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/crimes-in-boston/crime.csv",encoding='ISO-8859-1')

street_map = gpd.read_file("../input/boston-neighborhood/Boston_Neighborhoods.shp")
crs={'init':'epsg:4326'}

df.head()
geometry = [Point(xy) for xy in zip(df['Long'], df['Lat'])]

geometry[:3]
geo_df = gpd.GeoDataFrame(df, crs=crs, geometry = geometry)

geo_df.head()
fix, ax = plt.subplots(figsize = (8,8))

street_map.plot(ax=ax, alpha = 0.4, color='grey')

geo_df[(geo_df['SHOOTING']=='Y')&(geo_df['Lat']>10)].plot(ax=ax, markersize = 3, color='red')
df.OFFENSE_CODE_GROUP.value_counts(ascending=False).head()
import seaborn as sns

sns.countplot(data=df, x = 'DAY_OF_WEEK')
sns.countplot(data=df, x = 'MONTH')
fix, ax = plt.subplots(figsize = (8,8))

street_map.plot(ax=ax, alpha = 0.4, color='grey')

geo_df[(geo_df['DAY_OF_WEEK']=='Friday')&(geo_df['MONTH']==8)&(geo_df['Lat']>10)].plot(ax=ax, markersize = 3, color='red')

fix, ax = plt.subplots(figsize = (8,8))

street_map.plot(ax=ax, alpha = 0.4, color='grey')

geo_df[(geo_df['DAY_OF_WEEK']=='Sunday')&(geo_df['MONTH']==2)&(geo_df['Lat']>10)].plot(ax=ax, markersize = 3, color='green')
fig, ax = plt.subplots(figsize=(8,20))

sns.countplot(data=df, y = 'OFFENSE_CODE_GROUP', ax=ax, order=df['OFFENSE_CODE_GROUP'].value_counts().sort_values(ascending=False).index)
df.head()
fix, ax = plt.subplots(figsize = (8,8))

street_map.plot(ax=ax, alpha = 0.4, color='grey')

geo_df[(geo_df['DISTRICT']=='B2')&(geo_df['Lat']>10)].plot(ax=ax, markersize = 3, color='red')

geo_df[(geo_df['DISTRICT']=='A15')&(geo_df['Lat']>10)].plot(ax=ax, markersize = 3, color='green')
df.DISTRICT.value_counts().sort_values(ascending=False)
df[['Lat','OFFENSE_DESCRIPTION']][(df.Lat>10)&(df.DISTRICT=='A15')].sort_values('Lat').set_index('Lat').head(10)
df.head(3)
df.OCCURRED_ON_DATE = pd.to_datetime(df.OCCURRED_ON_DATE)
plt.figure(figsize=(16, 6))

x = df[(df.OCCURRED_ON_DATE>'June 2015')|(df.OCCURRED_ON_DATE<'September 2018')]

x = df[df.OFFENSE_CODE_GROUP == 'Larceny'].resample('M', on='OCCURRED_ON_DATE')['OFFENSE_DESCRIPTION'].count().plot()

df.OCCURRED_ON_DATE.max()
df.describe()