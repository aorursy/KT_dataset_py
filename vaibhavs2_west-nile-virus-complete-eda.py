# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import descartes 
import geopandas as gpd
from shapely.geometry import Point, Polygon
import warnings 
warnings.filterwarnings('ignore')


# Some standard settings
pd.set_option('display.max_colwidth',1000)
pd.set_option('display.max_columns',200)
pd.set_option('display.max_rows',1000)
sns.set()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/predict-west-nile-virus/train.csv.zip')
spray = pd.read_csv('../input/predict-west-nile-virus/spray.csv.zip')
weather = pd.read_csv('../input/predict-west-nile-virus/weather.csv.zip')
test = pd.read_csv('../input/predict-west-nile-virus/test.csv.zip')

train.shape

# aggregating data since the record is splitted by NumMosquitos above 50 as mentioned in the data description.
train = pd.DataFrame(train.groupby(by=[x for x in train.columns if x != 'NumMosquitos'])['NumMosquitos'].sum())
train.reset_index(inplace=True)
# train.shape

# setting the Date as index in both train and weather dataFrame and merging them together for the purpose of our analysis.

train['Date'] = pd.to_datetime(train['Date'])
train.reset_index(drop=True,inplace=True)
train.set_index('Date',inplace=True)

weather['Date'] = pd.to_datetime(weather['Date'])
weather.reset_index(drop=True,inplace=True)
weather.set_index(weather['Date'], inplace=True)
weather.index

# merging train and weather data.
df = train.join(weather[weather['Station']==1])
df.replace('M',np.NaN,inplace=True)
df.head()
# First thing first. Let's check our target class distribution and see if the dataset is balanced or not.
fig,  ax = plt.subplots(1,1)
sns.countplot(train['WnvPresent'], ax = ax)
for i in ax.patches:
    height = i.get_height()
    ax.text(i.get_x()+i.get_width()/2,height,'{:.2f}{}'.format(height/len(train)*100,'%'))
plt.show()
# Feature distribution  by data types.
df.get_dtype_counts()
# Summarizing key statistics.
stats = pd.concat([df.dtypes,df.skew(),df.kurtosis(),df.isna().sum(),pd.DataFrame(df.describe(include='all').T).loc[:,['unique','top','freq','mean','std','min','max','50%']]],axis=1)
stats.columns = ['dtype','skew','kurtosis','missing_count','unique','top','freq','mean','std','min','max','median']
stats = stats.reindex(columns=['dtype','missing_count','unique','top','freq','mean','median','std','skew','kurtosis','min','max'])
stats.index.name = 'Features'
stats.sort_values(by='dtype')
stats
df.drop(['Water1','Depth','Heat','SnowFall','NumMosquitos','Station'], axis=1,inplace=True)
df['Year'] = df.index.year
fig, ax = plt.subplots(1,1, figsize=(20,10))
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu" ,ax = ax)
plt.show()
# 1. Address Features
# Count of streets per block
x= df.groupby(['Block'])['Street'].nunique().sort_values(ascending=False)

# Count no. of blocks per street type 
x.value_counts()
# Streets that are more likely to have infected mosquitos.
year = df.index.year
x = df.groupby([year,'Block','Street'])['WnvPresent'].sum()
y = df.groupby([year,'Block','Street'])['Street'].count()

z = pd.concat([x,y], axis=1)
z['% Count'] = np.round(x/y,4)*100

z.index.rename(['Year','Block','StreetName'],inplace=True)
z.reset_index(inplace=True, )
z.set_index(['Block','StreetName'], inplace=True)
z.rename({'Stree':'Strt_Rec_Cnt'}, inplace=True)
z[z['% Count']>0].sort_values(by=['Year','% Count'], ascending=[True,False])

df.columns
# df.head(65)
fig, ax = plt.subplots(2,2, figsize = (20,10))
chicago_map = gpd.read_file(r'../input/geo-filesandas/geo_export_5cc115c5-d68c-417b-a39f-c1c848ba54e4.shp')

# Converting data into right format by
# Converting pandas df into a geo_df ( original_df, co-ordinate_reference_system, geometry)
crs =  {'init':'epsg:4326'}
geometry = [Point(xy) for xy in zip(train['Longitude'],train['Latitude'])]
geo_df = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)

# Plotting the data from GeoDataFrame to our original map.
# 1. 2007
chicago_map.boundary.plot(ax=ax[0,0])
geo_df[(geo_df['WnvPresent']==0) & (geo_df['Year']==2007)].plot(color='green', ax =ax[0,0] )
geo_df[geo_df['WnvPresent']==1 & (geo_df['Year']==2007)].plot(color='red', ax =ax[0,0])
ax[0,0].set_title('Year: 2007')

# # 2. 2009
chicago_map.boundary.plot(ax=ax[0,1])
geo_df[(geo_df['WnvPresent']==0) & (geo_df['Year']==2009)].plot(color='green', ax =ax[0,1] )
geo_df[geo_df['WnvPresent']==1 & (geo_df['Year']==2009)].plot(color='red', ax =ax[0,1])
ax[0,1].set_title('Year: 2009')

# 3. 2011
chicago_map.boundary.plot(ax=ax[1,0])
geo_df[(geo_df['WnvPresent']==0) & (geo_df['Year']==2011)].plot(color='green', ax =ax[1,0] )
geo_df[geo_df['WnvPresent']==1 & (geo_df['Year']==2011)].plot(color='red', ax =ax[1,0])
ax[1,0].set_title('Year: 2011')

# 4. 2013
chicago_map.boundary.plot(ax=ax[1,1])
geo_df[(geo_df['WnvPresent']==0) & (geo_df['Year']==2007)].plot(color='green', ax =ax[1,1] )
geo_df[geo_df['WnvPresent']==1 & (geo_df['Year']==2007)].plot(color='red', ax =ax[1,1])
ax[1,1].set_title('Year: 2013')

plt.show()
fig, ax = plt.subplots(1,1, figsize = (20,5))
sns.pointplot(z[(z['% Count']>5) & (z['Year']==2007)].index, z[(z['% Count']>5) & (z['Year']==2007)]['% Count'], color='g', ax=ax)
sns.pointplot(z[(z['% Count']>5) & (z['Year']==2009)].index, z[(z['% Count']>5) & (z['Year']==2009)]['% Count'], color='b', ax=ax)
sns.pointplot(z[(z['% Count']>5) & (z['Year']==2011)].index, z[(z['% Count']>5) & (z['Year']==2011)]['% Count'], color='r', ax=ax)
sns.pointplot(z[(z['% Count']>5) & (z['Year']==2013)].index, z[(z['% Count']>5) & (z['Year']==2013)]['% Count'], color='y', ax=ax)
ax.tick_params('x',labelrotation=90, labelsize='small', )
ax.set_title('Virus spread distribution in blocks and their streets in 2007, 2009, 2011 & 2013', fontdict = {'fontsize':20})
plt.legend()
plt.show()
# Identifying Mosquito Species that are the most likely carriers of Virus.
tmp = pd.crosstab(train['Species'],train['WnvPresent'],normalize='index')
tmp.reset_index(drop=True)
tmp.columns = ['WnvAbsent','WnvPresent']

fig,ax = plt.subplots(1,1)
g = sns.countplot(train['Species'], ax=ax, order = list(tmp.index))
ax.tick_params(axis='x',rotation=90)
g2 = g.twinx()
sns.pointplot(tmp.index,tmp['WnvPresent'], ax=g2, order = list(tmp.index))
plt.show()
# Effect of temperature.
# Let us see the distribution of temperatures for min, max and avg wrt. the mosquito virus infection.

fig, ax = plt.subplots(1,4, figsize=(20,5))
sns.distplot(df['Tmin'], ax=ax[0])
ax[0].xaxis.set(ticks=np.arange(40,80,5))
ax[0].set_title('Tmin')

sns.distplot(df['Tmax'], ax=ax[1])
ax[1].xaxis.set(ticks=np.arange(50,100,5))
ax[1].set_title('Tmax')

sns.distplot(df['Tavg'].apply(int), ax=ax[2])
ax[2].xaxis.set(ticks=np.arange(40,90,5))
ax[2].set_title('Tavg')

sns.distplot(df['Depart'].apply(int), ax=ax[3])
ax[3].set_title('Depart')

plt.show()


fig, ax = plt.subplots(1,1, figsize=(20,5))
sns.distplot(df.loc[df.index.year==2007,'Tavg'].apply(int), ax=ax,color='g',hist=False)
sns.distplot(df.loc[df.index.year==2009,'Tavg'].apply(int), ax=ax,color='b',hist=False)
sns.distplot(df.loc[df.index.year==2011,'Tavg'].apply(int), ax=ax,color='r',hist=False)
sns.distplot(df.loc[df.index.year==2013,'Tavg'].apply(int), ax=ax,color='y',hist=False)
ax.xaxis.set(ticks=np.arange(40,90,5))
ax.set_title('Tavg')

plt.show()
fig, ax = plt.subplots(1,1, figsize=(20,5))
sns.boxplot(df['Year'],df['Tavg'].apply(int), hue = df['WnvPresent'], ax=ax)
ax.set_title('Tavg')
plt.show()
fig, ax = plt.subplots(1,1, figsize=(20,5))
sns.boxplot(df['Year'],df['Depart'].apply(int), hue = df['WnvPresent'], ax=ax)
ax.set_title('Depart from Normal for Virus Absence/Presence')
plt.show()
# Moisture content in Atmosphere using DewPoint & WetBulb test
# Let us plot the data distribution and how are variable is distributed in the space.
fig, ax = plt.subplots(1,2, figsize=(20,5))
sns.distplot(df['DewPoint'], ax=ax[0])
ax[0].set_title('DewPoint')

sns.distplot(df['WetBulb'].dropna().apply(float), ax=ax[1])
ax[1].set_title('WetBulb')
plt.show()
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20,5))
sns.violinplot(df['Year'], df['Tavg'].apply(int)-df['DewPoint'], hue = df['WnvPresent'], ax=ax1)
ax1.set_title('Humidity Estimate from DewPoint', fontdict= {'fontsize':15})
sns.violinplot(df['Year'], df['Tavg'].dropna().apply(int)- df['WetBulb'].replace(np.NaN,0).apply(int), hue = df['WnvPresent'])
ax2.set_title('Humidity Estimate from Wet Bulb Temperature', fontdict= {'fontsize':15})
ax2.set_ylim(0,20)
plt.show()
# Let us see if the timings of Sunrise seem to be of any use to us.
plt.figure(figsize=(20,5))
tmp = df.groupby(['Year','Sunrise'])['WnvPresent'].sum().reset_index()
sns.pointplot(tmp.loc[tmp['Year']==2007,'Sunrise'],tmp['WnvPresent'],color='g')
sns.pointplot(tmp.loc[tmp['Year']==2009,'Sunrise'],tmp['WnvPresent'],color='b')
sns.pointplot(tmp.loc[tmp['Year']==2011,'Sunrise'],tmp['WnvPresent'],color='r')
sns.pointplot(tmp.loc[tmp['Year']==2013,'Sunrise'],tmp['WnvPresent'],color='y')
plt.tick_params(rotation=90)
plt.show()
plt.figure(figsize=(20,5))
tmp = df.groupby(['Year','Sunset'])['WnvPresent'].sum().reset_index()
sns.pointplot(tmp.loc[tmp['Year']==2007,'Sunset'],tmp['WnvPresent'],color='g',dodge=True)
sns.pointplot(tmp.loc[tmp['Year']==2009,'Sunset'],tmp['WnvPresent'],color='b')
sns.pointplot(tmp.loc[tmp['Year']==2011,'Sunset'],tmp['WnvPresent'],color='r')
sns.pointplot(tmp.loc[tmp['Year']==2013,'Sunset'],tmp['WnvPresent'],color='y')
plt.tick_params(rotation=90)
plt.show()
# Effect of Climatic Conditions
print(f'Count of distinct weather condition: {df["CodeSum"].nunique()}')
fig, ax = plt.subplots(1,1, figsize=(20,5))
tmp = pd.crosstab(df['CodeSum'],df['WnvPresent'],normalize='index')
tmp.reset_index()

g = sns.countplot(df['CodeSum'],order = list(tmp.index), ax =ax)
g.tick_params(axis='x',labelrotation=90)
g.set_title('Effect of Weather Condition on Virus', fontsize=20)
g1 = g.twinx()
sns.pointplot(tmp.index,tmp[1], ax=g1)

plt.show()
fig, ax = plt.subplots(1,1, figsize=(20,5))
tmp = pd.crosstab(df['PrecipTotal'],df['WnvPresent'],normalize='index')
tmp.reset_index()

g = sns.countplot(df['PrecipTotal'],order = list(tmp.index), ax =ax)
g.tick_params(axis='x',labelrotation=90)
g.set_title('Effect of Precipitation on Virus Presence', fontsize=20)
g1 = g.twinx()
sns.pointplot(tmp.index,tmp[1], ax=g1)

plt.show()
# Effect of Stn and Sea Level pressure variables.
fig, (ax1) = plt.subplots(1,1, figsize=(20,10))
sns.boxplot(df['Year'], df['StnPressure'].replace(np.NaN,0.0).apply(float), hue = df['WnvPresent'], ax=ax1)
ax1.set_ylim(28.75,29.75)
ax1.set_title('StnPressure', fontdict= {'fontsize':15})
plt.show()
fig, (ax1) = plt.subplots(1,1, figsize=(20,5))
sns.boxplot(df['Year'], df['ResultSpeed'].replace(np.NaN,0.0).apply(float), hue = df['WnvPresent'], ax=ax1)
# ax1.set_ylim(28,30)
ax1.set_title('ResultSpeed', fontdict= {'fontsize':15})
plt.show()
# Effect  of Wind Speed and Dir
fig, (ax1) = plt.subplots(1,1, figsize=(20,5))
sns.boxplot(df['Year'], df['ResultSpeed'].replace(np.NaN,0.0).apply(float), hue = df['WnvPresent'], ax=ax1)
# ax1.set_ylim(28,30)
ax1.set_title('ResultSpeed', fontdict= {'fontsize':15})
plt.show()
