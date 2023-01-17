import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline

import seaborn as sns

import datetime
airbnb = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv",engine = 'python')
airbnb.head()
airbnb.shape
airbnb.dtypes
airbnb.isnull().sum()
airbnb.drop(['host_name'], axis = 1, inplace = True)

airbnb.head()
airbnb['reviews_per_month'].fillna(0, inplace = True)

airbnb.head(20)
airbnb.reviews_per_month.isnull().sum()
airbnb['last_review'] = pd.to_datetime(airbnb['last_review'], format = '%Y-%m-%d')

airbnb['last_review'].fillna('1990-01-01', inplace = True)

airbnb.head(6)
airbnb.describe()
Number_of_Host = len(airbnb.host_id.unique())

Number_of_Host
top_host = airbnb.host_id.value_counts()

top_host_df = pd.DataFrame(top_host)

top_host_df.reset_index(inplace = True)

top_host_df.rename(columns = {'index':'Host_ID','host_id':'P_Count'}, inplace = True)

top_host_df['Total Percent of Place'] = top_host_df.P_Count/len(airbnb.host_id)

top_host_df['Cumulative_Percent'] = top_host_df['Total Percent of Place'].cumsum()

top_host_df
airbnb.neighbourhood_group.unique()
sub_group = airbnb[['neighbourhood_group','price']]

sub_group2 = sub_group.groupby('neighbourhood_group')

sub_group2.describe()
sns.set(rc = {'figure.figsize':(8,6)})

sns.set_style('whitegrid')

price_constrains = airbnb[airbnb['price']< 500]

pic_1=sns.violinplot(data=price_constrains, x='neighbourhood_group', y='price',palette = "RdBu")

pic_1.set_title('Density and distribution of prices for each neighberhood_group')
len(airbnb.neighbourhood.unique())
airbnb.room_type.unique()
grouped_room_type = airbnb.groupby('room_type')

agg_room_type = grouped_room_type['price'].agg([np.sum, np.mean, np.std,np.count_nonzero])

agg_room_type
airbnb.price.describe()
airbnb_price_adjusted = airbnb.price[(airbnb.price <= 500)]

airbnb_price_adjusted
df_airbnb_price_adjusted = pd.DataFrame(airbnb_price_adjusted)

df_airbnb_price_adjusted
df_airbnb_price_adjusted.to_csv('df_airbnb_price_adjusted.csv')
agg_room_type.to_csv('agg_room_type.csv')
airbnb.corr().style.background_gradient(cmap='Blues')