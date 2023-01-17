# import packages

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import datetime as dt



%matplotlib inline
# load data

df=pd.read_csv('../input/renfe.csv')

df.head(1)
df.info()
df.shape
# handling missing data

# check missing data rows

sum(df.isna().any(axis=1))
# check columns with missing data

df.isna().any(axis=0)
# View rows with missing data

df[df.isna().any(axis=1)].tail(10)
# Drop rows with missing value

df.dropna(inplace=True)

sum(df.isna().any(axis=1))
# change data types

df['insert_date']=pd.to_datetime(df['insert_date'])

df['start_date'] =pd.to_datetime(df['start_date'])

df['end_date']=pd.to_datetime(df['end_date'])
# create new columns

df['duration']=df['end_date']-df['start_date']

df['duration_mins']=df['duration']/np.timedelta64(1, 'm')

df['start_time']=df['start_date'].dt.time

df.head(1)
#drop unwanted columns

df.drop(['insert_date'],axis=1,inplace=True)
# histgram of price 

df['price'].hist()
# histgram of duration

df['duration_mins'].hist()
# plot train type

df['train_type'].value_counts().plot.bar()
# plot train class

df['train_class'].value_counts().plot.bar()
# plot fare

df['fare'].value_counts().plot.bar()
df.groupby(by=['origin','destination'])['duration','duration_mins','price'].agg({'count','mean'})
#calc total counts and revenue by orgin, destination,and train class

df_mjr=df.groupby(by=['origin','destination','train_type','train_class'])['price'].agg({'count',('sum',lambda x: x.astype(float).sum())}).reset_index()

df_mjr.head(10)
df_mjr['count_per']=df_mjr['count']/sum(df_mjr['count'])

df_mjr['sum_per']=df_mjr['sum']/sum(df_mjr['sum'])
df_mjr.sort_values('count_per',ascending=False, inplace=True)
df_mjr['origin_destination']=df_mjr['origin']+"-"+df_mjr['destination']

df_mjr.set_index('origin_destination',inplace=True)

df_mjr.head(6).sum()
df_mjr.head(6).plot(kind='pie',y='count_per',legend=False)
df_mjr.head(6).plot(kind='pie',y='sum_per',legend=False)
#create dataset for trips between Barcelona and Mardrid only

df_bm=df.query('origin in ("BARCELONA","MADRID") and destination in ("BARCELONA","MADRID")')

df_bm.head(1)
# boxplot price groupped by train class and fare

df_bm.boxplot(column=['price','duration_mins'],by=['train_class','fare'],rot=45,layout=(2, 1),figsize=(10,9))
# group data by origin, destination, train class, and fare

# since most travllers travel between Madrid and Barcelona, the investigation will focus on Barcelona and Madrid 

df_app=df_bm.groupby(by=['origin','destination','train_type','train_class','fare','duration_mins'])['price'].describe().reset_index()

df_app.head(6)
# view info in order

df_app.sort_values(['mean']).head(2)
# investigate fixed price and non-fixed price

# fixed price 

df_app_fixed=df_app[df_app['std'] == 0]

df_app_fixed.sort_values(['min','origin'])
#boxplot fixed travel packages groupped by train class and fare

df_app_fixed.boxplot(column=['mean','duration_mins'],by=['train_class','fare'],rot=45,layout=(3, 1),figsize=(10,9))
# investigate if there is travel time restrictions.

df_fixed_cheapest=df_bm.query('train_class=="Turista" and fare=="Adulto ida"')

df_fixed_cheapest.head(1)
# check available travel time

df_fixed_cheapest['start_time'].unique()
# non-fixed price

df_app_vary=df_app[df_app['std'] != 0]

df_app_vary.sort_values(['std','min'],ascending=True).head(4)
# investigate non-fixed trips

df_app_vary.sort_values(by='min',ascending=True).head(2)
# plot non-fixed trips

df_app_vary.boxplot(column=['mean','duration_mins'],by=['train_class','fare'],rot=45,layout=(3, 1),figsize=(10,9))
# merge two dataset to get travel start time.

df_bm_vary=pd.merge(df_bm,df_app_vary, how='left',on=['origin','destination','train_type','train_class','fare','duration_mins'])

df_bm_vary.head(1)
# find out available travel time

df_bm_tp=df_bm_vary.query('train_class =="Turista" and fare=="Promo"')

df_bm_tp.head(5)

# convert start_time data type to time

df_bm_tp['start_time'].astype(dt.datetime, inplace=True)

df_bm_tp.head(1)
# plot average price by start time

df_bm_tp.groupby('start_time')['price'].mean().plot.line()
# plot average price by start time

df_bm_tp.groupby('start_time')['price'].count().plot.line()