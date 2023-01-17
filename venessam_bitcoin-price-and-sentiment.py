# This Python 3 environment comes with many helpful analytics libraries installed



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

price_filepath = "../input/bitcoin_hist_price.csv"

trend_filepath= "../input/bitcoin_google_trends.csv"
df_price=pd.read_csv(price_filepath,index_col='date')
df_trend=pd.read_csv(trend_filepath,index_col='month')
df_price.head(1)
df_trend.head(1)
df_price.dtypes
df_price.index=pd.to_datetime(df_price.index)

df_price.dtypes
df_trend.dtypes
df_trend.index=pd.to_datetime(df_trend.index)

df_trend.dtypes
df_trend['trend'].unique()
df_trend.loc[df_trend['trend']=='<1'].head(1)
#df_trend.loc[df_trend['trend']=='<1']['trend'].map({'<1':'0'})



#df_trend['trend']

df_trend['trend']=df_trend['trend'].apply(lambda x : '0' if x =='<1' else x ) 

df_trend['trend'].unique()
df_trend['trend']=pd.to_numeric(df_trend['trend'])

df_trend.dtypes
plt.figure(figsize=(10,8))

sns.distplot(a=df_trend['trend'],kde=True)
plt.figure(figsize=(15,9))

sns.lineplot(x=df_price.index,y=df_price['high']/1000,color='orange',data=df_price, label='Bitcoin Price High in 000')

sns.lineplot(x=df_trend.index,y='trend',color='purple',data=df_trend,label='Bitcoin Google Trend')

plt.title('Bitcoin Price and Google Trend')
df_price.index.max()
plt.figure(figsize=(15,9))

sns.lineplot(x=df_price.index,y=df_price['high']/1000,color='orange',data=df_price.query('date > "2013-04-01"'),label="Bitcoin Price High in 000")# toshow

sns.lineplot(x=df_trend.index,y=df_trend['trend'],color='purple',data=df_trend.query('month > "2013-04-01"'),label="Bitcoin Google Trend")



plt.xlim(pd.Timestamp("2013-04-01"), pd.Timestamp("2019-06-18"))

plt.title("Bitcoin Price and Google Trend from Apr2013 to Jun2019")

plt.xlabel("Date")

plt.ylabel("Trend /Price")

plt.show()
df_price.head(5)
df_price['month']=df_price.index.strftime('%Y-%m-01')

df_price.head(2)
df_price_mean=df_price.groupby('month').mean().reset_index()

df_price_mean.set_index=df_price_mean['month']

df_price_mean['month']=pd.to_datetime(df_price_mean['month'])

df_price_mean.head(1)
df_price_mean.dtypes
df= pd.merge(df_price_mean, df_trend, how='inner',on='month')

df.head(1)
df.set_index('month',inplace=True)

df.head(1)
plt.figure(figsize=(15,9))

sns.lineplot(x=df.index, y='trend',data=df,color='purple',label='Trend')

sns.lineplot(x=df.index, y=df['high']/1000,data=df,color='orange',label='Monthly Average Price in 000')

plt.xlim(pd.Timestamp("2013-04-01"), pd.Timestamp("2019-06-18"))

plt.ylabel('Trend and Price')

plt.xlabel('Date')

plt.title('Price and Trend')
plt.figure(figsize=(15,9))

sns.regplot(x=df['high']/1000, y='trend',data=df,color='purple')

plt.title('Price and Trend')