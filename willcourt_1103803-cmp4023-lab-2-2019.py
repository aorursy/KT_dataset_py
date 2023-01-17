# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import dateutil

from datetime import datetime

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from pandasql import sqldf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_path='/kaggle/input/dwdm-petrol-prices/Petrol Prices.csv'

data = pd.read_csv(data_path)

data

#view dataset
data.head(7) #display first 7 records
data.tail(4)
data.columns #identify column names for later usage
data.dtypes #identify data types for conversion later
# Find count of non-NA across the columns 

#data.count(axis = 1) 
mnthcount = data['Date'].value_counts() #

mnthcount #displaying details of month variable
mnthcount = data.groupby(['Date']).count() 

print(count) 
dataf_results = sqldf("select count(Date) as Records, Date as Month  " +

                   "from data group by Month")

dataf_results #display data frame with label
#data['Date'] = pd.to_datetime(data['Date'])
data.loc[data['Date'] == 'ug 18 2016'] #loc function used to locate row with bad info
data.at[143,'Date']='Aug 18 2016' #correct info in the dataset

data #reload data
data['Date'] = pd.to_datetime(data['Date'])

data
data.loc[data['Date'] == 'Aug 18 2016']
data["months"]=pd.DatetimeIndex(data['Date']).month

data
do = data.groupby('months').count()

do
rng = pd.DataFrame() 

rng['date'] = pd.date_range('1/1/2011', periods = 72, freq ='H') 

  

# Print the dates in dd-mm-yy format 

rng[:5] 

  

# Create features for year, month, day, hour, and minute 

rng['year'] = rng['date'].dt.year 

rng['month'] = rng['date'].dt.month 

rng['day'] = rng['date'].dt.day 



  

# Print the dates divided into features 

rng.head(3) 
do.plot(title="Graph #1", kind='bar',

            figsize=(20,12))
data["day"]=pd.DatetimeIndex(data['Date']).day

data["year"]=pd.DatetimeIndex(data['Date']).year

data["month"]=pd.DatetimeIndex(data['Date']).month

data

data['Timestamp']= pd.to_datetime(data['Date']).values.astype(datetime)

data
# replacing na values in NaN with Null

data["months"].fillna("Null", inplace = True) 

data["day"].fillna("Null", inplace = True)  

data["year"].fillna("Null", inplace = True) 



data 
data2 = data[['Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane', 'Butane', 'HFO', 'Asphalt', 'ULSD', 'Ex_Refinery'

]]

data2
data3 = data[['Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane', 'Butane', 'HFO', 'Asphalt', 'ULSD', 'Ex_Refinery','Timestamp'

]]

data3
data3.plot(kind='line',x='Timestamp', y=['Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane', 'Butane', 'HFO', 'Asphalt', 'ULSD', 'Ex_Refinery'

],

           title="Line",

            figsize=(20,20))
data.Gasolene_90[0]
help(pd.date_range)
intrest_col= data['Gasolene_90'].pct_change(periods=4)

intrest_col

intrest_col.head(20)
intrest_col.plot(kind='bar', figsize=(20,20))
new={'HFO':data.HFO, 'Asphalt':data.Asphalt,'months':data.months,'year':data.year,'day':data.year,'Timestamp':data.Timestamp}

data11=pd.DataFrame(new)

data11
data11.info()

data11=data11.dropna(subset=['HFO'])
data11
data11['months'].astype(np.int64)
kmeans = KMeans(n_clusters=3).fit(data11)

centroids = kmeans.cluster_centers_

print(centroids)
kmeans = KMeans(n_clusters=3).fit(data11)
new={'HFO':data.HFO, 'Asphalt':data.Asphalt}

data12=pd.DataFrame(new)

data12
data12=data12.dropna(subset=['HFO'])
data11

data11['cluster']=kmeans.fit_predict(data12)
plt.scatter(data11['months'], data11['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)

plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)