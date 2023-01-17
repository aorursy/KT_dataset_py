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



from pandasql import sqldf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_path='/kaggle/input/dwdm-petrol-prices/Petrol Prices.csv' # Path to data file

data = pd.read_csv(data_path)

data
print("Displaying first 7 records in the Dataset")

data.head(7)
print('Displaying Last 4 Records in  Dataset')

data.tail(4)

data.shape #Shows a total number of rows and columns found in dataset
data.columns #displays the names of the columns
#Change anamoly in data - ug to Aug

data.loc[data['Date'] == 'ug 18 2016']
data.at[143,'Date']='Aug 18 2016'

data
#Change the date format of the DATE column that it becomes readable

data['Date'] = pd.to_datetime(data['Date'])

data
#Check to ensure that change was done.

data.loc[data['Date'] == 'Aug 18 2016']
#Count how many records exist for each month in the data se

sorted_data= data.sort_values(by='Date')

output = sorted_data['Date'].groupby(sorted_data.Date.dt.month).agg(Total=pd.NamedAgg(column='', aggfunc='count'))

print(output)
data["months"]=pd.DatetimeIndex(data['Date']).month

data
dout=data.groupby('months').count()

dout
dout.plot(title="Line Graph", kind='line',

            figsize=(12,8))
data
data["Day"]=pd.DatetimeIndex(data['Date']).day

data
data["Year"]=pd.DatetimeIndex(data['Date']).year

data
data["Time"]=pd.DatetimeIndex(data['Date']).time

data
data
print('#8. Interesting columns from dataset.')

InterestCol= data[['Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane', 'Butane', 'HFO', 'Asphalt', 'ULSD', 'Ex_Refinery'

]]

InterestCol
InterestCol2= data[['Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane', 'Butane', 'HFO', 'Asphalt', 'ULSD', 'Ex_Refinery', 'Year'

                   ]]

InterestCol2
print('#9.Plot a line graph showing the gas prices over the particular period for all the interesting columns.')



InterestCol2.plot(kind='line', figsize=(12,8),

    use_index = True,

    y= ['Gasolene_87','Gasolene_90','Auto_Diesel','Kerosene', ]

      )

plt.suptitle("Gas Prices Over Through Jan to Aug 2019",size=20)

plt.xlabel("Month", color='c',size=15)

plt.ylabel("Prices", color='b',size=15)
InterestCol2.plot(kind='line',x='Year',y=['Gasolene_87', 'Gasolene_90' ,'Auto_Diesel' ,'Kerosene' ,'Propane', 'Butane' ,'HFO' ,'Asphalt', 'ULSD' ,'Ex_Refinery'] ,

                  title="Line Graph Showing Gas Prices Over The Years",

                  figsize=(20,20))

print('#10.a Percentage Change')

interestPerc = data[['Gasolene_87']]

print(interestPerc)
Int_Col= data['Gasolene_87'].pct_change(periods=4)

Int_Col
print('#10 b. Plot these values using a suitable Graph')

interestPerc.plot(title="Line Graph Showing Percentage Change in 87 Gas", kind='line',

figsize=(12,8))
data
print('# 11 Dataset with 2 Interesting columns from task 6')

dataset={'HFO':data.HFO, 'Asphalt':data.Asphalt,'Months':data.months,'Year':data.Year,'Day':data.Year,'Time':data.Time}

datanew=pd.DataFrame(dataset)

datanew
#Replace NAT Values with null

data=data.fillna('0')

data
#Replace NAT Values with null

datanew=datanew.fillna('0')

datanew
data['Gasolene_87']=data['Gasolene_87'].astype(int)

data['Gasolene_90']=data['Gasolene_90'].astype(int)

data['Auto_Diesel']=data['Auto_Diesel'].astype(int)

data['Kerosene']=data['Kerosene'].astype(int)

data['Propane']=data['Propane'].astype(int)

data['Butane']=data['Butane'].astype(int)

data['HFO']=data['HFO'].astype(int)

data['Asphalt']=data['Asphalt'].astype(int)

data['ULSD']=data['ULSD'].astype(int)

data['87_Change']=data['87_Change'].astype(int)

data['Ex_Refinery']=data['Ex_Refinery'].astype(int)

data['Day']=data['Day'].astype(int)

data['Year']=data['Year'].astype(int)





data.dtypes
datanew
print ('#12. Perform K-Means on the dataset using the two interesting columns')

kmeans = KMeans(n_clusters = 3)

kmeans

y_predicted = kmeans.fit_predict(datanew[['HFO', 'Asphalt']])

y_predicted
datanew['cluster'] = y_predicted

datanew.tail(50)
#dataset['cluster']=kmeans.fit_predict(datanew)

#dataset
print('#14. Provide a suitable plot of each cluster.')

from matplotlib.pyplot import figure

datanew1 = datanew[datanew.cluster==0]

datanew2 = datanew[datanew.cluster==1]

datanew3 = datanew[datanew.cluster==2]



plt.figure(figsize=(8, 8), dpi=80)

plt.scatter(datanew1.HFO,datanew1.Asphalt,color='green')

plt.scatter(datanew2.HFO,datanew2.Asphalt,color='red')

plt.scatter(datanew3.HFO,datanew3.Asphalt,color='black')



#plotting the centroids

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='blue',marker='*',label='centroid', s=150)





plt.xlabel('HFO')

plt.ylabel('Asphalt')

plt.legend()

datanew.dtypes
print('centroids of each cluster.')

kmeans.cluster_centers_
data.plot(kind='scatter',x='HFO',y='Asphalt',c=kmeans.labels_, cmap='rainbow', figsize=(12,8))
Year = data[['Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane', 'Butane', 'HFO', 'Asphalt', 'ULSD', 'Ex_Refinery','Year'

]]

Year
data.groupby('Year')['HFO','Asphalt'].mean()
Group_year=Year.groupby('Year').sum()

Group_year
print('#15. a')

Group_mean=Group_year.groupby('Year').mean()

Group_mean