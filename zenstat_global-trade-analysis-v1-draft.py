# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import pandas as pd 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt 
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
#set plot size
plt.rcParams['figure.figsize']=(20,10)

dt = pd.read_csv("../input/commodity_trade_statistics_data.csv", na_values=["No Quantity",0.0,''],sep=',')
# Any results you write to the current directory are saved as output.
#first things first, find total number of rows and columns 
print("Shape................ :\n" + str(dt.shape))
#List all columns after data is laoded 
#temp.columns.values # so we have 8,225,871 rows × 10 columns present in the data
#see names and data types 
print("Data type Information.............. :\n" +  str(dt.dtypes))
#sort the data by year to tidy things before moving on 
dt = dt.sort_values("year")
#delete temp dataset 
#del(temp)
# Preview the data 
dt.head(10)
# see missing values 
dt.count()
#Preview preliminary statistics as is from the dat awithout cleanup 
dt.describe(include="all")
dt = dt.dropna(how='any').reset_index(drop=True)  
dt.describe(include="all")
#all missing values 'NaN' are messing up with prelim statisitics, lets treat missing cases by each variable 
#dt.describe(include="all")
#First lets filter the data where there is no missing values 
#pick first variable 
#find unique values present other than nan
print('Country or Area')
dt['country_or_area'].nunique()
dt['country_or_area'].unique()
#there are 209 overall country or areas present in the data, Now lets see frequencies of each of these catergories.. only top 15, since there are a lot of categories  
dt['country_or_area'].value_counts()[:15]
print('Commodity')
dt['commodity'].nunique()
dt['commodity'].unique()
dt['commodity'].value_counts()[:20]
dt.commodity.unique()
print('flow variable')
dt['flow'].nunique()
dt['flow'].unique()
dt['flow'].value_counts()
print('category')
dt['category'].nunique()
dt['category'].unique()
#there are 209 overall country or areas present in the data, Now lets see frequencies of each of these catergories, pick first 20 most occured 
dt['category'].value_counts()[:20]
dfSheeps = dt[dt['commodity']=='Sheep, live'].reset_index(drop=True)  
dfGoats = dt[dt['commodity']=='Goats, live'].reset_index(drop=True)  
dfSheeps.head()
dfSheepsGrouped = pd.DataFrame({'weight_kg' : dfSheeps.groupby( ["year","flow","commodity"] )["weight_kg"].sum()}).reset_index()
dfGoatsGrouped = pd.DataFrame({'weight_kg' : dfGoats.groupby( ["year","flow","commodity"] )["weight_kg"].sum()}).reset_index()
dfSheepsGrouped.head()

f, ax = plt.subplots(1, 1)
dfgr = pd.concat([dfSheepsGrouped,dfGoatsGrouped])
ax = sns.pointplot(ax=ax,x="year",y="weight_kg",data=dfgr[dfgr['flow']=='Import'],hue='commodity')
_ = ax.set_title('Global imports of kgs by animal')



dfSheepsGrouped = pd.DataFrame({'weight_kg' : dfSheeps.groupby( ["country_or_area","flow","commodity"] )["weight_kg"].sum()}).reset_index()
dfSheepsGrouped.head()

sheepsImportsCountry = dfSheepsGrouped[dfSheepsGrouped['flow']=='Import']
sheepsExportsCountry = dfSheepsGrouped[dfSheepsGrouped['flow']=='Export']
sheepsImportsCountry.head()

ax = sns.barplot(x="weight_kg", y="country_or_area", data=sheepsImportsCountry.sort_values('weight_kg',ascending=False)[:15])
_ = ax.set(xlabel='Kgs', ylabel='Country or area',title = "Countries or areas that imported more kgs of Sheeps")

ax = sns.barplot(x="weight_kg", y="country_or_area", data=sheepsExportsCountry.sort_values('weight_kg',ascending=False)[:15])
_ = ax.set(xlabel='Kgs', ylabel='Country or area',title = "Countries or areas that exported more kgs of Sheeps")
#List of top 10 countries having highest trade in terms of dollars 
dt.groupby(['country_or_area'])['trade_usd'].aggregate(np.sum).nlargest(10)
#List of top 10 countries having smallest trade in terms of dollars 
dt.groupby(['country_or_area'])['trade_usd'].aggregate(np.sum).nsmallest(10)
#List of top 10 countries having highest trade in terms of weight
dt.groupby(['country_or_area'])['weight_kg'].aggregate(np.sum).nlargest(10)
#List of top 10 countries having smallest trade in terms of weight
dt.groupby(['country_or_area'])['weight_kg'].aggregate(np.sum).nsmallest(10)