# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#import zomato files 
import pandas as pd
zomato = pd.read_csv('../input/zomato.csv',encoding='latin-1')
zomato.head()
#read country code file
country = pd.read_excel('../input/Country-Code.xlsx',encoding='latin-1')
country.head()
import seaborn as sns
%matplotlib inline
# map the two files on country code and get country name which is useful for further analysis
data = pd.merge(zomato,country, on = 'Country Code')
# verify the country column
data.head()
#Lets perform EDA on Indian restaurants
#sort for indian restaurant data
indian_res = data[data['Country']=='India']
print(indian_res.Country.unique())
print(indian_res.head())
# let's check on variable information\
indian_res.info()
res_cost = pd.DataFrame()
res_cost['Restaurant ID'] = indian_res['Restaurant ID']
res_cost['Average Cost for two'] = indian_res['Average Cost for two']
res_cost.head()
type(res_cost)
res_cost_dsc = res_cost.sort_values(by ='Average Cost for two', ascending = False)
res_cost_dsc.head()
type(res_cost_dsc)
# let's have a look into countplot to see relationship b/w average cost for two people Vs no.of restaurants
import matplotlib.pyplot as plt
plt.figure(figsize=(25,6))
ax=sns.countplot(x='Average Cost for two',data = res_cost_dsc)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.show()
print("from above plot most of the restaurants have average cost within Rs. 1000")
#Let's plot a countplot for Restaurants Vs rating text to see how each other is related
sns.countplot(x='Rating text', data = indian_res)
plt.show()
print('from above plot most of the restaurants have average, good ratings and in remaining many of restaurants are not rated')
#Let's plot a countplot for Restaurants Vs Aggregate rating to see how each other is related
plt.figure(figsize=(15,6))
sns.countplot(x='Aggregate rating', data = indian_res)
plt.show()
# from above plot most of the restaurants have aggregate rating b/w 2.8 to 3.9 and in remaining restaurants many are 0 rated
# plot count plot for online delivery 
sns.countplot(indian_res['Has Online delivery'], data= indian_res)
plt.show()
#from above plot, most of restaurants are old fashioned only
#let's plot number of restaurants by city
plt.figure(figsize = (15,4))
ax = sns.countplot(x='City',data = indian_res)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.show()
#from above plot most of restaurants are located in New Delhi from the given data set
# Let's plot bar chart for top 20 cities with highest no. of. restaurants
res_sort = indian_res.groupby('City')['Restaurant ID'].count().sort_values(ascending = False)
res_sort.head()
type(res_sort)
#convert series to data frame for convinient
res_sort = pd.DataFrame({'City': res_sort.index, 'Restaurants': res_sort.values})
type(res_sort)
#plot bar plot for top 10 cities have highest restaurants
plt.figure(figsize=(10,6))
sns.barplot(y= res_sort[:20].City,x=res_sort[:20].Restaurants,data = res_sort)
plt.show()
# plot the bar chart to identify the city with less number of restaurants
plt.figure(figsize=(10,6))
sns.barplot(y= res_sort[-20:].City,x=res_sort[-20:].Restaurants,data = res_sort)
plt.show()