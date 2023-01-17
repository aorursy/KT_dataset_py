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
import matplotlib.pyplot as plt 
data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")
import seaborn as sns
data.head()
data.info()
data["Date"] = pd.to_datetime(data["Date"])
data['days'] = data["Date"].apply(lambda days:days.day)
data[data["days"]==31].sum()
data[data["days"]==31].head()
summed_data = data[data['days']==31]
total_cases_confirmed = summed_data['ConfirmedCases'].sum()
print(total_cases_confirmed)
data = data.drop(["Province_State"],axis = 1)
data = data.drop(["Date"] , axis =1)
data.head()
data.groupby("days").sum()["ConfirmedCases"].plot()
sns.barplot(x ="days" , y = "ConfirmedCases" , data = data)
data[data["days"]==31].groupby("Country_Region").sum()
data = data[data['ConfirmedCases'] !=0]
plt.figure(figsize =(40 , 8))
sns.barplot(x= 'Country_Region',y="ConfirmedCases" , data=data)
pivoted_data = pd.pivot_table(data,values = "ConfirmedCases" , columns = "Country_Region" , index = 'days')
pivoted_data.plot()
new_data = data[data["days"]==31]
new_data.groupby("Country_Region").sum()["ConfirmedCases"].plot.bar(figsize = (100,8))
data_india = data[data["Country_Region"]=="India"]
data_india.groupby('days').sum()
data_india.drop(data_india.index[[0,1]] , inplace = True)
data_india.head()
india_list = data_india["ConfirmedCases"].to_list()
print(len(india_list))
print(india_list)
india_list = [i for i in india_list if i>=28]
print(len(india_list))
india_arr = np.array(india_list)
y_train = (india_arr).T
print(y_train)
a=[]
b=[]
for i in range(1,29):
    a.append(i**2)
    b.append(1)
arr = np.array(a)
brr = np.array(b)

x = np.vstack((brr , arr))
x_train = x.T


    
print(len(x_train))
theta = np.linalg.inv((x).dot(x_train))
theta = theta.dot(x)
theta = theta.dot(y_train)
print(theta)
hyp = theta.T.dot(x)
plt.plot(hyp)
plt.plot(india_list , "ro")
