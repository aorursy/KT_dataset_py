# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix,mean_squared_error
data= pd.read_csv('../input/real-estate-price-prediction/Real estate.csv')
data=data.drop(['No'],axis=1)
data.info()
plt.figure(figsize=(10,10))
plt.subplot(3,2,1)
sns.distplot(data['X2 house age'])
plt.subplot(3,2,2)
sns.distplot(data['X3 distance to the nearest MRT station'])
plt.subplot(3,2,3)
sns.distplot(data['X4 number of convenience stores'])
plt.subplot(3,2,4)
sns.distplot(data['X5 latitude'])
plt.subplot(3,2,5)
sns.distplot(data['X6 longitude'])
plt.figure(figsize=(10,10))
plt.subplot(3,2,1)
plt.scatter(data['X2 house age'],data['Y house price of unit area'])
plt.xlabel('age')
plt.ylabel('price')
plt.subplot(3,2,2)
plt.scatter(data['X3 distance to the nearest MRT station'],data['Y house price of unit area'])
plt.xlabel('mrt dist')
plt.ylabel('price')
plt.subplot(3,2,3)
plt.scatter(data['X4 number of convenience stores'],data['Y house price of unit area'])
plt.xlabel('number of conv')
plt.ylabel('price')
plt.subplot(3,2,4)
plt.scatter(data['X5 latitude'],data['Y house price of unit area'])
plt.xlabel('lattitude')
plt.ylabel('price')
plt.subplot(3,2,5)
plt.scatter(data['X6 longitude'],data['Y house price of unit area'])
plt.xlabel('longitude')
plt.ylabel('price')
plt.figure(figsize=(10,10))
plt.subplot(3,2,1)
sns.boxplot(data['X2 house age'])
plt.subplot(3,2,2)
sns.boxplot(data['X3 distance to the nearest MRT station'])
plt.subplot(3,2,3)
sns.boxplot(data['X4 number of convenience stores'])
plt.subplot(3,2,4)
sns.boxplot(data['X5 latitude'])
plt.subplot(3,2,5)
sns.boxplot(data['X6 longitude'])
iqr=data['X3 distance to the nearest MRT station'].quantile(0.75)-data['X3 distance to the nearest MRT station'].quantile(0.25)
upper_iqr=data['X3 distance to the nearest MRT station'].quantile(0.75) + (1.5*iqr)
outl=data[data['X3 distance to the nearest MRT station'] > upper_iqr].index
for i in range(0,len(outl)):
    data.loc[outl[i],['X3 distance to the nearest MRT station']] = [upper_iqr]                                      
iqr=data['X5 latitude'].quantile(0.75)-data['X5 latitude'].quantile(0.25)
upper_iqr=data['X5 latitude'].quantile(0.75) + (1.5*iqr)
lower_iqr=data['X5 latitude'].quantile(0.25) - (1.5*iqr)
outl_u=data[data['X5 latitude'] > upper_iqr].index
outl_l=data[data['X5 latitude'] < lower_iqr].index
for i in range(0,len(outl_u)):
    data.loc[outl_u[i],['X5 latitude']] = data['X5 latitude'].mean()
for i in range(0,len(outl_l)):
    data.loc[outl_l[i],['X5 latitude']] = data['X5 latitude'].mean()
iqr=data['X6 longitude'].quantile(0.75)-data['X6 longitude'].quantile(0.25)
upper_iqr=data['X6 longitude'].quantile(0.75) + (1.5*iqr)
lower_iqr=data['X6 longitude'].quantile(0.25) - (1.5*iqr)
outl_u=data[data['X6 longitude'] > upper_iqr].index
outl_l=data[data['X6 longitude'] < lower_iqr].index
for i in range(0,len(outl_u)):
    data.loc[outl_u[i],['X6 longitude']] = upper_iqr
for i in range(0,len(outl_l)):
    data.loc[outl_l[i],['X6 longitude']] = lower_iqr
plt.figure(figsize=(10,10))
plt.subplot(3,2,1)
sns.boxplot(data['X2 house age'])
plt.subplot(3,2,2)
sns.boxplot(data['X3 distance to the nearest MRT station'])
plt.subplot(3,2,3)
sns.boxplot(data['X4 number of convenience stores'])
plt.subplot(3,2,4)
sns.boxplot(data['X5 latitude'])
plt.subplot(3,2,5)
sns.boxplot(data['X6 longitude'])
a=data[(data['X2 house age'] < 10.0)].index
b=data[(data['X2 house age'] >= 10.0) & (data['X2 house age'] < 20.0)].index
c=data[(data['X2 house age'] >= 20.0) & (data['X2 house age'] < 30.0)].index
d=data[(data['X2 house age'] >= 30.0) & (data['X2 house age'] < 40.0)].index
e=data[(data['X2 house age'] >= 40.0)].index
for i in range(0,len(a)):
    data.loc[a[i],['X2 house age']] = 'b10'
for i in range(0,len(b)):
    data.loc[b[i],['X2 house age']] = '10b20'
for i in range(0,len(c)):
    data.loc[c[i],['X2 house age']] = '20b30'
for i in range(0,len(d)):
    data.loc[d[i],['X2 house age']] = '30b40'
for i in range(0,len(e)):
    data.loc[e[i],['X2 house age']] = 'a40'


    
plt.figure(figsize=(10,10))
plt.subplot(3,2,1)
plt.scatter(data['X2 house age'],data['Y house price of unit area'])
plt.xlabel('age')
plt.ylabel('price')
plt.subplot(3,2,2)
plt.scatter(data['X3 distance to the nearest MRT station'],data['Y house price of unit area'])
plt.xlabel('mrt dist')
plt.ylabel('price')
plt.subplot(3,2,3)
plt.scatter(data['X4 number of convenience stores'],data['Y house price of unit area'])
plt.xlabel('number of conv')
plt.ylabel('price')
plt.subplot(3,2,4)
plt.scatter(data['X5 latitude'],data['Y house price of unit area'])
plt.xlabel('lattitude')
plt.ylabel('price')
plt.subplot(3,2,5)
plt.scatter(data['X6 longitude'],data['Y house price of unit area'])
plt.xlabel('longitude')
plt.ylabel('price')
data=pd.get_dummies(data,drop_first=True)
data.columns
data=data[['X1 transaction date', 'X3 distance to the nearest MRT station',
       'X4 number of convenience stores', 'X5 latitude', 'X6 longitude','X2 house age_20b30',
       'X2 house age_30b40', 'X2 house age_a40', 'X2 house age_b10','Y house price of unit area',]]
x=np.array(data.iloc[:,0:9])
y=np.array(data.iloc[:,-1])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
mean_squared_error(y_pred,y_test)

