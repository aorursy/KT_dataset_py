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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.compose import ColumnTransformer

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
data=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

data.head()
colmiss=[col for col in data.columns if data[col].isnull().any()]

print(colmiss)
for col in colmiss:

    if(data[col].dtype == np.dtype('O')):

         data[col]=data[col].fillna(data[col].value_counts().index[0])    #replace nan with most frequent

    else:

        data[col] = data[col].fillna(data[col].mean()) 
data.isnull().sum()
data.head(5)
x=data.iloc[:,1:-1]

y=data.iloc[:,-1]
data.info
data.columns
x.shape
x.isnull().sum()   
x.select_dtypes(include=['object']).head(6)
i=0

for name in data.columns: 

    print(name,i)

    i+=1
for co in x.select_dtypes(include=['object']):

    print(x[co].describe())
LE = LabelEncoder()

for col in x.select_dtypes(include=['object']):

    x[col] = LE.fit_transform(x[col])
dc=data.corr()

dc
plt.figure(figsize=(30,30))

sns.heatmap(data.corr())
data.hist(figsize=(20,20))
corr = data.corr()

f,ax=plt.subplots(figsize=(20,1))

sns.heatmap(corr.sort_values(by=['SalePrice'],ascending=False).head(1), cmap='Blues')

plt.title("features correlation with the Research", weight='bold', fontsize=18)

plt.xticks(weight='bold')

plt.yticks(weight='bold', color='dodgerblue', rotation=0)

plt.show()
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.25,random_state=0)
reg=RandomForestRegressor(n_estimators=300)

reg.fit(train_x,train_y)

pred_y=reg.predict(test_x)
print(reg.score(train_x,train_y)*100)

print(reg.score(test_x,test_y)*100)
data1=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
data1.head()
data1.isnull().sum()

data1 = data1.iloc[:,1:]
data1.isnull().sum()
test_col_miss_val = [col for col in data1.columns if data1[col].isnull().any()]

print(test_col_miss_val)
for col in test_col_miss_val:

    if(data1[col].dtype == np.dtype('O')):

        data1[col] = data1[col].fillna(data1[col].value_counts().index[0])    #replace nan with most frequent

        

    else:

        data1[col] = data1[col].fillna(data1[col].mean()) 
for col in data1.select_dtypes(include=['object']):

    data1[col] = LE.fit_transform(data1[col])  
Predictions=reg.predict(data1)
print(Predictions)
Submission=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
Result=pd.DataFrame({'Id':Submission.Id,"SalePrice":Predictions})

Result.to_csv('submission1.csv', index=False)

Result.head()