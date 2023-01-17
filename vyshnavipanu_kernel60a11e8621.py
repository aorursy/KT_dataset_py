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
data=pd.read_csv('../input/automobile-dataset/Automobile_data.csv')
data.head()
data.columns
data.shape
data.info()





data['fuel-type'].value_counts()

data.describe().transpose()
data=data.replace('?',np.nan)
data.isnull().sum()
data.head()
l=data.columns[data.isnull().any()].tolist()
for i in l:
  a=data[i].isnull().sum()
  if a>0:
    b=(a/72983)*100
    print(i,':',b,'%')
data['normalized-losses'].fillna(data['normalized-losses'].median(),inplace=True)
data['num-of-doors']=data['num-of-doors'].replace('?','four')
data['bore'].fillna(data['bore'].median(),inplace=True)
data['stroke'].fillna(data['stroke'].median(),inplace=True)
data['horsepower'].fillna(data['horsepower'].median(),inplace=True)
data['price'].fillna(data['price'].median(),inplace=True)

data['peak-rpm'].fillna(data['peak-rpm'].median(),inplace=True)
data.isnull().any()
data['peak-rpm']=data['peak-rpm'].astype(int)
data['horsepower']=data['horsepower'].astype(int)
data['price']=data['price'].astype(int)
data['stroke']=data['stroke'].astype(float)
data['bore']=data['bore'].astype(float)
data['normalized-losses']=data['normalized-losses'].astype(int)
data[['engine-size','peak-rpm','curb-weight','horsepower','normalized-losses','wheel-base','price']].hist(figsize=(10,8),bins=6,color='Y')
# 2 plt.figure(figsize=(10,8))
plt.tight_layout()
plt.show()
import seaborn as sns
sns.distplot(data['highway-mpg'], kde=False, bins=10,color='r');

import matplotlib.pyplot as plt
%matplotlib inline
fig1, ax1 = plt.subplots()
ax1.pie(data['bore'],data['horsepower'])
pd.crosstab(data['fuel-type'],data['num-of-doors'])
pd.crosstab(data['engine-location'],data['body-style'])
pd.crosstab(data['drive-wheels'],data['engine-location'])
data.corr()
sns.heatmap(data.corr(),annot=True)
t