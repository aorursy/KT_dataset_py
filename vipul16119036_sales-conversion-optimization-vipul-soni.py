# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set()

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.linear_model import LinearRegression



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/clicks-conversion-tracking/KAG_conversion_data.csv')
df.head()
df.shape
df.columns
df.dtypes
df.describe()
df.corr()
df['interest'].plot.box()
df['Impressions'].plot.box()
df['Clicks'].plot.box()
df['Spent'].plot.box()
df['Total_Conversion'].plot.box()
df['Approved_Conversion'].plot.box()
df['age'].value_counts()
df['gender'].value_counts()
df['age'].value_counts().plot.bar()
df['gender'].value_counts().plot.bar()
train = df[0:1000]
test = df[1000:]
df
train
test
X_train = train.drop('Approved_Conversion',axis=1)
y_train = train['Approved_Conversion']
X_test = test.drop('Approved_Conversion',axis=1)
true_p = test['Approved_Conversion']
lreg=LinearRegression()
X_train = pd.get_dummies(X_train)
X_train.shape
X_test=pd.get_dummies(X_test)
lreg.fit(X_train,y_train)
pred = lreg.predict(X_test)
lreg.score(X_test,true_p)
lreg.score(X_train,y_train)
rmse_test = np.sqrt(np.mean(np.power((np.array(true_p)-np.array(pred)),2)))
rmse_train = np.sqrt(np.mean(np.power((np.array(y_train)-np.array(lreg.predict(X_train))),2)))
print(rmse_test,rmse_train)