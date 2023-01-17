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
!pip install quandl
import quandl

import math

from sklearn import preprocessing, svm

from sklearn.model_selection import cross_validate

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
#Stock price of the Walt Disney Company from Quandl



df=quandl.get("EOD/DIS", authtoken="Uahee8xUkqEPsNSnDc2J")
df=df[['Open', 'High', 'Low', 'Volume','Close',]]

df['Pct_change']=(df['Close']-df['Open'])/df['Open']*100

df=df[['Open', 'Close', 'Pct_change', 'Volume']]
#Printing the time frame of the forecast

forecast_col='Close'

df.fillna(-99999, inplace=True)

forecast_out=int(math.ceil(0.01*len(df)))

print("Time frame:")

print(forecast_out)
df['Forecasted price']=df[forecast_col].shift(-forecast_out)

df.dropna(inplace=True)
#Defining the independent and dependent variables

import numpy as np

X= np.array(df.drop(['Forecasted price'], 1))

y= np.array(df['Forecasted price'])
#Pre-processing the data

X= preprocessing.scale(X);

y=np.array(df['Forecasted price'])
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)
clf=LinearRegression()

clf.fit(X_train, y_train)
Accuracy=clf.score(X_test, y_test)

print(y)

print("Accuracy:")

print(Accuracy)