# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#import sklearn

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import OrdinalEncoder



#import matplotlib for visualization

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

df_test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")

submission=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/submission.csv")
df.head()
df.info()
df.describe()
df_test.head()
df_test.info()
df_test.describe()
df['Date']=pd.to_datetime(df['Date'], format='%Y-%m-%d')

df_test['Date']=pd.to_datetime(df_test['Date'], format='%Y-%m-%d')
df.info()
df['Country_Region'].value_counts()
ax=plt.scatter(df['ConfirmedCases'],df['Fatalities'],marker=r'$\clubsuit$')

plt.xlabel('Confired Cases')

plt.ylabel('Fatalities')
USCases=df.loc[df.Country_Region == 'US']
NYCases=USCases.loc[USCases.Province_State=='New York']
NoZeroNYCases=NYCases.loc[NYCases.ConfirmedCases > 0]
fig, ax = plt.subplots()



# Using set_dashes() to modify dashing of an existing line

line1, = ax.plot(NYCases['Date'], NYCases['ConfirmedCases'], label='Confirmed Cases')

line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break



# Using plot(..., dashes=...) to set the dashing when creating a line

line2, = ax.plot(NYCases['Date'], NYCases['Fatalities'], dashes=[6, 2], label='Fatalities')



ax.legend()

plt.show()
fig, ax = plt.subplots()



# Using set_dashes() to modify dashing of an existing line

line1, = ax.plot(NoZeroNYCases['Date'], NoZeroNYCases['ConfirmedCases'], label='Confirmed Cases')

line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break



# Using plot(..., dashes=...) to set the dashing when creating a line

line2, = ax.plot(NoZeroNYCases['Date'], NoZeroNYCases['Fatalities'], dashes=[6, 2], label='Fatalities')

ax.set_xlabel('Date')

ax.legend()

plt.show()
China=df.loc[df.Country_Region=='China']
Hubei=China.loc[China.Province_State=='Hubei']
Hubei
fig, ax = plt.subplots(figsize=(10,6))



# Using set_dashes() to modify dashing of an existing line

line1, = ax.plot(NYCases['Date'], NYCases['ConfirmedCases'], label='New York Cases')

line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break



# Using plot(..., dashes=...) to set the dashing when creating a line

line2, = ax.plot(Hubei['Date'], Hubei['ConfirmedCases'], dashes=[6, 2], label='Hubei Cases')



# Using set_dashes() to modify dashing of an existing line

line3, = ax.plot(NYCases['Date'], NYCases['Fatalities'], label='New York Fatalities')

line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break



# Using plot(..., dashes=...) to set the dashing when creating a line

line4, = ax.plot(Hubei['Date'], Hubei['Fatalities'], linestyle='--', label='Hubei Fatalities')



ax.set_ylim([1, 1000000])

plt.yscale('log')

ax.set_xlabel('Date')

ax.legend()

plt.show()
def create_features(df):

    df['day'] = df['Date'].dt.day

    df['month'] = df['Date'].dt.month

    df['dayofweek'] = df['Date'].dt.dayofweek

    df['dayofyear'] = df['Date'].dt.dayofyear

    df['quarter'] = df['Date'].dt.quarter

    df['weekofyear'] = df['Date'].dt.weekofyear

    return df
def categoricalToInteger(df):

    #convert NaN Province State values to a string

    df.Province_State.fillna('NaN', inplace=True)

    #Define Ordinal Encoder Model

    oe = OrdinalEncoder()

    df[['Province_State','Country_Region']] = oe.fit_transform(df.loc[:,['Province_State','Country_Region']])

    return df
df=categoricalToInteger(df)

df=create_features(df)
df.head()
df.describe()
x=['Id','Province_State','Country_Region','day','month','dayofweek','dayofyear','quarter','weekofyear']
train_x=df[x]
train_x.info()
y=['ConfirmedCases','Fatalities']
train_y=df[y]
train_y


import matplotlib.pyplot as plt

import numpy as np

from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error, r2_score

model_lr = linear_model.LinearRegression() 

model_lr.fit(train_x,train_y)

df_test
df_test=categoricalToInteger(df_test)

df_test=create_features(df_test)
df_test.head()
d=df_test.rename(columns={"ForecastId":"Id"})
d
test_x=d[x]
test_x
y_pred= model_lr.predict(test_x)
y_pred
y_pr=pd.DataFrame(y_pred, columns=['ConfirmedCases','Fatalities'])
y_pr=y_pr.astype(int)
y_pr
df_test.head()
df_test
fig, ax = plt.subplots(figsize=(10,6))



# Using set_dashes() to modify dashing of an existing line

line1, = ax.plot(df_test['Date'], y_pr['ConfirmedCases'], label='Cases')



# Using plot(..., dashes=...) to set the dashing when creating a line

line2, = ax.plot(df_test['Date'], y_pr['Fatalities'],label='Fatalities')



#ax.set_ylim([1, 1000000])

#plt.yscale('log')

ax.set_xlabel('Date')

ax.legend()

plt.show()
submission[['ConfirmedCases','Fatalities']]=y_pr[['ConfirmedCases','Fatalities']]
submission.to_csv('submission.csv', index= False)
