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
import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
#import data set



df_train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")

df_test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")
df_train.head(10)
df_train.describe()
df_train['Country_Region'].value_counts()
df_train.info()
df_test.info()
df_global = df_train[df_train['Country_Region']!='China' ].groupby(

    ['Date', 'Country_Region'], as_index=False

        ).agg(

            {

                'ConfirmedCases': sum,

                'Fatalities':sum

            }

            )

df_global.head()
#Change Date column to type datetime

df_global['Date'] = pd.to_datetime(df_global['Date'])
df_global['Date'].head()
#This is not best practice, legend is too long and colors are indistinguishable to my color blinded eyes, I tried to find a way

# to annote country name to each line, but failed.



plt.figure(figsize=(16,8))

plt.title('Global Confirmed Case Trend')

sns.lineplot(x='Date', y='ConfirmedCases', data=df_global, hue='Country_Region', legend='brief')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.figure(figsize=(12,6))

plt.title('Global Deceased Case Trend')

sns.lineplot(x='Date', y='Fatalities', data=df_global, hue='Country_Region', legend='brief')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# import plotly to create a interactive chart

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

init_notebook_mode()
fig1 = go.Figure()

for region, country in df_global.groupby('Country_Region'):

    fig1.add_scatter(x=country.Date, y=country.ConfirmedCases, name=region, mode='lines')



iplot(fig1)
fig2 = go.Figure()

for region, country in df_global.groupby('Country_Region'):

    fig2.add_scatter(x=country.Date, y=country.Fatalities, name=region, mode='lines')



iplot(fig2)
# get a list of Country names that are top ten in Confirmed Cases on last day in Train dataset

#They are 'US','Italy','Spain','Germany','France','Iran','United Kingdom','Switzerland','Korea, South','Netherlands'

# Udate, South Korea is no longger among the top ten countries in terms of total confirmed cases

top_fifteen=df_global[df_global['Date']==df_global['Date'].max()].sort_values(

    by=['ConfirmedCases'], ascending =False)['Country_Region'].head(15)

top_fifteen
df_fifteen = df_global[df_global.Country_Region.isin(top_fifteen)]

fig3 = go.Figure()

for region, country in df_fifteen.groupby('Country_Region'):

    fig3.add_scatter(x=country.Date, y=country.ConfirmedCases, name=region, mode='lines')



iplot(fig3)

fig4 = go.Figure()

for region, country in df_fifteen.groupby('Country_Region'):

    fig4.add_scatter(x=country.Date, y=country.Fatalities, name=region, mode='lines')



iplot(fig4)
df_cn = df_train[df_train['Country_Region']=='China' ].groupby(

    ['Date', 'Country_Region'], as_index=False

        ).agg(

            {

                'ConfirmedCases': sum,

                'Fatalities':sum

            }

            )

df_cn.head()
#change Column date to datetime datatype

df_cn['Date'] = pd.to_datetime(df_cn['Date'])
fig5 = go.Figure()

for region, country in df_cn.groupby('Country_Region'):

    fig5.add_scatter(x=country.Date, y=country.ConfirmedCases, name=region, mode='lines')



iplot(fig5)
fig6 = go.Figure()

for region, country in df_cn.groupby('Country_Region'):

    fig6.add_scatter(x=country.Date, y=country.Fatalities, name=region, mode='lines')



iplot(fig6)
# Try with Neural Network

#Prepare X, Y dataset

#change Province_State, Country_Region to Dummy variables

df = df_train

#change data type of Date to datetime

df['Date'] = pd.to_datetime(df['Date'])
#get # of days since the earlist date available in dataset, with the earlist date as day 1 

Day1=df['Date'].min()

df['Days'] =(df['Date']-Day1).dt.days+1

df = df.drop('Date', axis=1)

df.head()
dummy_Province = pd.get_dummies(df['Province_State'], drop_first=True)

dummy_Country = pd.get_dummies(df['Country_Region'], drop_first=True)

df = pd.concat([df.drop(['Province_State', 'Country_Region','Id'], axis=1), dummy_Province, dummy_Country], axis=1)



X= df.drop(['ConfirmedCases', 'Fatalities'], axis=1).values

y_ConfirmedCases= df['ConfirmedCases'].values

y_Fatalities=df['Fatalities'].values
from sklearn.model_selection import train_test_split



X_train1, X_test1, y_train_CC, y_test_CC = train_test_split(X,y_ConfirmedCases,test_size=0.3,random_state=101)



X_train2, X_test2, y_train_Fa, y_test_Fa = train_test_split(X,y_Fatalities,test_size=0.3,random_state=101)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Dropout
model_CC = Sequential()



model_CC.add(Dense(97, activation='relu'))

model_CC.add(Dropout(0.2))



#hidden layer

model_CC.add(Dense(56, activation='relu'))

model_CC.add(Dropout(0.2))



#hidden layer

model_CC.add(Dense(28, activation='relu'))

model_CC.add(Dropout(0.2))



model_CC.add(Dense(1))



model_CC.compile(optimizer='adam',loss='mse')
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
model_CC.fit(x=X_train1, 

          y=y_train_CC, 

          epochs=600,

          validation_data=(X_test1, y_test_CC), verbose=1,

          callbacks=[early_stop]

          )
losses_CC = pd.DataFrame(model_CC.history.history)

losses_CC.plot()
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
prediction_CC = model_CC.predict(X_test1)
mean_absolute_error(y_test_CC,prediction_CC)
np.sqrt(mean_squared_error(y_test_CC,prediction_CC))
explained_variance_score(y_test_CC,prediction_CC)
model_Fa = Sequential()



model_Fa.add(Dense(97, activation='relu'))

model_Fa.add(Dropout(0.2))



#hidden layer

model_Fa.add(Dense(56, activation='relu'))

model_Fa.add(Dropout(0.2))



#hidden layer

model_Fa.add(Dense(28, activation='relu'))

model_Fa.add(Dropout(0.2))



model_Fa.add(Dense(1))



model_Fa.compile(optimizer='adam',loss='mse')
model_Fa.fit(x=X_train2, 

          y=y_train_Fa, 

          epochs=600,

          validation_data=(X_test2, y_test_Fa), verbose=1,

          callbacks=[early_stop]

          )
losses_Fa = pd.DataFrame(model_Fa.history.history)

losses_Fa.plot()
prediction_Fa = model_Fa.predict(X_test2)
mean_absolute_error(y_test_Fa,prediction_Fa)
np.sqrt(mean_squared_error(y_test_Fa,prediction_Fa))

explained_variance_score(y_test_Fa,prediction_Fa)
#process test data

test = df_test

test.head()
#change data type of Date to datetime

test['Date'] = pd.to_datetime(test['Date'])
#get # of days since the earlist date available in dataset, with the earlist date as day 1 

test['Days'] =(test['Date']-Day1).dt.days+1

test = test.drop('Date', axis=1)

test.head()
dummy_Province = pd.get_dummies(test['Province_State'], drop_first=True)

dummy_Country = pd.get_dummies(test['Country_Region'], drop_first=True)

test = pd.concat([test.drop(['Province_State', 'Country_Region','ForecastId'], axis=1), dummy_Province, dummy_Country], axis=1)



test.head()
test_predict_CC = model_CC.predict(test)

test_predict_Fa = model_Fa.predict(test)
test_predict_CC.shape
test_predict_Fa
np.hstack(test_predict_Fa)
df_test.ForecastId.values
#prepare Submission

my_submission = pd.DataFrame({'ForecastId': df_test.ForecastId.values, 'ConfirmedCases': np.hstack(test_predict_CC), 'Fatalities': np.hstack(test_predict_Fa)})

my_submission.to_csv('submission.csv', index=False)