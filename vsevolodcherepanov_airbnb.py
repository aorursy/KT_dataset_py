import numpy as np

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv('/kaggle/input/airbnb-user-pathways/airbnb.csv')
df.isna().sum()
df.nunique()
import datetime

df_dropped=df[['id_visitor','dim_session_number','dim_device_app_combo','ds','ts_min',

              'ts_max','did_search','sent_message','sent_booking_request']]

for n in range(df_dropped.shape[0]):

    df_dropped['ts_max'][n]=datetime.datetime.strptime(df_dropped['ts_max'][n], '%Y-%m-%d %H:%M:%S')

    df_dropped['ts_min'][n]=datetime.datetime.strptime(df_dropped['ts_min'][n], '%Y-%m-%d %H:%M:%S')



df_dropped['session_time']=df_dropped['ts_max']-df_dropped['ts_min']

for n in range(df_dropped.shape[0]):

    df_dropped['session_time'][n]=df_dropped['session_time'][n].seconds

df_dropped.drop(['ts_min','ts_max'],axis=1,inplace=True)
print('unique entries for dim_device_app_combo:',len(df_dropped['dim_device_app_combo'].unique()))

print('unique entries for id_visitor:',len(df_dropped['id_visitor'].unique()))
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df_dropped['id_visitor'] = label_encoder.fit_transform(df_dropped['id_visitor'])
import seaborn as sns

import matplotlib.pyplot as plt

df_dropped['session_time, min']=df_dropped['session_time']/60

plt.figure(figsize=(20,10))

plt.xlim(0, 200)

sns.kdeplot(df_dropped['session_time, min'], )

print('median session time, min:', df_dropped['session_time, min'].median())

print('mean session time, min:', df_dropped['session_time, min'].mean())
search=df_dropped[(df_dropped['did_search']==1)]

search_sent=df_dropped[(df_dropped['did_search']==1) & (df_dropped['sent_message']==1)]

sent_booking=df_dropped[(df_dropped['sent_booking_request']==1) & (df_dropped['sent_message']==1)]

search_booking=df_dropped[(df_dropped['did_search']==1) & (df_dropped['sent_booking_request']==1)]

booking=df_dropped[(df_dropped['sent_booking_request']==1)]



dicti=pd.DataFrame([{'conversion rate search & booking / search':search_booking.shape[0]/search.shape[0]},

       {'conversion rate search & booking / search&sent a message':search_booking.shape[0]/search_sent.shape[0]},

       {'conversion rate sent a message & booking / booking':sent_booking.shape[0]/booking.shape[0]}])

fig, ax = plt.subplots()

dicti.plot(kind='barh', figsize=(10,10))

ax.set_xlabel('Number of bookings')

df_dropped[['dim_device_app_combo','sent_booking_request']].groupby('dim_device_app_combo').sum().sort_values(by='sent_booking_request').plot(kind='barh', figsize=(10,10), ax=ax)

sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.distplot(search["session_time, min"],fit_kws={'linewidth':5})

sns.distplot(search_sent["session_time, min"])

sns.distplot(search_booking["session_time, min"])

plt.legend(labels=['just search','search&sent a message','search&request a booking'],prop={'size': 20})

plt.show()
df_dropped['dim_device_app_combo'] = label_encoder.fit_transform(df_dropped['dim_device_app_combo'])

df_dropped.drop('ds',axis=1,inplace=True)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score,train_test_split

y_train=df_dropped['sent_booking_request']

X_train=df_dropped.drop(['sent_booking_request'], axis=1)

ran = RandomForestClassifier()

ran.fit(X_train,y_train)

crossval=cross_val_score(ran, X_train, y_train,cv=5)

print('mean cross-validation score:', crossval.mean())