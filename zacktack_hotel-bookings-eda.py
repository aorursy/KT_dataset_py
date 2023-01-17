# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_original = pd.read_csv("/kaggle/input/hotel-booking-demand/hotel_bookings.csv")

df_original.head()
df_original.tail()
df_original = pd.concat([df_original,df_original[(df_original.arrival_date_year == 2017) & ((df_original.arrival_date_month == 'August') | (df_original.arrival_date_month == 'July'))]]).drop_duplicates(keep=False)
df_original.tail()
display(f'Dataset shape is: {df_original.shape}',df_original.describe())
ax = sns.distplot(df_original.is_canceled,kde=False,color='r')

plt.title('Not Canceled VS. Canceled')

for p in ax.patches:

    if p.get_height() > 0: ax.text(p.get_x()+p.get_width()/2,p.get_height(),f"{int(p.get_height())}",fontsize=16) 
bookings_by_year_not_canceled = df_original[df_original.is_canceled == 0].groupby('arrival_date_year').arrival_date_year.count()

ax = sns.barplot(bookings_by_year_not_canceled.index,bookings_by_year_not_canceled.values)

plt.title('Number of Confirmed Bookings')

for p in ax.patches:

    ax.text(p.get_x()+p.get_width()/4,p.get_height(),f"{int(p.get_height())}",fontsize=16)
booking_by_year_month_not_canceled = df_original[df_original.is_canceled == 0].groupby(['arrival_date_year','arrival_date_month']).arrival_date_month.count().sort_values(ascending=False)

plt.figure(figsize=(10,30))

plt.title('Number of Confirmed Bookings by Year-Month')

ax = sns.barplot(booking_by_year_month_not_canceled.values,booking_by_year_month_not_canceled.index)

for p in ax.patches:

    ax.text(p.get_width(),p.get_y()+p.get_height()/2,f"{int(p.get_width())}",fontsize=16)
booking_by_month_not_canceled = df_original[df_original.is_canceled == 0].groupby('arrival_date_month').arrival_date_month.count().sort_values(ascending=False)

plt.figure(figsize=(10,15))

plt.title('Number of Confirmed Bookings by Month')

ax = sns.barplot(booking_by_month_not_canceled.values,booking_by_month_not_canceled.index)

for p in ax.patches:

    ax.text(p.get_width(),p.get_y() + p.get_height()/2,f"{int(p.get_width())}",fontsize=20)
bookings_by_year_canceled = df_original[df_original.is_canceled == 1].groupby('arrival_date_year').arrival_date_year.count()

ax = sns.barplot(bookings_by_year_canceled.index,bookings_by_year_canceled.values)

plt.title('Number of Canceled Bookings')

for p in ax.patches:

    ax.text(p.get_x()+p.get_width()/4,p.get_height(),f"{int(p.get_height())}",fontsize=16)
booking_by_year_month_canceled = df_original[df_original.is_canceled == 1].groupby(['arrival_date_year','arrival_date_month']).arrival_date_month.count().sort_values(ascending=False)

plt.figure(figsize=(10,30))

plt.title('Number of Canceled Bookings by Year-Month')

sns.set(font_scale=1.1)

ax = sns.barplot(booking_by_year_month_canceled.values,booking_by_year_month_canceled.index)

for p in ax.patches:

    ax.text(p.get_width(),p.get_y()+p.get_height()/2,f"{int(p.get_width())}",fontsize=16)
booking_by_month_canceled = df_original[df_original.is_canceled == 1].groupby('arrival_date_month').arrival_date_month.count().sort_values(ascending=False)

plt.figure(figsize=(10,15))

plt.title('Number of Canceled Bookings by Month')

ax = sns.barplot(booking_by_month_canceled.values,booking_by_month_canceled.index)

for p in ax.patches:

    ax.text(p.get_width(),p.get_y() + p.get_height()/2,f"{int(p.get_width())}",fontsize=20)
cancel_rate_by_month = ((df_original[df_original.is_canceled == 1].groupby('arrival_date_month').arrival_date_month.count() / df_original[df_original.is_canceled == 0].groupby('arrival_date_month').arrival_date_month.count()) * 100).sort_values(ascending=False)

plt.figure(figsize=(10,15))

ax = sns.barplot(cancel_rate_by_month.values,cancel_rate_by_month.index)

plt.title('Cancel Rates by Month')

for p in ax.patches:

    ax.text(p.get_width(),p.get_y() + p.get_height()/2,f"{round(p.get_width(),2)}%",fontsize=20)
hotel_dist = pd.concat([df_original.hotel.value_counts(),df_original[df_original.is_canceled == 0].hotel.value_counts(),df_original[df_original.is_canceled==1].hotel.value_counts()],axis=1)

hotel_dist.columns = ['Total','Confirmed','Canceled']

ax = hotel_dist.plot.bar(rot=0,figsize=(8,6))

for p in ax.patches:

    ax.text(p.get_x(),p.get_height(),f"{int(p.get_height())}",fontsize=16)
df_original.lead_time.agg(['min','mean','max'])
lead_time_diff = pd.concat([df_original[df_original.is_canceled == 0].lead_time.agg(['min','mean','max']),df_original[df_original.is_canceled == 1].lead_time.agg(['min','mean','max'])],axis=1)

lead_time_diff.columns=['Confirmed','Canceled']

ax = lead_time_diff.plot.bar(rot=0,figsize=(8,6))

for p in ax.patches:

    ax.text(p.get_x(),p.get_height(),f"{int(p.get_height())}",fontsize=16)
zero_night_stays = df_original[(df_original.stays_in_weekend_nights == 0) & (df_original.stays_in_week_nights == 0)]

zero_night_stays
zero_night_stays.is_canceled.value_counts()