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

import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression import linear_model

dat = pd.read_csv('/kaggle/input/chicago-divvy-bicycle-sharing-data/data.csv')
dat.head()
dat_s = dat.sample(n=int(dat.shape[0]*0.2), random_state=1)
dat_s.shape
# Visualize the missing values as a bar chart 
msno.bar(dat_s) 
# based on the week number, derive weekend and weekday flag var. We suppose the trip distribution is different in weekday and weekend 
dat_st['weekend_flag'] = dat_st.apply(lambda row: 1 if row.day== 5 or row.day==6 else 0, axis=1)
# based on the hour, we derive rush hour / none rush hour 
dat_st['rush_hour_flag'] = dat_st.apply(lambda row: 1 if row.hour == 8 or row.hour == 9 or row.hour == 12 or row.hour == 17 or row.hour == 18 else 0, axis=1)
# trip freq by usertype, gender, events

trip_usertype = dat_st.groupby('usertype').trip_id.count().reset_index()
trip_gender = dat_st.groupby('gender').trip_id.count().reset_index()
trip_events = dat_st.groupby('events').trip_id.count().reset_index()
trip_events['pct'] = trip_events['trip_id']/trip_events['trip_id'].sum();trip_events
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20,5))

ax1.bar(trip_usertype.usertype, trip_usertype.trip_id)
ax1.set_title('User type distribution')
ax1.set_xlabel('User type')
ax1.set_ylabel('Freq')
ax1.yaxis.grid()

ax2.bar(trip_gender.gender, trip_gender.trip_id)
ax2.set_title('Gender distribution')
ax2.set_xlabel('Gender')
ax2.set_ylabel('Freq')
ax2.yaxis.grid()

ax3.bar(trip_events.events, trip_events.trip_id)
ax3.set_title('Events distribution')
ax3.set_xlabel('Events')
ax3.set_ylabel('Freq')
plt.xticks(rotation=45)
ax3.yaxis.grid()
dat_st['temper_round'] = round(dat_st.temperature)
momth_temp_trip = dat_st.groupby(['temper_round','month']).trip_id.count().reset_index()
fig, ([ax1,ax2],[ax3, ax4],[ax5,ax6],[ax7,ax8], [ax9,ax10], [ax11,ax12]) = plt.subplots(nrows=6, ncols=2, figsize=(30,10))

ax1.bar(momth_temp_trip[momth_temp_trip.month==1].temper_round, momth_temp_trip[momth_temp_trip.month==1].trip_id)
# ax1.set_title('Jan')
ax1.set_xlabel('Temperature')
ax1.set_ylabel('Jan Freq')
ax1.yaxis.grid()
ax1.set_xlim([-10, 90])

ax2.bar(momth_temp_trip[momth_temp_trip.month==2].temper_round, momth_temp_trip[momth_temp_trip.month==2].trip_id)
# ax2.set_title('Feb')
ax2.set_xlabel('Temperature')
ax2.set_ylabel('Feb Freq')
ax2.yaxis.grid()
ax2.set_xlim([-10, 90])

ax3.bar(momth_temp_trip[momth_temp_trip.month==3].temper_round, momth_temp_trip[momth_temp_trip.month==3].trip_id)
# ax3.set_title('Mar')
ax3.set_xlabel('Temperature')
ax3.set_ylabel('Mar Freq')
ax3.yaxis.grid()
ax3.set_xlim([-10, 90])

ax4.bar(momth_temp_trip[momth_temp_trip.month==4].temper_round, momth_temp_trip[momth_temp_trip.month==4].trip_id)
# ax4.set_title('Apr')
ax4.set_xlabel('Temperature')
ax4.set_ylabel('Apr Freq')
ax4.yaxis.grid()
ax4.set_xlim([-10, 90])

ax5.bar(momth_temp_trip[momth_temp_trip.month==5].temper_round, momth_temp_trip[momth_temp_trip.month==5].trip_id)
# ax5.set_title('May')
ax5.set_xlabel('Temperature')
ax5.set_ylabel('May Freq')
ax5.yaxis.grid()
ax5.set_xlim([-10, 90])

ax6.bar(momth_temp_trip[momth_temp_trip.month==6].temper_round, momth_temp_trip[momth_temp_trip.month==6].trip_id)
# ax6.set_title('June')
ax6.set_xlabel('Temperature')
ax6.set_ylabel('June Freq')
ax6.yaxis.grid()
ax6.set_xlim([-10, 90])

ax7.bar(momth_temp_trip[momth_temp_trip.month==7].temper_round, momth_temp_trip[momth_temp_trip.month==7].trip_id)
# ax7.set_title('July')
ax7.set_xlabel('Temperature')
ax7.set_ylabel('July Freq')
ax7.yaxis.grid()
ax7.set_xlim([-10, 90])

ax8.bar(momth_temp_trip[momth_temp_trip.month==8].temper_round, momth_temp_trip[momth_temp_trip.month==8].trip_id)
# ax8.set_title('Aug')
ax8.set_xlabel('Temperature')
ax8.set_ylabel('Aug Freq')
ax8.yaxis.grid()
ax8.set_xlim([-10, 90])

ax9.bar(momth_temp_trip[momth_temp_trip.month==9].temper_round, momth_temp_trip[momth_temp_trip.month==9].trip_id)
# ax9.set_title('Sep')
ax9.set_xlabel('Temperature')
ax9.set_ylabel('Sep Freq')
ax9.yaxis.grid()
ax9.set_xlim([-10, 90])

ax10.bar(momth_temp_trip[momth_temp_trip.month==10].temper_round, momth_temp_trip[momth_temp_trip.month==10].trip_id)
# ax10.set_title('Oct')
ax10.set_xlabel('Temperature')
ax10.set_ylabel('Oct Freq')
ax10.yaxis.grid()
ax10.set_xlim([-10, 90])

ax11.bar(momth_temp_trip[momth_temp_trip.month==11].temper_round, momth_temp_trip[momth_temp_trip.month==11].trip_id)
# ax11.set_title('Nov')
ax11.set_xlabel('Temperature')
ax11.set_ylabel('Nov Freq')
ax11.yaxis.grid()
ax11.set_xlim([-10, 90])

ax12.bar(momth_temp_trip[momth_temp_trip.month==12].temper_round, momth_temp_trip[momth_temp_trip.month==12].trip_id)
# ax12.set_title('Dec')
ax12.set_xlabel('Temperature')
ax12.set_ylabel('Dec Freq')
ax12.set_xlim([-10, 90])
ax12.yaxis.grid()
def season_derive(month):
    if month>=12 or month <=2:
        return 'winter'
    if month >=3 and month <=6:
        return 'spring'
    if month >=7 and month <=9:
        return 'summer'
    else:
        return 'fall'
dat_st['season'] = dat_st.apply(lambda row: season_derive(row.month), axis=1)
dim = ['year', 'month', 'day', 'hour', 'usertype', 'gender', 'events', 'weekend_flag', 'season']
dat_agg = dat_st.groupby(dim).aggregate({'temperature': 'mean',
                                         'tripduration': 'mean',
                                         'trip_id': 'count'
                                         }).reset_index()
dat_agg2 = dat_agg.groupby(['season', 'hour']).aggregate({'tripduration': 'mean'}).reset_index()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30,10))

ax.plot(dat_agg2[dat_agg2.season=='summer'].hour, dat_agg2[dat_agg2.season=='summer'].tripduration, label='summer')
ax.plot(dat_agg2[dat_agg2.season=='winter'].hour, dat_agg2[dat_agg2.season=='winter'].tripduration, label='winter')
ax.plot(dat_agg2[dat_agg2.season=='spring'].hour, dat_agg2[dat_agg2.season=='spring'].tripduration, label='spring')
ax.plot(dat_agg2[dat_agg2.season=='fall'].hour, dat_agg2[dat_agg2.season=='fall'].tripduration, label='fall')

ax.legend(loc='lower right')
ax.set_title('Average trip duration by season and hour')
ax.set_xlabel('hour')
ax.set_ylabel('trip duration mean')

ax.yaxis.grid()

day_hour_trip = dat_st.groupby(['day', 'hour']).trip_id.count().reset_index()
fig, ([ax1,ax2], [ax3,ax4],[ax5,ax6],[ax7, ax8]) = plt.subplots(nrows=4, ncols=2, figsize=(25,15))

ax1.bar(day_hour_trip[day_hour_trip.day==0].hour, day_hour_trip[day_hour_trip.day==0].trip_id)
ax1.set_title('Monday')
ax1.set_xlabel('Hour')
ax1.set_ylabel('Freq')
ax1.yaxis.grid()

ax2.bar(day_hour_trip[day_hour_trip.day==1].hour, day_hour_trip[day_hour_trip.day==1].trip_id)
ax2.set_title('Tuesday')
ax2.set_xlabel('Hour')
ax2.set_ylabel('Freq')
ax2.yaxis.grid()

ax3.bar(day_hour_trip[day_hour_trip.day==2].hour, day_hour_trip[day_hour_trip.day==2].trip_id)
ax3.set_title('Wednesday')
ax3.set_xlabel('Hour')
ax3.set_ylabel('Freq')
ax3.yaxis.grid()

ax4.bar(day_hour_trip[day_hour_trip.day==3].hour, day_hour_trip[day_hour_trip.day==3].trip_id)
ax4.set_title('Thursday')
ax4.set_xlabel('Hour')
ax4.set_ylabel('Freq')
ax4.yaxis.grid()

ax5.bar(day_hour_trip[day_hour_trip.day==4].hour, day_hour_trip[day_hour_trip.day==4].trip_id)
ax5.set_title('Friday')
ax5.set_xlabel('Hour')
ax5.set_ylabel('Freq')
ax5.yaxis.grid()

ax6.bar(day_hour_trip[day_hour_trip.day==5].hour, day_hour_trip[day_hour_trip.day==5].trip_id)
ax6.set_title('Saturday')
ax6.set_xlabel('Hour')
ax6.set_ylabel('Freq')
ax6.yaxis.grid()

ax7.bar(day_hour_trip[day_hour_trip.day==6].hour, day_hour_trip[day_hour_trip.day==6].trip_id)
ax7.set_title('Sunday')
ax7.set_xlabel('Hour')
ax7.set_ylabel('Freq')
ax7.yaxis.grid()

dat_daily = dat_st.groupby(['year','month', 'week', 'day']).agg({'temperature': 'mean', 'events': lambda x:x.value_counts().index[0],
                                                      'trip_id': 'count', 'tripduration':'sum'}).reset_index()
dat_daily['avg_duration'] = dat_daily['tripduration']/ dat_daily['trip_id']
dat_daily.head()
dat_daily['year'] = dat_daily['year'].astype(str)
dat_daily['month'] = dat_daily['month'].astype(str)
dat_daily['week'] = dat_daily['week'].astype(str)
dat_daily['day'] = dat_daily['day'].astype(str)

dat_daily_dummy = pd.get_dummies(dat_daily, prefix=['year', 'month', 'week', 'day', 'events'])
dat_daily
x = dat_daily_dummy.drop(['trip_id', 'avg_duration'], axis=1)
y = dat_daily_dummy['trip_id']

model = linear_model.OLS(y,x).fit()
model.summary()