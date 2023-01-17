import pandas as pd

import numpy as np

from datetime import datetime
trip = pd.read_csv('../input/trip.csv')

station = pd.read_csv('../input/station.csv')

weather = pd.read_csv('../input/weather.csv')
trip.head()
station.head()
weather.head()
result = trip.groupby('id')['start_date'].count().sort_values(ascending = False)

result.head()
trip.isnull().sum()
weather.isnull().sum()
station.isnull().sum()
df1 = trip.drop(columns = ['zip_code'])
##Transform start and end date to datetime objects

df1['start_date'] = df1['start_date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M'))

df1['end_date'] = df1['end_date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M'))

##Extracc only year, month and date to join on weather data later on

df1['date_for_join'] = df1['start_date'].apply(lambda x: x.strftime('%Y-%m-%d'))

df1['date_for_join'] = df1['date_for_join'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
city_zip = pd.DataFrame({'city': ['San Jose', 'Redwood City', 'Mountain View', 'Palo Alto','San Francisco'], \

                         'zip_for_join': [95113,94063,94041,94301,94107]})

merge1 = station.merge(city_zip, how = 'left', left_on = 'city', right_on = 'city')
merge2 = merge1.copy()

merge2.columns = ['start_station_id','start_name','start_lat','start_long','start_dock_count','start_city','start_installation_date','start_zip']
merge3 = merge1.copy()

merge3.columns =  ['end_station_id','end_name','end_lat','end_long','end_dock_count','end_city','end_installation_date','end_zip']
merge4 = df1.merge(merge2, how = 'left', left_on = 'start_station_id',right_on = 'start_station_id')
merge5 = merge4.merge(merge3,how = 'left', left_on = 'end_station_id',right_on = 'end_station_id' )
merge6 = merge5.drop(columns = ['start_name','end_name'])
weather['date'] = weather['date'].apply(lambda x: datetime.strptime(x,'%m/%d/%Y'))
start_weather = weather.copy()

columns = list(start_weather.columns)

new_columns = []

for i in columns:

    i = 'start_' + i

    new_columns.append(i)

start_weather.columns = new_columns
end_weather = weather.copy()

columns = list(end_weather.columns)

new_columns = []

for i in columns:

    i = 'end_' + i

    new_columns.append(i)

end_weather.columns = new_columns
merge7 = merge6.merge(start_weather, how = 'left', left_on = ['date_for_join','start_zip'], \

                      right_on = ['start_date','start_zip_code'])
merge8 = merge7.merge(end_weather,how = 'left', left_on = ['date_for_join','end_zip'], \

                      right_on = ['end_date','end_zip_code'])
merge8.head(5).transpose()
merge9 = merge8.drop(columns = ['end_zip_code','end_date_y','start_date_y',\

                                'start_zip_code','date_for_join'])

merge9.rename(columns={'start_date_x':'start_date','end_date_x':'end_date'}, inplace=True)
merge9.shape
na_list = pd.DataFrame(merge9.isnull().sum())

na_list['column_name'] = na_list.index

na_list.columns = ['count_na','column_name']

na_column = na_list[na_list['count_na']>0]
na_column.sort_values(by = 'count_na')
merge9['start_events'] = merge9['start_events'].fillna('No Special Events')
merge9['end_events'] = merge9['end_events'].fillna('No Special Events')
merge9 = merge9.drop(columns = ['start_max_gust_speed_mph','end_max_gust_speed_mph'])
merge10 = merge9.dropna()
merge10.shape
merge10.isna().sum()
merge10.to_csv('SF_Bay_Area_Bike_Share_Data_Cleaned.csv', index = False)