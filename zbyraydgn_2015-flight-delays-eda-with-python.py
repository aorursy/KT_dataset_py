import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime, warnings, scipy 



import plotly.offline as py

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

from plotly import tools

from plotly.subplots import make_subplots
flights = pd.read_csv("../input/flight-delays/flights.csv")
airports = pd.read_csv("../input/flight-delays/airports.csv")

airlines = pd.read_csv("../input/flight-delays/airlines.csv")
flights.info()
airports.info()
airlines.info()
flights.describe()
flights.head()

flights.tail()
airlines.head()
airlines.tail()
airports.head(10)
airports.tail(15)
airports.isnull().sum()
airports = airports.dropna(subset = ["LATITUDE","LONGITUDE"])
airports.isnull().sum()
flights_null = flights.isnull().sum()*100/flights.shape[0]

flights_null
flights1 = flights.dropna(subset = ["TAIL_NUMBER",'DEPARTURE_TIME','DEPARTURE_DELAY','TAXI_OUT','WHEELS_OFF','SCHEDULED_TIME',

             'ELAPSED_TIME','AIR_TIME','WHEELS_ON','TAXI_IN','ARRIVAL_TIME','ARRIVAL_DELAY'])
flights1.isnull().sum()
# Creting Dataset w.r.t different Types of Delays

flights11 = flights1.dropna(subset = ['AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY'])

flights11 = flights11.drop(['YEAR','MONTH','DAY','DAY_OF_WEEK','TAIL_NUMBER','SCHEDULED_DEPARTURE','DEPARTURE_TIME','SCHEDULED_TIME',

                     'SCHEDULED_ARRIVAL','ARRIVAL_TIME','DIVERTED','CANCELLED','CANCELLATION_REASON','FLIGHT_NUMBER','WHEELS_OFF',

                     'WHEELS_ON','AIR_TIME'],axis = 1)

flights11.info()
# The other Dataset

Flight_Delays = flights11
# Creating Dataset by removing null values by not focussing fully on different types of Delays

flights2 = flights1.drop(['CANCELLATION_REASON','AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY',

                    'LATE_AIRCRAFT_DELAY','WEATHER_DELAY'],axis = 1)

flights2.isnull().sum()
flights2.info()
#we need to change the data type to datetime format

flights2.DEPARTURE_TIME.head(10)
# Creating a function to change the way of representation of time in the column

def Format_Hourmin(hours):

        if hours == 2400:

            hours = 0

        else:

            hours = "{0:04d}".format(int(hours))

            Hourmin = datetime.time(int(hours[0:2]), int(hours[2:4]))

            return Hourmin
flights2['Actual_Departure'] =flights1['DEPARTURE_TIME'].apply(Format_Hourmin)

flights2['Actual_Departure'].head(10)
flights2.columns
flights2['DATE'] = pd.to_datetime(flights2[['YEAR','MONTH','DAY']])

flights2.DATE.head(10)
flights2['DAY'] = flights2['DATE'].dt.weekday_name
# Applying the function to required variables in the dataset

flights2['Actual_Departure'] =flights2['DEPARTURE_TIME'].apply(Format_Hourmin)

flights2['Scheduled_Arrival'] =flights2['SCHEDULED_ARRIVAL'].apply(Format_Hourmin)

flights2['Scheduled_Departure'] =flights2['SCHEDULED_DEPARTURE'].apply(Format_Hourmin)

flights2['Actual_Arrival'] =flights2['ARRIVAL_TIME'].apply(Format_Hourmin)
# Merging on AIRLINE and IATA_CODE

flights2 = flights2.merge(airlines, left_on='AIRLINE', right_on='IATA_CODE', how='inner')
flights2 = flights2.drop(['AIRLINE_x','IATA_CODE'], axis=1)

flights2 = flights2.rename(columns={"AIRLINE_y":"AIRLINE"})
flights2 = flights2.merge(airports, left_on='ORIGIN_AIRPORT', right_on='IATA_CODE', how='inner')

flights2 = flights2.merge(airports, left_on='DESTINATION_AIRPORT', right_on='IATA_CODE', how='inner')
flights2.columns
flights2 = flights2.drop(['LATITUDE_x', 'LONGITUDE_x',

       'STATE_y', 'COUNTRY_y', 'LATITUDE_y', 'LONGITUDE_y','STATE_x', 'COUNTRY_x'], axis=1)

flights2 = flights2.rename(columns={'IATA_CODE_x':'Org_Airport_Code','AIRPORT_x':'Org_Airport_Name','CITY_x':'Origin_city',

                             'IATA_CODE_y':'Dest_Airport_Code','AIRPORT_y':'Dest_Airport_Name','CITY_y':'Destination_city'})

flights2
#airports with the most flights



F=flights2.Org_Airport_Name.value_counts().sort_values(ascending=False)[:20]

print(F)
F=flights2.Org_Airport_Name.value_counts().sort_values(ascending=False)[:15]

label=F.index

size=F.values

colors = ['skyblue', '#FEBFB3', '#96D38C', '#D0F9B1', 'gold', 'orange', 'lightgrey', 

          'lightblue','lightgreen','aqua','yellow','#D4E157','#D1C4E9','#1A237E','#64B5F6','#009688',

          '#1DE9B6','#66BB6A','#689F38','#FFB300']

trace =go.Pie(labels=label, values=size, marker=dict(colors=colors), hole=.1)

data = [trace]

layout = go.Layout(title='Origin Airport Distribution')

fig=go.Figure(data=data,layout=layout)

py.iplot(fig)
#cities with the most flights

F=flights2.Origin_city.value_counts().sort_values(ascending=False)[:15]

print(F)
#Cities from most flights

F=flights2.Destination_city.value_counts().sort_values(ascending=False)[:15]

print(F)
F=flights2.Origin_city.value_counts().sort_values(ascending=False)[:15]



trace1 = go.Bar(x=F.index,y=F.values,marker=dict(color = '#009688'))

                                                      

F = flights2.Destination_city.value_counts().sort_values(ascending=False)[:15]



trace2 = go.Bar(x=F.index, y=F.values, marker=dict( color ='#689F38' ))

 

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Origin City','Destination City'))                                                        

 

fig.add_trace(trace1, 1,1) 

fig.add_trace(trace2, 1,2)  

fig['layout'].update(yaxis = dict(title = 'Values'), height=500, width=900, showlegend=False)

py.iplot(fig)                                                          
F=flights2.AIRLINE.value_counts().sort_values(ascending=False)[:7]

print(F)
F=flights2.AIRLINE.value_counts().sort_values(ascending=True)[:7]

print(F)
#airlines flight rankings

F=flights2.AIRLINE.value_counts().sort_values(ascending=False)[:8]

trace1 = go.Scatter(x=F.index, y=F.values,name='Most Flights',marker=dict(color='blue'))



F=flights2.AIRLINE.value_counts().sort_values(ascending=True)[:7].iloc[::-1]

trace2 = go.Scatter(x=F.index, y=F.values,name='Least Flights',marker=dict(color='yellow'))



data=[trace1,trace2]

layout = dict(title = 'Airline distribution')

fig = dict(data=data,layout = layout)

py.iplot(fig)



F=flights2.MONTH.value_counts().to_frame().reset_index().sort_values(by='index')

F.columns=['Month','Flight_Values']

print(F)
#number of flights per month

F=flights2.MONTH.value_counts().to_frame().reset_index().sort_values(by='index')

F.columns=['Month','Flight_Values']

Month={1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',6: 'Jun',

       7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

F.Month=F.Month.map(Month)

colors = ['skyblue', '#FEBFB3', '#96D38C', '#D0F9B1','lightblue','lightgreen',

          'aqua','yellow','#D4E157','#D1C4E9','#1A237E','#64B5F6']

trace=go.Bar(x=F.Month,y=F.Flight_Values,marker=dict(color=colors))

data=[trace]

layout = go.Layout(title='Monthly Flights',yaxis=dict(title='Flights Value'),height=500, width=800)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)   



#number of flights per week

Flight_Valume=flights2.pivot_table(index='Origin_city',columns='DAY_OF_WEEK',values='DAY',aggfunc=lambda x:x.count())

F=Flight_Valume.sort_values(by=1,ascending=False)[:7]

print(F)

Flight_Valume=flights2.pivot_table(index='Origin_city',columns='DAY_OF_WEEK',values='DAY',aggfunc=lambda x:x.count())

F=Flight_Valume.sort_values(by=1,ascending=False)[:7]

F=F.iloc[::-1]

fig = plt.figure(figsize=(16,9))

sns.heatmap(F, cmap='RdBu', linecolor='black', linewidths=1)

plt.title('Air Traffic by Cities',size=16)

plt.ylabel('City',size=16)

plt.xticks(rotation=45)

plt.show()
#number of departure delays

F=flights2.groupby('AIRLINE').DEPARTURE_DELAY.mean().to_frame().sort_values(by='DEPARTURE_DELAY',ascending=False).round(3)

print(F)

#number of arrival delays

F=flights2.groupby('AIRLINE').ARRIVAL_DELAY.mean().to_frame().sort_values(by='ARRIVAL_DELAY',ascending=False).round(3)

print(F)
F=flights2.groupby('AIRLINE').DEPARTURE_DELAY.mean().to_frame().sort_values(by='DEPARTURE_DELAY',ascending=False).round(3)

trace1=go.Bar(x=F.index, y=F.DEPARTURE_DELAY ,name='DEPARTURE_DELAY', marker=dict(color='navy'))

F=flights2.groupby('AIRLINE').ARRIVAL_DELAY.mean().to_frame().sort_values(by='ARRIVAL_DELAY',ascending=False).round(3)

trace2=go.Bar(x=F.index, y=F.ARRIVAL_DELAY, name='ARRIVAL_DELAY', marker=dict(color='red'))

data=[trace1, trace2]

layout= go.Layout(xaxis=dict(tickangle=90),title='Mean Arrival & Departure Delay by Airlines', yaxis=dict(title='Minute'), barmode='stack')

fig=go.Figure(data=data, layout=layout)

py.iplot(fig)

flights2['Delay_Difference'] = flights2['DEPARTURE_DELAY'] - flights2['ARRIVAL_DELAY']

F=flights2.groupby('AIRLINE').Delay_Difference.mean().to_frame().sort_values(by='Delay_Difference', ascending=False).round(3)

print(F)
flights2['Delay_Difference'] = flights2['DEPARTURE_DELAY'] - flights2['ARRIVAL_DELAY']

F=flights2.groupby('AIRLINE').Delay_Difference.mean().to_frame().sort_values(by='Delay_Difference', ascending=False).round(3)

trace=go.Bar(x=F.index, y=F.Delay_Difference, marker=dict(color = F.Delay_Difference, colorscale='Cividis',showscale=True))

data=[trace]

layout = go.Layout(xaxis=dict(tickangle=45),title='Mean (Departure Delay - Arrival Delay) by Airlines', yaxis = dict(title = 'Minute'))

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)                                
flights2['Taxi_Difference'] = flights2['TAXI_OUT'] - flights2['TAXI_IN']

F = flights2.groupby('AIRLINE').Taxi_Difference.mean().to_frame().sort_values(by='Taxi_Difference',ascending=False).round(2)

print(F)
flights2['Taxi_Difference'] = flights2['TAXI_OUT'] - flights2['TAXI_IN']

F = flights2.groupby('AIRLINE').Taxi_Difference.mean().to_frame().sort_values(by='Taxi_Difference',ascending=False).round(2)



trace = go.Bar(x=F.index, y=F.Taxi_Difference, marker=dict(color = F.Taxi_Difference,colorscale='viridis',showscale=True))



data = [trace]

layout = go.Layout(xaxis=dict(tickangle=45),title='Mean (Taxi Out - Taxi In) by Airlines', yaxis = dict(title = 'Minute'))

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)