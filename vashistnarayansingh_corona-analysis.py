# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) #do not miss this line



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
# covid_recoverd = pd.read_csv("../input/Novel ")

input_dir = "../input/corona-virus-time-series-dataset/COVID-19/csse_covid_19_data/csse_covid_19_time_series/"



series_covid_19_confirmed = pd.read_csv(input_dir + "time_series_19-covid-Confirmed.csv")

series_covid_19_recovered = pd.read_csv(input_dir + "time_series_19-covid-Recovered.csv")

series_covid_19_deaths = pd.read_csv(input_dir + "time_series_19-covid-Deaths.csv")
series_covid_19_confirmed
series_covid_19_confirmed.shape
series_covid_19_recovered.head()
series_covid_19_deaths.head()
clean_data  = series_covid_19_confirmed.iloc[:,0:4]

# clean_data[clean_data['Country/Region']=='Mainland China'].shape

clean_data['Date'] = 'NA'

clean_data['Confirmed'] ='NA'

clean_data['Recovery'] ='NA'

clean_data['Death'] ='NA'

clean_data
only_date_data = series_covid_19_confirmed.iloc[:,4:]

only_date_data_columns = only_date_data.columns.tolist()

only_date_data_columns[0:6] , only_date_data.shape
clean_data.shape
# # series_covid_19_confirmed.drop(['Lat','Long'], axis=1 ,inplace=True)

# Code for cleaning the data

row_count = 0 

start_time = time.time()

for date in only_date_data_columns :

#     print(date)

    for idx,row in series_covid_19_confirmed.iterrows():

#         print(idx)

#         print(row)

        row_count = row_count + 1

#         print("row_count + ")

        if row_count > clean_data.shape[0] :

#             print(row_count)

            clean_data.loc[row_count] = [row['Province/State'],row['Country/Region'],row['Lat'],row['Long'],date,row[date],

                                         series_covid_19_recovered.loc[idx][date],series_covid_19_deaths.loc[idx][date]]

#             row_count = row_count + 1

        else:

#             print("msg from else part")

            clean_data.at[idx, 'Date'] = date

            clean_data.at[idx,'Confirmed'] = row[date]

            clean_data.at[idx,'Recovery'] = series_covid_19_recovered.loc[idx][date]

            clean_data.at[idx,'Death'] = series_covid_19_deaths.loc[idx][date]

#             row_count = row_count + 1

#         break

    print(row_count)

#     break

print("--- %s seconds ---" % (time.time() - start_time))        

# clean_data.head(500)

pwd
clean_data.to_csv('clean_covid_19_data.csv', index=False)
clean_data = pd.read_csv('clean_covid_19_data.csv',parse_dates = ['Date'])

clean_data['Country/Region'].replace('Mainland China','China',inplace=True)

clean_data['still_infected'] = clean_data['Confirmed'] - clean_data['Recovery'] - clean_data['Death']

clean_data['Province/State'] = clean_data['Province/State'].fillna('NA')



clean_data.head()
clean_data[clean_data.Death == clean_data.Death.max()] # maximun deaths
clean_data[clean_data.Recovery == clean_data.Recovery.max()] # Maximum recovery
clean_data[clean_data.Confirmed == clean_data.Confirmed.max()] # Maximum confirmed cases
total_country = list(set(clean_data['Country/Region'].values.tolist()))

total_country = sorted(total_country)

len(total_country),total_country
group_data = clean_data.groupby(['Country/Region','Province/State'])['Province/State','Confirmed','Recovery','Death','still_infected'].max()

group_data.reset_index(drop=True)

group_data.style.background_gradient(cmap='Pastel1_r')
# Taking the most recent data by date

clean_data_latest = clean_data[clean_data['Date'] == max(clean_data['Date'])].reset_index(drop=True)

clean_data_latest
 !pip install folium
import folium

from IPython.display import display



m = folium.Map(location=[0, 0], tiles='cartodbpositron',

               min_zoom=1, max_zoom=4, zoom_start=1)

tooltip = 'Click me!'

# folium.Marker([30.9756,112.2707], popup='<i>Mt. Hood Meadows</i>', tooltip=tooltip).add_to(m)

for idx, row in clean_data_latest.iterrows():

    folium.Circle(

    radius=100,

    location=[row['Lat'],row['Long']],

    tooltip='<ul><li><b>Country:</b> '+str(row['Country/Region'])+'</li><li><b>State:</b> '+str(row['Province/State'])+'</li><li><b>Confirmed Cases: </b>' + str(row['Confirmed']) + '</li><li><b>Recovery Count:</b> '+ str(row['Recovery']) + '</li><li><b>Death Count:</b> ' +str(row['Death']) + '</li><li><b>Still Affected:</b> ' + str(row['still_infected']),

    color='red',

    fill=False,

    ).add_to(m)

#     break

    



display(m)

# country wise analysis



# most confirmed cases

country_data = clean_data_latest.groupby('Country/Region')['Confirmed', 'Death', 'Recovery', 'still_infected'].sum().reset_index()

country_data.sort_values(by='Confirmed', ascending=False).reset_index(drop=True).style.background_gradient(cmap='Reds')
# country with all recovered cases

all_recovered_cases = country_data[country_data['Recovery'] == country_data['Confirmed']].reset_index(drop=True)

all_recovered_cases[['Country/Region','Confirmed','Recovery']].style.background_gradient(cmap='Greens')
# Countey reported death

country_with_death =  country_data[country_data['Death']>0].reset_index(drop=True)

country_with_death.sort_values('Death', ascending=False).style.background_gradient(cmap='Reds')
# country with all casses daied

all_cases_died =  country_data[country_data['Death'] == country_data['Confirmed']].reset_index(drop=True)

all_cases_died.style.background_gradient(cmap='Reds')
# number of states in china with no still infected patient is 

china_state_with_no_cases = clean_data_latest[clean_data_latest['Country/Region'] == 'China']

china_state_with_no_cases[china_state_with_no_cases['Confirmed'] == china_state_with_no_cases['Recovery']][['Province/State','Confirmed','Recovery']].reset_index(drop=True).style.background_gradient('Greens')
# number o states in china where all the cases died

# china_state_with_no_cases = clean_data_latest[clean_data_latest['Country/Region'] == 'China']

china_state_with_no_cases[china_state_with_no_cases['Confirmed'] == china_state_with_no_cases['Death']][['Province/State','Confirmed','Recovery']].reset_index(drop=True).style.background_gradient('Reds')
china_state_with_no_cases[china_state_with_no_cases['Confirmed'] == china_state_with_no_cases['still_infected']][['Province/State','Confirmed','Recovery']].reset_index(drop=True).style.background_gradient('Reds')
# Spread of virus over time in the world

import plotly.express as px

cnf = '#393e46' # confirmed - grey

dth = '#ff2e63' # death - red

rec = '#21bf73' # recovered - cyan

act = '#fe9801' # active case - yellow

temp = clean_data.groupby('Date')['Recovery', 'Death', 'still_infected'].sum().reset_index()

temp
temp = temp.melt(id_vars="Date", value_vars=['Recovery', 'Death', 'still_infected'],

    var_name='Case', value_name='Count')

# temp

# temp[temp['Case'] == 'Death']
fig = px.area(temp, x="Date", y="Count", color='Case',

             title='Cases over time', color_discrete_sequence = ['green', 'red', 'orange'])

fig.show()
# clean_data

# recovery_vs_death


recovery_vs_death = clean_data.groupby('Date').sum().reset_index()

# temp

recovery_vs_death['recovery_rate'] = round(recovery_vs_death['Recovery']/recovery_vs_death['Confirmed'],3)*100

recovery_vs_death['mortality_rate'] = round(recovery_vs_death['Death']/recovery_vs_death['Confirmed'],3)*100

recovery_vs_death = recovery_vs_death.melt(id_vars="Date", value_vars=['recovery_rate', 'mortality_rate'],

    var_name='rate', value_name='Rate_Count')



fig = px.line(recovery_vs_death, x="Date", y="Rate_Count", color='rate',log_y=True,

             title='Recovery Vs Mortality rate over time', color_discrete_sequence = ['green', 'red'])

fig.show()