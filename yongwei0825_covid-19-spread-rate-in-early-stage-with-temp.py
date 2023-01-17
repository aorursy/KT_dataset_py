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
data=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv',  header = 0, parse_dates = True)

data = data.dropna(subset=['Lat'],axis='rows')

lat=data['Lat'].unique()  

#load data and put all latitude value to a list
from pprint import pprint 

Max_by_lat={}  #dic of latitude and the total number of cofirmed case

Start_date_by_lat = {}   #dic of latitude and the data when the fist case appeared

for la in lat:

    Max_by_lat[la]=(data.groupby(['Lat']).get_group(la)['ConfirmedCases'].max(),\

                          data.groupby(['Lat']).get_group(la)['Fatalities'].max(),\

                            data.groupby(['Lat']).get_group(la)['Country/Region'].unique()[0] if data.groupby(['Lat']).get_group(la)['Province/State'].isnull().unique()[0] == True else data.groupby(['Lat']).get_group(la)['Province/State'].unique()[0])

    Start_date_by_lat[la] = data.groupby(['Lat']).get_group(la).query('ConfirmedCases != 0.0').head(1)['Date']

pprint(Max_by_lat)

pprint(Start_date_by_lat)
def early_stage_rate(stop_percent):    #stop_percent is the percent we use to determine the end of early stage

    data["min_gap"] = 0

    for k, v in Max_by_lat.items():

        stop_cases = stop_percent * v[0]

        mask=[la==k for la in data['Lat']]

        data["min_gap"][mask] = abs(data[mask]['ConfirmedCases']-stop_cases)

    

    end_date_by_lat={}

    for k, v in Max_by_lat.items():

        i=data.groupby('Lat').get_group(k)['min_gap'].idxmin() 

        end_date_by_lat[k]= (data['Date'][i], data['ConfirmedCases'][i]) #find the date of early stage end and put it in a dic

    

    average_rate = {}

    for k, v in end_date_by_lat.items():

        try:

            start_date = Start_date_by_lat[k].values[0]

            end_date = v[0]

            lat = k

            start_cases = 0

            end_cases = v[1]

            if start_date==end_date:

                average_rate[k]=0

            else:

                average_rate[k] = calculate_rate(start_date, start_cases, end_date, end_cases)

        except:

            continue

    return average_rate     #average rate=# of cases in early stage/# of days in early stage
from datetime import datetime

import matplotlib.pyplot as plt

rate=early_stage_rate(0.15)

plt.scatter([abs(la) for la in rate.keys()],[rate for rate in rate.values()])

plt.ylim([-5, 200])   # when we set the threshold of early stage is 5%, let's see the rate with temperature. Lower latitude, higher temperature