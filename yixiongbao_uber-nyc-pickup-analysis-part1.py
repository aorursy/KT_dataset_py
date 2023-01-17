import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
uber_data = pd.read_csv("../input/uber-raw-data-janjune-15.csv")
uber_data.shape
uber_data['Hour'] = uber_data['Pickup_date'].apply(lambda x: x[11:13])

uber_data['Date'] = uber_data['Pickup_date'].apply(lambda x: x[0:10])

uber_data['Month'] = uber_data['Date'].apply(lambda x: x[5:7])
Month = ['Jan','Feb','Mar','Apr','May','Jun']

Index = [0,1,2,3,4,5]

Monthly_pickup = uber_data.groupby(['Month']).size()

plt.figure(1,figsize=(12,6))

plt.bar(Index,Monthly_pickup)

plt.xticks(Index,Month)

plt.title('UBER Monthly Pickup Summary in NYC')
month = ['01','02','03','04','05','06']

idx = [0,7,14,21,27]

def daily_pickup_plot(month):

    plot_data = uber_data[uber_data['Month'] == month]

    plot_data = plot_data.groupby(['Date']).size()

    plot_data.plot(kind='bar',rot=45)

    plt.xlabel("")

    plt.xticks(idx,plot_data.index[idx])



plt.figure(1,figsize=(12,24))

for i in range(0,6):

    plt.subplot(3,2,i+1)

    daily_pickup_plot(month[i])

    plt.ylim(0,140000)

    plt.title('Daily Pickup of '+ Month[i])
Hourly_pickup = uber_data.groupby(['Hour']).size()

mean = Hourly_pickup.mean()

hour = [i for i in range(0,24)]

plt.figure(1,figsize=(12,6))

plt.bar(hour,Hourly_pickup)

plt.title('UBER Hourly Pickup Summary of NYC, Jan 2015 - Jun 2015')

plt.xlabel("")

plt.xticks(hour,hour)

plt.show()
def hourly_pickup_plot(month):

    plot_data = uber_data[uber_data['Month'] == month]

    plot_data = plot_data.groupby(['Hour']).size()

    plot_data.plot(kind='bar')

    plt.xlabel("")

    plt.xticks(hour,plot_data.index[hour])



plt.figure(1,figsize=(12,24))

for i in range(0,6):

    plt.subplot(3,2,i+1)

    hourly_pickup_plot(month[i])

    plt.title('Hourly Pickup of '+ Month[i] + ' 2015')

    plt.xlabel("")

    plt.ylim(0,200000)