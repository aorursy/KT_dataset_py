# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
week_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

week_df = week_df[week_df['Province_State'] == 'California']

week_df = week_df[39:]

start_date = week_df['Id'].tolist()[0]

week_df['day'] = week_df['Id'] - start_date + 1

week_df.shape
week_df.head(25)


fig, ax1 = plt.subplots(figsize=(12,6))

color_pos = 'tab:green'

color_neg = 'tab:blue'

color_total = 'tab:brown'

ax1.set_xlabel('Days Since Mar 01 2020')

ax1.set_ylabel('Confirmed Cases')

p1 = ax1.plot(week_df['day'], week_df['ConfirmedCases'], color = color_pos, label = 'Confirmed Cases')

ax1.tick_params(axis = 'x')

#ax1.legend(bbox_to_anchor=(1.15, 0.8))

ax1.legend(loc = 'upper left')

ax2 = ax1.twinx()

color = 'tab:red'

ax2.set_ylabel('Deaths')

p3 = ax2.plot(week_df['day'], week_df['Fatalities'], color=color, label='Fatalities', linewidth=4)

ax2.tick_params(axis = 'y', labelcolor=color)

plt.title('COVID-19 in California')

#ax2.legend(bbox_to_anchor=(1.13, 0.9))

ax2.legend(loc = 'upper center')

plt.xticks(np.arange(min(week_df['day']), max(week_df['day'])+1, 1.0))

fig.tight_layout()

plt.grid(True)

plt.savefig('/kaggle/working/ca_covid19_trends_week2.png')

plt.show()
week_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

week_df = week_df[week_df['Province_State'] == 'New York']

week_df = week_df[39:]

start_date = week_df['Id'].tolist()[0]

week_df['day'] = week_df['Id'] - start_date + 1

week_df.head(25)



fig, ax1 = plt.subplots(figsize=(12,6))

color_pos = 'tab:green'

color_neg = 'tab:blue'

color_total = 'tab:brown'

ax1.set_xlabel('Days Since Mar 01 2020')

ax1.set_ylabel('Confirmed Cases')

p1 = ax1.plot(week_df['day'], week_df['ConfirmedCases'], color = color_pos, label = 'Confirmed Cases')

ax1.tick_params(axis = 'x')

#ax1.legend(bbox_to_anchor=(1.15, 0.8))

ax1.legend(loc = 'upper left')

ax2 = ax1.twinx()

color = 'tab:red'

ax2.set_ylabel('Deaths')

p3 = ax2.plot(week_df['day'], week_df['Fatalities'], color=color, label='Fatalities', linewidth=4)

ax2.tick_params(axis = 'y', labelcolor=color)

plt.title('COVID-19 in New York')

#ax2.legend(bbox_to_anchor=(1.13, 0.9))

ax2.legend(loc = 'upper center')

plt.xticks(np.arange(min(week_df['day']), max(week_df['day'])+1, 1.0))

fig.tight_layout()

plt.grid(True)

plt.savefig('/kaggle/working/ny_covid19_trends_week2.png')

plt.show()
ca_df = pd.read_csv('/kaggle/input/covid19-in-usa/us_states_covid19_daily.csv')

ca_df = ca_df[ca_df['state'] == 'CA']

ca_df = ca_df.iloc[::-1]

ca_df = ca_df.fillna(0)

ca_df.head(20)
start_date = ca_df['date'].tolist()[0]

ca_df['day'] = ca_df['date'] - start_date + 1


fig, ax1 = plt.subplots(figsize=(12,6))

color_pos = 'tab:green'

color_neg = 'tab:blue'

color_total = 'tab:brown'

ax1.set_xlabel('Days Since Mar 03 2020')

ax1.set_ylabel('Cases')

p1 = ax2.plot(ca_df['day'], ca_df['positive'], color = color_pos, label = 'positive')

p2 = ax1.plot(ca_df['day'], ca_df['negative'], color = color_neg, label = 'negative')

p2 = ax1.plot(ca_df['day'], ca_df['total'], color = color_total, label = 'total', linestyle = '-.')

ax1.tick_params(axis = 'x')

#ax1.legend(bbox_to_anchor=(1.15, 0.8))

ax1.legend(loc = 'upper left')

ax2 = ax1.twinx()

color = 'tab:red'

ax2.set_ylabel('Deaths')

p3 = ax2.plot(ca_df['day'], ca_df['death'], color=color, label='death', linewidth=4)

ax2.tick_params(axis = 'y', labelcolor=color)

plt.title('COVID-19 in California')

#ax2.legend(bbox_to_anchor=(1.13, 0.9))

ax2.legend(loc = 'upper center')

plt.xticks(np.arange(min(ca_df['day']), max(ca_df['day'])+1, 1.0))

fig.tight_layout()

plt.grid(True)

plt.savefig('/kaggle/working/ca_covid19_trends.png')

plt.show()
ca_train = pd.read_csv('/kaggle/input/covid19-local-uscaforecasting/covid19-local-us-ca-forecasting-week1/ca_train.csv')

## drop the rows with zero confirmed cases 

ca_train = ca_train[ca_train['ConfirmedCases'] != 0]

ca_data= ca_train[['ConfirmedCases', 'Fatalities']]

ca_data 

## augment the data using other data set about covid19 cases in usa, available on kaggle 



for i in range(15,ca_df.shape[0]):

    ca_data = ca_data.append({

        'ConfirmedCases': ca_df.iloc[i, 2],

        'Fatalities': ca_df.iloc[i, 6]

    }, ignore_index = True)

ca_data
## more to follow