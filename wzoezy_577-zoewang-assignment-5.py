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
# load data sets

df_flights = pd.read_csv("/kaggle/input/flight-delays/flights.csv", na_values=['NA', '?'])

df_airports = pd.read_csv("/kaggle/input/flight-delays/airports.csv", na_values=['NA', '?'])

df_airlines = pd.read_csv("/kaggle/input/flight-delays/airlines.csv", na_values=['NA', '?'])



display(df_flights[:5])

display(df_airports[:5])

display(df_airlines[:5])

# Problem 1.1

p1_1 = df_flights.groupby('ORIGIN_AIRPORT')['FLIGHT_NUMBER'].agg(['sum']).sort_values(by='sum', ascending=False)

display(p1_1)
# Problem 1.2

p1_2 = df_flights.loc[df_flights['MONTH']<=3]

p1_2 = p1_2.groupby('AIRLINE')['FLIGHT_NUMBER'].agg(['sum']).sort_values(by='sum', ascending=False)

display(p1_2)
# Problem 1.3

p1_3 = p1_2.copy()

cat = pd.cut(p1_3['sum'], 3, labels=['small', 'medium', 'big'])

p1_3['category'] = cat

display(p1_3)



# find out means

bins = p1_3.groupby('category')['sum'].agg(['mean'])

display(bins)
# Problem 2

import matplotlib.pyplot as plt

import seaborn

from datetime import datetime



p2 = df_flights.loc[df_flights['MONTH']<=3].loc[df_flights['ORIGIN_AIRPORT']=='ATL']



p2 = p2.groupby('MONTH',as_index=False)['FLIGHT_NUMBER'].agg('mean')

p2['MONTH'] = ['2015-01','2015-02','2015-03']

display(p2)



seaborn.lineplot(x='MONTH', y='FLIGHT_NUMBER', data=p2)
# Problem 3

p3 = df_flights.loc[df_flights['ORIGIN_AIRPORT']=='ATL']

p3 = pd.merge(p3, p1_3, on='AIRLINE')



p3 = p3.groupby('category', as_index=False)['DEPARTURE_DELAY'].agg('mean')

display(p3)

seaborn.barplot(x='category', y='DEPARTURE_DELAY', data=p3)