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

import matplotlib.pyplot as plt

import seaborn as sns

# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')
data.info()
country = data.groupby('Country').sum().apply(list).reset_index()

country
country.plot(kind='barh', x='Country', y='Confirmed', title ="V comp", figsize=(15, 10), legend=True, fontsize=12)
country['%Recovered'] = ((country['Recovered'] / country['Confirmed'] ) * 100)

country['%Recovered'] = country['%Recovered'].round(2)

country
from datetime import datetime,date



data['date'] = pd.to_datetime(data['Last Update']).dt.date

data
date_conf_cases = data.groupby('date').sum().apply(list).reset_index()

date_conf_cases
date_conf_cases = date_conf_cases.iloc[1:(len(date_conf_cases)-2)]

date_conf_cases
date_conf_cases['Confirmed'].sum()
i=1

tot_conf = 0

date_conf_cases['Total Confirmed Cases'] = 1

date_conf_cases['Days'] = 1

for ind in date_conf_cases.index: 

    date_conf_cases['Days'][ind] = i

    i=i+1

    date_conf_cases['Total Confirmed Cases'][ind] = date_conf_cases['Confirmed'][ind] + tot_conf

    tot_conf = date_conf_cases['Total Confirmed Cases'][ind]

date_conf_cases
date_conf_cases.drop(['date', 'Sno','Deaths','Recovered','Confirmed'], axis=1)
from sklearn.linear_model import LinearRegression

reg=LinearRegression()
x= date_conf_cases['Days']

y=date_conf_cases['Total Confirmed Cases']

x_matrix=x.values.reshape(-1,1)
reg.fit(x_matrix,y)
reg.score(x_matrix,y)
reg.intercept_
reg.coef_
new_data=pd.DataFrame(data=[3650,7300,10950],columns=['Days'])

new_data
new_data_matrix=new_data.values.reshape(-1,1)

reg.predict(new_data_matrix)