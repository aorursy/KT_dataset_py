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
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

# read the data

data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

# count active cases column

data['Active'] = data['Confirmed'] - data['Deaths'] - data['Recovered']

# choose country

china = data[data['Country/Region'].str.contains('Mainland China') == True]

china.plot(x='ObservationDate', y='Confirmed', figsize=(12,6), kind='line')

plt.style.use('fivethirtyeight')

plt.title('Mainland China Confirmed Total')
data.head()
italy = data[data['Country/Region'].str.contains('Italy') == True]

pivot = pd.pivot_table(italy, index='ObservationDate', columns='Country/Region', values='Deaths', aggfunc='sum')
pivot1 = pd.pivot_table(italy, index='ObservationDate', columns='Country/Region', values='Recovered', aggfunc='sum')
pivot2 = pd.pivot_table(italy, index='ObservationDate', columns='Country/Region', values='Active', aggfunc='sum')


pivot.plot(figsize=(10,6), kind='bar')
plt.title('Italy Deaths from Covid-19 Cases')

pivot1.plot(figsize=(10,6), kind='bar')
plt.title('Italy Recovered from Covid-19 Cases')

pivot2.plot(figsize=(10,6), kind='bar')
plt.title('Italy Active Covid-19 Cases')


italy['Deaths_Diff'] = italy['Deaths'].diff()
italy['Confirmed_Diff'] = italy['Confirmed'].diff()
italy['Recovered_Diff'] = italy['Recovered'].diff()

italy = italy.set_index('ObservationDate')

#italy.plot(y='Deaths_Diff', figsize=(10,6), kind='bar')
#plt.title('Italy New Deaths from Covid-19')
#italy.plot(y='Confirmed_Diff', figsize=(10,6), kind='bar')
#plt.title('Italy New Confirmed Covid-19 Cases')
#italy.plot(y='Recovered_Diff', figsize=(10,6), kind='bar')
#plt.title('Italy New Recovered from Covid-19')
nordics = data[data['Country/Region'].str.contains('Finland|Sweden|Iceland|Norway|Denmark') == True]


pivot = pd.pivot_table(nordics, index='ObservationDate', columns='Country/Region', values='Confirmed', aggfunc='sum')
pivot2 = pd.pivot_table(nordics, index='ObservationDate', columns='Country/Region', values='Deaths', aggfunc='sum')
pivot3 = pd.pivot_table(nordics, index='ObservationDate', columns='Country/Region', values='Recovered', aggfunc='sum')

pivot.plot(figsize=(10,6), linewidth=2)
plt.title('Nordics Confirmed Covid-19 Cases by Country')

pivot2.plot(figsize=(10,6), linewidth=2)
plt.title('Nordics Deaths from Covid-19 by Country')

asia = data[data['Country/Region'].str.contains('Mainland China|Singapore|Taiwan|Japan|Hong Kong') == True]

pivot = pd.pivot_table(asia, index='ObservationDate', columns='Country/Region', values='Active', aggfunc='sum')
pivot2 = pd.pivot_table(asia, index='ObservationDate', columns='Country/Region', values='Recovered', aggfunc='sum')
pivot3 = pd.pivot_table(asia, index='ObservationDate', columns='Country/Region', values='Deaths', aggfunc='sum')


pivot.plot(figsize=(8,5), linewidth=2,logy=True)
plt.title('East Asia Active by Country (log scale)')

pivot2.plot(figsize=(8,5), linewidth=2,logy=True)
plt.title('East Asia Recovered by Country (log scale)')

pivot3.plot(figsize=(8,5), linewidth=2,logy=True)
plt.title('East Asia Deaths by Country (log scale)')

china = data[data['Country/Region'].str.contains('Mainland China') == True]

europe = data[data['Country/Region'].str.contains('Austria|Switzerland|Germany|Portugal|Belgium|Estonia|Turkey|Greece|Hungary|Latvia|Lithuania|Luxembourg|Finland|Sweden|Iceland|Norway|Denmark|UK|Spain|France|Italy|Netherlands') == True]

USA = data[data['Country/Region'].str.contains('US') == True]


Europe = pd.pivot_table(europe, index='ObservationDate', values='Active', aggfunc='sum')

China = pd.pivot_table(china, index='ObservationDate', values='Active', aggfunc='sum')

usa =  pd.pivot_table(USA, index='ObservationDate', values='Active', aggfunc='sum')

ax = Europe.plot(figsize=(8,5), linewidth=1.7)
China.plot(ax=ax, figsize=(8,5), linewidth=1.7)
usa.plot(ax=ax, figsize=(8,5), linewidth=1.7)

plt.title('China vs Europe vs USA Active COVID-19 cases')
plt.style.use('fivethirtyeight')
plt.legend(['European Countries', 'China', 'USA'])
ax = italy.tail(60).plot(y='Active', figsize=(8,5), linewidth=1.7)
China.plot(ax=ax, figsize=(8,5), linewidth=1.7)
plt.title('Italy vs China Active Covid-19 Cases')
plt.legend(['Italy', 'China'])
Europe = pd.pivot_table(europe, index='ObservationDate', columns="Country/Region", values='Active', aggfunc='sum')

Europe.plot(figsize=(15,10), linewidth=0.8)
plt.title('Europe, Active COVID-19 cases')
plt.legend(fontsize=10)

Europe = pd.pivot_table(europe, index='ObservationDate', values='Deaths', aggfunc='sum')
China = pd.pivot_table(china, index='ObservationDate', values='Deaths', aggfunc='sum')
usa =  pd.pivot_table(USA, index='ObservationDate', values='Deaths', aggfunc='sum')

ax = Europe.plot(figsize=(8,5), linewidth=1.7)
China.plot(ax=ax, figsize=(8,5), linewidth=1.7)
usa.plot(ax=ax, figsize=(8,5), linewidth=1.7)

plt.title('China vs Europe vs USA Deaths COVID-19 cases')
plt.style.use('fivethirtyeight')
plt.legend(['European Countries', 'China', 'USA'])

china = data[data['Country/Region'].str.contains('Mainland China') == True]

europe = data[data['Country/Region'].str.contains('Austria|Switzerland|Germany|Portugal|Belgium|Estonia|Turkey|Greece|Hungary|Latvia|Lithuania|Luxembourg|Finland|Sweden|Iceland|Norway|Denmark|UK|Spain|France|Italy|Netherlands') == True]

usa = data[data['Country/Region'].str.contains('US') == True]


Europe = pd.pivot_table(europe, index='ObservationDate', values='Recovered', aggfunc='sum')

China = pd.pivot_table(china, index='ObservationDate', values='Recovered', aggfunc='sum')

usa =  pd.pivot_table(usa, index='ObservationDate', values='Recovered', aggfunc='sum')

ax = Europe.plot(figsize=(8,5), linewidth=1.7)
China.plot(ax=ax, figsize=(8,5), linewidth=1.7)
usa.plot(ax=ax, figsize=(8,5), linewidth=1.7)

plt.title('China vs Europe vs USA Recovered COVID-19 cases')
plt.style.use('fivethirtyeight')
plt.legend(['European Countries', 'China', 'USA'])
