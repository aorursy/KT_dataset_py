# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_death = pd.read_csv('/kaggle/input/coronavirus-dataset-update-0206/time_series_2019-ncov-Deaths.csv')

df_confirmed = pd.read_csv('/kaggle/input/coronavirus-dataset-update-0206/time_series_2019-ncov-Confirmed.csv')
df_death_modf = df_death

df_death_modf['log_of_sum'] = df_death['2/14/20 11:23'].apply(lambda x: math.log2(x+1))

df_death_modf = df_death_modf.sort_values(by = 'log_of_sum', ascending = False)

df_death_modf = df_death_modf[df_death['2/14/20 11:23']!=0]

df_death_modf.head()
fig, ax = plt.subplots(1,1,figsize = (6,15))

fig = sns.barplot(y = 'Province/State', x = '2/14/20 11:23', orient = 'h',data = df_death_modf)

plt.xlabel('Total_of_Death_Unitl_02_14_20', fontsize = 15)

plt.ylabel('Province in China', fontsize = 15)

plt.title('Totoal Death vs Province in China', fontsize = 20)

plt.xticks(rotation = 0)

fig, ax = plt.subplots(1,1,figsize = (6,15))

fig = sns.barplot(y = 'Province/State', x = 'log_of_sum', orient = 'h',data = df_death_modf)

plt.xlabel('Total_of_Death_Unitl_02_14_20', fontsize = 15)

plt.ylabel('Province in China', fontsize = 15)

plt.title('Totoal Death vs Province in China', fontsize = 20)

plt.xticks(rotation = 0)
fig = px.scatter_geo(data_frame= df_death_modf, lat = 'Lat', lon = 'Long',size = 'log_of_sum' ,scope= 'asia')

fig.update_layout(title_text = 'Death Distribution Across the World Casused By Novle Corona Virus')

fig.show()
df_death_modf_plot = df_death_modf[df_death['2/14/20 11:23']!=0].drop(['Country/Region','Lat','Long','log_of_sum'], axis = 1)

df_death_modf_plot.fillna(value = 0, inplace = True)

index = df_death_modf_plot[df_death_modf_plot['Province/State'] == 'Hubei'].T.index[1:]

df_death_hubei = df_death_modf_plot[df_death_modf_plot['Province/State'] == 'Hubei'].T[1:]



sns.set()

fig, ax = plt.subplots(1,1, figsize = (15,4))

ax = sns.scatterplot(x = index ,y = df_death_hubei[12])

plt.title('Hubei_Death_total_Unitl_Feb_14th', fontsize = 20)

plt.ylabel('Death_total', fontsize = 15)

plt.xlabel('Date', fontsize = 15)

plt.xticks(rotation = 90)

plt.show()
df_henan_death = df_death_modf.iloc[1,4:-1]

df_helongjiang = df_death_modf.iloc[2,4:-1]



sns.set()

fig, ax = plt.subplots(1,1, figsize = (15,4))

ax = sns.scatterplot(x = df_henan_death.index ,y = df_henan_death)

plt.title('Henan_Death_total_Unitl_Feb_14th', fontsize = 20)

plt.ylabel('Death_total', fontsize = 15)

plt.xlabel('Date', fontsize = 15)

plt.xticks(rotation = 90)

plt.show()
total_death_rate = sum(df_death['2/14/20 11:23'])/sum(df_confirmed['2/14/2020 11:23'])

total_death_rate
df_death_rate = df_death['2/14/20 11:23']/df_confirmed['2/14/2020 11:23']

df_death_rate =pd.concat([df_death['Province/State'],df_death_rate], axis = 1)

df_death_rate.rename(columns={0: "death_rate"}, inplace = True)

df_death_rate_modf = df_death_rate[df_death_rate['death_rate'] != 0]

df_death_rate_modf =df_death_rate_modf.dropna()

df_death_rate_modf.sort_values(by = 'death_rate',ascending = False, inplace = True)



fig, ax = plt.subplots(1,1,figsize = (6,15))

fig = sns.barplot(y = 'Province/State', x = 'death_rate', orient = 'h',data = df_death_rate_modf)

plt.xlabel('Death_Rate_Unitl_02_14_20', fontsize = 15)

plt.ylabel('Province in China', fontsize = 15)

plt.title('Death_Rate vs Province in China', fontsize = 20)

plt.xticks(rotation = 0)

df_HN = pd.read_csv('/kaggle/input/coronavirus-dataset-update-0206/Pandemic (H1N1) 200905-200907.csv', encoding='latin')

df_HN['death_rate'] = df_HN['Deaths']/df_HN['Cases']

df_HN.sort_values(by = 'death_rate', ascending = False)

df_HN_trunc = df_HN.sort_values(by = 'Deaths', ascending = False)

df_HN_trunc = df_HN_trunc.drop_duplicates(subset = 'Country').head(18).sort_values(by = 'death_rate', ascending = False)




df_HN_total = df_HN[df_HN['Country'] == 'Grand Total'].sort_values(by = 'Update Time')



fig, ax = plt.subplots(1,1,figsize = (6,15))

fig = sns.barplot(y = 'Update Time', x = 'death_rate', orient = 'h',data = df_HN_total)

plt.xlabel('Death_Rate', fontsize = 15)

plt.ylabel('Date', fontsize = 15)

plt.title('Global Death Rate of H1N1 Virus with Time', fontsize = 20)

plt.xticks(rotation = 0)




fig, ax = plt.subplots(1,1,figsize = (6,15))

fig = sns.barplot(y = 'Country', x = 'death_rate', orient = 'h',data = df_HN_trunc)

plt.xlabel('Death_Rate', fontsize = 15)

plt.ylabel('Country', fontsize = 15)

plt.title('Death_Rate of H1N1 Virus', fontsize = 20)

plt.xticks(rotation = 0)