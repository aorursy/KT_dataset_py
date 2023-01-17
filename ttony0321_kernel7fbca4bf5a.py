# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import random

from matplotlib import colors as mcolors

#pd.options.display.float_format = '{:.2f}'.format
df = pd.read_csv('/kaggle/input/corona-virus-report/worldometer_data.csv')

world_data = df[df['Country/Region']!='Diamond Princess']

world_data
unique_continent = list(world_data['Continent'].unique())

unique_countries =  list(world_data['Country/Region'].unique())
confirmed_cases = []

death_cases = []

recover_cases = []

total_tests = []

mortality_rate = []



for i in unique_continent:

    confirmed_cases.append(world_data[world_data['Continent'] == i]['TotalCases'].sum())

    



for i in range(len(unique_continent)):   

    death_cases.append(world_data[world_data['Continent'] == unique_continent[i]]['TotalDeaths'].sum())

    recover_cases.append(world_data[world_data['Continent'] == unique_continent[i]]['TotalRecovered'].sum())

    total_tests.append(world_data[world_data['Continent'] == unique_continent[i]]['TotalTests'].sum())

    mortality_rate.append(death_cases[i]/confirmed_cases[i])





visual_unique_continent = []

visual_Totalcases = []

visual_deathcases = []

visual_recovercases = []

visual_totaltests = []

continent_totalcases = []





for i in range(len(confirmed_cases)):

    visual_unique_continent.append(unique_continent[i])

    visual_Totalcases.append(confirmed_cases[i])

    

for i in range(len(death_cases)):

    visual_unique_continent.append(unique_continent[i])

    visual_deathcases.append(death_cases[i])

    

for i in range(len(recover_cases)):

    visual_unique_continent.append(unique_continent[i])

    visual_recovercases.append(recover_cases[i])

    

for i in range(len(total_tests)):

    visual_unique_continent.append(unique_continent[i])

    visual_totaltests.append(total_tests[i])
NorthAmerica = list(world_data[world_data['Continent'] == 'North America']['Country/Region'].unique())

SouthAmerica = list(world_data[world_data['Continent'] == 'South America']['Country/Region'].unique())

Asia = list(world_data[world_data['Continent'] == 'Asia']['Country/Region'].unique())

Europe = list(world_data[world_data['Continent'] == 'Europe']['Country/Region'].unique())

Africa = list(world_data[world_data['Continent'] == 'Africa']['Country/Region'].unique())

Australia_Oceania = list(world_data[world_data['Continent'] == 'Australia/Oceania']['Country/Region'].unique())

def pie_charts(x,y,title):

    Color = random.choices(list(mcolors.CSS4_COLORS.values()), k = len(unique_countries))

    

    plt.figure(figsize = (20,15))

    plt.title(title, size=20)

    plt.pie(y, colors = Color)

    plt.legend(x, loc='best', fontsize=15)

    plt.show()

    

def bar_charts(x,y,z,w,title):

    plt.figure(figsize = (16,9))

    plt.barh(x,y)

    plt.barh(x,z)

    plt.barh(x,w)

    plt.title(title, size=20)

    #plt.ylabel('Continent', fontsize=15)

    plt.xlabel('Cases')

    #plt.pie(y, colors = Color)

    plt.legend(['ConfirmedCases','Recovered Cases', 'Death Cases' ], fontsize=15)

    plt.show()

    

def make_data(x):

    for i in range(len(x)):

        total_tests.append(world_data[world_data['Country/Region'] == x[i]]['TotalTests'].sum())

        confirmed_cases.append(world_data[world_data['Country/Region'] == x[i]]['TotalCases'].sum())

        death_cases.append(world_data[world_data['Country/Region'] == x[i]]['TotalDeaths'].sum())

        recover_cases.append(world_data[world_data['Country/Region'] == x[i]]['TotalRecovered'].sum())

        mortality_rate.append(death_cases[i]/confirmed_cases[i])
continent_df = pd.DataFrame({

    'Continent': unique_continent,

    'Total Confirmed' : confirmed_cases,

    'Total Deaths' : death_cases,

    'Total Recovered' : recover_cases,

    'Mortality Rate' : mortality_rate

})
continent_df.style.background_gradient(cmap = 'Blues')
pie_charts(visual_unique_continent, visual_Totalcases, 'Covid-19 Confirmed Cases per Continent')
pie_charts(visual_unique_continent, visual_deathcases, 'Covid-19 Death Cases per Continent')
pie_charts(visual_unique_continent, visual_recovercases, 'Covid-19 Recovered Cases per Continent')
bar_charts(unique_continent,visual_Totalcases,visual_recovercases,visual_deathcases, 'Confirmed, Recovered Cases, Death Cases')
confirmed_cases = []

death_cases = []

recover_cases = []

total_tests = []

mortality_rate = []



make_data(unique_countries)



world_data_df = pd.DataFrame({

    'Continent': unique_countries,

    'Total Confirmed' : confirmed_cases,

    'Total Deaths' : death_cases,

    'Total Recovered' : recover_cases,

    'Mortality Rate' : mortality_rate

})
world_data_df.style.background_gradient(cmap = 'Blues')

confirmed_cases = []

death_cases = []

recover_cases = []

total_tests = []

mortality_rate = []



make_data(NorthAmerica)



NorthAmerica_df = pd.DataFrame({

    'Continent': NorthAmerica,

    'Total Confirmed' : confirmed_cases,

    'Total Deaths' : death_cases,

    'Total Recovered' : recover_cases,

    'Mortality Rate' : mortality_rate

})
NorthAmerica_df.style.background_gradient(cmap = 'Blues')
pie_charts(NorthAmerica, confirmed_cases, 'COVID-19 Confirmed Cases in NorthAmerica')
pie_charts(NorthAmerica, death_cases, 'COVID-19 Death Cases in NorthAmerica')
pie_charts(NorthAmerica, recover_cases, 'COVID-19 Recovered Cases in NorthAmerica')
bar_charts(NorthAmerica,confirmed_cases,recover_cases,death_cases, 'Total Test,and Confirmed Cases in NorthAmerica')
confirmed_cases = []

death_cases = []

recover_cases = []

total_tests = []

mortality_rate = []



make_data(SouthAmerica)



SouthAmerica_df = pd.DataFrame({

    'Continent': SouthAmerica,

    'Total Confirmed' : confirmed_cases,

    'Total Deaths' : death_cases,

    'Total Recovered' : recover_cases,

    'Mortality Rate' : mortality_rate

})
SouthAmerica_df.style.background_gradient(cmap = 'Blues')
pie_charts(SouthAmerica, confirmed_cases, 'COVID-19 Confirmed Cases in SouthAmerica')
pie_charts(SouthAmerica, death_cases, 'COVID-19 Death Cases in SouthAmerica')
pie_charts(SouthAmerica, recover_cases, 'COVID-19 Recovered Cases in SouthAmerica')
bar_charts(SouthAmerica,confirmed_cases,recover_cases,death_cases, 'Total Test,and Confirmed Cases in SouthAmerica')
confirmed_cases = []

death_cases = []

recover_cases = []

total_tests = []

mortality_rate = []



make_data(Asia)



Asia_df = pd.DataFrame({

    'Continent': Asia,

    'Total Confirmed' : confirmed_cases,

    'Total Deaths' : death_cases,

    'Total Recovered' : recover_cases,

    'Mortality Rate' : mortality_rate

})
Asia_df.style.background_gradient(cmap = 'Blues')
pie_charts(Asia, confirmed_cases, 'COVID-19 Confirmed Cases in Asia')
pie_charts(Asia, death_cases, 'COVID-19 Death Cases in Asia')
pie_charts(Asia, recover_cases, 'COVID-19 Recovered Cases in Asia')
bar_charts(Asia,confirmed_cases,recover_cases,death_cases, 'Total Test,and Confirmed Cases in Asia')
confirmed_cases = []

death_cases = []

recover_cases = []

total_tests = []

mortality_rate = []



make_data(Europe)



Europe_df = pd.DataFrame({

    'Continent': Europe,

    'Total Confirmed' : confirmed_cases,

    'Total Deaths' : death_cases,

    'Total Recovered' : recover_cases,

    'Mortality Rate' : mortality_rate

})
Europe_df.style.background_gradient(cmap = 'Blues')
pie_charts(Europe, confirmed_cases, 'COVID-19 Confirmed Cases in Europe')
pie_charts(Europe, death_cases, 'COVID-19 Death Cases in Europe')
pie_charts(Europe, recover_cases, 'COVID-19 Recovered Cases in Europe')
bar_charts(Europe,confirmed_cases,recover_cases,death_cases, 'Total Test,and Confirmed Cases in Europe')
confirmed_cases = []

death_cases = []

recover_cases = []

total_tests = []

mortality_rate = []



make_data(Africa)



Africa_df = pd.DataFrame({

    'Continent': Africa,

    'Total Confirmed' : confirmed_cases,

    'Total Deaths' : death_cases,

    'Total Recovered' : recover_cases,

    'Mortality Rate' : mortality_rate

})
Africa_df.style.background_gradient(cmap = 'Blues')
pie_charts(Africa, confirmed_cases, 'COVID-19 Confirmed Cases in Africa')
pie_charts(Africa, death_cases, 'COVID-19 Death Cases in Africa')
pie_charts(Africa, recover_cases, 'COVID-19 Recovered Cases in Africa')
bar_charts(Africa,confirmed_cases,recover_cases,death_cases, 'Total Test,and Confirmed Cases in Africa')
confirmed_cases = []

death_cases = []

recover_cases = []

total_tests = []

mortality_rate = []



make_data(Australia_Oceania)



Australia_Oceania_df = pd.DataFrame({

    'Continent': Australia_Oceania,

    'Total Confirmed' : confirmed_cases,

    'Total Deaths' : death_cases,

    'Total Recovered' : recover_cases,

    'Mortality Rate' : mortality_rate

})
death_cases
Australia_Oceania_df.style.background_gradient(cmap = 'Blues')
pie_charts(Australia_Oceania, confirmed_cases, 'COVID-19 Confirmed Cases in Australia_Oceania')
pie_charts(Australia_Oceania, death_cases, 'COVID-19 Death Cases in Australia_Oceania')
pie_charts(Australia_Oceania, recover_cases, 'COVID-19 Recovered Cases in Australia_Oceania')
bar_charts(Australia_Oceania,confirmed_cases,recover_cases,death_cases, 'Total Test,and Confirmed Cases in Australia_Oceania')