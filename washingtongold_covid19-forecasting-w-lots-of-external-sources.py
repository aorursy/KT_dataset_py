import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib import dates

import datetime



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')
train.head()
train['date_datetime'] = train['Date'].apply(lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d')))
def days_convert(date):

    return (date - datetime.datetime(2020,1,22)).days

train['days_since_start'] = train['date_datetime'].apply(days_convert)
train.head()
def rename_countries(country):

    if country == 'US':

        country = 'United States'

    elif country == 'Gambia, The' or country == 'The Gambia':

        country = "Gambia"

    elif country == 'The Bahamas':

        country = 'Bahamas'

    elif country == 'Taiwan*':

        country = 'Taiwan'

    elif country == 'Republic of the Congo' or country == 'Congo (Kinshasa)' or country == 'Congo (Brazzaville)':

        country = 'DR Congo'

    elif country == 'Korea, South':

        country = 'South Korea'

    elif country == 'Czechia':

        country = 'Czech Republic'

    return country

train['Country/Region'] = train['Country/Region'].apply(rename_countries)
population = pd.read_html('https://en.wikipedia.org/wiki/List_of_countries_by_population_%28United_Nations%29')[3]

population.head()
def clean_name(country):

    country = country.split('[')[0]

    return country

population['country'] = population['Country or area'].apply(clean_name)

population['pop'] = population['Population(1 July 2019)'] #cleaner name
population
def get_population(country_name):

    try:

        return population[population['country'] == country_name]['pop'].reset_index()['pop'][0]

    except:

        if country_name == 'Taiwan':

            return 23_510_000

        elif country_name == 'Reunion':

            return 859_959

        elif country_name == 'Kosovo':

            return 1_880_000

        elif country_name == 'Jersey':

            return 106_000

        elif country_name == 'Holy See':

            return 1_000

        elif country_name == 'Guernsey':

            return 66_300

        elif country_name == 'Cruise Ship':

            return 3_700

        elif country_name == "Cote d'Ivoire":

            return 22_700_000

        else:

            print(country_name)

train['population'] = train['Country/Region'].apply(get_population)
train.head()
age = pd.read_html('https://en.wikipedia.org/wiki/List_of_countries_by_median_age')[0]

age.head()
def change_countries(country):

    if country == 'DR Congo':

        country = 'Democratic Republic of the Congo'

    elif country == 'Bahamas':

        country = 'The Bahamas'

    return country

train['Country/Region2'] = train['Country/Region'].apply(change_countries)

def get_median(country):

    try:

        return age[age['Country/Territory'] == country]['Median(Years)'].reset_index()['Median(Years)'][0]

    except:

        if country == 'Cruise Ship':

            return 53 #best guess since there are no stats, lots of old people on a cruise ship I'd presume

        elif country == 'Eswatini':

            return 20.5

        elif country == 'French Guiana':

            return 24.8

        elif country == 'Gambia':

            return 20.7

        elif country == 'Guadeloupe':

            return 35

        elif country == 'Holy See':

            return 40

        elif country == 'Martinique':

            return 45.4

        elif country == 'Mayotte':

            return 20.1

        elif country == 'Reunion':

            return 45

        

        print(country)

train['median_age'] = train['Country/Region2'].apply(get_median)
train
life = pd.read_csv('/kaggle/input/world-bank-data-1960-to-2016/life_expectancy.csv')[['Country Name','2016']]

life.head()
def change_countries(country):

    if country == 'DR Congo':

        country = 'Democratic Republic of the Congo'

    elif country == 'Bahamas':

        country = 'The Bahamas'

    return country

#train['Country/Region2'] = train['Country/Region'].apply(change_countries)

def get_expectancy(country):

    try:

        return life[life['Country Name'] == country]['2016'].reset_index()['2016'][0]

    except:

        #all error countries. it is very fast to just quickly google the answer and fill in manually than to write exceptions, 

        #look up the dataset name for the country, etc. unfortunate fact of life that

        if country == 'Brunei':

            return 77

        elif country == 'Democratic Republic of the Congo':

            return 60

        elif country == 'Cruise Ship':

            return 85 #probably well off

        elif country == 'Egypt':

            return 72.7

        elif country == 'Eswatini':

            return 42

        elif country == 'French Guiana':

            return 75.9

        elif country == 'Gambia':

            return 64.9

        elif country == 'Guadeloupe':

            return 81.84

        elif country == 'Guernsey':

            return 82.5

        elif country == 'Holy See':

            return 78

        elif country == 'Iran':

            return 71.4

        elif country == 'Jersey':

            return 81.9

        elif country == 'South Korea':

            return 82.7

        elif country == 'Kyrgyzstan':

            return 71.4

        elif country == 'Martinique':

            return 79.6

        elif country == 'Mayotte':

            return 76.83

        elif country == 'North Macedonia':

            return 76

        elif country == 'Reunion':

            return 76.4

        elif country == 'Russia':

            return 70.8

        elif country == 'Saint Lucia':

            return 77.8

        elif country == 'Saint Vincent and the Grenadines':

            return 75.3

        elif country == 'Slovakia':

            return 77.1

        elif country == 'Taiwan':

            return 80.2

        elif country == 'The Bahamas':

            return 72.4

        elif country == 'Gambia':

            return 64.9

        elif country == 'Venezuela':

            return 75.8

        print(country)

train['life_expectancy'] = train['Country/Region2'].apply(get_expectancy)
for index in train[train['Country/Region']=='Andorra']['life_expectancy'].index:

    train.loc[index,'life_expectancy'] = 82.8

for index in train[train['Country/Region']=='Greenland']['life_expectancy'].index:

    train.loc[index,'life_expectancy'] = 72.4

for index in train[train['Country/Region']=='Monaco']['life_expectancy'].index:

    train.loc[index,'life_expectancy'] = 89.5

for index in train[train['Country/Region']=='San Marino']['life_expectancy'].index:

    train.loc[index,'life_expectancy'] = 85.6

    

#These four countries have no data in the dataset we used.
train[train['life_expectancy'].isna()==True]
X = train[['Lat','Long','days_since_start','population','median_age']] #including life expectancy increases error by 1

y = train['ConfirmedCases']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train,y_train)
from sklearn.metrics import mean_absolute_error

mean_absolute_error(rf.predict(X_test),y_test)
X1 = train[['Lat','Long','days_since_start','population','median_age','life_expectancy']]

y1 = train['Fatalities']

from sklearn.model_selection import train_test_split

X_train1,X_test1,y_train1,y_test1 = train_test_split(X1,y1,test_size=0.3)
from sklearn.ensemble import RandomForestClassifier

rf1 = RandomForestClassifier()

rf1.fit(X_train1,y_train1)
from sklearn.metrics import mean_absolute_error

mean_absolute_error(rf1.predict(X_test1),y_test1)
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

test.head()
test['date_datetime'] = test['Date'].apply(lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d')))

def days_convert(date):

    return (date - datetime.datetime(2020,1,22)).days

test['days_since_start'] = test['date_datetime'].apply(days_convert)

test['Country/Region'] = test['Country/Region'].apply(rename_countries)

test['population'] = test['Country/Region'].apply(get_population)

def change_countries(country):

    if country == 'DR Congo':

        country = 'Democratic Republic of the Congo'

    elif country == 'Bahamas':

        country = 'The Bahamas'

    return country

test['Country/Region2'] = test['Country/Region'].apply(change_countries)

test['median_age'] = test['Country/Region2'].apply(get_median)

test['life_expectancy'] = test['Country/Region2'].apply(get_expectancy)

for index in test[test['Country/Region']=='Andorra']['life_expectancy'].index:

    test.loc[index,'life_expectancy'] = 82.8

for index in test[test['Country/Region']=='Greenland']['life_expectancy'].index:

    test.loc[index,'life_expectancy'] = 72.4

for index in test[test['Country/Region']=='Monaco']['life_expectancy'].index:

    test.loc[index,'life_expectancy'] = 89.5

for index in test[test['Country/Region']=='San Marino']['life_expectancy'].index:

    test.loc[index,'life_expectancy'] = 85.6

test.head()
predictions = rf.predict(test[['Lat','Long','days_since_start','population','median_age']])

predictions1 = rf1.predict(test[['Lat','Long','days_since_start','population','median_age','life_expectancy']])
submit = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')

submit['ConfirmedCases'] = predictions

submit['ConfirmedCases'] = submit['ConfirmedCases'].apply(int)

submit['Fatalities'] = predictions1

submit['Fatalities'] = submit['Fatalities'].apply(int)

submit.head()
submit.to_csv('submission.csv',index=False)
test.to_csv('test.csv')

train.to_csv('train.csv')