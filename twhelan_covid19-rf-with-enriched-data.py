import numpy as np

import pandas as pd

import datetime

import os

from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype

import math



from matplotlib import pyplot as plt

import plotly.graph_objects as go 

import seaborn as sns

import plotly

import plotly.express as px



from fastai.tabular import *

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb



def display_all(df):

    with pd.option_context("display.max_rows", 1000): 

        with pd.option_context("display.max_columns", 1000): 

            display(df.tail().transpose())



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_raw = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")

test_raw = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")

sub =  pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")

country_data_raw = pd.read_csv("../input/countries-of-the-world/countries of the world.csv")

pop_data_raw = pd.read_csv("../input/population-by-country-2020/population_by_country_2020.csv")

pollution_raw = pd.read_csv("../input/pollution-by-country-for-covid19-analysis/region_pollution.csv")

nurses_raw = pd.read_csv("../input/doctors-and-nurses-per-1000-people-by-country/Nurses_Per_Capital_By_Country.csv")

doctors_raw = pd.read_csv("../input/doctors-and-nurses-per-1000-people-by-country/Doctors_Per_Capital_By_Country.csv")

overweight_raw = pd.read_csv("../input/who-overweight-by-country-2016/WHO_overweightByCountry_2016.csv")

obese_raw = pd.read_csv("../input/who-obesity-by-country-2016/WHO_obesityByCountry_2016.csv")

econ_raw = pd.read_csv("../input/the-economic-freedom-index/economic_freedom_index2019_data.csv", encoding= "ISO-8859-1")

countryinfo_raw = pd.read_csv("../input/countryinfo/covid19countryinfo.csv")

weather_raw = pd.read_csv("../input/weather-data/training_data_with_weather_info.csv")

happiness_raw = pd.read_csv("../input/world-happiness-report-2020/WHR20_DataForFigure2.1.csv")
# Data on country response

def clean_response(df_raw):

    df = df_raw.copy().drop(['pop', 'density', 'medianage', 'urbanpop'], axis=1)

    print('The following countries are missing or have different names to those in train_raw:\n', list(set(list(train_raw['Country/Region'])) - set(list(df['country']))))

    for col in ['quarantine', 'schools', 'restrictions']:

        df[col] = pd.to_datetime(df[col])

    return df.dropna(axis=1, how='all').rename(columns={'country': 'Country'})



response = clean_response(countryinfo_raw)

display_all(response)
# Air pollution data

pollution = pollution_raw.copy().rename(columns={'Region': 'Country'})

print('The following countries are missing or have different names to those in train_raw:\n', list(set(list(train_raw['Country/Region'])) - set(list(pollution['Country']))))

display_all(pollution)
# Overweight data

def clean_overweight(df_raw):

    df = df_raw.copy()

    df.columns = ['Country', 'CombinedOverweight', 'MaleOverweight', 'FemaleOverweight']

    

    try_me = df['Country'] == 'Congo'

    df_try = df.copy()[try_me]

    df_try['Country'] = 'Republic of the Congo'

    df = df.append([df_try], ignore_index=True)

    

    try_me = df['Country'] == 'Gambia'

    df_try = df.copy()[try_me]

    df_try['Country'] = 'The Gambia'

    df = df.append([df_try], ignore_index=True)

    

    df.loc[df['Country']=='Côte d\'Ivoire','Country'] = 'Cote d\'Ivoire'

    df.loc[df['Country']=='United States of America','Country'] = 'US'

    df.loc[df['Country']=='United Kingdom of Great Britain and Northern Ireland','Country'] = 'United Kingdom'

    df.loc[df['Country']=='Republic of North Macedonia','Country'] = 'North Macedonia'

    df.loc[df['Country']=='Taiwan','Country'] = 'Taiwan*'

    df.loc[df['Country']=='Republic of Korea','Country'] = 'Korea, South'

    df.loc[df['Country']=='Gambia','Country'] = 'Gambia, The'

    df.loc[df['Country']=='Congo','Country'] = 'Congo (Brazzaville)'

    df.loc[df['Country']=='Democratic Republic of the Congo','Country'] = 'Congo (Kinshasa)'

    df.loc[df['Country']=='Bahamas','Country'] = 'The Bahamas'

    df.loc[df['Country']=='United Republic of Tanzania','Country'] = 'Tanzania'

    df.loc[df['Country']=='Viet Nam','Country'] = 'Vietnam'

    df.loc[df['Country']=='Republic of Moldova','Country'] = 'Moldova'

    df.loc[df['Country']=='Iran (Islamic Republic of)','Country'] = 'Iran'

    df.loc[df['Country']=='Brunei Darussalam','Country'] = 'Brunei'

    df.loc[df['Country']=='Russian Federation','Country'] = 'Russia'

    df.loc[df['Country']=='Venezuela (Bolivarian Republic of)','Country'] = 'Venezuela'

    df.loc[df['Country']=='Bolivia (Plurinational State of)','Country'] = 'Bolivia'

    

    print('The following countries are missing or have different names to those in train_raw:\n', list(set(list(train_raw['Country/Region'])) - set(list(df['Country']))))

    

    return df

    

overweight = clean_overweight(overweight_raw)

display_all(overweight)
# Obesity data

def clean_obesity(df_raw):

    df = df_raw.copy()

    df.columns = ['Country', 'CombinedObesity', 'MaleObesity', 'FemaleObesity']

    

    try_me = df['Country'] == 'Congo'

    df_try = df.copy()[try_me]

    df_try['Country'] = 'Republic of the Congo'

    df = df.append([df_try], ignore_index=True)

    

    try_me = df['Country'] == 'Gambia'

    df_try = df.copy()[try_me]

    df_try['Country'] = 'The Gambia'

    df = df.append([df_try], ignore_index=True)

    

    df.loc[df['Country']=='Côte d\'Ivoire','Country'] = 'Cote d\'Ivoire'

    df.loc[df['Country']=='United States of America','Country'] = 'US'

    df.loc[df['Country']=='United Kingdom of Great Britain and Northern Ireland','Country'] = 'United Kingdom'

    df.loc[df['Country']=='Republic of North Macedonia','Country'] = 'North Macedonia'

    df.loc[df['Country']=='Taiwan','Country'] = 'Taiwan*'

    df.loc[df['Country']=='Republic of Korea','Country'] = 'Korea, South'

    df.loc[df['Country']=='Gambia','Country'] = 'Gambia, The'

    df.loc[df['Country']=='Congo','Country'] = 'Congo (Brazzaville)'

    df.loc[df['Country']=='Democratic Republic of the Congo','Country'] = 'Congo (Kinshasa)'

    df.loc[df['Country']=='Bahamas','Country'] = 'The Bahamas'

    df.loc[df['Country']=='United Republic of Tanzania','Country'] = 'Tanzania'

    df.loc[df['Country']=='Viet Nam','Country'] = 'Vietnam'

    df.loc[df['Country']=='Republic of Moldova','Country'] = 'Moldova'

    df.loc[df['Country']=='Iran (Islamic Republic of)','Country'] = 'Iran'

    df.loc[df['Country']=='Brunei Darussalam','Country'] = 'Brunei'

    df.loc[df['Country']=='Russian Federation','Country'] = 'Russia'

    df.loc[df['Country']=='Venezuela (Bolivarian Republic of)','Country'] = 'Venezuela'

    df.loc[df['Country']=='Bolivia (Plurinational State of)','Country'] = 'Bolivia'

    

    print('The following countries are missing or have different names to those in train_raw:\n', list(set(list(train_raw['Country/Region'])) - set(list(df['Country']))))

    

    return df

    

obesity = clean_obesity(obese_raw)

display_all(obesity)
# Economy data

def clean_econ(df_raw):

    df = df_raw.copy()

    

    try_me = df['Country'] == 'Congo, Republic of'

    df_try = df.copy()[try_me]

    df_try['Country'] = 'Republic of the Congo'

    df = df.append([df_try], ignore_index=True)

    

    try_me = df['Country'] == 'Gambia'

    df_try = df.copy()[try_me]

    df_try['Country'] = 'The Gambia'

    df = df.append([df_try], ignore_index=True)

    

    df.loc[df['Country']=='Côte d\'Ivoire','Country'] = 'Cote d\'Ivoire'

    df.loc[df['Country']=='United States','Country'] = 'US'

    #df.loc[df['Country']=='Czech Republic'] = 'Czechia'

    df.loc[df['Country']=='Republic of North Macedonia','Country'] = 'North Macedonia'

    df.loc[df['Country']=='Taiwan ','Country'] = 'Taiwan*'

    df.loc[df['Country']=='Gambia','Country'] = 'Gambia, The'

    df.loc[df['Country']=='Congo, Republic of','Country'] = 'Congo (Brazzaville)'

    df.loc[df['Country']=='Congo, Democratic Republic of the Congo','Country'] = 'Congo (Kinshasa)'

    df.loc[df['Country']=='Bahamas','Country'] = 'The Bahamas'

    df.loc[df['Country']=='Slovak Republic','Country'] = 'Slovakia'

    df.loc[df['Country']=='Kyrgyz Republic','Country'] = 'Kyrgyzstan'

    df.loc[df['Country']=='Brunei Darussalam','Country'] = 'Brunei'

    df.loc[df['Country']=='Macedonia','Country'] = 'North Macedonia'

    

    df1 = pd.DataFrame({'Country':['Guernsey','Andorra','Greenland','Aruba','Cruise Ship','San Marino','Jersey','Antigua and Barbuda','French Guiana',

                                   'Puerto Rico','Mayotte','Holy See','Reunion','Guam','Martinique','Guadeloupe','Monaco','Czechia'],

                        'Region':['Europe','Europe','Americas','Americas','Asia-Pacific','Europe','Europe','Americas','Americas',

                                   'Americas','Sub-Saharan Africa','Europe','Sub-Saharan Africa','Asia-Pacific','Americas','Americas','Europe','Europe']})

    

    df = df.append(df1, sort=True)

    

    print('The following countries are missing or have different names to those in train_raw:\n', list(set(list(train_raw['Country/Region'])) - set(list(df['Country']))))

    

    df = df.drop(['CountryID', 'Country Name', 'WEBNAME', 'Population (Millions)'], axis=1)

    

    df['GDP (Billions, PPP)'] = df['GDP (Billions, PPP)'].str.strip('$').str.split(' ').str.get(0).str.replace(',', '').astype(float)

    df['GDP per Capita (PPP)'] = df['GDP per Capita (PPP)'].str.strip('$').str.split(' ').str.get(0).str.replace(',', '').astype(float)

    df['Unemployment (%)'] = df['Unemployment (%)'].str.split(' ').str.get(0).astype(float)

    df['FDI Inflow (Millions)'] = df['FDI Inflow (Millions)'].str.replace(',', '').astype(float)

    

    for col in df.columns[df.isna().any()].tolist():

        df[col] = df[col].fillna(df.groupby('Region')[col].transform('median')) # fill missing values with region medians

    

    return df.rename(columns={'Region': 'EconRegion'})

    

econ = clean_econ(econ_raw)

display_all(econ)
# Population data

def clean_pop_data(df):

    df.columns = ['Country', 'Pop', 'YearlyPopChange', 'NetPopChange', 'PopDensity', 'LandArea', 

                  'NetMigrants', 'FertilityRate', 'MedianAge', 'UrbanPop', 'WorldShare']

    

    try_me = df['Country'] == 'Congo'

    df_try = df.copy()[try_me]

    df_try['Country'] = 'Republic of the Congo'

    df = df.append([df_try], ignore_index=True)

    

    try_me = df['Country'] == 'Gambia'

    df_try = df.copy()[try_me]

    df_try['Country'] = 'The Gambia'

    df = df.append([df_try], ignore_index=True)

    

    df.loc[df['Country']=='Côte d\'Ivoire','Country'] = 'Cote d\'Ivoire'

    df.loc[df['Country']=='Channel Islands','Country'] = 'Guernsey'

    df.loc[df['Country']=='United States','Country'] = 'US'

    df.loc[df['Country']=='Réunion','Country'] = 'Reunion'

    df.loc[df['Country']=='Taiwan','Country'] = 'Taiwan*'

    df.loc[df['Country']=='South Korea','Country'] = 'Korea, South'

    df.loc[df['Country']=='Gambia','Country'] = 'Gambia, The'

    df.loc[df['Country']=='Congo','Country'] = 'Congo (Brazzaville)'

    df.loc[df['Country']=='DR Congo','Country'] = 'Congo (Kinshasa)'

    df.loc[df['Country']=='Bahamas','Country'] = 'The Bahamas'

    df.loc[df['Country']=='Czech Republic (Czechia)','Country'] = 'Czechia'

    df.loc[df['Country']=='St. Vincent & Grenadines','Country'] = 'Saint Vincent and the Grenadines'

    

    try_me = df['Country'] == 'Guernsey'

    df_try = df.copy()[try_me]

    df_try['Country'] = 'Jersey'

    df = df.append([df_try], ignore_index=True)

    

    for col in ['YearlyPopChange', 'UrbanPop', 'WorldShare']:

        df[col] = df[col].str.rstrip('%')

    

    # Add Kosovo and Cruise Ship manually

    # https://en.wikipedia.org/wiki/Demographics_of_Kosovo

    # https://www.indexmundi.com/kosovo/#Demographics

    # https://en.wikipedia.org/wiki/Diamond_Princess_(ship)

    df1 = pd.DataFrame({'Country':['Kosovo','Cruise Ship'],

                        'Pop':[1793000,3711],

                        'YearlyPopChange':[0.64,0],

                        'NetPopChange':[1147,0],

                        'PopDensity':[165,26],

                        'LandArea':[10887,141],

                        'NetMigrants':[-7340,0],

                        'FertilityRate':[2.09,0],

                        'MedianAge':[30,62],

                        'UrbanPop':[65,100],

                        'WorldShare':[0.02,0.00]})

    

    df = df.append(df1)

    

    print('The following countries are missing or have different names to those in train_raw:\n', list(set(list(train_raw['Country/Region'])) - set(list(df['Country']))))

    

    return df



pop_data = clean_pop_data(pop_data_raw)

display_all(pop_data)
# Additional country data

def clean_country_data(df):

    country_data = df.copy()

    country_data['Country'] = country_data['Country'].str.strip()

    country_data['Region'] = country_data['Region'].str.strip()

    for col in list(set(list(country_data))-set(['Country', 'Region'])):

        if is_string_dtype(country_data[col]):

            country_data[col] = country_data[col].str.replace(',','.').astype(float)

            

    df1 = pd.DataFrame({'Country':['Cruise Ship', 'Holy See', 'Kosovo', 'Montenegro'],

                        'Region':['ASIA (EX. NEAR EAST)', 'WESTERN EUROPE', 'EASTERN EUROPE', 'EASTERN EUROPE']})

    

    country_data = country_data.append(df1, sort=True)

    

    for col in country_data.columns[country_data.isna().any()].tolist():

        country_data[col] = country_data[col].fillna(country_data.groupby('Region')[col].transform('median')) # fill missing values with region medians

        

    try_me = country_data['Country'] == 'Congo, Repub. of the'

    df_try = country_data.copy()[try_me]

    df_try['Country'] = 'Republic of the Congo'

    country_data = country_data.append([df_try], ignore_index=True)

    

    try_me = country_data['Country'] == 'Gambia, The'

    df_try = country_data.copy()[try_me]

    df_try['Country'] = 'The Gambia'

    country_data = country_data.append([df_try], ignore_index=True)



    country_data.loc[country_data['Country']=='United States','Country'] = 'US'

    country_data.loc[country_data['Country']=='Mainland China','Country'] = 'China'

    country_data.loc[country_data['Country']=='Viet Nam','Country'] = 'Vietnam'

    country_data.loc[country_data['Country']=='UK','Country'] = 'United Kingdom'

    country_data.loc[country_data['Country']=='Taiwan','Country'] = 'Taiwan*'

    country_data.loc[country_data['Country']=='Hong Kong SAR, China','Country'] = 'Hong Kong'

    country_data.loc[country_data['Country']=='Bosnia & Herzegovina','Country'] = 'Bosnia and Herzegovina'

    country_data.loc[country_data['Country']=='Antigua & Barbuda','Country'] = 'Antigua and Barbuda'

    country_data.loc[country_data['Country']=='Central African Rep.','Country'] = 'Central African Republic'

    country_data.loc[country_data['Country']=='Czech Republic','Country'] = 'Czechia'

    country_data.loc[country_data['Country']=='Swaziland','Country'] = 'Eswatini'

    country_data.loc[country_data['Country']=='Macedonia','Country'] = 'North Macedonia'

    country_data.loc[country_data['Country']=='Congo, Repub. of the','Country'] = 'Congo (Brazzaville)'

    country_data.loc[country_data['Country']=='Congo, Dem. Rep.','Country'] = 'Congo (Kinshasa)'

    country_data.loc[country_data['Country']=='Bahamas, The','Country'] = 'The Bahamas'

    country_data.loc[country_data['Country']=='Trinidad & Tobago','Country'] = 'Trinidad and Tobago'



    print('The following countries are missing or have different names to those in train_raw:\n', list(set(list(train_raw['Country/Region'])) - set(list(country_data['Country']))))

    

    country_data['Climate'] = country_data['Climate'].astype(str)

    

    return country_data.drop(['GDP ($ per capita)'], axis=1)



country_data = clean_country_data(country_data_raw)

display_all(country_data)
# Weather data

# https://www.kaggle.com/davidbnn92/weather-data/

def clean_weather(df_raw):

    df = df_raw.copy()[['Id', 'temp', 'min', 'max', 'stp', 'wdsp', 'prcp', 'fog']]

    # Note: we will handle missing values later, so convert to nan to make things easier

    for col in ['temp', 'min', 'max', 'stp']:

        df[col] = df[col].replace(9999.9, np.nan)

    df['wdsp'] = df['wdsp'].replace(999.9, np.nan)

    df['prcp'] = df['prcp'].replace(99.99, np.nan)

    #df['Date'] = pd.to_datetime(df['Date'])

    

    return df.rename(columns={'min': 'min_temp', 'max': 'max_temp'})



weather = clean_weather(weather_raw)

display_all(weather)
# Happiness report 2020 data

def clean_happiness_data(df):

    df = df.rename(columns={'Country name': 'Country'}).drop('Regional indicator', axis=1)

   

    try_me = df['Country'] == 'Congo (Brazzaville)'

    df_try = df.copy()[try_me]

    df_try['Country'] = 'Republic of the Congo'

    df = df.append([df_try], ignore_index=True)

    

    try_me = df['Country'] == 'Gambia'

    df_try = df.copy()[try_me]

    df_try['Country'] = 'The Gambia'

    df = df.append([df_try], ignore_index=True)

    

    df.loc[df['Country']=='Ivory Coast','Country'] = 'Cote d\'Ivoire'

    df.loc[df['Country']=='United States','Country'] = 'US'

    df.loc[df['Country']=='Macedonia','Country'] = 'North Macedonia'

    df.loc[df['Country']=='Taiwan Province of China','Country'] = 'Taiwan*'

    df.loc[df['Country']=='South Korea','Country'] = 'Korea, South'

    df.loc[df['Country']=='Gambia','Country'] = 'Gambia, The'

    df.loc[df['Country']=='Bahamas','Country'] = 'The Bahamas'

    df.loc[df['Country']=='Czech Republic','Country'] = 'Czechia'

    df.loc[df['Country']=='Swaziland','Country'] = 'Eswatini'

       

    print('The following countries are missing or have different names to those in train_raw:\n', list(set(list(train_raw['Country/Region'])) - set(list(df['Country']))))

    print('Impute missing values later')

    

    return df



happiness = clean_happiness_data(happiness_raw)

display_all(happiness)
# Cleaning and feature engineering

combined_raw = train_raw.append(test_raw, sort=True)



def clean_covid19_data(df):

    # Basic cleaning

    df['Date'] = pd.to_datetime(df['Date'])

    df['Province/State'] = df['Province/State'].fillna('None')

    df = df.sort_values(['Country/Region', 'Province/State', 'Date'])

    df['Id'].fillna(-1, inplace=True)

    df['ForecastId'].fillna(-1, inplace=True)

    df['ConfirmedCases'].fillna(0, inplace=True)

    df['Fatalities'].fillna(0, inplace=True)



    # Add some extra features from the date column

    def make_date(df, date_field):

        "Make sure `df[date_field]` is of the right date type."

        field_dtype = df[date_field].dtype

        if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):

            field_dtype = np.datetime64

        if not np.issubdtype(field_dtype, np.datetime64):

            df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)



    def add_datepart(df, field_name, drop=True, time=False):

        "Helper function that adds columns relevant to a date in the column `field_name` of `df`."

        make_date(df, field_name)

        field = df[field_name]

        attr = ['Year', 'Month', 'Week', 'Day', 'DayOfWeek', 'DayOfYear']

        if time: attr = attr + ['Hour', 'Minute', 'Second']

        for n in attr: df[n] = getattr(field.dt, n.lower())

        df['Elapsed'] = field.astype(np.int64) // 10 ** 9

        if drop: df.drop(field_name, axis=1, inplace=True)

        return df



    df = add_datepart(df, 'Date', drop=False)

    

    # Add additional data sources

    df = pd.merge(df.rename(columns={'Country/Region':'Country'}), country_data, on=['Country'], how='left')

    df = pd.merge(df, pop_data, on=['Country'], how='left')

    df = pd.merge(df, pollution, on=['Country'], how='left')

    df = pd.merge(df, overweight, on=['Country'], how='left')

    df = pd.merge(df, obesity, on=['Country'], how='left')

    df = pd.merge(df, econ, on=['Country'], how='left')

    df = pd.merge(df, response, on=['Country'], how='left')

    df = pd.merge(df, weather, on=['Id'], how='left')

    df = pd.merge(df, happiness, on=['Country'], how='left')

    df = df.drop_duplicates()

    

    # Fix missing (typically fill missing values with region medians)

    df['NetMigrants'].fillna((df['Net migration']/100)*df['Pop'], inplace=True)

    df['Net migration'] = (df['NetMigrants'] / df['Pop']) * 100

    for col in ['FertilityRate', 'MedianAge', 'UrbanPop']:

        df[col] = df[col].str.replace('N.A.', '99000')

        df[col] = df[col].astype(float)

        df[col] = df[col].replace(99000, (df.groupby('Region')[col].transform('median')))

        df[col] = df[col].fillna(df.groupby('Region')[col].transform('median'))

        

    for col in ['temp', 'min_temp', 'max_temp', 'stp', 'wdsp', 'prcp', 'fog']:

        df[col] = df[col].fillna(df.groupby(['Country', 'Month', 'Week'])[col].transform('median'))

        df[col] = df[col].fillna(df.groupby(['Region', 'Month', 'Week'])[col].transform('median'))

        df[col] = df[col].fillna(df.groupby(['Country', 'Month'])[col].transform('median'))

        df[col] = df[col].fillna(df.groupby(['Region', 'Month'])[col].transform('median'))

        df[col] = df[col].fillna(df.groupby(['Country'])[col].transform('median'))

        df[col] = df[col].fillna(df.groupby(['Region'])[col].transform('median'))

        

    for col in ['CombinedOverweight', 'MaleOverweight', 'FemaleOverweight', 'CombinedObesity', 'MaleObesity', 'FemaleObesity', 'hospibed', 'smokers', 

                'sex0', 'sex14', 'sex25', 'sex54', 'sex64', 'sex65plus', 'sexratio', 'lung', 'femalelung', 'malelung', 'tests', 'testpop',

                'Ladder score', 'Standard error of ladder score', 'upperwhisker', 'lowerwhisker', 'Logged GDP per capita', 'Social support', 'Healthy life expectancy',

                'Freedom to make life choices', 'Generosity', 'Perceptions of corruption', 'Ladder score in Dystopia', 'Explained by: Log GDP per capita',

                'Explained by: Social support', 'Explained by: Healthy life expectancy', 'Explained by: Freedom to make life choices', 'Explained by: Generosity',

                'Explained by: Perceptions of corruption', 'Dystopia + residual']:

        df[col] = df[col].fillna(df.groupby('Region')[col].transform('median'))

        

    df['pct_tested'] = df['tests'] / df['Pop']

    df['tests'] = df['tests'].fillna(df['pct_tested'].median()*df['Pop'])

    df['testpop'] = df['Pop'] / df['tests']

    

    # Fix channel island population metrics

    guer_pop = df[df['Country']=='Guernsey']['Population'].mean() / (df[df['Country']=='Guernsey']['Population'].mean() + df[df['Country']=='Jersey']['Population'].mean())

    guer_la = df[df['Country']=='Guernsey']['Area (sq. mi.)'].mean() / (df[df['Country']=='Guernsey']['Area (sq. mi.)'].mean() + df[df['Country']=='Jersey']['Area (sq. mi.)'].mean())

    guer_pop_den = df[df['Country']=='Guernsey']['Pop. Density (per sq. mi.)'].mean() / (df[df['Country']=='Guernsey']['Pop. Density (per sq. mi.)'].mean() + df[df['Country']=='Jersey']['Pop. Density (per sq. mi.)'].mean())

    df['Pop'] = np.where(df['Country']=='Guernsey', guer_pop*df['Pop'], 

                             np.where(df['Country']=='Jersey', (1-guer_pop)*df['Pop'], df['Pop']))

    df['LandArea'] = np.where(df['Country']=='Guernsey', guer_la*df['LandArea'], 

                             np.where(df['Country']=='Jersey', (1-guer_la)*df['LandArea'], df['LandArea']))

    df['PopDensity'] = np.where(df['Country']=='Guernsey', guer_pop_den*df['PopDensity'], 

                             np.where(df['Country']=='Jersey', (1-guer_pop_den)*df['PopDensity'], df['PopDensity']))

    

    # Add features to further explain country response to the outbreak

    df['HasQuarantine'] = np.where(df['quarantine']<=df['Date'], 1, 0)

    df['HasSchoolClosure'] = np.where(df['schools']<=df['Date'], 1, 0)

    df['HasRestrictions'] = np.where(df['restrictions']<=df['Date'], 1, 0)

    df.loc[df['quarantine'].notnull(), 'DaysSinceQuarantine'] = (df['Date'] - df['quarantine']).dt.days

    df.loc[df['schools'].notnull(), 'DaysSinceSchoolClosure'] = (df['Date'] - df['schools']).dt.days

    df.loc[df['restrictions'].notnull(), 'DaysSinceRestrictions'] = (df['Date'] - df['restrictions']).dt.days



    for col in ['DaysSinceQuarantine', 'DaysSinceSchoolClosure', 'DaysSinceRestrictions']:

        df[col] = df[col].fillna(0)

        df[col] = np.where(df[col]<0, 0, df[col])

    

    def add_cases_fatalities_feats(df):

        # Add features to explain days since first case and fatality for different groups

        df1 = df.copy()[df['ConfirmedCases']>0].groupby('Country')['Date'].min().reset_index().rename(columns={'Date':'FirstCase'})

        df = pd.merge(df, df1, on='Country', how='left')

        df.loc[df['FirstCase'].notnull(), 'DaysSinceFirstCaseCountry'] = (df['Date'] - df['FirstCase']).dt.days

        df1 = df.copy()[df['Fatalities']>0].groupby('Country')['Date'].min().reset_index().rename(columns={'Date':'FirstFatality'})

        df = pd.merge(df, df1, on='Country', how='left')

        df.loc[df['FirstFatality'].notnull(), 'DaysSinceFirstFatalityCountry'] = (df['Date'] - df['FirstFatality']).dt.days

        df = df.drop(['FirstCase', 'FirstFatality'], axis=1)



        df1 = df.copy()[df['ConfirmedCases']>0].groupby('Region')['Date'].min().reset_index().rename(columns={'Date':'FirstCase'})

        df = pd.merge(df, df1, on='Region', how='left')

        df.loc[df['FirstCase'].notnull(), 'DaysSinceFirstCaseRegion'] = (df['Date'] - df['FirstCase']).dt.days

        df1 = df.copy()[df['Fatalities']>0].groupby('Region')['Date'].min().reset_index().rename(columns={'Date':'FirstFatality'})

        df = pd.merge(df, df1, on='Region', how='left')

        df.loc[df['FirstFatality'].notnull(), 'DaysSinceFirstFatalityRegion'] = (df['Date'] - df['FirstFatality']).dt.days

        df = df.drop(['FirstCase', 'FirstFatality'], axis=1)



        df1 = df.copy()[(df['ConfirmedCases']>0)&(df['Province/State']!='None')].groupby('Province/State')['Date'].min().reset_index().rename(columns={'Date':'FirstCase'})

        df = pd.merge(df, df1, on='Province/State', how='left')

        df.loc[df['FirstCase'].notnull(), 'DaysSinceFirstCaseProvince'] = (df['Date'] - df['FirstCase']).dt.days

        df1 = df.copy()[(df['Fatalities']>0)&(df['Province/State']!='None')].groupby('Province/State')['Date'].min().reset_index().rename(columns={'Date':'FirstFatality'})

        df = pd.merge(df, df1, on='Province/State', how='left')

        df.loc[df['FirstFatality'].notnull(), 'DaysSinceFirstFatalityProvince'] = (df['Date'] - df['FirstFatality']).dt.days

        df = df.drop(['FirstCase', 'FirstFatality'], axis=1)



        df = df.fillna(0)

        return df

    

    df = add_cases_fatalities_feats(df)

    

    def add_cases_fatalities_100(df):

        # Add features to explain days since 100th case and fatality for different groups

        df1 = df.copy()[df['ConfirmedCases']>100].groupby('Country')['Date'].min().reset_index().rename(columns={'Date':'Case100'})

        df = pd.merge(df, df1, on='Country', how='left')

        df.loc[df['Case100'].notnull(), 'DaysSinceCase100Country'] = (df['Date'] - df['Case100']).dt.days

        df1 = df.copy()[df['Fatalities']>100].groupby('Country')['Date'].min().reset_index().rename(columns={'Date':'Fatality100'})

        df = pd.merge(df, df1, on='Country', how='left')

        df.loc[df['Fatality100'].notnull(), 'DaysSinceFatality100Country'] = (df['Date'] - df['Fatality100']).dt.days

        df = df.drop(['Case100', 'Fatality100'], axis=1)



        df1 = df.copy()[df['ConfirmedCases']>100].groupby('Region')['Date'].min().reset_index().rename(columns={'Date':'Case100'})

        df = pd.merge(df, df1, on='Region', how='left')

        df.loc[df['Case100'].notnull(), 'DaysSinceCase100Region'] = (df['Date'] - df['Case100']).dt.days

        df1 = df.copy()[df['Fatalities']>100].groupby('Region')['Date'].min().reset_index().rename(columns={'Date':'Fatality100'})

        df = pd.merge(df, df1, on='Region', how='left')

        df.loc[df['Fatality100'].notnull(), 'DaysSinceFatality100Region'] = (df['Date'] - df['Fatality100']).dt.days

        df = df.drop(['Case100', 'Fatality100'], axis=1)



        df1 = df.copy()[(df['ConfirmedCases']>100)&(df['Province/State']!='None')].groupby('Province/State')['Date'].min().reset_index().rename(columns={'Date':'Case100'})

        df = pd.merge(df, df1, on='Province/State', how='left')

        df.loc[df['Case100'].notnull(), 'DaysSinceCase100Province'] = (df['Date'] - df['Case100']).dt.days

        df1 = df.copy()[(df['Fatalities']>100)&(df['Province/State']!='None')].groupby('Province/State')['Date'].min().reset_index().rename(columns={'Date':'Fatality100'})

        df = pd.merge(df, df1, on='Province/State', how='left')

        df.loc[df['Fatality100'].notnull(), 'DaysSinceFatality100Province'] = (df['Date'] - df['Fatality100']).dt.days

        df = df.drop(['Case100', 'Fatality100'], axis=1)



        df['DaysSinceFirstCaseGlobal'] = (df['Date'] - df[df['ConfirmedCases']>0]['Date'].min()).dt.days

        df['DaysSinceFirstFatalityGlobal'] = (df['Date'] - df[df['Fatalities']>0]['Date'].min()).dt.days

        df['DaysSinceCase100Global'] = (df['Date'] - df[df['ConfirmedCases']>100]['Date'].min()).dt.days

        df['DaysSinceFatality100Global'] = (df['Date'] - df[df['Fatalities']>100]['Date'].min()).dt.days



        df = df.fillna(0)

        return df

    

    df = add_cases_fatalities_100(df)

    

    # Remove surplus columns

    df = df.drop(['Population', 'Area (sq. mi.)', 'Pop. Density (per sq. mi.)', 'pct_tested', 'quarantine', 'schools', 'restrictions', 'EconRegion'], axis=1)

    

    # Convert remaining string cols to floats

    for col in ['YearlyPopChange', 'WorldShare']:

        df[col] = df[col].astype(float)

    

    return df.drop_duplicates()



combined = clean_covid19_data(combined_raw)

combined.columns[combined.isna().any()].tolist()
# https://www.kaggle.com/saga21/covid-global-forecast-sir-model-ml-regressions

def calculate_trend(df, lag_list, column):

    for lag in lag_list:

        trend_column_lag = "Trend_" + column + "_" + str(lag)

        df[trend_column_lag] = (df[column]-df[column].shift(lag, fill_value=-999))/df[column].shift(lag, fill_value=0)

    return df





def calculate_lag(df, lag_list, column):

    for lag in lag_list:

        column_lag = column + "_" + str(lag)

        df[column_lag] = df[column].shift(lag, fill_value=0)

    return df



combined = calculate_lag(combined, range(1,7), 'ConfirmedCases')

combined = calculate_lag(combined, range(1,7), 'Fatalities')

combined = calculate_trend(combined, [1], 'ConfirmedCases')

combined = calculate_trend(combined, [1], 'Fatalities')

combined.replace([np.inf, -np.inf], 0, inplace=True)

combined.fillna(0, inplace=True)
def add_trends(df):

    master = pd.DataFrame()

    for country in df['Country'].unique():

        df1 = df.copy()[df['Country']==country].sort_values(by='Date')



        df1['NewConfirmed'] = df1['ConfirmedCases'] - df1['ConfirmedCases'].shift(1)

        df1['NewConfirmed'] = df1['NewConfirmed'].fillna(df1['ConfirmedCases'])

        df1['PreviousDayNewConfirmed'] = df1['NewConfirmed']

        df1['PreviousDayNewConfirmed'] = df1['PreviousDayNewConfirmed'].fillna(0)

        df1['GrowthFactor'] = df1['NewConfirmed'] / df1['PreviousDayNewConfirmed'] # https://www.youtube.com/watch?v=Kas0tIxDvrg

        df1['GrowthFactor'] = df1['GrowthFactor'].rolling(3).median() # add smoothing

        



        df1['NewFatalities'] = df1['Fatalities'] - df1['Fatalities'].shift(1)

        df1['NewFatalities'] = df1['NewFatalities'].fillna(df1['Fatalities'])

        df1['PreviousDayNewFatalities'] = df1['Fatalities'].shift(1) - df1['Fatalities'].shift(2)

        df1['PreviousDayNewFatalities'] = df1['PreviousDayNewFatalities'].fillna(0)

        df1['MortalityFactor'] = df1['NewFatalities'] / df1['PreviousDayNewFatalities']

        df1['MortalityFactor'] = df1['MortalityFactor'].rolling(3).median() # add smoothing

        

        df1['CaseFatalityRate'] = df1['Fatalities'] / df1['ConfirmedCases']

        df1['InfectionRate'] = df1['ConfirmedCases'] / df1['Pop']

        df1['MortalityRate'] = df1['Fatalities'] / df1['Pop']

        

        df1 = df1.replace([np.inf, -np.inf], np.nan)

        df1['GrowthFactor'] = np.where(df1['ConfirmedCases']==0, 0, df1['GrowthFactor'].fillna(1.25)) # assumed

        df1['MortalityFactor'] = np.where(df1['Fatalities']==0, 0, df1['MortalityFactor'].fillna(1.25)) # assumed

        df1['CaseFatalityRate'] = df1['CaseFatalityRate'].fillna(0)

        df1['InfectionRate'] = df1['InfectionRate'].fillna(0)

        df1['MortalityRate'] = df1['MortalityRate'].fillna(0)

        

        master = master.append(df1)

        

    # Add expected cases based on different mortality rates

    #master['ExpectedCases0.5'] = master['Fatalities'].shift(1) / 0.005

    #master['ExpectedCases1'] = master['Fatalities'].shift(1) / 0.01

    #master['ExpectedCases2'] = master['Fatalities'].shift(1) / 0.02

    #master['ExpectedCases5'] = master['Fatalities'].shift(1) / 0.05

    #for col in ['ExpectedCases0.5', 'ExpectedCases1', 'ExpectedCases2', 'ExpectedCases5']:

     #   master[col] = np.where(master[col]==0, master['ConfirmedCases'].shift(1), master[col])

        

    return master.drop(['NewConfirmed', 'PreviousDayNewConfirmed', 'NewFatalities', 'PreviousDayNewFatalities'], axis=1)



combined = add_trends(combined)

display_all(combined)
# Label encode categorical features

cat = combined.copy()[['Province/State', 'Region', 'Climate', 'Country']]

cont = combined.drop(['Province/State', 'Region', 'Climate', 'Country'], axis=1)



# Use label encoding to convert categorical features to numeric

# https://stackoverflow.com/a/37038257

def label_encode(df):

    # Convert df to label encoded

    df_le = pd.DataFrame({col: df[col].astype('category').cat.codes for col in df}, index=df.index)

    # Save mappings as a dict

    mappings = {col: {n: cat for n, cat in enumerate(df[col].astype('category').cat.categories)} 

     for col in df}

    return df_le, mappings



cat_le, mappings = label_encode(cat)

combined = pd.merge(cat_le, cont, left_index=True, right_index=True)
# Apply log transformations

cols = [x for x in list(combined) if 'ConfirmedCases' in x and 'Trend' not in x or 'Fatalities' in x and 'Trend' not in x]

combined[cols] = combined[cols].astype('float64').apply(lambda x: np.log(x))

combined.replace([np.inf, -np.inf], 0, inplace=True)



# Split into train, test and validation sets

train = combined[combined['ForecastId']==-1]

test = combined[combined['ForecastId']!=-1]

valid = train.copy()[train['Date']>=test['Date'].min()]

train = train.copy()[train['Date']<test['Date'].min()]

index = test['ForecastId']



def split_xs_ys(df):

    y1 = df['ConfirmedCases']

    y2 = df['Fatalities']

    X = df.drop(['ConfirmedCases', 'Fatalities', 'Date', 'ForecastId', 'Id'], axis=1)

    return X, y1, y2



X_train, y1_train, y2_train = split_xs_ys(train)

X_valid, y1_valid, y2_valid = split_xs_ys(valid)

X_test, y1_test, y2_test = split_xs_ys(test)
# Fit a model to predict confirmed cases

#m = RandomForestRegressor(n_estimators=100, max_features=0.5, max_depth=15, n_jobs=-1)

m = RandomForestRegressor(n_estimators=100, max_features=0.5, min_samples_leaf=5, n_jobs=-1, bootstrap=True)



def r_mse(pred, y): return round(math.sqrt(((pred-y)**2).mean()), 6)

def m_rmse(m, X, y): return r_mse(m.predict(X), y)



m.fit(X_train, y1_train)

m_rmse(m, X_train, y1_train), m_rmse(m, X_valid, y1_valid), m_rmse(m, X_test, y1_test)
def rf_feat_importance(m, df):

    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}

                       ).sort_values('imp', ascending=True)



fi = rf_feat_importance(m, X_train)



# Remove low importance variables

to_keep = fi[fi.imp>0.001].cols

X_train_c = X_train.copy()[to_keep]

X_test_c = X_test.copy()[to_keep]

X_valid_c = X_valid.copy()[to_keep]



m.fit(X_train_c, y1_train)

m_rmse(m, X_train_c, y1_train), m_rmse(m, X_valid_c, y1_valid), m_rmse(m, X_test_c, y1_test)
def plot_fi(fi):

    return fi.plot('cols', 'imp', 'barh', figsize=(12,20), legend=False)



fi = rf_feat_importance(m, X_train_c)

plot_fi(fi);
#from sklearn.inspection import plot_partial_dependence

#fig,ax = plt.subplots(figsize=(12, 4))

#plot_partial_dependence(m, X_valid, ['DaysSinceFirstCase', 'Pop', 'WorldShare'], ax=ax);
!pip install treeinterpreter

!pip install waterfallcharts

from treeinterpreter import treeinterpreter

from waterfall_chart import plot as waterfall



row = X_valid_c.iloc[:5]

prediction,bias,contributions = treeinterpreter.predict(m, row.values)

waterfall(X_valid_c.columns, contributions[0], threshold=0.08, 

          rotation_value=45,formatting='{:,.3f}');
pred_confirmed = m.predict(X_test_c)
m.fit(X_train, y2_train)



m_rmse(m, X_train, y2_train), m_rmse(m, X_valid, y2_valid), m_rmse(m, X_test, y2_test)
fi = rf_feat_importance(m, X_train)



# Remove low importance variables

to_keep = fi[fi.imp>0.001].cols

X_train_f = X_train.copy()[to_keep]

X_test_f = X_test.copy()[to_keep]

X_valid_f = X_valid.copy()[to_keep]



m.fit(X_train_f, y2_train)

m_rmse(m, X_train_f, y2_train), m_rmse(m, X_valid_f, y2_valid), m_rmse(m, X_test_f, y2_test)
fi = rf_feat_importance(m, X_train_f)

plot_fi(fi);
#fig,ax = plt.subplots(figsize=(12, 4))

#plot_partial_dependence(m, X_valid, ['DaysSinceFirstCase', 'Pop', 'WorldShare'], ax=ax);
row = X_valid_f.iloc[:5]

prediction,bias,contributions = treeinterpreter.predict(m, row.values)

waterfall(X_valid_f.columns, contributions[0], threshold=0.08, 

          rotation_value=45,formatting='{:,.3f}');
pred_fatalities = m.predict(X_test_f)
submission = pd.DataFrame({

        "ForecastId": index, 

        "ConfirmedCases": np.exp(pred_confirmed),

        "Fatalities": np.exp(pred_fatalities)

    })

submission['ForecastId'] = submission['ForecastId'].astype('int32')

display_all(submission)
submission.drop_duplicates('ForecastId').to_csv('submission.csv', index=False)
mappings['Country']
def plot_preds(country):

    hist = train[(train['Country']==country)&(train['ConfirmedCases']>0)]

    hist = hist.append(valid[(valid['Country']==country)&(valid['ConfirmedCases']>0)], sort=True)

    hist['ConfirmedCases'] = np.exp(hist['ConfirmedCases'])

    hist['Fatalities'] = np.exp(hist['Fatalities'])

    hist = hist[['Date', 'ConfirmedCases', 'Fatalities']]



    preds = test[(test['Country']==country)].rename(columns={'ConfirmedCases':'PredictedCases', 'Fatalities':'PredictedFatalities'})

    preds['PredictedCases'] = np.exp(preds['PredictedCases'])

    preds['PredictedFatalities'] = np.exp(preds['PredictedFatalities'])

    hist = hist.groupby(['Date']).agg({'ConfirmedCases':'sum', 'Fatalities':'sum'}).reset_index()

    preds = preds.groupby(['Date']).agg({'PredictedCases':'sum', 'PredictedFatalities':'sum'}).reset_index()



    fig, ax = plt.subplots(figsize=(16,10))



    conf = hist.set_index(['Date'])['ConfirmedCases'].plot(ax=ax, marker='o')

    fatal = hist.set_index(['Date'])['Fatalities'].plot(ax=ax, marker='o')

    pred_conf = preds.set_index(['Date'])['PredictedCases'].plot(ax=ax, marker='o')

    pred_fatal = preds.set_index(['Date'])['PredictedFatalities'].plot(ax=ax, marker='o')

    plt.legend()

    plt.tight_layout();

    

    df = pd.merge(hist, preds, on='Date', how='left')

    df = df[df['Date']>=test['Date'].min()]

    return df

    

output = plot_preds(82)

output
# Based on the plot above, this isn't a very good model!!