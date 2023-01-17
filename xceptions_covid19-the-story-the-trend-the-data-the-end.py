import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



pd.set_option("display.width", 300)
DIR = '/kaggle/input/corona-virus-report/'



data = pd.read_csv(DIR + 'covid_19_clean_complete.csv', parse_dates=['Date'])

data.sample(7)
def engineer_features(data):

    data['year'] = data.Date.dt.year

    data['month'] = data.Date.dt.month

    data['day'] = data.Date.dt.day

    data['dayofweek'] = data.Date.dt.dayofweek

    return data

    

data = engineer_features(data)



# rename for convenience

data.rename({'Confirmed': 'cases',

             'Country/Region': 'country',

             'Province/State': 'state',

             'Deaths': 'fatalities',

             'Date': 'date'}, axis='columns', inplace=True)



def basic_eda(df):

    print("First five rows of data")

    print("=" * 100)

    print(df.head())

    print("\n")

    print("Data Information")

    print("=" * 100)

    print(df.info())

    print("\n")

    print("Data Statistics")

    print("=" * 100)

    print(df.describe())

    

basic_eda(data)
# Total number worldwide



data['state'] = data['state'].fillna('')

temp = data[[col for col in data.columns if col != 'state']]



latest = temp[temp['date'] == max(temp['date'])].reset_index()

latest_grouped = latest.groupby('country')['cases', 'fatalities'].sum().reset_index()



sns.set_style("darkgrid")

plt.figure(figsize=(25, 10))

sns.barplot(x="cases", y="country", data=latest_grouped.sort_values('cases', ascending=False)[:20])

plt.title("WORLDWIDE CONFIRMED CASES")

plt.show()



sns.set_style("darkgrid")

plt.figure(figsize=(25, 10))

sns.barplot(x="fatalities", y="country", data=latest_grouped.sort_values('fatalities', ascending=False)[:20])

plt.title("WORLDWIDE FATALITIES")

plt.show()
# world wide confirmed cases overtime

group = data.groupby('date')['date', 'cases', 'fatalities'].sum().reset_index()



sns.set_style("ticks")

sns.set(rc={'axes.facecolor':'#80d8e0', 'figure.facecolor':'#80d8e0'})



plt.figure(figsize=(25, 10))

fig = sns.lineplot(x="date", y="cases", data=data)

plt.title("darkgrid")

plt.show()



sns.set_style("darkgrid")

sns.set(rc={'axes.facecolor':'#9697c8', 'figure.facecolor':'#9697c8'})

plt.figure(figsize=(25, 10))

fig = sns.lineplot(x="date", y="fatalities", data=data)

plt.title("darkgrid")

plt.show()
grouped_china = data[data['country'] == "China"].reset_index()

grouped_china_date = grouped_china.groupby('date')['date', 'cases', 'fatalities'].sum().reset_index()



sns.set_style("ticks")

sns.set(rc={'axes.facecolor':'#80d8e0', 'figure.facecolor':'#80d8e0'})



plt.figure(figsize=(25, 10))

fig = sns.lineplot(x="date", y="cases", data=grouped_china_date)

plt.title("darkgrid")

plt.show()
grouped_country = data[data['country'] == "Italy"].reset_index()

grouped_country_date = grouped_country.groupby('date')['date', 'cases', 'fatalities'].sum().reset_index()



sns.set_style("ticks")

sns.set(rc={'axes.facecolor':'#cccccc', 'figure.facecolor':'#cccccc'})



plt.figure(figsize=(25, 10))

fig = sns.lineplot(x="date", y="cases", data=grouped_country_date)

plt.title("darkgrid")

plt.show()



grouped_country = data[data['country'] == "US"].reset_index()

grouped_country_date = grouped_country.groupby('date')['date', 'cases', 'fatalities'].sum().reset_index()



sns.set_style("ticks")

sns.set(rc={'axes.facecolor':'#cccccc', 'figure.facecolor':'#cccccc'})



plt.figure(figsize=(25, 10))

fig = sns.lineplot(x="date", y="cases", data=grouped_country_date)

plt.title("darkgrid")

plt.show()



grouped_country = data[data['country'] == "Nigeria"].reset_index()

grouped_country_date = grouped_country.groupby('date')['date', 'cases', 'fatalities'].sum().reset_index()



sns.set_style("ticks")

sns.set(rc={'axes.facecolor':'#cccccc', 'figure.facecolor':'#cccccc'})



plt.figure(figsize=(25, 10))

fig = sns.lineplot(x="date", y="cases", data=grouped_country_date)

plt.title("darkgrid")

plt.show()
grouped_rest = data[~data['country'].isin(['China', 'Italy', 'US', 'Nigeria'])].reset_index()

grouped_rest_date = grouped_rest.groupby('date')['date', 'cases', 'fatalities'].sum().reset_index()



sns.set_style("ticks")

sns.set(rc={'axes.facecolor':'cornflowerblue', 'figure.facecolor':'cornflowerblue'})



plt.figure(figsize=(25, 10))

fig = sns.lineplot(x="date", y="cases", data=grouped_rest_date)

plt.title("darkgrid")

plt.show()
cleanest_countries = ["Finland", "Iceland", "Sweden", "Denmark", "Slovenia"]



for country in cleanest_countries:

    grouped_country = data[data['country'] == country].reset_index()

    grouped_country_date = grouped_country.groupby('date')['date', 'cases', 'fatalities'].sum().reset_index()



    sns.set_style("ticks")

    sns.set(rc={'axes.facecolor':'#8ac7d7', 'figure.facecolor':'#8ac7d7'})



    plt.figure(figsize=(25, 10))

    fig = sns.lineplot(x="date", y="cases", data=grouped_country_date)

    plt.title("Cases for " + country)

    plt.show()
print(f"Cases as at {max(data['date'])}:")

for country in cleanest_countries:

    max_vals = data.loc[data['country'] == country]

    max_case = max(max_vals['cases'])

    max_fatalities = max(max_vals['fatalities'])

    print(f"{ country } ---> { max_case } cases, { max_fatalities } fatalities {(max_fatalities * 100) / max_case:.4f} %")
lawful_countries = ["Norway", "Netherlands", "Germany", "New Zealand", "Austria"]



for country in lawful_countries:

    grouped_country = data[data['country'] == country].reset_index()

    grouped_country_date = grouped_country.groupby('date')['date', 'cases', 'fatalities'].sum().reset_index()



    sns.set_style("ticks")

    sns.set(rc={'axes.facecolor':'#8ac7d7', 'figure.facecolor':'#8ac7d7'})



    plt.figure(figsize=(25, 10))

    fig = sns.lineplot(x="date", y="cases", data=grouped_country_date)

    plt.title("Cases for " + country)

    plt.show()
print(f"Cases as at {max(data['date'])}:")

for country in lawful_countries:

    max_vals = data.loc[data['country'] == country]

    max_case = max(max_vals['cases'])

    max_fatalities = max(max_vals['fatalities'])

    print(f"{ country } ---> { max_case } cases, { max_fatalities } fatalities {(max_fatalities * 100) / max_case:.4f} %")
countries_ = ["China", "Italy", "US", "Spain", "Germany", "Iran", "France", "United Kingdom", "Netherlands", "Belgium"]



for country in countries_:

    grouped_country = data[data['country'] == country].reset_index()

    grouped_country_date = grouped_country.groupby('date')['Recovered'].sum().reset_index()



    sns.set_style("ticks")

    sns.set(rc={'axes.facecolor':'#ffe8d1', 'figure.facecolor':'#ffe8d1'})



    plt.figure(figsize=(25, 10))

    fig = sns.lineplot(x="date", y="Recovered", data=grouped_country_date)

    plt.title("Recovery for " + country)

    plt.show()
print(f"Cases as at {max(data['date'])}:")

for country in countries_:

    max_vals = data.loc[data['country'] == country]

    max_case = max(max_vals['cases'])

    max_recoveries = max(max_vals['Recovered'])

    print(f"{ country } ---> { max_case } cases, { max_recoveries } recoveries. {(max_recoveries * 100) / max_case:.4f} %")