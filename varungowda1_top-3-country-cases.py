import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

from scipy.stats import pearsonr

import pandas as pd

import numpy as np

test=pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')

train=pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')
test.head(5)
train.head(5)
# rename columns

train = train.rename(columns={'Province/State': 'Province_State', 'Country/Region': 'Country_Region','Date':'train_Date'})

test = test.rename(columns={'Province/State': 'Province_State', 'Country/Region': 'Country_Region','Date':'test_Date'})
train.head()
test.head()
train.isnull().sum()
test.isnull().sum()
covid19 = pd.merge(train, test)

covid19
# Finding missing values
covid19.isnull().sum()
covid19['Province_State'].value_counts()
covid19.Province_State.fillna(covid19.Country_Region, inplace=True)
covid19.head()
covid19.isnull().sum()
# Total Counts
covid19['Province_State'].value_counts()
covid19['Country_Region'].value_counts()
Total = covid19['ConfirmedCases'].sum()

print (Total)
Total = covid19['Fatalities'].sum()

print (Total)
Total = covid19['ForecastId'].sum()

print (Total)
covid19['train_Date'].max(), covid19['train_Date'].min()
covid19['test_Date'].max(), covid19['test_Date'].min()
# Top 10 with covid19

plot = covid19.Country_Region.value_counts().nlargest(10).plot(kind='bar', title="Top 10 Status", figsize=(12,6))
# Top 10 province with covid19

plot = covid19.Province_State.value_counts().nlargest(15).plot(kind='bar', title="Top 10 Status", figsize=(12,6))
confirmed_total_date = covid19.groupby(['train_Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date = covid19.groupby(['train_Date']).agg({'Fatalities':['sum']})

total_date = confirmed_total_date.join(fatalities_total_date)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))

total_date.plot(ax=ax1)

ax1.set_title("Global confirmed cases", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

fatalities_total_date.plot(ax=ax2, color='orange')

ax2.set_title("Global deceased cases", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
# Plotting a bar graph of the number of stores in each city, for the first ten cities listed

# in the column 'City'

Country_count  = covid19['Country_Region'].value_counts()

Country_count = Country_count[:10,]

plt.figure(figsize=(10,5))

sns.barplot(Country_count.index, Country_count.values, alpha=0.8)

plt.title('Most Dangerous Cases')

plt.ylabel('ConfirmedCases', fontsize=12)

plt.xlabel('Country_Region', fontsize=12)

plt.show()
data_country = covid19.groupby('Country_Region')
plt.figure(figsize=(35,20))

height = 100*data_country['ConfirmedCases'].sum()/covid19['ConfirmedCases'].sum()

x = data_country['ConfirmedCases'].sum().keys()

plt.bar(x = x, height= height)

plt.xticks(rotation='vertical')

for i,v in enumerate(height):

    plt.text(i, v, " "+str(round(v,2)), color='red', ha='center', rotation='vertical', va='bottom')

plt.show()
height['China'] + height['Italy'] + height['Iran']
data_Country = covid19.groupby('Country_Region')
plt.figure(figsize=(35,20))

height = 100*data_Country['Fatalities'].sum()/covid19['Fatalities'].sum()

x = data_Country['Fatalities'].sum().keys()

plt.bar(x = x, height= height)

plt.xticks(rotation='vertical')

for i,v in enumerate(height):

    plt.text(i, v, " "+str(round(v,2)), color='red', ha='center', rotation='vertical', va='bottom')

plt.show()
height['China'] + height['Italy'] + height['Iran']
grouped = covid19.groupby(['train_Date','Country_Region','Province_State','ConfirmedCases','Fatalities']).size()

grouped
confirmed_total_date_China = train[train['Country_Region']=='China'].groupby(['train_Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_China = train[train['Country_Region']=='China'].groupby(['train_Date']).agg({'Fatalities':['sum']})

total_date_China = confirmed_total_date_China.join(fatalities_total_date_China)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

total_date_China.plot(ax=ax1)

ax1.set_title("China confirmed cases", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

fatalities_total_date_China.plot(ax=ax2, color='orange')

ax2.set_title("China deceased cases", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
confirmed_total_date_Italy = train[train['Country_Region']=='Italy'].groupby(['train_Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_Italy = train[train['Country_Region']=='Italy'].groupby(['train_Date']).agg({'Fatalities':['sum']})

total_date_Italy = confirmed_total_date_Italy.join(fatalities_total_date_Italy)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

total_date_Italy.plot(ax=ax1)

ax1.set_title("Italy confirmed cases", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

fatalities_total_date_China.plot(ax=ax2, color='orange')

ax2.set_title("Italy deceased cases", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
confirmed_total_date_Iran = train[train['Country_Region']=='Iran'].groupby(['train_Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_Iran = train[train['Country_Region']=='Iran'].groupby(['train_Date']).agg({'Fatalities':['sum']})

total_date_Iran = confirmed_total_date_Iran.join(fatalities_total_date_Iran)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

total_date_Iran.plot(ax=ax1)

ax1.set_title("Iran confirmed cases", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

fatalities_total_date_China.plot(ax=ax2, color='orange')

ax2.set_title("Iran deceased cases", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)