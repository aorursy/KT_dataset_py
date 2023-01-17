# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





import matplotlib.pyplot as plt

%matplotlib inline





from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objs as go

import plotly.offline as py

import pycountry

py.init_notebook_mode(connected=True)







import seaborn as sns

sns.set()



from collections import Counter



import plotly.express as px

import folium

from folium import plugins



from sklearn.linear_model import LinearRegression



from datetime import date



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_df = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

data_df .head()
data_df.columns
data_df.describe
data_df.info()


#data_df.drop(['Sno'],axis=1,inplace=True)

data_df.head()
data_df['Country'].replace({'Mainland China':'China'},inplace=True)

country = data_df['Country'].unique().tolist()

print(country)

print("\n Affected contries by virus: ", len(country))
union_deneme = []



union_deneme = data_df[["Country","Confirmed"]].groupby(["Country"], as_index = False).sum().sort_values(by="Confirmed", ascending = False).reset_index(drop=True)

union_deneme

death_data = data_df[['Country', 'Deaths']].groupby(["Country"], as_index = False).sum().sort_values(by="Deaths", ascending=False).reset_index(drop=True)

death_data = death_data[death_data['Deaths']>0]

death_data
import datetime

data_df['Last Update'] = pd.to_datetime(data_df['Last Update']) 

data_df['Date'] = [datetime.datetime.date(d) for d in data_df ['Last Update']]

data_df['Time'] = [datetime.datetime.time(d) for d in data_df['Last Update']]
data_df['Date'] = data_df['Date'].astype(str)

day = data_df["Date"].values

day = [my_str.split("-")[2] for my_str in day]

data_df["Date"] = day
dates = data_df['Date'].unique()

dates = np.flipud(dates) 



dates
all_cases = []



for i in dates:

    all_cases.append(data_df[data_df['Date']==i].Confirmed.sum())



plt.figure(figsize=(15, 10));

plt.plot(dates, all_cases);

plt.title('Daily Case Numbers', size=15);

plt.xlabel('Days', size= 10)

plt.ylabel('Number of Cases', size=15);

plt.show();
all_cases = []



for i in dates:

    all_cases.append(data_df[data_df['Date']==i].Deaths.sum())



plt.figure(figsize=(15, 10));

plt.plot(dates, all_cases);

plt.title('Daily Death Numbers', size=15);

plt.xlabel('Days', size= 10)

plt.ylabel('Number of Cases', size=15);

plt.show();





fig = px.scatter_geo(data_df, locations= union_deneme["Country"], locationmode='country names', 

                     color= union_deneme["Confirmed"], hover_name= union_deneme["Country"], range_color= [0, 500], projection="natural earth",

                    title='Countries Where Cases Spread')

fig.show()


fig = px.scatter_geo(data_df, locations= death_data["Country"], locationmode='country names', 

                     color= death_data["Deaths"], hover_name= death_data["Country"], range_color= [0, 500], projection="natural earth",

                    title='Countries where Deaths Occurred')

fig.show()
f, ax = plt.subplots(figsize=(10, 16))





sns.barplot(x="Confirmed", y="Province/State", data=data_df[1:],

            label="Confirmed", color="b")





sns.barplot(x="Recovered", y="Province/State", data=data_df[1:],

            label="Recovered", color="g")



ax.legend(ncol=6, loc="lower right", frameon=True)

ax.set(xlim=(0, 1800), ylabel="",

       xlabel="Stats")

sns.despine(left=True, bottom=True)
stateCountry = data_df.groupby(['Country', 'Province/State']).size().unstack()
plt.figure(figsize=(15,10))

sc = sns.heatmap(stateCountry,square=True, cbar_kws={'fraction' : 0.01}, cmap='afmhot_r', linewidth=1)
data_df.shape
data_df.describe()
data_df.isnull().sum()
data_df.drop(['Sno'], axis=1, inplace=True)
data_df['Last Update'] =data_df['Last Update'].apply(pd.to_datetime)
data_df.tail()
data_df['Province/State'].value_counts()
countries = data_df['Country'].unique().tolist()

print(countries)



print("\nTotal countries affected by virus: ",len(countries))

data_df
# Latest Numbers



print('Confirmed Cases around the globe : ',data_df['Confirmed'].sum())

print('Deaths Confirmed around the globe: ',data_df['Deaths'].sum())

print('Recovered Cases around the globe : ',data_df['Recovered'].sum())
tempState = data_df['Province/State'].mode()

print(tempState)
## Countries Currently affected by it.

allCountries = data_df['Country'].unique().tolist()

print(allCountries)



print("\nTotal countries affected by virus: ",len(allCountries))
CountryWiseData = pd.DataFrame(data_df.groupby('Country')['Confirmed', 'Deaths', 'Recovered'].sum())

CountryWiseData['Country'] = CountryWiseData.index

CountryWiseData.index = np.arange(1, len(allCountries)+1)



CountryWiseData = CountryWiseData[['Country','Confirmed', 'Deaths', 'Recovered']]



#formatted_text('***Country wise Analysis of ''Confirmed'', ''Deaths'', ''Recovered'' Cases***')

CountryWiseData
data_df.plot(subplots=True,figsize=(18,18))

plt.show()