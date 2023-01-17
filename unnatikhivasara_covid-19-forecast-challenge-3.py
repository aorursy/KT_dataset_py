# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

import statsmodels.formula.api as sm



%matplotlib inline

import matplotlib



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#read train and test data

train = pd.read_csv(r"/kaggle/input/covid19-global-forecasting-week-3/train.csv")

test = pd.read_csv(r"/kaggle/input/covid19-global-forecasting-week-3/test.csv")
print(train.head())

print(test.head())
#check for null values in training data

train.isna().mean().round(4)
#fill the empty cells with Unknown values

train.fillna("Unknown",inplace=True)
#Group the train data country wise by summing up the values of other columns -  ConfirmedCases and Fatalities

data = (train.groupby(["Date","Country_Region"]).agg({"ConfirmedCases":"sum","Fatalities":"sum"}).reset_index())
#Similarly group test data

test_data = ( test.groupby(["Date","Country_Region"]).last().reset_index()[["Date","Country_Region"]])
#changing the format of date column for train and test data

data["Date"] = pd.to_datetime(data.Date)

test_data["Date"] = pd.to_datetime(test_data.Date)
#As mentioned in the instructions train and test data have overlap of 1 week

print(f"Data last date: {data.Date.max()}")

print(f"Test date first: {test_data.Date.min()}")
#Finding unique countries in train and test data

train_countries = data["Country_Region"].unique()

test_countries = test_data["Country_Region"].unique()
len(train_countries)
#Check if Countries are same in Train and Test data

set(train_countries) == set(test_countries)
countries = ['US',"France", "Spain", "Italy", "India", "United Kingdom","China"]
covid_country = data[data.Country_Region.isin(countries)]
fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(10,10))

covid_country.groupby('Country_Region')['ConfirmedCases'].plot(ax=axes[0], legend=True)

covid_country.groupby('Country_Region')['Fatalities'].plot(ax=axes[1],legend=True)

axes[0].set_title('Confirmed Cases') 

axes[1].set_title('Fatalities') 
plt.figure(1,figsize=(16,16*8))

for i in range(1, len(train_countries)):

           _data = data[data['Country_Region'] == train_countries[i-1]]

           plt.subplots_adjust(top=2.5)

           plt.subplot(60,3,i)

           plt.xticks(rotation=45)

           plt.title(train_countries[i-1])

           plt.plot(_data.Date, _data.ConfirmedCases, color="grey")

           plt.plot(_data.Date,_data.Fatalities,color="r")
#Combine data by Country summing up the confirmed cases and fatalities values per countrywise

combined_data = data.groupby("Country_Region").agg({"ConfirmedCases":"sum","Fatalities":"sum"})
combined_data.shape
#picking up the countries having confirmed cases 

combined_data = combined_data[combined_data.ConfirmedCases > 1000]

combined_data.shape
combined_data["FatalityPercentage"] = (combined_data["Fatalities"]*100)/combined_data["ConfirmedCases"]
#Finding Mortality Rate

mortality_rate = combined_data["FatalityPercentage"].sort_values(ascending=False)

#Top 10 countries with highest mortality rate

print(mortality_rate.head(20))
print(f"Median mortality rate: {mortality_rate.median()}")
#Check overlapping date values between train and test set

print(f"Data last date: {data.Date.max()}")

print(f"Test Data first date: {test_data.Date.min()}")
#Remove overlapping values from both the sets

valid = data[data.Date >= min(test_data.Date)]

train = data[data.Date < min(test_data.Date)]
#assigning null values to the columns to be predicted

test_data["ConfirmedCases"] = 0.0

test_data["Fatalities"] = 0.0
#using linear regression predict the values on country-by-country basis

for country in train_countries:

    _train = train[train["Country_Region"]==country]

    _test = test_data[test_data["Country_Region"]==country]

    confirmed = _train["ConfirmedCases"].values[-10:]

    fatalities = _train["Fatalities"].values[-10:]

    if np.sum(confirmed) > 0:

        X = np.arange(len(confirmed)).reshape(-1,1)

        X_test = len(confirmed) + np.arange(len(_test)).reshape(-1,1)

        

        model = LinearRegression()

        model.fit(X,confirmed)

        

        p_conf = (model.predict(X_test))

        p_conf = np.clip(p_conf,0,None)

        p_conf = p_conf - np.min(p_conf) + confirmed[-1]

        p_conf = np.ceil(p_conf)

        test_data.loc[test_data["Country_Region"] == country, "ConfirmedCases"] = p_conf

                  

        model = LinearRegression()

        model.fit(X,fatalities)

        

        p_fatal = (model.predict(X_test))

        p_fatal = np.clip(p_fatal,0,None)

        p_fatal = p_fatal - np.min(p_fatal) + fatalities[-1]

        p_fatal = np.ceil(p_fatal)

        test_data.loc[test_data["Country_Region"] == country, "Fatalities"] = p_fatal

test_data[["ConfirmedCases","Fatalities"]].fillna(0,inplace=True)

        
test_data.head()
test_data.index.name = "ForecastId"
test_data = pd.DataFrame(test_data).rename_axis("ForecastId", axis=1)

print (test_data.head())
test_data_filtered = test_data[["ConfirmedCases","Fatalities"]]

test_data_filtered.head()
#exporting data to csv

test_data_filtered.to_csv('submission.csv')