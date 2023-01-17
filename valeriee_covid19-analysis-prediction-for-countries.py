# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #visualization
# Read dataset

dataset = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")
dataset.head()
# You can change the country variable to your prefer

print(dataset['Country_Region'].unique())

country = 'Turkey'
# Country's data was taken since the first case

country_cases = dataset[dataset.Country_Region == country]

df =  country_cases[country_cases.ConfirmedCases != 0.0]

print(df)
print(df.info())

print(df.describe())

total_days = max(df.Id) - min(df.Id) + 1 

total_cases = max(df.ConfirmedCases)

total_deaths = max(df.Fatalities)

print("Number of days since the first case:", total_days)
plt.figure(figsize=(30,10))    

plt.plot(df.Date, df.ConfirmedCases, color = 'blue', label = 'Cases')

plt.title('Coronavirus Cases Graph', fontsize=30)

plt.legend(frameon=True, fontsize=20)

plt.xticks(np.arange(0, total_days, int(total_days/10)), fontsize = 15) #xlabel data freq

plt.yticks(np.arange(0, total_cases, int(total_cases/5)), fontsize = 15) #ylabel data freq

plt.xlabel('Dates', fontsize=20)

plt.ylabel('Cases', fontsize=20)

plt.grid(True)

plt.show()
plt.figure(figsize=(30,10)) 

plt.plot(df.Date ,df.Fatalities, color = 'red', label = 'Deaths')

plt.title('Coronavirus Deaths Graph', fontsize=30)

plt.legend(frameon=True, fontsize=20)

plt.xticks(np.arange(0, total_days, int(total_days/10)), fontsize = 15) 

plt.yticks(np.arange(0, total_deaths, int(total_deaths/5)),fontsize = 15) 

plt.xlabel('Dates', fontsize=20)

plt.ylabel('Deaths', fontsize=20)

plt.grid(True)

plt.show()
plt.figure(figsize=(30,10))

plt.bar(df.Date, df.ConfirmedCases, color = 'blue', label = 'Cases')

plt.xticks(np.arange(0, total_days, int(total_days/10)),fontsize = 15)

plt.yticks(np.arange(0, total_cases, int(total_cases/5)), fontsize = 15)

plt.xlabel('Dates', fontsize=20)

plt.ylabel('Cases',fontsize=20)

plt.legend(frameon=True, fontsize=20)

plt.grid(True)

plt.show()
plt.figure(figsize=(30,10))

plt.bar(df.Date, df.Fatalities, color = 'red', label = 'Deaths')

plt.xticks(np.arange(0, total_days, int(total_days/10)),fontsize = 15)

plt.yticks(np.arange(0, total_deaths, int(total_deaths/5)),fontsize = 15)

plt.xlabel('Dates', fontsize=20)

plt.ylabel('Deaths',fontsize=20)

plt.legend(frameon=True, fontsize=20)

plt.grid(True)

plt.show()
days_array = np.arange(1,total_days+1).reshape(total_days,1)

case_array = np.array(df.ConfirmedCases).reshape(total_days,1)

death_array = np.array(df.Fatalities).reshape(total_days,1)

print(days_array.shape)

print(case_array.shape)

print(death_array.shape)

# Now we got same shapes for models
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

case_poly_reg = PolynomialFeatures(degree = 4)

X_poly = case_poly_reg.fit_transform(days_array) # X axis

case_poly_reg.fit(X_poly, case_array) # model fit to predict y axis

case_lin_reg = LinearRegression()

case_lin_reg.fit(X_poly, case_array)
plt.scatter(days_array, case_array, color = 'blue')

plt.plot(days_array, case_lin_reg.predict(case_poly_reg.fit_transform(days_array)), color = 'green')

plt.title('Case Graph between days 1 and '+ str(total_days))

plt.xlabel('Dates')

plt.ylabel('Cases')

plt.grid(True)

plt.show()

# Dots are accurate datas

# Line is predict datas
death_poly_reg = PolynomialFeatures(degree = 4)

X_poly = death_poly_reg.fit_transform(days_array) # X axis

death_poly_reg.fit(X_poly, death_array) # model fit to predict y axis

death_lin_reg = LinearRegression()

death_lin_reg.fit(X_poly, death_array)
plt.scatter(days_array, death_array, color = 'red')

plt.plot(days_array, death_lin_reg.predict(death_poly_reg.fit_transform(days_array)), color = 'green')

plt.title('Death Graph between days 1 and '+ str(total_days) )

plt.xlabel('Days')

plt.ylabel('Deaths')

plt.grid(True)

plt.show()

# Dots are accurate datas

# Line is predict datas
total_cases = max(df.ConfirmedCases)

total_deaths = max(df.Fatalities)

pred_now_cases = int(case_lin_reg.predict(case_poly_reg.fit_transform([[total_days]])))

pred_now_deaths = int(death_lin_reg.predict(death_poly_reg.fit_transform([[total_days]])))

fault_margin_cases = int(pred_now_cases - total_cases)

fault_margin_deaths = int(pred_now_deaths - total_deaths)

print("Predict Today Cases and Deaths: ", pred_now_cases, " ", pred_now_deaths)

print("Fault Margin for Cases and Deaths: ", fault_margin_cases, " ", fault_margin_deaths)
# Predict cases and deaths for next day

pred_nextday_cases = int(case_lin_reg.predict(case_poly_reg.fit_transform([[total_days+1]])))

pred_nextday_deaths = int(death_lin_reg.predict(death_poly_reg.fit_transform([[total_days+1]])))

print("Predict Next Day Cases and Deaths")

print("Cases: ", pred_nextday_cases , " Deaths: ", pred_nextday_deaths)