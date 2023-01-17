import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import scipy

from sklearn.metrics import mean_squared_error, r2_score
def prediction_poly(country, how_far):

    

    #Importing data

    data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

    #print(data.dtypes)

    

    #Choosing country for analysis

    if country == "France":

        data_country = data.loc[(data["Country/Region"] == country) & (data["Province/State"] == country)]

    elif country == "UK":

        data_country = data.loc[(data["Country/Region"] == country) & (data["Province/State"] == "United Kingdom")]

    #elif country == "US":

    else:

        data_country = data.loc[data["Country/Region"] == country]

    #print(data_country.head(len(data_country)).sort_values(by=["Confirmed"], ascending=True))

    

    #Preparing data for curve fitting

    days_country=np.array(np.arange(1, len(data_country)+1), dtype=int)

    cases_country=np.array(data_country["Confirmed"], dtype=int)

    #print(f"Shape of input {days_country.shape}")

    #print(f"Shape of output {cases_country.shape}")

    

    #Polynomial curve fitting

    tmp = 0

    for i in range(1, 100):

        predictor = np.poly1d(np.polyfit(days_country,cases_country,i))

        if int(predictor(np.array([len(days_country)+1]))) > tmp:

            tmp = int(predictor(np.array([len(days_country)+how_far])))

        else:

            power = i

            break

            

    print(f"Prediction of number of cases in {country} after {how_far} days from today is {int(predictor(np.array([len(days_country)+how_far])))} by polynom of power {power}")

    

    #Graph of polynomial curve fitting

    plt.figure(figsize=(5,5))

    plt.xlabel("Days")

    plt.ylabel("Cases")

    plt.title(f"{country} polynomial graph")



    plt.scatter(days_country, cases_country, label = "Real data")

    plt.plot(days_country, predictor(days_country), color="orange", label = "Prediction")

    plt.legend(prop={'size': 15})

    

    #Evaluating errors

    rmse = np.sqrt(mean_squared_error(cases_country, predictor(days_country)))

    r2 = r2_score(cases_country, predictor(days_country))

    print(f"RMSE error is {rmse}")

    print(f"R2 error is {r2}")



def prediction_exp(country, how_far):

    

    #Importing data

    data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

    #print(data.dtypes)

    

    #Choosing country for analysis

    if country == "France":

        data_country = data.loc[(data["Country/Region"] == country) & (data["Province/State"] == country)]

    elif country == "UK":

        data_country = data.loc[(data["Country/Region"] == country) & (data["Province/State"] == "United Kingdom")]

    #elif country == "US":

    else:

        data_country = data.loc[data["Country/Region"] == country]

    #print(data_country.head(len(data_country)).sort_values(by=["Confirmed"], ascending=True))

    

    #Preparing data for curve fitting

    days_country=np.array(np.arange(1, len(data_country)+1), dtype=int)

    cases_country=np.array(data_country["Confirmed"], dtype=int)

    #print(f"Shape of input {days_country.shape}")

    #print(f"Shape of output {cases_country.shape}")

    

    #Exponential curve fitting

    def func(x, a, b, c):

        return a+np.exp(b*x+c)

    params, pcov = scipy.optimize.curve_fit(func, days_country, cases_country, maxfev=10000000)

    #print(params)



    print(f"Prediction of number of cases in {country} after {how_far} days from today is {int(func(np.array([len(days_country)+how_far]), params[0], params[1], params[2]))} by exponential function {round(params[0], 3)} + exp({round(params[1], 3)} * x + {round(params[2], 3)}")



    #Graph of exponential curve fitting

    plt.figure(figsize=(5,5))

    plt.xlabel("Days")

    plt.ylabel("Cases")

    plt.title(f"{country} exponential graph")



    plt.scatter(days_country, cases_country, label = "Real data")

    plt.plot(days_country, func(days_country, params[0], params[1], params[2]), color="orange", label = "Prediction")

    plt.legend(prop={'size': 15})

    

    #Evaluating errors

    rmse = np.sqrt(mean_squared_error(cases_country, func(days_country, params[0], params[1], params[2])))

    r2 = r2_score(cases_country, func(days_country, params[0], params[1], params[2]))

    print(f"RMSE error is {rmse}")

    print(f"R2 error is {r2}")



prediction_poly("Italy", 1)

prediction_exp("Italy", 1)
data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

print(data.dtypes)

countries_list = sorted(list(set(data["Country/Region"])))

print(countries_list)

data.head(5)
data_italy = data.loc[data['Country/Region'] == 'Italy']

data_italy.head(len(data_italy)).sort_values(by=['Confirmed'], ascending=True)
days_italy=np.array(np.arange(1, len(data_italy)+1), dtype=int)

cases_italy=np.array(data_italy["Confirmed"], dtype=int)

cases_true = np.array(number for number in cases_italy if number != 0)

        

print(type(days_italy))

print(type(cases_italy))

print(days_italy.shape)

print(cases_italy.shape)
plt.xlabel("Days")

plt.ylabel("Cases")

plt.title("Italy virus spreading graph")



plt.plot(days_italy, cases_italy)
tmp = 0

for i in range(1, 100):

    predictor = np.poly1d(np.polyfit(days_italy,cases_italy,i))

    if int(predictor(np.array([len(days_italy)+1]))) > tmp:

        tmp = int(predictor(np.array([len(days_italy)+1])))

    else:

        power = i

        print(f"Final polynomial of power {i}")

        break
plt.figure(figsize=(15,15))

plt.xlabel("Days")

plt.ylabel("Cases")

plt.title("Italy polynomial graph")



plt.plot(days_italy, cases_italy, label = "Real data")

plt.plot(days_italy, predictor(days_italy), label = "Prediction")

plt.legend(prop={'size': 25})
rmse = np.sqrt(mean_squared_error(cases_italy, predictor(days_italy)))

r2 = r2_score(cases_italy, predictor(days_italy))

  

print(f"RMSE error is {rmse}")

print(f"R2 error is {r2}")
print(f"Prediction of number of cases in Italy after 1 day from today is {int(predictor(np.array([len(days_italy)+1])))}")
def func(x, a, b, c):

    return a+np.exp(b*x+c)

params, pcov = scipy.optimize.curve_fit(func, days_italy, cases_italy, maxfev=10000000)

print(f"Final expression of exponential is {round(params[0], 3)} + exp({round(params[1], 3)} * x + {round(params[2], 3)})")
plt.figure(figsize=(15,15))

plt.xlabel("Days")

plt.ylabel("Cases")

plt.title("Italy exponential graph")



plt.plot(days_italy, cases_italy, label = "Real data")

plt.plot(days_italy, func(days_italy, params[0], params[1], params[2]), label = "Prediction")

plt.legend(prop={'size': 25})
rmse = np.sqrt(mean_squared_error(cases_italy, func(days_italy, params[0], params[1], params[2])))

r2 = r2_score(cases_italy, func(days_italy, params[0], params[1], params[2]))

  

print(f"RMSE error is {rmse}")

print(f"R2 error is {r2}")
print(f"Prediction of number of cases in Italy after 1 day from today is {int(func(np.array([len(days_italy)+1]), params[0], params[1], params[2]))}")
prediction_poly("Italy", 1)

prediction_exp("Italy", 1)
prediction_poly("Spain", 1)

prediction_exp("Spain", 1)
prediction_poly("Germany", 1)

prediction_exp("Germany", 1)
prediction_poly("UK", 1)

prediction_exp("UK", 1)
prediction_poly("France", 1)

prediction_exp("France", 1)
prediction_poly("Israel", 1)

prediction_exp("Israel", 1)
prediction_poly("Turkey", 1)

prediction_exp("Turkey", 1)
prediction_poly("Russia", 1)

prediction_exp("Russia", 1)
prediction_poly("Azerbaijan", 1)

prediction_exp("Azerbaijan", 1)