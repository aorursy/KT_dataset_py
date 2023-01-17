import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from scipy.optimize import minimize
train_dataset = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

test_dataset = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')

submission = pd.read_csv('../input/covid19-global-forecasting-week-4/submission.csv')
submission.head()
train_dataset.info()
train_dataset.describe()
train_dataset.head()
test_dataset.head()
train_dataset.isna().sum()
test_dataset.isna().sum()
train_dataset['Province_State'].fillna('', inplace = True)

test_dataset['Province_State'].fillna('', inplace = True)
train_dataset['Country_Region'] = train_dataset['Country_Region'] + ' ' + train_dataset['Province_State']

test_dataset['Country_Region'] = test_dataset['Country_Region'] + ' ' + test_dataset['Province_State']

del train_dataset['Province_State']

del test_dataset['Province_State']
train_dataset.head()
test_dataset.head()
# How many countries

train_dataset['Country_Region'].describe()
country_list = train_dataset['Country_Region'].unique()
train_date = train_dataset.Date.unique()

train_date
test_date = test_dataset.Date.unique()

test_date
train_days = np.arange(len(train_date))

train_days
train_days[train_date == '2020-04-02']
train_days[train_date == '2020-04-22']
test_days = np.arange(len(test_date)) + 71

test_days
train_end = train_days[train_date == '2020-04-22']

test_start = test_days[0]
train_end
test_start
Day = np.zeros(len(train_dataset))

for ii in range(len(train_date)):

    Day[train_dataset.Date == train_date[ii]] = train_days[ii]

train_dataset['Day'] = Day
train_dataset.head(5)
Day = np.zeros(len(test_dataset))

for ii in range(len(test_date)):

    Day[test_dataset.Date == test_date[ii]] = test_days[ii]

test_dataset['Day'] = Day
test_dataset.head(5)
top_comfirmedcases = train_dataset[train_dataset.Date == '2020-04-22'].sort_values(by = 'ConfirmedCases', ascending = False)

top_comfirmedcases.head(10)
def country_plot(country):

    train = train_dataset[train_dataset['Country_Region'] == country]

    test = test_dataset[test_dataset['Country_Region'] == country]

    

    # X_train

    x_train = train.Day.values

    confirmed_train = train.ConfirmedCases.values

    fatalities_train = train.Fatalities.values

    

    # Plot figures

    # Confirmed cases

    plt.figure(figsize = (15, 3))

    plt.subplot(1, 2, 1)

    plt.xlabel('Days')

    plt.ylabel('Confirmed cases')

    plt.title(country)

    plt.plot(x_train, confirmed_train)

    plt.grid()



    # Fatalities

    plt.subplot(1, 2, 2)

    plt.xlabel('Days')

    plt.ylabel('Fatalities')

    plt.title(country)

    plt.plot(x_train, fatalities_train, color = 'orange')

    plt.grid()

    plt.show()
for country in top_comfirmedcases.Country_Region[0:9].values:

    country_plot(country)
def Gompertz(t, theta1, theta2, theta3):

    '''

    theta1: The asymptote.

    theta2: The displacement along the x-axis.

    theta3: The growth rate.

    '''

    f = theta1 * np.exp(-theta2 * np.exp(-theta3 * t))

    return f
x = np.linspace(start = -2, stop = 5, num = 50)

y1 = Gompertz(x, theta1 = 5, theta2 = 1, theta3 = 1)

y2 = Gompertz(x, theta1 = 5, theta2 = 1.5, theta3 = 1)

y3 = Gompertz(x, theta1 = 5, theta2 = 2, theta3 = 1)



plt.figure(figsize = (12, 8))

plt.plot(x, y1, label = 'y1')

plt.plot(x, y2, label = 'y1')

plt.plot(x, y3, label = 'y1')

plt.legend()

plt.grid()

plt.show()
x = np.linspace(start = -2, stop = 5, num = 50)

y1 = Gompertz(x, theta1 = 5, theta2 = 1, theta3 = 0.1)

y2 = Gompertz(x, theta1 = 5, theta2 = 1, theta3 = 0.5)

y3 = Gompertz(x, theta1 = 5, theta2 = 1, theta3 = 1)



plt.figure(figsize = (12, 8))

plt.plot(x, y1, label = 'y1')

plt.plot(x, y2, label = 'y1')

plt.plot(x, y3, label = 'y1')

plt.legend()

plt.grid()

plt.show()
country = 'Korea, South '

train = train_dataset[train_dataset['Country_Region'] == country]

test = test_dataset[test_dataset['Country_Region'] == country]



# X_train

x_train = train.Day.values

confirmed_train = train.ConfirmedCases.values

fatalities_train = train.Fatalities.values



# X_test

x_test = test.Day.values

country_plot(country)
popt_confirmed, pcov_confirmed = curve_fit(f = Gompertz, 

                                           xdata = x_train, 

                                           ydata = confirmed_train, 

                                           p0 = [3 * max(confirmed_train), 1, 1], 

                                           maxfev = 800)
popt_confirmed
pcov_confirmed
def curve_plot(x_train, y_train, x_test, est):

    plt.figure(figsize = (12, 5))

    plt.xlabel('Days')

    plt.ylabel('Cases')

    plt.title(country)

    plt.scatter(x_train, y_train, color = 'r')

    plt.plot(x_train, Gompertz(x_train, *est), label = 'Fitting curve (train)')

    plt.plot(x_test, Gompertz(x_test, *est), label = 'Fitting curve (test)')

    plt.axvline(x = test_start, color = 'r', linestyle = ':', label = 'test_start = %.f' % (test_start))

    plt.axvline(x = train_end, color = 'b', linestyle = ':', label = 'train_end = %.f' % (train_end))

    plt.legend()

    plt.show()
curve_plot(x_train = x_train, y_train = confirmed_train, x_test = x_test, est = popt_confirmed)
popt_fatalities, pcov_fatalities = curve_fit(f = Gompertz, 

                                             xdata = x_train, 

                                             ydata = fatalities_train, 

                                             p0 = [3 * max(fatalities_train), 1, 1], 

                                             maxfev = 800)
popt_fatalities
pcov_fatalities
curve_plot(x_train = x_train, y_train = fatalities_train, x_test = x_test, est = popt_fatalities)
# Minimize the Loss function: MSE

def growth_curve(x, y):

    # Loss function

    def l_fun(params): 

        theta1 = np.exp(params[0])

        theta2 = np.exp(params[1])

        theta3 = np.exp(params[2])

        mse = np.mean((y - Gompertz(x, theta1, theta2, theta3)) ** 2)

        return mse



    p0 = [np.log(3 * max(y)), 0, 0]

    res = minimize(fun = l_fun, x0 = p0, method = 'L-BFGS-B')

    mse = res.fun



    # convergence_res

    convergence_res = {'MSE': mse,

                       'nfev': res.nfev, 

                       'nit': res.nit, 

                       'status': res.status}



    # Estimator

    est = np.exp(res.x)

    return est, convergence_res
# Confirmed cases

est_confirmed, convergence_res = growth_curve(x = x_train, y = confirmed_train)

convergence_res
curve_plot(x_train = x_train, y_train = confirmed_train, x_test = x_test, est = est_confirmed)
# Confirmed cases

est_fatalities, convergence_res = growth_curve(x = x_train, y = fatalities_train)

convergence_res
curve_plot(x_train = x_train, y_train = fatalities_train, x_test = x_test, est = est_fatalities)
confirmed_pred = np.zeros(len(test_dataset))

fatalities_pred = np.zeros(len(test_dataset))
for country in country_list:

    train = train_dataset[train_dataset['Country_Region'] == country]

    test = test_dataset[test_dataset['Country_Region'] == country]

    

    # X_train

    x_train = train.Day.values

    confirmed_train = train.ConfirmedCases.values

    fatalities_train = train.Fatalities.values

    

    # X_test

    x_test = test.Day.values



    # Confirmed cases

    confirmed_est, confirmed_convergence = growth_curve(x = x_train, y = confirmed_train)

    

    # Fatalities    

    fatalities_est, fatalities_convergence = growth_curve(x = x_train, y = fatalities_train)

    

    # Predictions

    confirmed_pred[test_dataset.Country_Region == country] = Gompertz(x_test, *confirmed_est)

    fatalities_pred[test_dataset.Country_Region == country] = Gompertz(x_test, *fatalities_est)
submission['ConfirmedCases'] = confirmed_pred

submission['Fatalities'] = fatalities_pred
submission.to_csv('submission.csv', index = False)