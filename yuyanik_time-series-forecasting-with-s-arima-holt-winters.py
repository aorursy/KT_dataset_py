# Importing required libraries

import itertools

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

from scipy import stats

from pandas import DataFrame

plt.style.use('fivethirtyeight')

import pandas as pd

import statsmodels.api as sm

import seaborn as sns

from sklearn.metrics import mean_squared_error

from math import sqrt

from statsmodels.tsa.api import ExponentialSmoothing





# There are 2 products named X and Y. We will forecast only the sales of product X.

df = pd.read_excel("../input/alcoholic-beverage-sales-in-unidentified-regions/XY_2013_2017.xlsx",parse_dates=True, index_col=1)

Total = df.iloc[:,1:2]

Total.drop(["Yıl-Ay"],inplace=True)

Total= Total.astype(float)

Total.columns=["Liters"]

Total.index=pd.to_datetime(Total.index)

Train=Total.iloc[:48,:]

Test=Total.iloc[48:,]

Train.tail()


from pandas import read_csv

from matplotlib import pyplot

from statsmodels.graphics.tsaplots import plot_acf



plot_acf(Total, lags=50)

pyplot.show()
Total.plot(figsize=(15,6))

plt.show()
from pylab import rcParams

rcParams['figure.figsize'] = 18, 8

decomposition = sm.tsa.seasonal_decompose(Total.iloc[:,0], model='multiplicative')

fig = decomposition.plot()

plt.show()
Total
Bc_Train,fitted_lambda=stats.boxcox(Total["Liters"])

print("Lambda Value is:  %f " %fitted_lambda)

Bc_Test = stats.boxcox(Test["Liters"], fitted_lambda)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.scatterplot(data=Bc_Train)
fig,ax =plt.subplots(1,3)

sns.distplot(Train, ax=ax[0]).set_title('Original Train Data')

sns.distplot(Bc_Train, ax=ax[1]).set_title('Transformed Train Data')

sns.distplot(Bc_Test, ax=ax[2]).set_title('Transformed Test Data')

plt.show()
# defining a root mean squared error method to compare the success of different methods

def rmse(actual,forecast):

    rms = sqrt(mean_squared_error(actual, forecast))

    return rms
warnings.filterwarnings("ignore")

fit1 = ExponentialSmoothing(Train["Liters"], seasonal_periods=12, trend='add', seasonal='add').fit(use_boxcox=True)

fit2 = ExponentialSmoothing(Train["Liters"], seasonal_periods=12, trend='add', seasonal='mul').fit(use_boxcox=True)

fit3 = ExponentialSmoothing(Train["Liters"], seasonal_periods=12, trend='mul', seasonal='mul').fit(use_boxcox=True)

fit4 = ExponentialSmoothing(Train["Liters"], seasonal_periods=12, trend='add', seasonal='mul', damped=True).fit(use_boxcox=True)



results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$\gamma$",r"$l_0$","$b_0$","SSE","RMSE"])

params = ['smoothing_level', 'smoothing_slope', 'damping_slope', 'smoothing_seasonal', 'initial_level', 'initial_slope']

results["Additive"]       = [fit1.params[p] for p in params] + [fit1.sse] + [rmse(Test["Liters"],fit1.forecast(12))]

results["Mult. Seasonal"] = [fit2.params[p] for p in params] + [fit2.sse] + [rmse(Test["Liters"],fit2.forecast(12))]

results["Multiplicative"]   = [fit3.params[p] for p in params] + [fit3.sse] + [rmse(Test["Liters"],fit3.forecast(12))]

results["Multiplica Dam"] = [fit4.params[p] for p in params] + [fit4.sse] + [rmse(Test["Liters"],fit4.forecast(12))]

                                                                

ax = Test["Liters"].plot(figsize=(15,8), marker='o', color='black', title="Forecasts from Holt-Winters' multiplicative method" )

ax.set_ylabel("Liters")

ax.set_xlabel("Month")

Train["Liters"].plot(ax=ax, color= "black")

fit1.fittedvalues.plot(ax=ax, style='--', color='red')

fit2.fittedvalues.plot(ax=ax, style='--', color='green')



fit1.forecast(12).rename('Holt-Winters (add-add-seasonal)').plot(ax=ax, style='--', marker='o', color='red', legend=True)

fit2.forecast(12).rename('Holt-Winters (add-mul-seasonal)').plot(ax=ax, style='--', marker='o', color='green', legend=True)



plt.show()

print("Figure 1: Forecasting Sales using Holt-Winters method with both additive and multiplicative seasonality.")



results                                                                
fit4.params
def find_best_sarima(train, eval_metric):

    

    p = d = q = range(0, 2)

    pdq = list(itertools.product(p, d, q))

    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]



    counter = 0

    myDict = {}

    

    for param in pdq:

        for param_seasonal in seasonal_pdq:

            try:

                counter += 1

                mod = sm.tsa.statespace.SARIMAX(train,

                                                order=param,

                                                seasonal_order=param_seasonal,

                                                enforce_stationarity=False,

                                                enforce_invertibility=False)



                results = mod.fit()

                myDict[counter] = [results.aic, results.bic, param, param_seasonal]



            except:

                continue

                

    dict_to_df = pd.DataFrame.from_dict(myDict,orient='index')

    

    if eval_metric == 'aic':

        best_run = dict_to_df[dict_to_df[0] == dict_to_df[0].min()].index.values

        best_run = best_run[0]

    elif eval_metric == 'bic':

        best_run = dict_to_df[dict_to_df[1] == dict_to_df[1].min()].index.values

        best_run = best_run[0]

            

    model = sm.tsa.statespace.SARIMAX(train,

                                      order=myDict[best_run][2],

                                      seasonal_order=myDict[best_run][3],

                                      enforce_stationarity=False,

                                      enforce_invertibility=False).fit()

    

    best_model = {'model':model, 

                  'aic':model.aic,

                  'bic':model.bic,

                  'order':myDict[best_run][2], 

                  'seasonal_order':myDict[best_run][3]}

    

    return best_model
best = find_best_sarima(Bc_Train, 'aic')

print("Best SARIMA Model for our data is: {}x{} with aic= {} and bic= {}".format(best["order"], best["seasonal_order"], best["aic"], best["bic"]))

best
mod = sm.tsa.statespace.SARIMAX(Bc_Train,

                                order=(1, 1, 1),

                                seasonal_order=(1, 1, 0, 12),

                                enforce_stationarity=False,

                                enforce_invertibility=False)

results = mod.fit()

results.plot_diagnostics(figsize=(16, 8))

plt.show()
pred = best['model'].predict(start=48, end=59, dynamic=True)

pred
# invert a boxcox transform for one value

def power_transform_invert_value(value, lam):

	from math import log

	from math import exp

	# log case

	if lam == 0:

		return exp(value)

	# all other cases

	return exp(log(lam * value + 1) / lam)
# Invert all the predicted values

o_pred=[]

for prediction in pred:

    o_pred.append(power_transform_invert_value(prediction,fitted_lambda))

pred=pd.DataFrame(o_pred,index=Test.index, columns=["Liters"])    

pred
ax = Test["Liters"].rename("Original Data").plot(figsize=(15,8), marker='o', color='black', title="Forecasts from (1, 1, 1)x(1, 1, 0, 12) SARIMA model",legend=True)

ax.set_ylabel("Liters")

ax.set_xlabel("Month")

Train["Liters"].plot(ax=ax, color= "black")

pred["Liters"].rename("SARIMA Forecast").plot(ax=ax, style='--', color='red', legend=True)











plt.show()



print('The Root Mean Squared Error of our forecasts is {}'.format(rmse(Test["Liters"],pred["Liters"])))