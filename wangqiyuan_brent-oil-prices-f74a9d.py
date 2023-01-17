import pandas as pd

df = pd.read_csv("/kaggle/input/brent-oil-prices/BrentOilPrices.csv")

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
day = []

month = []

year = []

month_mapped = []

index = []
#Map text date to number

def monthToNum(shortMonth):

    return{

            'Jan' : 1,

            'Feb' : 2,

            'Mar' : 3,

            'Apr' : 4,

            'May' : 5,

            'Jun' : 6,

            'Jul' : 7,

            'Aug' : 8,

            'Sep' : 9, 

            'Oct' : 10,

            'Nov' : 11,

            'Dec' : 12

    }[shortMonth]
#Seperate date to month, date, year so we can work with month

for i in df['Date']:

    month.append(i[:3])

    day.append(i[3:6])

    year.append(i[8:12])
#Mapping month to number

for i in month:

    month_mapped.append(monthToNum(i))

#Taking difference of dates

from datetime import date

for i in range(0, len(day)):

    d0 = date(int(year[0]), int(month_mapped[0]), int(day[0]))

    d1 = date(int(year[i]), int(month_mapped[i]), int(day[i]))

    delta = d1 - d0

    index.append(delta.days + 1)
d = {'Index' : index, 

      'Price' : df['Price']} 

features = 'Index'

target = 'Price'

new_df = pd.DataFrame(d)
df_train = new_df[0:int((len(new_df)*0.98))]

df_test = new_df[int((len(new_df)*0.98)):len(new_df)]

df_train
plt.figure(figsize=(16,8))

plt.plot(df_train[features], df_train[target], 'magenta')
#Data is non-linear, we use polynomial regression

import operator



import numpy as np



from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import PolynomialFeatures



np.random.seed(0)



# transforming the data to include another axis

x = df_train[features]

y = df_train[target]

x = x[:, np.newaxis]

y = y[:, np.newaxis]



polynomial_features= PolynomialFeatures(degree=14)

x_poly = polynomial_features.fit_transform(x)



model = LinearRegression()

model.fit(x_poly, y)

y_poly_pred = model.predict(x_poly)



rmse = np.sqrt(mean_squared_error(y,y_poly_pred))

r2 = r2_score(y,y_poly_pred)

print("RMSE for polynomial regression is", rmse)

print(r2)



fig, ax = plt.subplots(figsize = (16,8))

ax.plot(x, y, '-b', label = 'Data')



# sort the values of x before line plot

sort_axis = operator.itemgetter(0)

sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)

x, y_poly_pred = zip(*sorted_zip)

ax.plot(x, y_poly_pred, '-m', label = 'Fit')

ax.legend()

plt.show()

#Using exponential smoothing

from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt

fit = SimpleExpSmoothing(df_train[target]).fit(optimized = True, use_brute = True)

plt.figure(figsize=(16,8))

plt.plot(df_train[features], fit.fittedvalues, 'blue', label = r'$\alpha=%s$'%fit.model.params['smoothing_level'])

plt.plot(x, y_poly_pred, '-m', label = 'Fit')

plt.legend()
N = len(df_test['Index'])

fcast = fit.forecast(N)

plt.figure(figsize=(16,8))

plt.plot(df_test['Index'], fcast, 'green', label = r'Forecast, $\alpha=%s$'%fit.model.params['smoothing_level'])

plt.plot(df_test[features], df_test[target], 'black', label = 'Data')

plt.legend()

rmse = np.sqrt(mean_squared_error(df_test[target],fcast))

print("RMSE for SES is", rmse)
fit1 = ExponentialSmoothing(df_train[target], seasonal_periods = 8, trend = 'mul', seasonal = 'None').fit()

fcast1 = fit1.forecast(N)

plt.figure(figsize=(16,8))

plt.plot(df_test[features], df_test[target], 'black', label = 'Data')

plt.plot(df_test[features], fcast1, 'green', label = r'Forecast, $\alpha=%s$'%fit1.model.params['smoothing_level'])

rmse = np.sqrt(mean_squared_error(df_test[target],fcast1))

print("RMSE for Holt is", rmse)
from statsmodels.tsa.statespace.sarimax import SARIMAX as sarimax

fit2 = sarimax(df_train[target], order = (2,1,6), seasonal_order = (0,1,1,8)).fit()

fcast2 = fit2.forecast(N)

plt.figure(figsize=(16,8))

plt.plot(df_test[features], df_test[target], 'black', label = 'Data')

plt.plot(df_test[features], fcast2, 'green', label = 'Sarimax')

rmse = np.sqrt(mean_squared_error(df_test[target],fcast2))

print("RMSE for ARIMA Is", rmse)