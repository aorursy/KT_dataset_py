# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import datetime

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

from pandas import datetime



# Any results you write to the current directory are saved as output.
def parser(x):

    return datetime.strptime(x,'%Y-%m')

sales = pd.read_csv('../input/sales-cars.csv', date_parser = parser ,index_col = 0, parse_dates = [0])

sales.head()





sales.dtypes
sales.plot()
from statsmodels.graphics.tsaplots import plot_acf

#we have created a an auto correltion 

sales_acf_plot = plot_acf(sales)
sales.head()
sales.shift(1)
sales.head()
#sales.diff gives us the differece in sales compared to the month before. 

#we do this to make the data stationary, for time series forecasting its important that the data is made stationary. 

sales_diff = sales.diff(periods = 1)

#we dropped the first row as it was showing NaN 

sales_diff = sales_diff[1:]

sales_diff.head(10)
#we have created a plot over the sales diff column here to check autocorrelation. 

sales_difference = plot_acf(sales_diff)

sales_diff.plot()
#Divide the data into train and test data. 

X = sales.values

# lets figure out the size of the x 

X.size # = 36 = 3 years sales data 12*3 = 36



train = X[0:27] # 27 as test data

test = X[26:] #9 as train data

predictions = [] 
from statsmodels.tsa.ar_model import AR

from sklearn.metrics import mean_squared_error



model_ar = AR(train)

model_ar_fit = model_ar.fit()

predictions = model_ar_fit.predict(start=27, end=36)



plt.plot(test)

plt.plot(predictions, color = 'red')
mean_squared_error(test, predictions)
from statsmodels.tsa.arima_model import ARIMA
#specify the parameters for arima

#the three parameters are p,d,q 

# p is periods taken for autoregressive model

# d integrated order, difference in number months

# q the period in moving average 

model_arima = ARIMA(train, order = (9, 1, 0))

model_arima_fit = model_arima.fit()

print(model_arima_fit.aic)
predictions = model_arima_fit.forecast(steps = 10)[0]

predictions
plt.plot(test)

plt.plot(predictions, color = 'red')
import warnings 

warnings.filterwarnings('ignore')

#here we will import iteration package.

import itertools

#to check the aic for various values we will create a range from 0-5

p = d = q = range(0,5)



pdq = list(itertools.product(p, d, q))

pdq

import warnings 

warnings.filterwarnings('ignore')

for param in pdq:

    try:

        model_arima = ARIMA(train, order = param)

        model_arima_fit = model_arima.fit()

        final_value = print(param, model_arima_fit.aic)

    except:

        continue

    