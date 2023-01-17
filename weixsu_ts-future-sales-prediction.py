# This block is from https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

import seaborn as sns #collection of functions for data visualization
print("seaborn version: {}". format(sns.__version__))

from sklearn.preprocessing import OneHotEncoder #OneHot Encoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
%matplotlib inline

#misc libraries
import random
import time
from pandas import datetime


#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
item_categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
sales_train_raw = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
sample_submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')
shops_raw = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
test_raw = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
sales_train_raw.info()
def parser(x):
    return datetime.strptime(x,'%d.%m.%Y')

sales_train_di = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv', index_col= 0, parse_dates=[0] ,date_parser=parser)
sales_train_raw.head()
sales_train_di.head()
fig = plt.figure(figsize=(5,5))
plt.subplot(1,2,1)
sns.boxplot(y='item_price', data=sales_train_di)
plt.subplot(1,2,2)
sns.boxplot(y='item_cnt_day', data=sales_train_di)
fig.tight_layout(pad=1.0)
sales_train_di.item_price = sales_train_di.item_price.apply(lambda x: 6000 if x > 10000 else x)
sales_train_di.item_cnt_day = sales_train_di.item_cnt_day.apply(lambda x: 700 if x > 700 else x)
fig = plt.figure(figsize=(5,5))
plt.subplot(1,2,1)
sns.boxplot(y='item_price', data=sales_train_di)
plt.subplot(1,2,2)
sns.boxplot(y='item_cnt_day', data=sales_train_di)
fig.tight_layout(pad=1.0)
#since we want to predict the sales through time, we are creating the feature we want to target
sales_train_di['sales'] = sales_train_di['item_price']*sales_train_di['item_cnt_day']
sales_train_di.index.value_counts()
sales = sales_train_di.drop(['date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day'], axis=1)
sales.head()
sns.boxplot(sales)
sales.sales = sales.sales.apply(lambda x: 800000 if x > 800000 else x)
sns.boxplot(sales)
#grouping the data by month
sales_gm = sales.resample("m").sum()
sales_gm
sales_gm.size
plt.figure(figsize=(14,6))
sns.lineplot(data=sales_gm.sales)
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(sales_gm.sales)
#split the data for trainning and validation purposes
train_index_m = int(np.rint(sales_gm.size*0.8))
sales_gm.size
train_index_m
X_train_m = sales_gm[:train_index_m]
X_train_m.size

X_train_m
X_test_m = sales_gm[train_index_m:]
X_test_m.size
X_test_m
#hyperparameter tuning for the ARIMA model
import itertools
from statsmodels.tsa.arima_model import ARIMA

def hyper_p (train):
    best_aic = np.inf 
    best_param = None
    best_model = None
    
    p=d=q=range(0,12)
    pdq = list(itertools.product(p,d,q))

    for param in pdq:
        try:
            arima = ARIMA(train,order=param)
            arima_fit = arima.fit()
            if arima_fit.aic < best_aic:
                best_aic = arima_fit.aic
                best_param = param
                best_model = arima_fit
        except:
            continue

    print('aic: {:6.5f} | pdq set: {}'.format(best_aic, best_param))
    return best_model
best_arima = hyper_p (X_train_m)
predictions_m= best_arima.forecast(steps=X_test_m.size)[0]
predictions_m
from sklearn.metrics import mean_squared_error
score = mean_squared_error(X_test_m, predictions_m)
score
p_df = pd.DataFrame({'sale': predictions_m}, index = X_test_m.index)
p_df
plt.plot(X_test_m)
plt.plot(p_df,color='red')
train_rindex = X_train_m.copy()
train_rindex.reset_index(level=0, inplace=True)
train_rindex.columns = ['ds', 'y']
train_rindex
#borrowed from https://www.kaggle.com/jagangupta/time-series-basics-exploring-traditional-ts
from fbprophet import Prophet
#prophet reqiures a pandas df at the below config 
# ( date column named as DS and the value column as Y)
model = Prophet( yearly_seasonality=True) #instantiate Prophet with only yearly seasonality as our data is monthly 
model.fit(train_rindex) #fit the model with your dataframe
future = model.make_future_dataframe(periods = 7, freq = 'MS')  
# now lets make the forecasts
forecast = model.predict(future)

forecast
model.plot(forecast)
sales_gm.plot()
y_pred = forecast[['ds', 'yhat_lower']].tail(7)
y_pred.columns = ['date', 'sales']
y_pred=y_pred.set_index('date')
plt.plot(X_test_m)
plt.plot(y_pred,color='red')