import pandas as pd 
import datetime
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from pylab import rcParams
import warnings
from itertools import chain
from pandas.core.nanops import nanmean as pd_nanmean

from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from math import sqrt

%matplotlib inline
df = pd.read_csv(r'../input/sputnik/train.csv', sep =',')
df.head(5)
t = [len(df[df['sat_id'] ==i ]) for i in range(600)]
plt.plot(range(600), t)
plt.plot(df[df['sat_id'] == 0]['x'][:30])
plt.show()
plt.plot(df[df['sat_id'] == 0]['x'][:5])
plt.show()
plt.plot(df[df['sat_id'] == 0]['x'][24:28])
plt.show()
plt.plot(df[df['sat_id'] == 0]['x'] - df[df['sat_id'] == 0]['x_sim'])
plt.show()
temp = df
temp['error'] = [0] * len(temp['id'])
temp['error']  = np.array(np.linalg.norm(temp[['x', 'y', 'z']].values - temp[['x_sim', 'y_sim', 'z_sim']].values, axis=1))
temp.head(5)
dt = temp['epoch'] 
df['error']  = np.array(np.linalg.norm(df[['x', 'y', 'z']].values - df[['x_sim', 'y_sim', 'z_sim']].values, axis=1))

temp2 = df.copy()
temp2.sort_values('epoch', axis=0, inplace=True)
temp2['epoch'] = pd.to_datetime(temp2.epoch,format='%Y-%m-%d %H:%M:%S') 
temp2.index = temp2.epoch
temp2 = temp2.drop(columns = 'epoch')
res = []
train_err = temp[temp.type == 'train']
test_err = temp[temp.type == 'test']
train_list = []
test_list = []
for sat in range(temp['sat_id'].max() + 1):
    train = train_err[train_err.sat_id == sat]
    
    train = train.iloc[-24:]

    train_list.append(train)
    
    test = test_err[test_err.sat_id == sat]
    
    test_list.append(test)
    
    model = ExponentialSmoothing(np.asarray(train.error) ,seasonal_periods=24 , seasonal='add').fit()
    forecast = pd.DataFrame(model.forecast(len(test)),index = test.index)
    res.append(forecast.values)
res = list(chain.from_iterable(res))
result = pd.DataFrame()
df1 = df[df.type == 'test']
result['id'] = np.array(test_err.id.sort_values().astype(int))
result['error']  = list(map(lambda x: x[0],res))
result.head(5)
with open('my_sub.csv', mode='w', encoding='utf-8') as f_csv:
    result.to_csv(f_csv, index=False)