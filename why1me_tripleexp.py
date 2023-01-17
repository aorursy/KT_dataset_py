import pandas as pd 

import datetime

import numpy as np 

import matplotlib.pyplot as plt 

import statsmodels.api as sm



from sklearn.model_selection import TimeSeriesSplit



from datetime import datetime



from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt



from tqdm.notebook import tqdm

from pylab import rcParams
def smape(A, F):

    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
from sklearn.metrics import mean_squared_error

from math import sqrt

    

def rmse(y_actual, y_predicted):

    rmse = sqrt(mean_squared_error(y_actual, y_predicted))

    return rmse
df = pd.read_csv('/kaggle/input/sputnik/train.csv')

df.rename(columns={'epoch':'Datetime'}, inplace=True)

df['Datetime'] = pd.to_datetime(df.Datetime)

df.index  = df.Datetime

df.drop('Datetime', axis = 1, inplace = True)

df.head(2)
df['error']  = np.linalg.norm(df[['x', 'y', 'z']].values - df[['x_sim', 'y_sim', 'z_sim']].values, axis=1)

df.head(2)
# from sklearn.model_selection import TimeSeriesSplit #здесь смотрел параметры для модели 

# df_sat = df[(df.sat_id == 1)&(df.type == 'train')].copy()

# df_sat.drop(['id','sat_id','type'], axis = 1, inplace = True)

# df_sat_x = df_sat['x'].copy()

# df_sat_y = df_sat['y'].copy()

# df_sat_z = df_sat['z'].copy()



# priznak = pd.DataFrame()

# for i in range(2):

#     best_error = float('inf')

#     df_sat = df[(df.sat_id == i)&(df.type == 'train')].copy()

#     df_sat.drop(['id','sat_id','type'], axis = 1, inplace = True)

#     df_sat_x = df_sat['x'].copy()

#     df_sat_y = df_sat['y'].copy()

#     df_sat_z = df_sat['z'].copy()

#     b_al = 0

#     b_be = 0

#     b_ga = 0

#     b_se = 0

#     for se in range(20,31):

#         for al in np.linspace(0.1,0.9,10):

#             for be in np.linspace(0.1,0.9,10):

#                 for ga in np.linspace(0.1,0.9,10):

#                     error = 0

#                     tscv = TimeSeriesSplit(n_splits=4) 

#                     for train_idx, test_idx in tscv.split(df_sat_z):

#                         model = ExponentialSmoothing(np.asarray(df_sat_z.iloc[train_idx]), 

#                                                      trend=None, seasonal='add', 

#                                                      seasonal_periods=se).fit(smoothing_level = al, 

#                                                                               smoothing_slope = be, 

#                                                                               smoothing_seasonal = ga, 

#                                                                               optimized=False)

#                         forecast = pd.Series(model.forecast(len(test_idx)))

#                         actual = df_sat_z.iloc[test_idx]

#                         error += rmse(actual.values, forecast.values)



#                     if error/4 < best_error:

#                         best_error = error/4

#                         b_al = al

#                         b_be = be

#                         b_ga = ga

#                         b_se = se

#     priznak.at[i, 'al'] = b_al

#     priznak.at[i, 'be'] = b_be

#     priznak.at[i, 'ga'] = b_ga

#     priznak.at[i, 'se'] = b_se

# print(priznak)

# print(best_error)

# rcParams['figure.figsize'] = 12, 7

# plt.plot(actual.values, label='Train')

# plt.plot(forecast.values, label='Test')

# plt.legend()
def naive_pred(df, sat_num):

    df_test = df[(df.sat_id == sat_num)&(df.type == 'test')].copy()

#     df_test.drop(['sat_id', 'type'], axis = 1, inplace = True)



    df_train = df[(df.sat_id == sat_num)&(df.type == 'train')].copy()

    df_train.drop(['sat_id', 'type', 'x_sim', 'y_sim', 'z_sim', 'id', 'error'], axis = 1, inplace = True)



    model = ExponentialSmoothing(np.asarray(df_train.x), 

                                     trend=None, seasonal='add', 

                                     seasonal_periods=24).fit(smoothing_level = 0.1, smoothing_slope = 0.1, 

                                                              smoothing_seasonal = 1, optimized=False)

    forecastx = pd.Series(model.forecast(len(df_test)))

    forecastx.index = df_test.index

    df_test['x'] = forecastx



    model = ExponentialSmoothing(np.asarray(df_train.y), 

                                    trend=None, seasonal='add', 

                                    seasonal_periods=24).fit(smoothing_level = 0.1, smoothing_slope = 0.1, 

                                                              smoothing_seasonal = 1, optimized=False)

    forecasty = pd.Series(model.forecast(len(df_test)))

    forecasty.index = df_test.index

    df_test['y'] = forecasty



    model = ExponentialSmoothing(np.asarray(df_train.z), 

                                     trend=None, seasonal='add', 

                                     seasonal_periods=24).fit(smoothing_level = 0.1, smoothing_slope = 0.1, 

                                                              smoothing_seasonal = 1, optimized=False)

    forecastz = pd.Series(model.forecast(len(df_test)))

    forecastz.index = df_test.index

    df_test['z'] = forecastz

    df_test['error']  = np.linalg.norm(df_test[['x', 'y', 'z']].values - 

                                       df_test[['x_sim', 'y_sim', 'z_sim']].values, axis=1)

    return df_test[['id','error']]
df_res = naive_pred(df, 0)
for i in tqdm(range(1,600)):

    df_res1 = naive_pred(df, i)

    df_res = pd.concat([df_res, df_res1])
df_res.shape
df_res.head(5)
df_res.to_csv('mysub_naiv.csv', index=False)