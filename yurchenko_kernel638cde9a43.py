import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt 

import statsmodels.api as sm

from pylab import rcParams

import warnings

from tqdm import tqdm

import lightgbm as lgb

from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/sputnik/train.csv')
df['rad'] = np.sqrt(df.x **2 + df.y **2 + df.z**2)

df['rad_sim'] = np.sqrt(df.x_sim ** 2 + df.y_sim ** 2 + df.z_sim ** 2)



df['phi'] = np.arctan(df.y / df.x)

df['phi_sim'] = np.arctan(df.y_sim / df.x_sim)



df['theta'] = np.arccos(df.z / df.rad)

df['theta_sim'] = np.arccos(df.z_sim / df.rad_sim)



df['epoch'] = pd.to_datetime(df.epoch,format='%Y-%m-%d %H:%M:%S') 

df = df.rename({'epoch': 'ds'}, axis=1)

df['error']  = np.linalg.norm(df[['x', 'y', 'z']].values - df[['x_sim', 'y_sim', 'z_sim']].values, axis=1)

df_train = df[df.type == 'train']

df_test = df[df.type == 'test']

df_train = df_train.drop(['type'], axis=1)

df_test = df_test.drop(['type'], axis=1)
error_predicts = []

for i in tqdm(range(600)):

    tmp = df[df.sat_id == i].drop(['id', 'sat_id', 'ds', 'x', 'y', 'z', 'rad', 'phi', 'theta'], axis=1)

    lag_period = df_test[df_test.sat_id == i].shape[0]

    for add in range(14):

        tmp["lag_period_{}".format(add)] = tmp.error.shift(add+lag_period)

    tmp.expanding_mean_error = tmp.error.expanding(3).mean()

    features = list(tmp.drop(['error'], axis=1).columns)

    df_train = tmp[tmp.type == 'train'].drop(['type'], axis=1).dropna()

    d_train = lgb.Dataset(df_train.drop(['error'], axis=1), df_train['error'])

    params = {'num_leaves': 31,

             'learning_rate': 0.7,

             'num_iterations': 300}

    clf = lgb.train(params, d_train)

    test = tmp[tmp.type == 'test'].drop(['error', 'type'], axis=1)

    pred = clf.predict(test)

    error_predicts.extend(pred)
sub = pd.read_csv('/kaggle/input/sputnik/sub.csv')

sub.error = abs(np.array(error_predicts))

sub.to_csv('submission.csv', index=False)