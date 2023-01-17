# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import time
import resource
from os import walk
from datetime  import datetime,timedelta,date
import re
import gc
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans

import lightgbm as lgb
import xgboost as xgb
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Flatten, LSTM
from keras.optimizers import Adam, SGD
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from keras.utils.vis_utils import plot_model

import os
print(os.listdir("../input"))

def save_obj(obj, file_path):    
    file_save = open(file_path, 'wb')
    pickle.dump(obj, file_save, pickle.HIGHEST_PROTOCOL)
    file_save.close()
    
def load_obj(path):
    file_save = open(path, 'rb')
    return pickle.load(file_save)

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
for df in [df_train, df_test]:
    for i in range(8):
        df['mpaa_rating_'+str(i)] = df.mpaa_rating.apply(lambda x: 1 if x == i else 0)   

drop_feats = ['people', 'studio', 'genre', 'id', 'lastseen', 'mpaa_rating',\
    'name', 'release_date', 'revenue', 'month', 'season', 'movie_season', 'year']


df_train.shape, df_test.shape
# Shuffle the trainning set:
df_train = df_train.sample(frac=1)

# Scale budget feature:
print ('Train budget range before:', df_train.budget.min(), df_train.budget.max())
print ('Test budget range before:', df_test.budget.min(), df_test.budget.max())

df_train.budget = df_train.budget + .5
df_train.budget = np.log1p(df_train.budget.values)
df_test.budget = df_test.budget + .5
df_test.budget = np.log1p(df_test.budget.values)

print ('Train budget range after:', df_train.budget.min(), df_train.budget.max())
print ('Test budget range: after', df_test.budget.min(), df_test.budget.max())

# Add a special feature:
for df in [df_train, df_test]:
    tmp_feat1 = -0.6571437267608864*df['budget_available'] + 0.46238344874564663*df['budget']\
    + 0.33543944867081527*df['movie_season'] + 0.307830534240863*df['num_actor'] + 0.3506438787242728*df['num_director']\
    + 0.34082499414410156*df['num_writer']

    df['s_feat1'] = tmp_feat1
from sklearn.decomposition import PCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans

# PCA & KMeans:
pca = PCA(n_components=3)
# lda = LDA(n_components=5)
kmean = KMeans(n_clusters=7)

pca_feats = pca.fit_transform(df_train.drop(drop_feats, axis=1).values)
kmean.fit(df_train.drop(drop_feats, axis=1).values)
kmean_label = kmean.labels_

# Concat them into a maxtrix:
morefeats = np.concatenate((kmean_label.reshape(kmean_label.shape[0],1), pca_feats), axis=1)

# Add 4 new features to df_train
for i in range(4):
    df_train['analysis_feat_%i'%i] = morefeats[:, i:i+1]
    
# Do the same in testset:
kmean_testset = kmean.transform(df_test.drop(drop_feats, axis=1).values)
pca_testset = pca.transform(df_test.drop(drop_feats, axis=1).values)
# Concat them into a maxtrix:
morefeats_test = np.concatenate((kmean_testset, pca_testset), axis=1)

# Add 4 new features to df_train
for i in range(4):
    df_test['analysis_feat_%i'%i] = morefeats_test[:, i:i+1]

# Apply CCA to 2 subsets: df['revenue'] and df[rest of columns] 
X = df_train.drop(drop_feats, axis=1).values
y = np.log1p(df_train.revenue.values)

# Add cca feature to trainset:
std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
cca = CCA()
cca.fit(X, y)
X_cca = cca.transform(X)
df_train['cca_feat'] = X_cca[:,0]

# Add cca feature to testset:
X_test_cca = cca.transform(df_test.drop(drop_feats, axis=1).values)
df_test['cca_feat'] = X_test_cca[:,0]
df_train.shape, df_test.shape
# Train:
X = df_train.drop(drop_feats, axis=1).values
y = np.log1p(df_train.revenue.values)
# Test:
X_test = df_test.drop(drop_feats, axis=1).values
y_test = np.log1p(df_test.revenue.values)

# Scale trainning data:
std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
X_test = std_scaler.transform(X_test)

# Split:
X_train = X[:6000]
X_val = X[6000:]
y_train = y[:6000]
y_val = y[6000:]

X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape
print('Revenue range: \t\t', (df_train.revenue.min(), df_train.revenue.max()))
print ('Log1p(revenue) range:   ', (y.min(), y.max()))
# # lr = [.0146, .0147, .0148, .0149]
# min_childs = [1,5,10,20,50]
# max_depth = [3, 4, 5, 7, 8, 10, 12]

# losses = []
# for depth in max_depth:
#     model_lgb = xgb.XGBRegressor(max_depth=depth, num_leaves=2**depth // 2 + 1, learning_rate=.0147,\
#                               n_estimators=5000, n_jobs=32)
#     model_lgb.fit(X_train, y_train, early_stopping_rounds=300, eval_set=[[X_val, y_val]], verbose=100)
#     losses.append(model_lgb.best_score)
# # pred_lgb = model_lgb.predict(X)

# losses
def NN_model():
    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(24, activation='relu'))
    model.add(Dropout(.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    adam = Adam(lr=.01, decay=.01)
    model.compile(optimizer=adam, loss='mse', metrics=['mse'])
    
    return model
# Train:
model = NN_model()
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=4)
# Predict:
pred = model.predict(X)
pred_test = model.predict(X_test)

pred.shape
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.summary()
from sklearn.linear_model import LinearRegression as Lrg
from sklearn.svm import SVR

# Train Linear Regression:
model_lrg = Lrg()
model_lrg.fit(X_train, y_train)
# Train SVR:
model_svr = SVR()
model_svr.fit(X_train, y_train)

# Predict:
pred_lgr = model_lrg.predict(X)
pred_lgr_test = model_lrg.predict(X_test)
pred_svr = model_svr.predict(X)
pred_svr_test = model_svr.predict(X_test)
from lightgbm import LGBMRegressor

# Turnned params:
depth = 4
lr = .0147

# Train LGBM:
model_lgb = LGBMRegressor(max_depth=depth, num_leaves=2**depth // 2 + 1, learning_rate=lr,\
                              n_estimators=5000, n_jobs=32)
model_lgb.fit(X_train, y_train, early_stopping_rounds=300, eval_set=[[X_val, y_val]], eval_metric='mae', verbose=100)

# Predict:
pred_lgb = model_lgb.predict(X)
pred_lgb_test = model_lgb.predict(X_test)

# Train XGBOOST:
model_xgb = xgb.XGBRegressor(max_depth=depth, num_leaves=2**depth // 2 + 1, learning_rate=lr,\
                              n_estimators=5000, n_jobs=32)
model_xgb.fit(X_train, y_train, early_stopping_rounds=300, eval_set=[[X_val, y_val]], verbose=100, eval_metric='mae')
              
# Predict:
pred_xgb = model_xgb.predict(X)
pred_xgb_test = model_xgb.predict(X_test)
df_pred1 = pd.DataFrame([y, pred.ravel(), pred_lgr,\
                        pred_svr, pred_lgb, pred_xgb]).T
df_pred1_test = pd.DataFrame([y_test, pred_test.ravel(), pred_lgr_test,\
                        pred_svr_test, pred_lgb_test, pred_xgb_test]).T

df_pred1.columns = ['actual', 'NN', 'Linear', 'SVR', 'LGBM', 'XGB']
df_pred1_test.columns = ['actual', 'NN', 'Linear', 'SVR', 'LGBM', 'XGB']

# Add prediction values to trainset and testset as new features:
# Trainset:
df_pred1['id'] = df_train['id']
df_train = pd.merge(df_train, df_pred1.drop(['actual'], axis=1), on='id', how='left')
# Testset:
df_pred1_test['id'] = df_test['id']
df_test = pd.merge(df_test, df_pred1_test.drop(['actual'], axis=1), on='id', how='left')

# Get prediction of valset:
df_pred11 = df_pred1.iloc[6000:]
print('Actual value and prediction of trainned models:')
df_pred11.head(3)
print ('Neuralnet RMSE:', np.sqrt(((df_pred11.NN - df_pred11.actual)**2).mean()))
print ('Linear regression RMSE:', np.sqrt(((df_pred11.Linear - df_pred11.actual)**2).mean()))
print ('SVR RMSE:', np.sqrt(((df_pred11.SVR - df_pred11.actual)**2).mean()))
print ('LGBM RMSE:', np.sqrt(((df_pred11.LGBM - df_pred11.actual)**2).mean()))
print ('XGBoost RMSE:', np.sqrt(((df_pred11.XGB - df_pred11.actual)**2).mean()))
print ('Neuralnet MAE:', (abs(df_pred11.NN - df_pred11.actual)).mean())
print ('Linear regression MAE:', np.sqrt((abs(df_pred11.Linear - df_pred11.actual)).mean()))
print ('SVR MAE:', np.sqrt((abs(df_pred11.SVR - df_pred11.actual)).mean()))
print ('LGBM MAE:', np.sqrt((abs(df_pred11.LGBM - df_pred11.actual)).mean()))
print ('XGBoost MAE:', np.sqrt((abs(df_pred11.XGB - df_pred11.actual)).mean()))
# df_train = df_train.sample(frac=1)
X = df_train.drop(drop_feats, axis=1).values
y = np.log1p(df_train.revenue.values)

std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)

X_train = X[:6000]
X_val = X[6000:]
y_train = y[:6000]
y_val = y[6000:]
X_train.shape, X_val.shape, y_train.shape, y_val.shape
model_2 = xgb.XGBRegressor(max_depth=depth, num_leaves=2**depth // 2 + 1, learning_rate=lr,\
                          n_estimators=5000, n_jobs=32)
model_2.fit(X_train, y_train, early_stopping_rounds=300, eval_set=[[X_val, y_val]], verbose=100, eval_metric='mae')
pred_2 = model_2.predict(X_val)
df_pred = pd.DataFrame([y_val, pred_2]).T
df_pred.columns = ['actual', 'pred']
print('XGBoost v2 MAE:', (abs(df_pred.pred - df_pred.actual)).mean())
print('XGBoost v2 RMSE:', np.sqrt(((df_pred.pred - df_pred.actual)**2).mean()))
lgr = Lrg()
lgr.fit(df_pred1.iloc[:6000, :].drop(['id', 'actual'], axis=1).values, df_pred1.iloc[:6000, :]['actual'].values)
ensemble_pred = lgr.predict(df_pred1.iloc[6000:, :].drop(['id', 'actual'], axis=1).values)
print('Ensemble model MAE:', (abs(ensemble_pred - df_pred1.iloc[6000:, :]['actual'].values)).mean())
print('Ensemble model RMSE:', np.sqrt(((ensemble_pred - df_pred1.iloc[6000:, :]['actual'].values)**2).mean()))
X_test = df_test.drop(drop_feats, axis=1).values
y_test = np.log1p(df_test.revenue.values)

std_scaler = StandardScaler()
X_test = std_scaler.fit_transform(X_test)

final_predict = model_2.predict(X_test)
print('Ensemble model MAE in test set:', (abs(final_predict - y_test)).mean())
