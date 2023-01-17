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

# Any results you write to the current directory are saved as output.
def save_obj(obj, file_path):    
    file_save = open(file_path, 'wb')
    pickle.dump(obj, file_save, pickle.HIGHEST_PROTOCOL)
    file_save.close()
    
def load_obj(path):
    file_save = open(path, 'rb')
    return pickle.load(file_save)
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
for df in [df_train, df_test]:
    for i in range(8):
        df['mpaa_rating_'+str(i)] = df.mpaa_rating.apply(lambda x: 1 if x == i else 0)   

drop_feats = ['people', 'studio', 'genre', 'id', 'lastseen', 'mpaa_rating',\
    'name', 'release_date', 'revenue', 'month', 'season', 'movie_season', 'year']


df_train.shape, df_test.shape
df_train.head()
print ('Train budget range before:', df_train.budget.min(), df_train.budget.max())
print ('Test budget range before:', df_test.budget.min(), df_test.budget.max())

df_train.budget = df_train.budget + .5
df_train.budget = np.log1p(df_train.budget.values)
df_test.budget = df_test.budget + .5
df_test.budget = np.log1p(df_test.budget.values)

print ('Train budget range after:', df_train.budget.min(), df_train.budget.max())
print ('Test budget range: after', df_test.budget.min(), df_test.budget.max())
df_train = df_train.iloc[:,:272]
df_test = df_test.iloc[:,:272]

for df in [df_train, df_test]:
    df['year'] = df.release_date.apply(lambda x: int(x[:4]))
    for feat in ['month', 'season', 'mpaa_rating', 'genre']:
        tmp_df_feat = df.groupby(feat).agg({'id':'count'}).sort_values('id', ascending=False).reset_index().reset_index()
        tmp_df_feat.columns = [feat+'_id', feat, 'freq']
        df = pd.merge(df, tmp_df_feat[[feat+'_id', feat]], on=feat, how='left')
        
df_train.shape, df_test.shape
df_train.head()
# df.columns[:272]
# df_train = df_train.sample(frac=1)
# from sklearn.decomposition import PCA
# # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from sklearn.cluster import KMeans

# pca = PCA(n_components=5)
# # lda = LDA(n_components=5)
# kmean = KMeans(n_clusters=7)

# pca_feats = pca.fit_transform(df_train.drop(drop_feats, axis=1).values)
# kmean.fit(df_train.drop(drop_feats, axis=1).values)
# kmean_label = kmean.labels_
# morefeats = np.concatenate((kmean_label.reshape(kmean_label.shape[0],1), pca_feats), axis=1)
# for i in range(6):
#     df_train['analysis_feat_%i'%i] = morefeats[:, i:i+1]
    
# df_train.head()

# pca = PCA(n_components=3)
# # lda = LDA(n_components=5)
# kmean = KMeans(n_clusters=7)

# pca_feats = pca.fit_transform(df_train.drop(drop_feats, axis=1).values)
# kmean.fit(df_train.drop(drop_feats, axis=1).values)
# kmean_label = kmean.labels_
# morefeats = np.concatenate((kmean_label.reshape(kmean_label.shape[0],1), pca_feats), axis=1)
# for i in range(4):
#     df_train['analysis_feat_%i'%i] = morefeats[:, i:i+1]
X = df_train.drop(drop_feats, axis=1).values
y = np.log1p(df_train.revenue.values)

std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)

X_train = X[:6000]
X_val = X[6000:]
y_train = y[:6000]
y_val = y[6000:]
X_train.shape, X_val.shape, y_train.shape, y_val.shape


print('Revenue range: \t\t', (df_train.revenue.min(), df_train.revenue.max()))
print ('Log1p(revenue) range:   ', (y.min(), y.max()))

def baseline_model():
    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=260))
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
model = baseline_model()
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=48)
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
plt.show()
pred = model.predict(X)
pred.shape
from sklearn.linear_model import LinearRegression as Lrg
from sklearn.svm import SVR

model_lrg = Lrg()
model_lrg.fit(X_train, y_train)

model_svr = SVR()
model_svr.fit(X_train, y_train)


pred_lgr = model_lrg.predict(X)
pred_svr = model_svr.predict(X)
from lightgbm import LGBMRegressor
model_lgb = LGBMRegressor(max_depth=8, num_leaves=129, learning_rate=.01, min_child_samples=20,\
                          n_estimators=2000, n_jobs=32)
model_lgb.fit(X_train, y_train, early_stopping_rounds=300, eval_set=[[X_val, y_val]], verbose=100)
pred_lgb = model_lgb.predict(X)

# XGBOOST:
model_xgb = xgb.XGBRegressor(max_depth=8, num_leaves=129, learning_rate=.01, n_estimators=2000, n_jobs=32)
model_xgb.fit(X_train, y_train, early_stopping_rounds=300, eval_set=[[X_val, y_val]], verbose=100)
              

pred_xgb = model_xgb.predict(X)
df_pred1 = pd.DataFrame([y, pred.ravel(), pred_lgr,\
                        pred_svr, pred_lgb, pred_xgb]).T
df_pred1.columns = ['actual', 'NN', 'Linear', 'SVR', 'LGBM', 'XGB']

df_pred1['id'] = df_train['id']
df_train = pd.merge(df_train, df_pred1.drop(['actual'], axis=1), on='id', how='left')

df_pred1 = df_pred1.iloc[6000:]
df_pred1.head(3)
print ('Neuralnet RMSE:', np.sqrt(((df_pred1.NN - df_pred1.actual)**2).mean()))
print ('Linear regression RMSE:', np.sqrt(((df_pred1.Linear - df_pred1.actual)**2).mean()))
print ('SVR RMSE:', np.sqrt(((df_pred1.SVR - df_pred1.actual)**2).mean()))
print ('LGBM RMSE:', np.sqrt(((df_pred1.LGBM - df_pred1.actual)**2).mean()))
print ('XGBoost RMSE:', np.sqrt(((df_pred1.XGB - df_pred1.actual)**2).mean()))
try:
    print ('Neuralnet MAE:', np.sqrt((abs(df_pred1.NN - df_pred1.actual)).mean()))
    print ('Linear regression MAE:', np.sqrt((abs(df_pred1.Linear - df_pred1.actual)).mean()))
    print ('SVR MAE:', np.sqrt((abs(df_pred1.SVR - df_pred1.actual)).mean()))
    print ('LGBM MAE:', np.sqrt((abs(df_pred1.LGBM - df_pred1.actual)).mean()))
    print ('XGBoost MAE:', np.sqrt((abs(df_pred1.XGB - df_pred1.actual)).mean()))
except:
    print ('Something failed!')
# from sklearn.decomposition import PCA
# # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from sklearn.cluster import KMeans

# pca = PCA(n_components=5)
# # lda = LDA(n_components=5)
# kmean = KMeans(n_clusters=7)

# pca_feats = pca.fit_transform(df_train.drop(drop_feats, axis=1).values)
# kmean.fit(df_train.drop(drop_feats, axis=1).values)
# kmean_label = kmean.labels_
# morefeats = np.concatenate((kmean_label.reshape(kmean_label.shape[0],1), pca_feats), axis=1)
# for i in range(6):
#     df_train['analysis_feat_%i'%i] = morefeats[:, i:i+1]
    
# df_train.head()
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
model_2 = LGBMRegressor(max_depth=8, num_leaves=129, learning_rate=.01, min_child_samples=20,\
                          n_estimators=2000, n_jobs=32)
model_2.fit(X_train, y_train, early_stopping_rounds=300, eval_set=[[X_val, y_val]], verbose=100)
pred_2 = model_2.predict(X_val)
df_pred = pd.DataFrame([y_val, pred_2]).T
df_pred.columns = ['actual', 'pred']
print('LGBM v2 mean abs error:', np.sqrt((abs(df_pred.pred - df_pred.actual)).mean()))
print('LGBM v2 RMSE:', np.sqrt(((df_pred.pred - df_pred.actual)**2).mean()))

df_pred.head(20)


# seed = 7
# np.random.seed(seed)
# estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
# kfold = KFold(n_splits=5, random_state=seed)
# results = cross_val_score(estimator, X, y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


