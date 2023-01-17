%matplotlib inline

import matplotlib.pyplot as plt



import numpy as np 

import pandas as pd 

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder



from sklearn.neighbors import KNeighborsRegressor

import warnings

warnings.filterwarnings('ignore')
# To be sure that R2 is calculated properly

def r2(y,f):

    SS_res = ((f - y)**2).sum()

    SS_tot = ((y - y.mean())**2).sum()

    R2 = 1 - SS_res / SS_tot

    return SS_res, SS_tot, R2
# Loading data

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
# features with only one values

zero_features =['X11','X93','X107','X233','X235','X268','X289','X290','X293','X297','X330','X347']
# preparing data

y_train = df_train[df_train.y < 250]['y']

x_train = df_train[df_train.y < 250].drop(['ID','y'] + zero_features,axis =1)

x_test = df_test.drop(['ID'] + zero_features,axis = 1)



print(x_train.shape, y_train.shape, x_test.shape)
# dealing with categorical variables

num_train = len(x_train)

x_all = pd.concat([x_train, x_test])



for c in x_all.columns:

    if x_all[c].dtype == 'object':

        lbl = LabelEncoder()

        lbl.fit(list(x_all[c].values))

        x_all[c] = lbl.transform(list(x_all[c].values))



x_train = x_all[:num_train]

x_test = x_all[num_train:]



print(x_train.shape, y_train.shape, x_test.shape)
# preparing xgboost. Parameters are defined by cross validation (xgb.cv)



xgb_params = {

    'eta': 0.01,

    'max_depth': 2,

    'subsample': 0.8,

    'colsample_bytree': 1.0,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



dtrain = xgb.DMatrix(x_train, y_train)

dtest = xgb.DMatrix(x_test)
# run xgboost

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round = 672)
# Look on data. Seems all is OK

y_predict = model.predict(dtest)

sub = pd.DataFrame({'id': df_test['ID'], 'y': y_predict})

sub.head()
# But check them on train set

z_train = df_train[df_train.y < 250].copy()

z_train['err'] = y_train - model.predict(dtrain)

z_train['y'] = y_train

z_train['predict'] = model.predict(dtrain)

fig, axes = plt.subplots(ncols=2)

fig.set_size_inches(15, 5)

z_train.plot.scatter(x = 'y',y = 'predict', ax=axes[0], label = 'prediction')

z_train.plot.scatter(x = 'y',y = 'y', color = 'Red',ax=axes[0], label = 'actual')

z_train.plot.scatter(x = 'y',y = 'err', ax=axes[1], label = 'error')
SS_res, SS_tot, R2 = r2(y_train,model.predict(dtrain))

print('SS_res: %.0f  SS_tot: %.0f  R2: %0.5f' %(SS_res, SS_tot, R2))
knr = KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',

          metric_params=None, n_jobs=1, n_neighbors=19, p=2, weights='distance')

knr.fit(x_train,y_train)

y_pred = knr.predict(x_test)

sub = pd.DataFrame({'id': df_test['ID'], 'y': y_pred})

sub.head()
# Let's look on predicting on train set

z_train = df_train[df_train.y < 250].copy()

z_train['err'] = y_train - knr.predict(x_train)

z_train['y'] = y_train

z_train['predict'] = knr.predict(x_train)

fig, axes = plt.subplots(ncols=2)

fig.set_size_inches(15, 5)

z_train.plot.scatter(x = 'y',y = 'predict', ax=axes[0], label = 'prediction')

z_train.plot.scatter(x = 'y',y = 'y', color = 'Red',ax=axes[0], label = 'actual')

z_train.plot.scatter(x = 'y',y = 'err', ax=axes[1], label = 'error')
SS_res, SS_tot, R2 = r2(y_train,knr.predict(x_train))

print('SS_res: %.0f  SS_tot: %.0f  R2: %0.5f' %(SS_res, SS_tot, R2))
sub.to_csv("knr.csv", index = False)