from sklearn.metrics import mean_squared_error, mean_squared_log_error,  r2_score



from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn import ensemble

from sklearn.experimental import enable_hist_gradient_boosting  # noqa

from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')
train.head()
test.head()
def limpa(train):

    train.LotFrontage.fillna(train.LotFrontage.mean(), inplace = True)

    train.Alley.fillna(0, inplace = True)

    train.Alley.replace('Grvl', 1, inplace = True)

    train.Alley.replace('Pave', 2, inplace = True)

    for index, var in enumerate(train.FireplaceQu.unique()):

        train.FireplaceQu.replace(var, index, inplace = True)



    train.PoolQC.fillna(0, inplace = True)

    for index, var in enumerate(train.PoolQC.unique()):

        train.PoolQC.replace(var, index, inplace = True)



    train.Fence.fillna(0, inplace = True)

    for index, var in enumerate(train.PoolQC.unique()):

        train.Fence.replace(var, index, inplace=  True)     
limpa(train)

limpa(test)
train = train.select_dtypes([np.number])

test  = test.select_dtypes([np.number])
Treino_miss = """

    LotFrontage

    MasVnrArea

    GarageYrBlt

"""

for var in Treino_miss.split():

    train[var].fillna(train[var].mean(), inplace = True)



Teste_miss = """

    LotFrontage

    MasVnrArea

    BsmtFinSF1

    BsmtFinSF2

    BsmtUnfSF

    TotalBsmtSF

    BsmtFullBath

    BsmtHalfBath

    GarageYrBlt

    GarageCars

    GarageArea 

"""



for var in Teste_miss.split():

    test[var].fillna(test[var].mean(), inplace = True)
labels = train.columns[1:-1]

X = train[labels].values

XX = test[labels].values

Y = train['SalePrice'].values
xgboost = XGBRegressor(base_score=0.5,

                        booster='gbtree',

                        colsample_bylevel=1,

                        colsample_bynode=1, 

                        colsample_bytree=0.6,

                        gamma=0.3,

                        importance_type='gain', 

                        learning_rate=0.1, 

                        max_delta_step=0,

                        max_depth=4, 

                        min_child_weight=4, 

                        missing=None, 

                        n_estimators=100,

                        n_jobs=1, 

                        nthread=-1, 

                        objective='reg:squarederror', 

                        random_state=0,

                        reg_alpha=0, 

                        reg_lambda=1, 

                        scale_pos_weight=1, 

                        seed=None,

                        silent=None,

                        subsample=1.0, 

                        verbosity=1).fit(X, Y)
preds3 = xgboost.predict(XX)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33)



gbr = ensemble.GradientBoostingRegressor(n_estimators=3000, 

                                            learning_rate=0.05,

                                            max_depth=4, 

                                            max_features='sqrt',

                                            min_samples_leaf=15, 

                                            min_samples_split=10, 

                                            loss='huber', 

                                            random_state =42).fit(X, Y.ravel())





lightgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=4,

                                       learning_rate=0.01, 

                                       n_estimators=5000,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       ).fit(X, Y.ravel())

                                       



xgboost = XGBRegressor(base_score=0.5,

                        booster='gbtree',

                        colsample_bylevel=1,

                        colsample_bynode=1, 

                        colsample_bytree=0.6,

                        gamma=0.3,

                        importance_type='gain', 

                        learning_rate=0.1, 

                        max_delta_step=0,

                        max_depth=4, 

                        min_child_weight=4, 

                        missing=None, 

                        n_estimators=100,

                        n_jobs=1, 

                        nthread=-1, 

                        objective='reg:squarederror', 

                        random_state=0,

                        reg_alpha=0, 

                        reg_lambda=1, 

                        scale_pos_weight=1, 

                        seed=None,

                        silent=None,

                        subsample=1.0, 

                        verbosity=1).fit(X, Y.ravel())
preds1 = gbr.predict(XX)

preds2 = lightgbm.predict(XX)

preds3 = xgboost.predict(XX)



preds_final = (preds3.ravel() + preds2.ravel() + preds1.ravel()) // 3
sub = pd.DataFrame({'Id': test['Id'], 'SalePrice': preds_final})

sub.to_csv('submission.csv', index = False)
sub