%%time



import numpy as np 

import pandas as pd



TRAIN_PATH = '/kaggle/input/ashrae-energy-prediction/train.csv'

TEST_PATH  = '/kaggle/input/ashrae-energy-prediction/test.csv'



train = pd.read_csv(TRAIN_PATH)

test  = pd.read_csv(TEST_PATH)
y_train = train['meter_reading']

del train['meter_reading']
test.set_index('row_id', inplace=True)
def datetime_features(x, attrs=['year', 'month', 'day'], prefix='dt_'):

    x = pd.to_datetime(x)

    features = [getattr(x.dt, attr).rename(prefix+attr) for attr in attrs]

    df = pd.concat(features, axis=1)

    return df
FEATURES = ['month', 'weekday', 'hour']

#FEATURES = ['month', 'hour']



dt_train = datetime_features(train['timestamp'], FEATURES)

dt_test  = datetime_features(test['timestamp'],  FEATURES)
cols = ['building_id', 'meter']



X_train = pd.concat([dt_train, train[cols]], axis=1)

X_test =  pd.concat([dt_test,  test[cols]],  axis=1)



X_test
del train, test
from sklearn.base import BaseEstimator, RegressorMixin





class TargetEncodingRegressor(BaseEstimator, RegressorMixin):



    def __init__(self, func='mean'):

        self.func = func

        

        

    def fit(self, X, y):

        

        self.X_cols = list(X)

        self.mapper = X.join(y).groupby(self.X_cols).agg(self.func)

        self.target_mean = y.apply(self.func)

        self.target = y.name

        

        return self





    def predict(self, X):

        Xt = X.merge(self.mapper, on=self.X_cols, how='left')

        return Xt[self.target].fillna(self.target_mean).values
from sklearn.base import BaseEstimator, RegressorMixin, clone







class TransformedTargetRegressor(BaseEstimator, RegressorMixin):

    

    def __init__(self, regressor=None, func=None, inverse_func=None):

    

        self.regressor = regressor

        self.func = func

        self.inverse_func = inverse_func





    def fit(self, X, y, sample_weight=None):

        

        self.target = y.name

        y = self.func(y)

        

        self.regressor_ = clone(self.regressor).fit(X, y)



        return self





    def predict(self, X):

        

        y = self.regressor_.predict(X)

        y = self.inverse_func(y)



        y = pd.Series(y, name=self.target, index=X.index)



        return y
regressor = TargetEncodingRegressor('mean')

estimator = TransformedTargetRegressor(regressor, np.log1p, np.expm1)
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)



y_pred = y_pred.clip(upper=y_train.max())

y_pred = y_pred.astype('float32')



y_pred
y_pred.to_csv('submission.csv', header=True)