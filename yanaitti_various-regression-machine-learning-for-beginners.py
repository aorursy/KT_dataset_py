# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install h5py
from matplotlib import pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

submit = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
print(train.shape)

print(train.describe())

print(train.info())
train.describe(include='O')
int_columns = train.select_dtypes(exclude='object').columns

int_columns
obj_columns = train.select_dtypes(include='object').columns

obj_columns
select_columns = int_columns[1:-1]

print(select_columns)

print(len(select_columns))
train[select_columns].isnull().any()
# imputer

from sklearn.impute import SimpleImputer

print(train[select_columns])



imp_mean = SimpleImputer()

train_imp = pd.DataFrame(imp_mean.fit_transform(train[select_columns]))

train_imp.columns = select_columns



test_imp = pd.DataFrame(imp_mean.transform(test[select_columns]))

test_imp.columns = select_columns



print(train_imp)
train_imp.isnull().any()
# select_columns = [col for col in select_columns if train[col].isnull().any() == False]

# print(select_columns)

# print(len(select_columns))
train_obj = train[obj_columns].fillna('nan')

print(train_obj.describe(include='O'))



test_obj = test[obj_columns].fillna('nan')

print(test_obj.describe(include='O'))
train_categorize = pd.get_dummies(train_obj)

print(train_categorize)



test_categorize = pd.get_dummies(test_obj)

print(test_categorize)
# X = train[select_columns]

# _X = pd.concat([train_imp, train_categorize], axis=1)

y = train['SalePrice']



# _X_val = pd.concat([test_imp, test_categorize], axis=1)

# X_val = test[select_columns]



from sklearn.preprocessing import StandardScaler



## StanderdScaler

sc = StandardScaler()

train_std = pd.DataFrame(sc.fit_transform(train_imp))

train_std.columns = train_imp.columns



test_std = pd.DataFrame(sc.fit_transform(test_imp))

test_std.columns = test_imp.columns

import category_encoders as ce



# OneHotEncodeしたい列を指定。Nullや不明の場合の補完方法も指定。

ce_ohe = ce.OneHotEncoder(handle_unknown='impute')



# pd.DataFrameをそのまま突っ込む

train_categorize = ce_ohe.fit_transform(train_obj)

test_categorize = ce_ohe.transform(test_obj)



print(train_categorize)

print(test_categorize)



X = pd.concat([train_std, train_categorize], axis=1)

X_val = pd.concat([test_std, test_categorize], axis=1)

# X = pd.concat([train_std, train_obj], axis=1)

# X_val = pd.concat([test_std, test_obj], axis=1)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train
names = []

names_h5 = []

pd.options.display.float_format = '{:.2f}'.format
X_train['KitchenAbvGr']
from sklearn.linear_model import LinearRegression



simple_lr = LinearRegression()



X_train_simple = X_train['KitchenAbvGr'].values.reshape(-1, 1)

X_test_simple = X_test['KitchenAbvGr'].values.reshape(-1, 1)



simple_lr.fit(X_train_simple, y_train)

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

import pickle



y_pred = simple_lr.predict(X_test_simple)



def lr_scores(model_name, model, y_test, y_pred):

    # MAE / Mean Absolute Error ...平均絶対誤差

    mae = mean_absolute_error(y_test, y_pred)



    # MSE / Mean Squared Error ...平均二乗誤差

    mse = mean_squared_error(y_test, y_pred)



    # RMSE / Root Mean Squared Error...二乗平均平方根誤差

    rmse = np.sqrt(mse)



    # R2_score / 決定係数...1に近い方がよい。マイナスもある

    r2 = r2_score(y_test, y_pred)

    

    linear_predict = {'name': model_name, 'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}

    print(linear_predict)



    names.append(model_name)

    with open(model_name + '.pickle', mode='wb') as fp:

        pickle.dump(model, fp)



def lr_scores_h5(model_name, model, y_test, y_pred):

    # MAE / Mean Absolute Error ...平均絶対誤差

    mae = mean_absolute_error(y_test, y_pred)



    # MSE / Mean Squared Error ...平均二乗誤差

    mse = mean_squared_error(y_test, y_pred)



    # RMSE / Root Mean Squared Error...二乗平均平方根誤差

    rmse = np.sqrt(mse)



    # R2_score / 決定係数...1に近い方がよい。マイナスもある

    r2 = r2_score(y_test, y_pred)

    

    linear_predict = {'name': model_name, 'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}

    print(linear_predict)



    names_h5.append(model_name)

    model.save(model_name + '.h5')



lr_scores('Simple Linear Regressor', simple_lr, y_test, y_pred)
multiple_lr = LinearRegression()

multiple_lr.fit(X_train, y_train)



y_pred = multiple_lr.predict(X_test)

lr_scores('Multiple Linear Regressor', multiple_lr, y_test, y_pred)
from sklearn.linear_model import SGDRegressor



sgd = SGDRegressor(eta0 = 0.0001, max_iter = 100)

sgd.fit(X_train, y_train)



y_pred = sgd.predict(X_test)

lr_scores('SGD Regressor', sgd, y_test, y_pred)
from sklearn.linear_model import Lasso



lasso = Lasso()

lasso.fit(X_train, y_train)



y_pred = lasso.predict(X_test)

lr_scores('Lasso', lasso, y_test, y_pred)
from sklearn.linear_model import ElasticNet



eln = ElasticNet()

eln.fit(X_train, y_train)



y_pred = eln.predict(X_test)

lr_scores('ElasticNet', eln, y_test, y_pred)
from sklearn.linear_model import Ridge



ridge = Ridge()

ridge.fit(X_train, y_train)



y_pred = ridge.predict(X_test)

lr_scores('Ridge Regressor', ridge, y_test, y_pred)
from sklearn.svm import SVR



svr_l = SVR(kernel='linear')

svr_l.fit(X_train, y_train)



y_pred = svr_l.predict(X_test)

lr_scores('SVM linear', svr_l, y_test, y_pred)
from sklearn.svm import SVR



svr_r = SVR(kernel='rbf')

svr_r.fit(X_train, y_train)



y_pred = svr_r.predict(X_test)

lr_scores('SVM rbf', svr_r, y_test, y_pred)
from sklearn.ensemble import RandomForestRegressor



random_forest = RandomForestRegressor(random_state=1, n_estimators=10)

random_forest.fit(X_train, y_train)



y_pred = random_forest.predict(X_test)

lr_scores('Random Forest Regressor', random_forest, y_test, y_pred)
from sklearn.ensemble import GradientBoostingRegressor



gradient_boost = GradientBoostingRegressor(random_state=1, n_estimators=10)

gradient_boost.fit(X_train, y_train)



y_pred = gradient_boost.predict(X_test)

lr_scores('Gradient Boosting Regressor', gradient_boost, y_test, y_pred)
# アルゴリズムを順に実行

result = []



for name in names:

    with open(name + '.pickle', 'rb') as fp:

        reg = pickle.load(fp)

    

    if name == 'Simple Linear Regressor':

        y_pred = reg.predict(X_test_simple)

    else:

        y_pred = reg.predict(X_test)



    # MAE

    mae = mean_absolute_error(y_test, y_pred)



    # MSE

    mse = mean_squared_error(y_test, y_pred)



    # RMSE

    rmse = np.sqrt(mse)



    # R2_score

    r2 = r2_score(y_test, y_pred)



    result.append([mae, mse, rmse, r2])

    

    print(name)



df_result = pd.DataFrame(result, columns=['mae', 'mse', 'rmse', 'r2'], index = names)

df_result
from xgboost import XGBRegressor



xgb = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100)

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)



lr_scores('XGBoost Regressor', xgb, y_test, y_pred)
from lightgbm import LGBMRegressor



lgb = LGBMRegressor(max_depth=3, learning_rate=0.1, n_estimators=100)

lgb.fit(X_train, y_train)

y_pred = lgb.predict(X_test)



lr_scores('LightGBM Regressor', lgb, y_test, y_pred)
from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



folds = KFold(n_splits = 5, shuffle = True, random_state = 100)



params = {'max_depth': list(range(2, 10)), 'random_state': [0], 'n_estimators': list(range(50, 200, 50))}



# XGBoost

xgb = XGBRegressor()



reg_cv = GridSearchCV(xgb, params, cv=folds, return_train_score=True)

reg_cv.fit(X, y)
reg_cv.best_params_

y_pred = reg_cv.predict(X_test)

lr_scores('XGBoost(GCV)', reg_cv, y_test, y_pred)

# LightGBM

lgb = LGBMRegressor()



reg_cv = GridSearchCV(lgb, params, cv=folds, return_train_score=True)

reg_cv.fit(X, y)
reg_cv.best_params_

y_pred = reg_cv.predict(X_test)

lr_scores('LightGBM(GCV)', reg_cv, y_test, y_pred)

# アルゴリズムを順に実行

result = []



for name in names:

    with open(name + '.pickle', 'rb') as fp:

        reg = pickle.load(fp)

    

    if name == 'Simple Linear Regressor':

        y_pred = reg.predict(X_test_simple)

    else:

        y_pred = reg.predict(X_test)



    # MAE

    mae = mean_absolute_error(y_test, y_pred)



    # MSE

    mse = mean_squared_error(y_test, y_pred)



    # RMSE

    rmse = np.sqrt(mse)



    # R2_score

    r2 = r2_score(y_test, y_pred)



    result.append([mae, mse, rmse, r2])

    

    print(name)



df_result = pd.DataFrame(result, columns=['mae', 'mse', 'rmse', 'r2'], index = names)

df_result

batch_size = 64

n_epochs = 100
import tensorflow as tf

from tensorflow import keras



model = keras.Sequential([

    keras.layers.Dense(512, activation='relu', input_shape=[X.shape[1]]),

    keras.layers.Dense(512, activation='relu'),

    keras.layers.Dense(1)

])

optimizer = tf.keras.optimizers.RMSprop(0.001)



model.compile(optimizer=optimizer, 

              loss='mse',

              metrics=['mae', 'mse'])



model.summary()
model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size)

y_pred = model.predict(X_test)

lr_scores_h5('TensorFlow', model, y_test, y_pred)

from keras.models import load_model



# アルゴリズムを順に実行

result = []



for name in names_h5:

    reg = load_model(name + '.h5')

    

    y_pred = reg.predict(X_test)



    # MAE

    mae = mean_absolute_error(y_test, y_pred)



    # MSE

    mse = mean_squared_error(y_test, y_pred)



    # RMSE

    rmse = np.sqrt(mse)



    # R2_score

    r2 = r2_score(y_test, y_pred)



    result.append([mae, mse, rmse, r2])

    

    print(name)



df_result = pd.DataFrame(result, columns=['mae', 'mse', 'rmse', 'r2'], index = names_h5)

df_result

submit
with open('XGBoost(GCV).pickle', 'rb') as fp:

    reg = pickle.load(fp)



y_pred_final = reg.predict(X_val)

y_pred_final
submit['SalePrice'] = y_pred_final
submit.to_csv('submit.csv', index=None)