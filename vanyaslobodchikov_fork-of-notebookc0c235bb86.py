# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import random 



import xgboost as xgb



from sklearn.model_selection import KFold, train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error

from sklearn.feature_selection import SelectFromModel



import scipy.sparse as sp



rng = np.random.RandomState(31337)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



df = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])



macro = pd.read_csv('../input/macro.csv', parse_dates=['timestamp'], )



#macro_floats = list(macro.select_dtypes(include=[np.float64]).columns)

#macro[macro_floats]

#macro[macro_floats].astype(np.float32)
df.duplicated('timestamp')


df = pd.merge(df, macro, on='timestamp')

df.duplicated('timestamp')
X = df.drop('price_doc', axis=1).select_dtypes(exclude=[object])

y = df['price_doc']
test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])

test.shape
test = pd.merge(test, macro, on='timestamp')

test.shape
obj_cols = list(df.select_dtypes(include=[object]).columns)

#obj_cols
dummy_train = pd.get_dummies(df[obj_cols])

dummy_test = pd.get_dummies(test[obj_cols])
ls_dummy_empty_cols = list(set(list(dummy_train.columns)) & set(list(dummy_test.columns)))

#ls_dummy_empty_cols
X = pd.concat([X, dummy_train[ls_dummy_empty_cols]], axis=1)

X_test = pd.concat([test, dummy_test[ls_dummy_empty_cols]], axis=1).select_dtypes(exclude=[object])
X = X.fillna(0)

X_test = X_test.fillna(0)
X = sp.hstack((X.drop(['id', 'timestamp'], axis=1), sp.csr_matrix(np.ones((X.shape[0], 1)))), format='csc')

X_test = sp.hstack((X_test.drop(['id', 'timestamp'], axis=1), sp.csr_matrix(np.ones((X_test.shape[0], 1)))), format='csc')
X.data
params = {'max_depth': 7, 'n_estimators': 250}
xgb_reg = xgb.XGBRegressor(**params).fit(X, y)

predictions = xgb_reg.predict(X_test)
print(predictions)
# select features using threshold

selection = SelectFromModel(xgb_reg, threshold=0.00001, prefit=True)

select_X_train = selection.transform(X)

# train model

selection_model = xgb.XGBRegressor(**params)

selection_model.fit(select_X_train, y)

# eval model

select_X_test = selection.transform(X_test)

y_pred = selection_model.predict(select_X_test)
boosting_sub = pd.DataFrame({"id":test["id"], "price_doc":y_pred})

boosting_sub.to_csv("boosting6-250-macro-select.csv", index=False)