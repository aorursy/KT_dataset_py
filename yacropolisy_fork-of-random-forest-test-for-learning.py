# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')
test.head(10)
train.describe()
train.describe(include='all')
train.shape
test['is_train'] = False
train['is_train'] = True

all_df = pd.concat([test, train], axis = 0)
test.head()
from sklearn import  preprocessing
le = preprocessing.LabelEncoder()

cat_vars = ["user_id", "region", "city", "parent_category_name", "category_name", "param_1", "param_2", "param_3", "user_type"]
for col in cat_vars:
    all_df[col] = all_df[col].astype('str')
    le.fit(all_df[col])
    all_df[col] = le.transform(all_df[col])
all_df.head()
cols_to_drop = ["item_id", "title", "description", "activation_date", "image"]
all_df = all_df.drop(cols_to_drop, axis = 1)
all_df.head()
all_df = all_df.fillna(0)
all_df.head()
all_df.loc[all_df['param_1']==110].head()
train_df= all_df.loc[all_df['is_train']==True].drop(['is_train'], axis = 1)
test_df = all_df.loc[all_df['is_train'] == False].drop(['is_train', 'deal_probability'], axis = 1)
from sklearn.model_selection import train_test_split
train_X, valid_X, train_y, valid_y = train_test_split(train_df.drop('deal_probability', axis=1), 
                                                      train_df['deal_probability'], test_size=0.2)
rf_params={
    'n_estimators' : [100, 200, 300],
    'n_jobs' : [-1], 
}
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
rf = RandomForestRegressor()
grid_search = GridSearchCV(rf, param_grid=rf_params)
grid_search.fit(train_X,train_y)
from sklearn import linear_model
reg = linear_model.Ridge (alpha = .5)
reg.fit(train_X,train_y)
rf
rf.score(valid_X, valid_y)

import xgboost as xgb 
xgb_model = xgb.XGBRegressor()
xgb_model.fit(train_X,train_y)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_jobs = -1, n_estimators = 100)
rf.fit(train_X,train_y)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(verbose = 1, n_estimators=20)
rf.fit(train_X,train_y)
from sklearn.metrics import mean_squared_error as mse
print(np.sqrt(mse(rf.predict(valid_X), valid_y)))
pred_test_y = rf.predict(test_df)
pred_test_y[:5]
pred_test_y[pred_test_y>1] = 1
pred_test_y[pred_test_y<0] = 0
test_id = test['item_id'].values
sub = pd.DataFrame({'item_id':test_id})
sub['deal_probability'] = pred_test_y
sub.to_csv('rf_test.csv', index=False)
sub.head()
