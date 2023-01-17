# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
%matplotlib inline
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
plt.style.use('ggplot') 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
items  = pd.read_csv('../input/items.csv')
train = pd.read_csv('../input/sales_train.csv')
test = pd.read_csv('../input/test.csv')
item_category = pd.read_csv('../input/item_categories.csv')
shops = pd.read_csv('../input/shops.csv')
new_train = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
new_train.columns = ['item_cnt_month']
new_train.reset_index(inplace=True)
X_train = new_train[new_train.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = new_train[new_train.date_block_num < 33]['item_cnt_month']
X_valid = new_train[new_train.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = new_train[new_train.date_block_num == 33]['item_cnt_month']
X_test = test
X_test['date_block_num'] = 34
del X_test['ID']
X_test = X_test[['date_block_num', 'shop_id', 'item_id']]
model = xgb.XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 10)
Y_pred = model.predict(X_valid).clip(0, 20)
Y_test = model.predict(X_test).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.to_csv('xgb_submission.csv', index=False)