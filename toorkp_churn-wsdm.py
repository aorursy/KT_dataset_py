# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import lightgbm as lgb

import time

import gc; gc.enable()

from sklearn.model_selection import ShuffleSplit

from sklearn import *



## loading data



#train = pd.read_csv('../input/train_v2.csv')

#test = pd.read_csv('../input/sample_submission_v2.csv')

#members = pd.read_csv('../input/members_v3.csv')

#transactions = pd.read_csv('../input/transactions_v2.csv')

#user_logs = pd.read_csv('../input/user_logs_v2.csv')
# referred from LGBM

train_input = pd.read_csv('../input/train.csv',dtype={'is_churn' : bool,'msno' : str})

members_input = pd.read_csv('../input/members_v3.csv',dtype={'registered_via' : np.uint8,

                                                      'gender' : 'category'})



train_input = pd.merge(left = train_input,right = members_input,how = 'left',on=['msno'])



del members_input

train_input.head()
transactions_input = pd.read_csv('../input/transactions.csv',dtype = {'payment_method' : 'category',

                                                                  'payment_plan_days' : np.uint8,

                                                                  'plan_list_price' : np.uint8,

                                                                  'actual_amount_paid': np.uint8,

                                                                  'is_auto_renew' : np.bool,

                                                                  'is_cancel' : np.bool})



transactions_input = pd.merge(left = train_input,right = transactions_input,how='left',on='msno')

grouped  = transactions_input.copy().groupby('msno')

grouped.head()
shuffle = grouped.agg({'msno' :{'msno_count': 'count'},

                         'plan_list_price' :{'plan_list_price':'sum'},

                         'actual_amount_paid' : {'actual_amount_paid_mean' : 'mean',

                                                  'actual_amount_paid_sum' : 'sum'},

                         'is_cancel' : {'is_cancel_sum': 'sum'}})



shuffle.head()
shuffle.columns = shuffle.columns.droplevel(0)

shuffle.head()
shuffle.reset_index(inplace=True)

shuffle.head()
train_input = pd.merge(left = train_input,right = shuffle,how='left',on='msno')

train_input.head()
del transactions_input,shuffle

train_input.head()
# referring from LGBM Starter

# merging user_logs

model = None 



for train_indices,val_indices in ShuffleSplit(n_splits=1,test_size = 0.1,train_size=0.4).split(train_input): 

    train_data = lgb.Dataset(train_input.drop(['msno','is_churn'],axis=1).loc[train_indices,:],label=train_input.loc[train_indices,'is_churn'])

    val_data = lgb.Dataset(train_input.drop(['msno','is_churn'],axis=1).loc[val_indices,:],label=train_input.loc[val_indices,'is_churn'])

    

    params = {

        'objective': 'binary',

        'metric': 'binary_logloss',

        'boosting': 'gbdt',

        'learning_rate': 0.05 , 

        'verbose': 0,

        'num_leaves': 108,

        'bagging_fraction': 0.95,

        'bagging_freq': 1,

        'bagging_seed': 1,

        'feature_fraction': 0.9,

        'feature_fraction_seed': 1,

        'max_bin': 128,

        'max_depth': 10,

        'num_rounds': 50,

        } 

    

    model = lgb.train(params, train_data, 50, valid_sets=[val_data])
test_input = pd.read_csv('../input/sample_submission_v2.csv',dtype = {'msno' : str})

members_input = pd.read_csv('../input/members_v3.csv',dtype={'registered_via' : np.uint8,

                                                      'gender' : 'category'})

test_input = pd.merge(left=test_input,right=members_input,how='left',on=['msno'])



del members_input



transactions_input = pd.read_csv('../input/transactions_v2.csv',dtype = {'payment_method' : 'category',

                                                                  'payment_plan_days' : np.uint8,

                                                                  'plan_list_price' : np.uint8,

                                                                  'actual_amount_paid': np.uint8,

                                                                  'is_auto_renew' : np.bool,

                                                                  'is_cancel' : np.bool})



transactions_input = pd.merge(left = test_input,right = transactions_input,how='left',on='msno')

grouped  = transactions_input.copy().groupby('msno')



shuffle = grouped.agg({'msno' : {'total_order' : 'count'},

                         'plan_list_price' : {'plan_net_worth' : 'sum'},

                         'actual_amount_paid' : {'mean_payment_each_transaction' : 'mean',

                                                  'total_actual_payment' : 'sum'},

                         'is_cancel' : {'cancel_times' : lambda x : sum(x==1)}})

             

shuffle.columns = shuffle.columns.droplevel(0)

shuffle.reset_index(inplace=True)

test_input = pd.merge(left = test_input,right = shuffle,how='left',on='msno')

del transactions_input



predictions = model.predict(test_input.drop(['msno','is_churn'],axis=1))

test_input['is_churn'] = predictions

test_input.drop(['city','bd','gender','registered_via','registration_init_time','total_order','plan_net_worth','mean_payment_each_transaction','total_actual_payment','cancel_times'],axis=1,inplace=True)

#test_input.head()

test_input.to_csv('submissions.csv',index=False)

#submissions.head()
output = pd.read_csv('submissions.csv')

output.head()