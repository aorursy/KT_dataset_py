# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')
## undersampling
true_txn_count = len(train[train.target == 1])

print('Number of traget rows is:' + str(true_txn_count))

true_txn_indices = np.array(train[train.target == 1].index)

false_txn_indices = train[train.target == 0].index



# Out of the indices we picked, randomly select "x" number (number_records_fraud)

random_txn_indices = np.random.choice(false_txn_indices, true_txn_count, replace = True)

random_txn_indices = np.array(random_txn_indices)



# Appending the 2 indices

under_sample_indices = np.concatenate([true_txn_indices,random_txn_indices])



# Under sample dataset

under_sample_data = train.iloc[under_sample_indices,:]



X_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'target']

y_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'target']



# Showing ratio

print("Percentage of false txn: ", len(under_sample_data[under_sample_data.target == 0])/len(under_sample_data))

print("Percentage of true txn: ", len(under_sample_data[under_sample_data.target == 1])/len(under_sample_data))

print("Total number of txn in resampled data: ", len(under_sample_data))
target = y_undersample

train_id = train['id']

test_id = test['id']

train = X_undersample.drop(['id'], axis=1)

test.drop('id', axis=1, inplace=True)



print(len(target), len(train))

print(len(test))
traintest = pd.concat([train, test], axis=0)

one_hot_traintest = pd.concat([pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True, dummy_na=True)], axis=1)

print(one_hot_traintest.shape)



one_hot_train = one_hot_traintest.iloc[:len(train),:]

one_hot_test = one_hot_traintest.iloc[len(train):, :]



print(one_hot_train.shape)

print(one_hot_test.shape)
## since data type of sparse columns is now dtype:Sparse[uint8, 0], we should use below technique:

## https://www.kaggle.com/peterhurford/why-not-logistic-regression



train_ohe = one_hot_train.sparse.to_coo().tocsr()

test_ohe = one_hot_test.sparse.to_coo().tocsr()
from sklearn.model_selection import train_test_split



features = list(train.columns) 

X_train, X_test, y_train, y_test = train_test_split(train_ohe, target, test_size=0.25, random_state=42)



print(f'Train set: {X_train.shape}')

print(f'Test set: {X_test.shape}')
from sklearn.metrics import roc_auc_score

import lightgbm as lgb



lgb_train = lgb.Dataset(X_train.astype('float32'), y_train)

lgb_test = lgb.Dataset(X_test.astype('float32'), y_test, reference=lgb_train)



params = {'num_leaves':200, 'objective':'binary','max_depth':24,'learning_rate':.15,'max_bin':32,

    'feature_fraction': .9,'bagging_fraction': 0.8,'bagging_freq': 15,'verbose': 0,

         'min_data_in_leaf' : 12, }

params['metric'] = ['auc', 'binary_logloss']



evals_result = {}  # to record eval results for plotting

gbm = lgb.train(params,

                lgb_train,

                num_boost_round=2000,

                valid_sets=[lgb_train, lgb_test],

                evals_result=evals_result,

                verbose_eval=100)
print('Plotting metrics recorded during training...')

ax = lgb.plot_metric(evals_result, metric='auc')

plt.show()



ax = lgb.plot_metric(evals_result, metric='binary_logloss')

plt.show()



print('Plotting feature importances...')

ax = lgb.plot_importance(gbm, max_num_features=10)

plt.show()



#print('Plotting split value histogram...')

#ax = lgb.plot_split_value_histogram(gbm, feature='f22', bins='auto')

#plt.show()
sub_df = pd.DataFrame(test_id, columns=['id'])

y_pred = gbm.predict(test_ohe.astype('float32'))

print(len(y_pred), len(test_id))



sub_df['target'] = y_pred

sub_df.head()

sub_df.to_csv('submission.csv', index=False)