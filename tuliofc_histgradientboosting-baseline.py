# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

from copy import deepcopy

import json

import os



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.multioutput import MultiOutputRegressor

from sklearn.preprocessing import LabelEncoder



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def mcrmse_loss(y_true, y_pred, N=3):

    """

    Calculates competition eval metric

    """

    assert len(y_true) == len(y_pred)

    n = len(y_true)

    return np.sum(np.sqrt(np.sum((y_true - y_pred)**2, axis=0)/n)) / N
data_train = [json.loads(line) for line in open('../input/stanford-covid-vaccine/train.json','r')]

data_test = [json.loads(linha) for linha in open('../input/stanford-covid-vaccine/test.json','r')]

test_set = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')
for jason in data_train:

    jason['step'] = list(range(jason['seq_scored']))

    jason['sequence'] = list(jason['sequence'])

    jason['structure'] = list(jason['structure'])

    jason['predicted_loop_type'] = list(jason['predicted_loop_type'])
for jason in data_test:

    jason['step'] = list(range(jason['seq_scored']))

    jason['sequence'] = list(jason['sequence'])

    jason['structure'] = list(jason['structure'])

    jason['predicted_loop_type'] = list(jason['predicted_loop_type'])
train = pd.json_normalize(data = data_train, 

                            record_path ='reactivity',  

                            meta =['id','signal_to_noise',

                                  'SN_filter','seq_length','seq_scored']) 

train.rename(columns={0:'reactivity'}, inplace=True)

train['step'] = pd.json_normalize(data = data_train, 

                            record_path ='step'

                                        )

train['sequence'] = pd.json_normalize(data = data_train, 

                            record_path ='sequence'

                                        )

train['structure'] = pd.json_normalize(data = data_train, 

                            record_path ='structure'

                                        )

train['predicted_loop_type'] = pd.json_normalize(data = data_train, 

                            record_path ='predicted_loop_type'

                                        )

train['reactivity_error'] = pd.json_normalize(data = data_train, 

                            record_path ='reactivity_error'

                                        )

train['deg_Mg_pH10'] = pd.json_normalize(data = data_train, 

                            record_path ='deg_Mg_pH10'

                                        )

train['deg_error_Mg_pH10'] = pd.json_normalize(data = data_train, 

                            record_path ='deg_error_Mg_pH10'

                                        )

train['deg_pH10'] = pd.json_normalize(data = data_train, 

                            record_path ='deg_pH10',

                                        )

train['deg_error_pH10'] = pd.json_normalize(data = data_train, 

                            record_path ='deg_error_pH10',

                                        )

train['deg_Mg_50C'] = pd.json_normalize(data = data_train, 

                            record_path ='deg_Mg_50C',

                                        )

train['deg_error_Mg_50C'] = pd.json_normalize(data = data_train, 

                            record_path ='deg_error_Mg_50C',

                                        )

train['deg_50C'] = pd.json_normalize(data = data_train, 

                            record_path ='deg_50C',

                                        )

train['deg_error_50C'] = pd.json_normalize(data = data_train, 

                            record_path ='deg_error_50C',

                                        )



train.set_index(['id','step'], inplace=True)
test = pd.json_normalize(data = data_test,

                         record_path = 'sequence',

                        meta = ['id','seq_length','seq_scored'])

test.rename(columns={0:'sequence'},inplace=True)

test['step'] = pd.json_normalize(data = data_test,

                                record_path = 'step')

test['sequence'] = pd.json_normalize(data = data_test,

                                    record_path = 'sequence')

test['structure'] = pd.json_normalize(data = data_test,

                                     record_path = 'structure')

test['predicted_loop_type'] = pd.json_normalize(data = data_test,

                                               record_path = 'predicted_loop_type')

test.set_index(['id','step'], inplace=True)
train
test
np.random.seed(2020) #Seed the randomness to be deterministic
enc = LabelEncoder()

category_cols_train = [cols for cols in train.columns if train[cols].dtype == 'object']

category_cols_test = [cols for cols in test.columns if test[cols].dtype == 'object']

print(category_cols_train)

print(category_cols_test)

X_train_enc = deepcopy(train)

X_test_enc = deepcopy(test)

for cols in category_cols_train:

    X_train_enc[cols] = enc.fit_transform(X_train_enc[cols])

for cols in category_cols_test:

    X_test_enc[cols] = enc.fit_transform(X_test_enc[cols])
X = X_train_enc.drop(['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C'],axis=1)

y = X_train_enc.loc[:,['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']]
hgbr = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter = 1750, max_depth = 15,early_stopping = True, n_iter_no_change = 10,

                                                          learning_rate = 0.0025, tol = 1e-6, validation_fraction = 0.2,

                                                          verbose = 2, max_leaf_nodes = 64),

                           n_jobs = 4

)



gbr = MultiOutputRegressor(GradientBoostingRegressor(loss = 'huber', n_estimators = 1000, max_depth = 15,

                                                     learning_rate = 0.0025, tol = 1e-7, validation_fraction = 0.2,

                                                     n_iter_no_change = 15, verbose = 2

    )

)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
X_train
hgbr.fit(X_train,y_train)
y_pred = hgbr.predict(X_test)
print(mean_squared_error(y_test,y_pred,squared=False))
X_train_enc[X_test_enc.columns]
hgbr.fit(X_train_enc[X_test_enc.columns],y)
y_pred_2 = hgbr.predict(X_test_enc)
y_pred_2.shape
submission = pd.DataFrame(np.concatenate([test_set.id_seqpos.values[:,np.newaxis],y_pred_2],axis=1),columns=test_set.columns)

display(submission.head(10))

submission.to_csv('submission.csv',index=False)