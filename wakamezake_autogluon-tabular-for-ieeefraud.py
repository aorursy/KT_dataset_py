# https://autogluon.mxnet.io/

!pip install --upgrade --quiet mxnet

!pip install --quiet autogluon
import gc

import pandas as pd

import numpy as np

from autogluon import TabularPrediction as task

from autogluon.utils.tabular.metrics import roc_auc
%ls ../input/ieee-fraud-detection/
label_column = 'isFraud' # name of target variable to predict in this competition

eval_metric = 'roc_auc' # Optional: specify that competition evaluation metric is AUC

directory = "../input/ieee-fraud-detection/"

output_directory = 'AutoGluonModels/'



train_identity = pd.read_csv(directory+'train_identity.csv')

train_transaction = pd.read_csv(directory+'train_transaction.csv')
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

train_data = task.Dataset(df=train) # convert to AutoGluon dataset

del train_identity, train_transaction, train # free unused memory

gc.collect()
predictor = task.fit(train_data=train_data, label=label_column, output_directory=output_directory,

                     eval_metric=eval_metric, verbosity=3, auto_stack=True, time_limits=3600)



# results = predictor.fit_summary()
test_identity = pd.read_csv(directory + 'test_identity.csv')

test_transaction = pd.read_csv(directory + 'test_transaction.csv')

test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left') # same join applied to training files

test_data = task.Dataset(df=test) # convert to AutoGluon dataset



del test_identity, test_transaction, test # free unused memory

gc.collect()
y_predproba = predictor.predict_proba(test_data)

print(y_predproba[:5]) # some example predicted fraud-probabilities
submission = pd.read_csv(directory + 'sample_submission.csv')

submission['isFraud'] = y_predproba

submission.head()

submission.to_csv('submission.csv', index=False)