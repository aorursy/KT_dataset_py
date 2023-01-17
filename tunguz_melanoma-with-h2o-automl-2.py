# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import h2o

print(h2o.__version__)

from h2o.automl import H2OAutoML



h2o.init(max_mem_size='16G')
train = h2o.import_file("../input/melanoma-train-test-creator/train_32.csv")

test = h2o.import_file("../input/melanoma-train-test-creator/test_32.csv")
train.head()
test.head()
x = test.columns

y = 'target'
# For binary classification, response should be a factor

train[y] = train[y].asfactor()
aml = H2OAutoML(max_models=2000, seed=47, max_runtime_secs=31000)

aml.train(x=x, y=y, training_frame=train)
# View the AutoML Leaderboard

lb = aml.leaderboard

lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)
# The leader model is stored here

aml.leader
preds = aml.predict(test)
preds['p1'].as_data_frame().values.flatten().shape
preds
sample_submission = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')

sample_submission.head()
sample_submission['target'] = preds['p1'].as_data_frame().values

sample_submission.to_csv('submission.csv', index=False)