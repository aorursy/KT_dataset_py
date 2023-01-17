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
import h2o

print(h2o.__version__)

from h2o.automl import H2OAutoML



h2o.init(max_mem_size='16G')
train = h2o.import_file("../input/train.csv")

test = h2o.import_file("../input/test.csv")
train.head()
train.shape

test.head()
test.shape
x = test.columns[1:]

y = 'target'
# For binary classification, response should be a factor

train[y] = train[y].asfactor()
aml = H2OAutoML(max_models=200, seed=47, max_runtime_secs=31000)

aml.train(x=x, y=y, training_frame=train)
# View the AutoML Leaderboard

lb = aml.leaderboard

lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)
# The leader model is stored here

aml.leader
preds = aml.predict(test)
preds['p1'].as_data_frame().values.flatten().shape

preds
sample_submission = pd.read_csv('../input/sample_submission.csv')

sample_submission.shape
sample_submission['target'] = preds['p1'].as_data_frame().values

sample_submission.to_csv('submission.csv', index=False)