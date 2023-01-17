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

from h2o.automl import H2OAutoML

print(h2o.__version__)



h2o.init(max_mem_size='17G')
train = h2o.import_file("../input/malware-feature-engineering-only/new_train.csv")

train.head()
x = train.columns[1:-1]

y = 'HasDetections'
# For binary classification, response should be a factor

train[y] = train[y].asfactor()
# Run AutoML for 20 base models (limited to 1 hour max runtime by default)

aml = H2OAutoML(max_runtime_secs=30000, seed=66)

aml.train(x=x, y=y, training_frame=train)
# View the AutoML Leaderboard

lb = aml.leaderboard

lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)
h2o.remove(train.frame_id)

test = h2o.import_file("../input/malware-feature-engineering-only/new_test.csv")

preds = aml.predict(test)

h2o.remove(test.frame_id)
sample_submission = pd.read_csv('../input/microsoft-malware-prediction/sample_submission.csv')

sample_submission.head()



sample_submission['HasDetections'] = preds['p1'].as_data_frame().values

sample_submission.to_csv('submission.csv', index=False)