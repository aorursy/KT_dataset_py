import pandas as pd

import h2o

from h2o.automl import H2OAutoML
h2o.init()

h2o.cluster().show_status()
train = h2o.import_file('/kaggle/input/dsa-automl-and-fe/train_fe.csv')

test = h2o.import_file('/kaggle/input/dsa-automl-and-fe/test_fe.csv')
train['target'] = train['target'].asfactor()
x = train.columns

x.remove('ID')

x.remove('v22')

x.remove('v56')

x.remove('v71')

y = 'target'

train[y] = train[y].asfactor()
%%time

aml = H2OAutoML(max_models=10, 

                nfolds=20, 

                max_runtime_secs=60*60*4,

                balance_classes=True,

                sort_metric='logloss', 

                exclude_algos=['DeepLearning'], 

                verbosity='info')

aml.train(x=x, y=y, training_frame=train)

lb = aml.leaderboard

lb.head(rows=lb.nrows)
aml.leaderboard
test['PredictedProb'] = aml.leader.predict(test)[:,2]
h2o.export_file(test[['ID', 'PredictedProb']],'submission.csv')