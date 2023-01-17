import pandas as pd

import h2o

from h2o.automl import H2OAutoML
h2o.init()
train = h2o.import_file('../input/titanic/train.csv')

test = h2o.import_file('../input/titanic/test.csv')
train.head()
X_train, y_test = train.split_frame(ratios=[.80], seed=2020)
y = "Survived"

x = train.columns

x.remove(y)
X_train[y] = X_train[y].asfactor()

y_test[y] = y_test[y].asfactor()
model = H2OAutoML(max_models=10, nfolds=10,  max_runtime_secs=300, max_runtime_secs_per_model=200, verbosity='info')
model.train(x = x, y = y, training_frame = X_train, leaderboard_frame = y_test)
model.leaderboard


model.leader
pred = model.leader.predict(test)
pred
submission = test['PassengerId'].as_data_frame()

submission['Survived'] = pred['predict'].as_data_frame()

submission.to_csv('submission.csv', index=False)