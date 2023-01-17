!pip install -U h2o
import pandas as pd



import h2o

from h2o.automl import H2OAutoML



h2o.init()
train = h2o.import_file('/kaggle/input/infopulsehackathon/train.csv')

test = h2o.import_file('/kaggle/input/infopulsehackathon/test.csv')



sample_submission = pd.read_csv('/kaggle/input/infopulsehackathon/sample_submission.csv')



y = "Energy_consumption"

x = list(train.columns) 

x.remove(y)
aml = H2OAutoML(max_runtime_secs = 18000, sort_metric='mse', stopping_metric='MSE' , stopping_rounds=100)

aml.train(x = x, y = y, training_frame = train)



aml.leaderboard
sample_submission['Energy_consumption'] = aml.leader.predict(test).as_data_frame()['predict']

sample_submission.loc[sample_submission['Energy_consumption'] < 0, 'Energy_consumption'] = 0
sample_submission[['Id','Energy_consumption']].to_csv('submission.csv', index=False)