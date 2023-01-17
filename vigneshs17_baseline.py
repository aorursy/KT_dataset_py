import os

print((os.listdir('../input/')))
import pandas as pd

from sklearn.experimental import enable_hist_gradient_boosting 

from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import roc_auc_score
df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')

df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')

test_index=df_test['Unnamed: 0'] #copying test index for later
df_train.head()
train_X = df_train.loc[:, 'V1':'V16']

train_y = df_train.loc[:, 'Class']
rf = HistGradientBoostingClassifier(random_state=123, learning_rate=0.1, max_depth=None, max_leaf_nodes=31)

#HistGradientBoostingClassifier gave the highest roc_auc score among GradientBoosting, AdaBoostClassifier

#, RandomForestClassifier and then I tuned the hyperparameters to yield the highest accuracy
rf.fit(train_X, train_y)
df_test = df_test.loc[:, 'V1':'V16']

pred = rf.predict_proba(df_test)
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred[:,1])

result.head()
result.to_csv('output.csv', index=False)