import numpy as np 

import pandas as pd 



import os

print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')
train_feat = train.drop(columns=['id', 'target'], axis=1)

test_feat = test.drop(columns='id', axis=1)

from sklearn.preprocessing import StandardScaler

std = StandardScaler()



train_feat = std.fit_transform(train_feat)

test_feat = std.fit_transform(test_feat)





train_label = train['target']



from sklearn.linear_model import LassoCV, LogisticRegressionCV, RidgeCV, SGDClassifier



l1 = LassoCV().fit(train_feat, train_label)

l2 = LogisticRegressionCV().fit(train_feat, train_label)

pred3 = l1.predict(test_feat)  # 0.846

pred4 = l2.predict(test_feat)  # 0.6

target = pd.DataFrame(pred3).rename(columns={0: 'target'})

sub_id = sub[['id']]

submission = pd.concat([sub_id, target], axis=1)

print(submission)



submission.to_csv('sub.csv', index=False)
