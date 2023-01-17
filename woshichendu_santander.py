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
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

sub=pd.read_csv('../input/sample_submission.csv')
train.isnull().any().any()
train.info()
train.head()
train_id=train['ID_code']

train_label=train['target']

test_id=test['ID_code']

train.drop(['ID_code','target'],axis=1,inplace=True)

test.drop('ID_code',axis=1,inplace=True)
train.head()
train.shape,test.shape
(train.columns==test.columns).sum()
from sklearn.ensemble import RandomForestClassifier

rfc1=RandomForestClassifier(n_estimators=50).fit(train,train_label)

pred1=rfc1.predict(test)

pred1
pred1.sum()
test_id=pd.DataFrame(test_id)

test_id
target1=pd.DataFrame(pred1,columns=['target'])

target1
result=pd.concat([test_id,target1],axis=1)

result
result.to_csv('result.csv',index=False)