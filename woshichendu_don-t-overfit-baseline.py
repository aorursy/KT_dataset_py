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

train.head()
train.info()
train_feat=train.drop(columns=['id','target'],axis=1)

train_feat.head()
test_feat=test.drop(columns='id',axis=1)

test_feat.head()
train_feat.shape,test_feat.shape
#from sklearn.tree import DecisionTreeClassifier

train_label=train['target']

#dcmodel1=DecisionTreeClassifier().fit(train_feat,train_label)

#pred1=dcmodel1.predict(test_feat)

#pred1

'''from sklearn.ensemble import RandomForestClassifier

rfcmodel1=RandomForestClassifier(n_estimators=500).fit(train_feat,train_label)

pred2=rfcmodel1.predict(test_feat)'''

from sklearn.linear_model import LassoCV,LogisticRegressionCV,RidgeCV,SGDClassifier

l1=LassoCV().fit(train_feat,train_label)

l2=LogisticRegressionCV().fit(train_feat,train_label)

pred3=l1.predict(test_feat)

pred4=l2.predict(test_feat)
#pd.DataFrame(pred1).rename(columns={0:'target'})
#sub
target=pd.DataFrame(pred3).rename(columns={0:'target'})

sub_id=sub[['id']]

submission=pd.concat([sub_id,target],axis=1)

submission

submission.to_csv('sub.csv',index=False)