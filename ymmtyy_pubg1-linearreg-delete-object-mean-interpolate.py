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
# import train and test to play with it
train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')
sample_sumission = pd.read_csv('../input/sample_submission_V2.csv')
#print(train.head())
#print(train.columns)
print(len(train))
# データの型の確認
#print(train.dtypes)

# グループに所属しているか
groupId_train=train['groupId']
groupBool=[]
groupId_test=test['groupId']
groupBool_train=[]
groupBool_test=[]
for gid in groupId_train:
    if gid in groupId_train:
        groupBool_train.append(1)
    else:
        groupBool_train.append(0)
train['groupBool']=groupBool_train

for gid in groupId_test:
    if gid in groupId_test:
        groupBool_test.append(1)
    else:
        groupBool_test.append(0)
test['groupBool']=groupBool_test

#for column in columns:
#    plt.scatter(X_train[column], X_train['winPlacePerc'])
#plt.show()
# fillna
train_Id=train['Id']
train=train.drop(['Id','groupId','matchId','matchType'], axis=1)
train=train.fillna(train.mean())
X_train=train.drop(['winPlacePerc'], axis=1)
Y_train=train['winPlacePerc']
test_Id=test['Id']
test=test.drop(['Id','groupId','matchId','matchType'], axis=1)
X_test=test.fillna(test.mean())
from sklearn import linear_model
clf = linear_model.LinearRegression()

#fitで説明変数、目的変数を元にモデルを作成します
#第一引数に説明変数、第二引数に目的変数を入れます！
clf.fit(X_train,Y_train)

Y_test = clf.predict(X_test)
# create submission.csv
y_pred=pd.DataFrame(Y_test)
result=pd.concat([test_Id,y_pred],axis=1)
result.columns=['Id','winPlacePerc']
print(result)

# you could use any filename. We choose submission here
result.to_csv('submission.csv', index=False)
