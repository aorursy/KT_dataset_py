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
train=pd.read_csv('../input/train/train.csv')

test=pd.read_csv('../input/test/test.csv')

sub=pd.read_csv('../input/test/sample_submission.csv')
train.head()
train.keys()
X1_train=train[['Type','Age','Breed1', 'Breed2', 'Gender', 'Color1', 'Color2','Color3',

               'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 

               'Health', 'Quantity', 'Fee','PhotoAmt']]

y=train['AdoptionSpeed']

X1_test=test[['Type','Age','Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',

             'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',

             'Sterilized', 'Health', 'Quantity', 'Fee','PhotoAmt']]
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC
#分割数据

X1_train_train, X1_train_valid, y_train, y_valid=train_test_split(X1_train, y, test_size=0.25)#75%用以训练，25用于验证

#对数据进行标准化

ss=StandardScaler()

X1_train_train_standard=ss.fit_transform(X1_train_train)

X1_train_valid_standard=ss.fit_transform(X1_train_valid)

#模型

lsvc=LinearSVC()

lsvc.fit(X1_train_train_standard, y_train)

y_valid_predict=lsvc.predict(X1_train_valid_standard)

#验证集准确性评测

print("The Accuracy of Linear SVC is ",lsvc.score(X1_train_valid_standard,y_valid))

#详细分析

from sklearn.metrics import classification_report

print(classification_report(y_valid,y_valid_predict))
X1_test_standard=ss.fit_transform(X1_test)

y_predict=lsvc.predict(X1_test_standard)
sub['AdoptionSpeed']=y_predict

sub.to_csv('submission.csv',index=False)