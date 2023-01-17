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
from time import strftime

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

le = LabelEncoder()
# load data
print('[{}] Load the data...'.format(strftime('%Y-%m-%d %H:%M:%S')))
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')
train['is_train'] = 1
test['is_train'] = 0

n_train, n_test = len(train), len(test)

data = pd.concat((train, test), axis=0, ignore_index=True)
"""
属性列表: 
Index(['Age', 'Cabin', 'Embarked', 'Fare', 'Name', 'Parch', 'PassengerId',
   'Pclass', 'Sex', 'SibSp', 'Survived', 'Ticket', 'is_train'],
  dtype='object')
"""
columns = ['Age', 'Family_cnt', 'Fare', 'Sex', 'Pclass', 'Embarked']
### Age: 236个为空,最小值0.17,最大值80; 年龄在25岁左右的人,生还概率和死亡概率差不多
print('[{}] Process `Age` column...'.format(strftime('%Y-%m-%d %H:%M:%S')))
data['Age'].fillna(25, inplace=True)

### Sex: 没有为空的
print('[{}] Process `Sex` column...'.format(strftime('%Y-%m-%d %H:%M:%S')))
data['Sex'] = le.fit_transform(data['Sex'].astype('category'))

### Pclass: 船票的等级: 没有为空的. 取1(顶级),2(中级),3(低级)三个值
print('[{}] Process `Pclass` column...'.format(strftime('%Y-%m-%d %H:%M:%S')))
data['Pclass'] = data['Pclass'].astype('category')

### Parch, SibSp: 最小值是0, 最大值是10, 99%分位点为7, 95%分位点为4
print('[{}] Process `Parch` & `SibSp` column...'.format(strftime('%Y-%m-%d %H:%M:%S')))
data['Family_cnt'] = data['Parch'] + data['SibSp']

### Fare
print('[{}] Process `Fare` column...'.format(strftime('%Y-%m-%d %H:%M:%S')))
data['Fare'].fillna(13.30, inplace=True)

### Cabin
print('[{}] Process `Cabin` column...'.format(strftime('%Y-%m-%d %H:%M:%S')))
#    data['Fare'].fillna(13.30, inplace=True)

### Embarked: 上船的港口: S, C, Q
print('[{}] Process `Embarked` column...'.format(strftime('%Y-%m-%d %H:%M:%S')))
data['Embarked'].fillna('S', inplace=True)
data['Embarked'] = le.fit_transform(data['Embarked'])
print('[{}] Using RF classifier for training...'.format(strftime('%Y-%m-%d %H:%M:%S')))
train_samples = data.loc[data['is_train'] == 1, columns]
y_train = data.loc[data['is_train']==1, 'Survived'].values
test_samples  = data.loc[data['is_train'] == 0, columns]
train_part, valid_part, y_train_part, y_valid_part = train_test_split(train_samples.values,
                                                                      y_train,
                                                                      test_size=0.1,
                                                                      random_state=42)

rf = RandomForestClassifier()
rf.fit(train_part, y_train_part)
print('[{}] Test on validation set...'.format(strftime('%Y-%m-%d %H:%M:%S')))
rf_valid_pred = rf.predict(valid_part)

valid_acc = (rf_valid_pred == y_valid_part).sum() / y_valid_part.shape[0]
print('[{}] Acc validation set is {:.2f}%...'.format(strftime('%Y-%m-%d %H:%M:%S'), valid_acc*100))

print('[{}] Make prediction on test set...'.format(strftime('%Y-%m-%d %H:%M:%S')))
rf_test_pred = rf.predict(test_samples.values)

print('[{}] Make submission...'.format(strftime('%Y-%m-%d %H:%M:%S')))
submission = pd.DataFrame({'PassengerId':test.PassengerId.values,
                           'Survived': rf_test_pred})
submission.to_csv('titanic_result_v3.csv', index=False)