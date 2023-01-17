# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

gender_submission = pd.read_csv('../input/titanic/gender_submission.csv') # 提出用のcsvファイルらしい
gender_submission.head()
test.head()
train.head()
# sort = Trueは、カラムがABC順にソートされる

data = pd.concat([train, test], sort=False)

data.head()
print(len(train), len(test), len(data)) # 891+418=1309
data.isnull().sum()
# inplaceは、変換後のpandas.DataFrameを置き換えるか

data["Sex"].replace(['male', 'female'], [0, 1], inplace=True)
# 欠損値は、最頻値のSで補完

data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
# 補完する値は、Fareの平均

data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
age_avg = data['Age'].mean()

age_std = data['Age'].std()



data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)
delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']

data.drop(delete_columns, axis = 1, inplace = True)
# これまで、trainとtestを連結して処理してきたdataを分割する

train = data[:len(train)]

test = data[len(train):]
y_train = train['Survived']

X_train = train.drop('Survived', axis=1)



X_test = test.drop('Survived', axis=1)
X_train.head()
y_train.head()
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l2', solver='sag', random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred[:20]
sub = pd.read_csv('../input/titanic/gender_submission.csv')

# subのSurvivedに、y_predを整数にした値を代入

sub['Survived'] = list(map(int, y_pred))

# ファイル保存時に、pandas.DataFrameの行のindex番号を付与しないようにするFalse

sub.to_csv('submission.csv', index=False) # outputのファイルにsubmission.csvが入っている