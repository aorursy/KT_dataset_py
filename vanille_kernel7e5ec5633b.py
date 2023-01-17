# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')



train_x = train.drop(['Survived'], axis=1)

test_x = test.copy()

train_y = train['Survived']



# PassengerIdを除外

train_x = train_x.drop(['PassengerId'], axis=1)

test_x = test_x.drop(['PassengerId'], axis=1)



# Name, Ticket, Cabinを除外

train_x = train_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)

test_x = test_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)
# label encodingを行う

for c in ['Sex', 'Embarked']:

    # 学習データに基づいてどう変換するかを定める

    le = LabelEncoder()

    le.fit(train_x[c].fillna('NA'))

    

    train_x[c] = le.transform(train_x[c].fillna('NA'))

    test_x[c] = le.transform(test_x[c].fillna('NA'))
model = XGBClassifier(n_estimators=20, random_state=71)

model.fit(train_x, train_y)



pred = model.predict_proba(test_x)[:, 1]

pred_label = np.where(pred > 0.5, 1, 0)



submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label})

submission.to_csv('submission_first.csv', index=False)



submission