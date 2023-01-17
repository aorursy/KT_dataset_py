# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")
data.head(300)
data.isnull().sum()
data.drop(['PassengerId','Ticket'], axis = 1, inplace = True)
data['title'] = data['Name'].apply(lambda x: x.split(', ')[1].split('. ')[0])
data['last_name'] = data['Name'].apply(lambda x: x.split(', ')[0])
# infer age from name title by group average
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(5, shuffle=False, random_state= 233)
data['age_title_enc'] = np.nan

for tr_ind, val_ind in kf.split(data, data.Survived):
    X_tr, X_val = data.iloc[tr_ind], data.iloc[val_ind]
    X_val['age_title_enc'] = X_val['title'].map(X_tr.groupby('title')['Age'].mean())
    data.iloc[val_ind] = X_val
prior = data['Age'].mean()
data['age_title_enc'].fillna(prior, inplace=True)
data['Age'].fillna(data['age_title_enc'], inplace = True)
data.drop('age_title_enc', axis = 1, inplace = True)
data[data['Cabin'].str.contains(pat = '[A-Z] [A-Z][0-9]*').fillna(False)]
data[data['Cabin'].str.contains(pat = 'T').fillna(False)]
data[data['Cabin'].str.contains(pat = '[A-Z][0-9]* ').fillna(False)]
data[data['Cabin'].str.contains(pat = '[A-Z][0-9]+ ').fillna(False)]
data['Cabin'].filter(regex = 'C')
data['cabin_count'] = data['Cabin'].apply(lambda x: len(str(x).split(' '))) # assuming those missing has one cabin
data['cabin_letter'] = data['Cabin'].apply(lambda x: 'NA' if pd.isnull(x) else set([i[0] for i in x.split(' ')]))
data[data['cabin_count'] > 1]
data['cabin_letter'] = data['Cabin'].str[0]
data['cabin_number'] = data['Cabin'].apply(lambda x:  np.nan if pd.isnull(x) else -999 if len(x) == 1 else x[1:])
data.cabin_letter.value_counts()
data['cabin_number'].value_counts()
data.groupby('Cabin')['Fare'].agg(['mean','std'])
data['']
# directly fed into sklearn cross validation.

data = pd.get_dummies(train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1))

X = data.drop('Survived',axis = 1)
y = data['Survived']
from sklearn.preprocessing import Imputer

X_i = Imputer(strategy = 'median', axis = 1).fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_i, y, test_size = .5, random_state = 233
)
xlf.fit(X_train, y_train)
xlf_prob = xlf.predict_proba(X_test)[:,1]
xlf_pred = xlf.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, auc

print(confusion_matrix(xlf_pred, y_test))
print(accuracy_score(xlf_pred, y_test))
importance = pd.DataFrame()
importance['Feature'] = X.columns
importance['importance'] = xlf.feature_importances_
importance.sort_values(ascending = False, by = 'importance')
X_2 = pd.get_dummies(pd.read_csv('../input/test.csv').drop(['PassengerId',
                                                            'Name', 'Ticket', 'Cabin'], axis = 1))
id_2 = pd.read_csv('../input/test.csv')['PassengerId']
submit = pd.DataFrame()

submit['PassengerId'] = id_2
submit['Survived'] = xlf.predict(np.array(X_2))

submit.to_csv('plain_xgb_2.csv', index = False)