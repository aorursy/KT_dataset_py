# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import re

from sklearn import preprocessing

import sklearn.ensemble as ske

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
test = pd.read_csv('../input/test.csv', dtype={'Age':np.float64})

train = pd.read_csv('../input/train.csv', dtype={'Age':np.float64})
train.info()
test.info()
test['Survived']=''
test.info()
test.head()
data = pd.concat([train, test], ignore_index=True)
data.head()
data.info()
((data['Name'][0].split(',')[1]).split()[0]).split('.')[0]
data['Title'] = data['Name'].apply(lambda x: (x.split(',')[1]).split()[0])
data['Title'].unique()
data['Title'][0].split('.')[0]
data['Title'].value_counts()
NA_age = np.array(['Ms.', 'Dr.', 'Master.', 'Mrs.', 'Miss.', 'Mr.'])

NA_age
data['Age'][data['Title']=='Mr.'].isnull()[0]
data['Age'][data['Title']=='Mr.'].mean()

        
mean_age = {}

for title in NA_age:

    mean_age[title] = data['Age'][data['Title']==title].mean()
mean_age
a = data[['Age','Title']]

a.loc[0][1]
def impute_age(row):

    if np.isnan(row['Age']) == True:

        return mean_age[row['Title']]

    else:

        return row['Age']
data['Age'] = data[['Age', 'Title']].apply(impute_age, axis=1)
data.head(10)
data.info()
data['Embarked'][data['Embarked'].isnull()] = 'S'
data['Embarked'][data['Embarked'].isnull()]
data['Fare'][data['Fare'].isnull()] = data[data['Pclass']==3]['Fare'].mean()
data['Fare'][data['Fare'].isnull()]
data['Family'] = data['SibSp'] + data['Parch'] + 1
data.drop(['SibSp', 'Parch', 'Name', 'Cabin'], axis=1, inplace=True)
data.head()
data['Ticket']
r1 = r'(.*) '

r2 = r' ([0-9]*)'
a = data['Ticket'][0]

b = data['Ticket'][1]

c = data['Ticket'][12]
re.sub(r1, '', a)
re.sub(r2, '', a)
re.sub(r1, '', b)
re.sub(r2, '', b)
re.sub(r1, '', c)
re.sub(r2, '', c)
data['Ticket_Pre'] = data['Ticket'].apply(lambda x: re.sub(r2, '', x))
data['Ticket_Num'] = data['Ticket'].apply(lambda x: re.sub(r1, '', x))
data.head(-25)
type(data['Ticket_Pre'][3])
data['Ticket_Pre'][3] == data['Ticket_Num'][3]
def remove_num_pre(row):

    if row['Ticket_Pre'] == row['Ticket_Num']:

        return 'None'

    else:

        return row['Ticket_Pre']
data['Ticket_Pre'] = data[['Ticket_Pre', 'Ticket_Num']].apply(remove_num_pre, axis=1)
data.head(10)
data['Ticket_Pre'] = data['Ticket_Pre'].apply(lambda x: x.split('.')[0])
data.head(10)
data['Ticket_Pre'].value_counts().sort_index()
data['Ticket_Pre'].replace(to_replace='STON/O', value='SOTON/O', inplace=True)
data['Ticket_Pre'].replace(to_replace='SC/Paris', value='SC/PARIS', inplace=True)
data['Ticket_Pre'].replace(to_replace='STON/OQ', value='SOTON/O', inplace=True)
data['Ticket_Pre'].replace(to_replace='STON/O2', value='SOTON/O2', inplace=True)
data['Ticket_Pre'].replace(to_replace='A/S', value='A/5', inplace=True)
data['Ticket_Pre'].replace(to_replace='A4', value='A/4', inplace=True)
data['Ticket_Pre'].replace(to_replace='P/PP', value='PP', inplace=True)
data.drop(['Ticket'], axis=1, inplace=True)
data.info()
objects = data.dtypes[data.dtypes == 'object']
objects.drop(['Survived'], inplace=True)

objects
le = preprocessing.LabelEncoder()

for var in objects.index:

    data[var] = le.fit_transform(data[var])
data.head()
train = data[data['Survived']!='']

test = data[data['Survived']=='']
X_train = train.drop(['Survived', 'PassengerId'], axis=1)

y_train = list(train['Survived'].values)
X_test = test.drop(['Survived', 'PassengerId'], axis=1)
rf = ske.RandomForestClassifier(n_estimators=50)

rf.fit(X_train, y_train)

pred = rf.predict(X_test)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred})

submission.to_csv('rf_titanic.csv', index=False)