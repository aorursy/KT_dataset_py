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
train = pd.read_csv('../input/train.csv')

train.set_index('PassengerId',inplace = True)

test = pd.read_csv('../input/test.csv')

test_id = list(test['PassengerId'])

test.set_index('PassengerId',inplace = True)
train.info()
train.Parch.value_counts()
import seaborn as sns

import matplotlib.pyplot as plt
train['Deck'] = train['Cabin'].apply(lambda x: str(x)[0] if x != 'NaN' else 'N')
import matplotlib

matplotlib.style.use('ggplot')

plt.figure()

sns.countplot(x="Deck", data=train,hue ='Survived')
g = sns.FacetGrid(train, col="Survived")

g.map(plt.hist,'Fare')
sns.countplot(x = 'Pclass',data = train,hue = 'Survived')
sns.countplot(x = 'Survived',data = train,hue = 'Sex')
male_servived = sum((train['Survived'] == 1) & (train['Sex'] == 'male' )) / train.Sex.value_counts()[0]

female_servived = sum((train['Survived'] == 1) & (train['Sex'] == 'female' )) / train.Sex.value_counts()[1]

male_servived,female_servived
sns.factorplot(x="Pclass", hue="Sex", col="Survived",data=train, kind="count",size=4, aspect=.7)
features = ['Pclass','Sex_new','Age','Fare'] 

train.dropna(subset=['Age'],inplace = True)

test.dropna(subset=['Age','Fare'],inplace = True)

from sklearn import svm

train['Sex_new'] = train['Sex'].apply(lambda x: 0 if x == 'female' else 1)

test['Sex_new'] = test['Sex'].apply(lambda x: 0 if x == 'female' else 1)

X_train = train[features]

y_train = train['Survived']

X_test = test[features]

#y_test = test['Survived']

clf = svm.SVC().fit(X_train,y_train)

#train.Sex_new.value_counts()

clf.score(X_train,y_train)
from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble     import RandomForestClassifier

from sklearn.ensemble     import RandomForestRegressor

classifiers = {

    "SGD": SGDClassifier(alpha=100),

    "ASGD": SGDClassifier(average=True),

    "Passive-Aggressive I": PassiveAggressiveClassifier(loss='hinge', C=1.0),

    "SAG": LogisticRegression(solver='liblinear',C=1.e4 / train[features].shape[0]),

    "RF_C": RandomForestClassifier(max_depth=20,n_estimators=13),

    "RF_R": RandomForestRegressor(n_estimators=15,max_depth=22)

}
for classifier_type in classifiers.keys():  

    # Train classifier

    clf2 = classifiers[classifier_type]

    clf2.fit(X_train, y_train)

    print(classifier_type, ':', clf2.score(X_train,y_train))