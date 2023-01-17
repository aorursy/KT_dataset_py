# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import seaborn as sns



from sklearn.pipeline import Pipeline



from sklearn.preprocessing import StandardScaler





from sklearn.model_selection import train_test_split



from sklearn.model_selection import KFold





from sklearn.model_selection import cross_val_score





from sklearn.model_selection import GridSearchCV



from sklearn.metrics import classification_report



from sklearn.metrics import accuracy_score



from sklearn.metrics import confusion_matrix



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier





from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier



import matplotlib.pyplot as plt

%matplotlib inline
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

combine=[train,test]
train.head()
train.describe().transpose()
train.groupby('Survived').size()
train.isnull().sum()
sns.set_style('whitegrid')

sns.countplot(x='Survived',data= train, palette='RdBu_r')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='RdBu_r')
train['Age'].hist(bins=30,color='darkred',alpha=0.7)
sns.countplot(x='SibSp',data=train)
sns.countplot(x='Survived',hue='SibSp',data=train)
train['Fare'].hist(color='green',bins=35)
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)

test
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.drop('Cabin',axis=1,inplace=True)

test.drop('Cabin',axis=1,inplace=True)
train.head()
test.head()
train.isnull().sum()
train.dropna(inplace=True)
test.isnull().sum()
test[test['Fare'].isnull()]
test.shape
test
test.set_value(152,'Fare',50)
train.isnull().sum()

test.isnull().sum()
sex = pd.get_dummies(train['Sex'],drop_first=True) # getting dummy of 'Sex' c

sex = pd.get_dummies(train['Sex'],drop_first=True) # getting dummy of 'Sex' c

embark = pd.get_dummies(train['Embarked'],drop_first=True) # getting dummy of 'Embarked'

sex_test = pd.get_dummies(test['Sex'],drop_first=True) # getting dummy of 'Sex' column

embark_test = pd.get_dummies(test['Embarked'],drop_first=True) # getting dummy of 'Embarked'

embark_test
# drop columns: 'Sex', 'Embarked', 'Name','Ticket','PassengerId'

train.drop(['Sex','Name','Ticket','Embarked'],axis=1,inplace=True)



# for test

test.drop(['Sex','Name','Ticket','Embarked'],axis=1,inplace=True)



# for train

X_train = pd.concat([train,sex],axis=1)

# for test

X_test = pd.concat([test,sex_test],axis=1)

Y_train=X_train['Survived']

X_train.drop(columns=['Survived'],inplace=True)

test
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })
submission