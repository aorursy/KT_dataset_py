# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

sns.set_style('whitegrid')

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.neighbors import KNeighborsClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data.head()
test_data.head()
print('Training set shape:', train_data.shape)

print('Testing set shape: ', test_data.shape)
train_data.describe()
sns.heatmap(train_data.corr(), annot=True);
train_data.groupby('Pclass')['Survived'].value_counts().plot(kind='bar');
sns.pairplot(train_data.dropna());
sns.distplot(train_data['Age'].dropna(),kde=False,color='darkred',bins=30);
sns.boxplot(x='Pclass',y='Age',data=train_data,palette='winter')
train_data.isnull().sum()
test_data.isnull().sum()
total = train_data.isnull().sum().sort_values(ascending=False)

percent_1 = train_data.isnull().sum()/train_data.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

missing_data.head()
sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train_data['Embarked'].value_counts()
train_data['Embarked'] = train_data['Embarked'].fillna('S')
test_data['Embarked'] = test_data['Embarked'].fillna('S')
train_data["Age"] = train_data["Age"].fillna(value=train_data["Age"].median())
test_data["Age"] = test_data["Age"].fillna(value=test_data["Age"].median())
train_data.head()
train_data["Fare"] = train_data["Fare"].fillna(value=train_data["Fare"].median())   
test_data["Fare"] = test_data["Fare"].fillna(value=test_data["Fare"].median())
train_data = train_data.drop('Cabin',axis=1)
test_data = test_data.drop('Cabin',axis=1)
def get_title(name):

    if '.' in name:

        return name.split(',')[1].split('.')[0].strip()

    else:

        return 'Unknown'



def title_number(title):

    if title == 'Mr':

        return 1

    elif title == 'Mrs':

        return 2

    elif title == 'Miss':

        return 3

    elif title == 'Master':

        return 4

    else:

        return 5  
train_data['get_title'] = train_data['Name'].apply(get_title).apply(title_number)
test_data['get_title'] = test_data['Name'].apply(get_title).apply(title_number)
train_data['get_title'].value_counts()
train_data.groupby('get_title')['Survived'].value_counts().plot(kind='bar')
train_data['Sex'] = train_data['Sex'].map({'male':0, 'female':1})
test_data['Sex'] = test_data['Sex'].map({'male':0, 'female':1})
def get_age(various):

    if various <= 11:

        return  0

    elif various <= 18: 

        return 1

    elif (various > 18) & (various <= 22):

        return 2

    elif (various > 22) & (various <= 27):

        return 3

    elif (various > 27) & (various <= 33):

        return 4

    elif (various > 33) & (various <= 40):

        return 5

    elif (various > 40) & (various <= 66):

        return 6

    else:

        return 7
train_data['get_age'] = train_data['Age'].apply(get_age)
test_data['get_age'] = test_data['Age'].apply(get_age)
train_d= train_data.copy()
sns.countplot(x='Survived', hue='get_age',data=train_d,palette='rainbow')
ages = pd.get_dummies(train_data['get_age'],prefix='age',drop_first=True)

ages_test = pd.get_dummies(test_data['get_age'],prefix='age',drop_first=True)
ages.shape
ages_test.shape
trainor = pd.concat([train_data,ages],axis=1)
testor= pd.concat([test_data,ages_test],axis=1)
print(trainor.shape)

print(testor.shape)
def get_embark(embark):

    if embark == 'S':

        return 0

    elif embark  == 'C':

        return 1

    else:

        return 2
trainor['Embarked'] = trainor['Embarked'].apply(get_embark)
testor['Embarked'] = testor['Embarked'].apply(get_embark)
def get_person(pers):

    age, sex = pers

    

    if age < 16: 

        return 3

    else: 

        return sex
trainor['person'] = trainor[['Age','Sex']].apply(get_person,axis=1)
testor['person'] = testor[['Age','Sex']].apply(get_person,axis=1)
trainor['person'].value_counts()
testor['person'].value_counts()
trainor[train_da['person']== 3].Survived.value_counts().plot(kind='barh')
trainor.hist('person');
trainor.head()
sns.countplot(x='Survived', hue='person',data=trainor,palette='rainbow')
print(train_da.shape)

print(test_da.shape)
X_train = trainor.drop(['PassengerId',"Survived",'Name','Age','Ticket','get_age'],axis=1)

Y_train = trainor["Survived"]

X_test  = testor.drop(['PassengerId','Name','Age','Ticket','get_age'],axis=1).copy()
X_train.head()
X_test.head()
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)
y_pred = logreg.predict(X_test)
logreg.score(X_train,Y_train)
y_pred
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3) 

knn.fit(X_train, Y_train)  

Y_pred = knn.predict(X_test)  

knn_round = round(knn.score(X_train, Y_train) * 100, 2)

knn_round
knn = KNeighborsClassifier(n_neighbors = 5) 

knn.fit(X_train, Y_train)  

Y_pred = knn.predict(X_test)  

knn_round = round(knn.score(X_train, Y_train) * 100, 2)

knn_round
submission = pd.DataFrame({

        "PassengerId": test_da["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic_submission.csv', index=False)
submit = pd.read_csv('titanic_submission.csv')

submit