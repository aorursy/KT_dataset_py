import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
test.head()
train.head()
print(test.shape)

print(train.shape)
train.info()
test.info()
import matplotlib.pyplot as plt

import seaborn as sns 
def bar_chart(feature):

    survived = train[train['Survived']==1][feature].value_counts()

    dead = train[train['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index = ['Survived','Dead']

    df.plot(kind='bar',stacked=True, figsize=(10,5))
bar_chart('Sex')
bar_chart('Pclass')
bar_chart('SibSp')
bar_chart('Parch')
bar_chart('Embarked')
#Finding the NaN char in the DataFrame

train.isnull().sum()
test.isnull().sum()
#combining of two dataset i.e train and test 
train_test_data=[train,test]
#Cabin feature contain most NaN 

#so we drop the Cabin feature column 

train.drop('Cabin',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)
#extrating the Name string that ends with . into a column "Title"

for dataset in train_test_data:

    dataset['Title']=dataset['Name'].str.extract('([A-Za-z]+)\.',expand=False)
train.Title.value_counts()
#creating dictornary

title_mapping={'Mr':0,'Miss':1,'Mrs':2,'Master':3,'Dr':3,'Rev':3,'Mlle':3,'Major':3,'Col':3,"Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Mme": 3,"Capt": 3,"Sir": 3}



for dataset in train_test_data:

    dataset['Title']=dataset['Title'].map(title_mapping)
#missing value in Title feature

train.Title.fillna(0,inplace=True)
test.Title.fillna(0,inplace=True)
# Lets Analysis the new feature i.e 'Title'



bar_chart('Title')
train.drop('Name',axis=1,inplace=True)

test.drop('Name',axis=1,inplace=True)
sex_mapping={'male': 0, "female": 1}

for dataset in train_test_data:

    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
age_mean1=train.Age.mean()

age_mean1
train.Age.fillna(age_mean1,inplace=True)
age_mean2=test.Age.mean()
test.Age.fillna(age_mean2,inplace=True)
for dataset in train_test_data:

    dataset.loc[ dataset['Age'] <= 18, 'Age'] = 0,

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 26), 'Age'] = 1,

    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,

    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 49), 'Age'] = 3,

    dataset.loc[ dataset['Age'] > 49, 'Age'] = 4
train.Embarked.value_counts()
# maximum value goes with "S"

#so fill NaN value with "S" embark

train.fillna('S',inplace=True)
embarked_mapping = {"S": 0, "C": 1, "Q": 2}

for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
train['Fare'].groupby(train['Pclass']).mean()
test['Fare'].groupby(test['Pclass']).mean()
test.Fare.isnull().value_counts()
test[test.Fare.isnull()]
test.Fare.fillna(12,inplace=True)
for dataset in train_test_data:

    dataset.loc[dataset['Fare'] <=17,'Fare']=0

    dataset.loc[(dataset['Fare'] >17) & (dataset['Fare'] <=30),'Fare']=1

    dataset.loc[(dataset['Fare'] >30) & (dataset['Fare'] <=100),'Fare']=2

    dataset.loc[dataset['Fare'] >100,'Fare' ]=3
test.isnull().sum()
train.isnull().sum()
train.head()
test.head()
train.SibSp.value_counts()
train.Parch.value_counts()
train.drop('Ticket',axis=1,inplace=True)
test.drop('Ticket',axis=1,inplace=True)
train.drop('PassengerId',axis=1,inplace=True)
test.drop('PassengerId',axis=1,inplace=True)
train_data=train.drop('Survived',axis=1,inplace=False)
train_target=train.Survived
train.head()
train_data.head()
test_1=pd.read_csv('../input/gender_submission.csv')
test_1.drop('PassengerId',axis=1,inplace=True)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression 
model1=KNeighborsClassifier(n_neighbors=7)
model1.fit(train_data,train_target)
predicted1=model1.predict(test)
score1=model1.score(test,test_1)
print (f'Accuracy Score :{score1}') 
model2=DecisionTreeClassifier()
model2.fit(train_data,train_target)
predicted2=model2.predict(test)
score2=model2.score(test,test_1)
print(f'Accuracy Score:{score2}')
model3=RandomForestClassifier(n_estimators=100,max_depth=2,random_state=0)
model3.fit(train_data,train_target)
predicted3=model3.predict(test)
score3=model3.score(test,test_1)
print(f'Accuracy Score:{score3}')
model4=LogisticRegression()
model4.fit(train_data,train_target)
predicted=model4.predict(test)
score4=model4.score(test,test_1)
print(f'Accuracy score:{score4}')
print(f'KNeighborsClassifier :{score1*100} \nDecisionTreeClassifier:{score2*100} \nRandomForestClassifier:{score3*100}\nLogisticRegression:{score4*100}') 