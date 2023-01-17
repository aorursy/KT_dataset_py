import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

import warnings



# Reading dataset

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



# Adding a column in each dataset before merging

train['kind'] = 'train'

test['kind'] = 'test'



# Merging train and test

dataset = train.append(test,sort=True).reset_index()

# Prepare data

print('Nan in data:\n',dataset.isnull().sum())



# clear unneed nan. (Embarked 2 nan, Fare 1 nan)

dataset['Embarked'].fillna('C', inplace = True)

dataset['Fare'].fillna(dataset['Fare'].mean(), inplace = True)





# Change Cabin to A...P Without the number 

def changeToLetter(letter,text):

    if pd.isnull(text):

        return text

    isContins = letter in text.lower()

    return letter if isContins  else text

        

for char in range(ord('a'), ord('p') + 1):

    dataset['Cabin'] = dataset['Cabin'].apply((lambda x: changeToLetter(chr(char),x)))
# Analysis data

sns.set()



# male Vs Female survived plot

groupby_sex = dataset.groupby('Sex')['Survived'].sum()

groupby_sex.plot.bar(title = "Male Vs Female Survived")

plt.show()



print(groupby_sex,'\n')

print("Much more female was saved than male - more then double")



dataset = pd.get_dummies(dataset,columns = ['Sex'],drop_first = True)
# Age survived plot

grid = sns.FacetGrid(train, col="Survived",height=5)

grid.map(sns.distplot,'Age',bins = 30);

plt.show()

print("The survived and not survived ages is between 20-40")

print("But we can see that the survived ages was also in ages 0-10 - children")
# Pclass survived plot

not_survived_Pclass= train.loc[(train['Survived'] == False),'Pclass']

survived_Pclass = train.loc[(train['Survived'] == True),'Pclass']

sns.kdeplot(not_survived_Pclass,shade=True,color='Red', label='Not Survived')

sns.kdeplot(survived_Pclass,shade=True,color='Green', label='Survived')

labels = ['Upper', 'Middle', 'Lower']

plt.xticks(sorted(train.Pclass.unique()), labels);



plt.show()

print("Must of the not survivedrs was in the Lower floor")

print("Must of the surviveders was in the Upper floor")





dataset = pd.get_dummies(dataset,columns = ['Pclass'])
# cabin survived plot



cabin =  dataset.dropna()

not_survived = cabin[cabin['Survived'] == False]

survived = cabin[cabin['Survived'] == True]



not_survived.groupby('Cabin').size().plot.bar(title = "Not Survived",color = 'red')

plt.show()



survived.groupby('Cabin').size().plot.bar(title = "Survived",color = 'blue')

plt.show()
# Create Cabin classification model



cabin_dataset= dataset.dropna()

cabin_dataset = cabin_dataset[['Fare','Cabin','Ticket','Pclass_1','Pclass_2','Pclass_3']] # Feature for Cabin prediction

# Create PC ticket feature and remove Ticket column

cabin_dataset['Is_PC_Ticket'] = cabin_dataset['Ticket'].str.contains('PC')

cabin_dataset.drop('Ticket',inplace = True, axis = 1)



X = cabin_dataset.drop('Cabin',axis = 1)

y = cabin_dataset['Cabin']



X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 666)



scoreTest = []

scoreTrain = []

for number in range(1,30):

    cls = DecisionTreeClassifier(max_depth = number)

    cls.fit(X_train,y_train)

    scoreTest.append(accuracy_score(y_true=y_test, y_pred=cls.predict(X_test)))

    scoreTrain.append(accuracy_score(y_true=y_train, y_pred=cls.predict(X_train)))

pd.DataFrame({'test score':scoreTest,'train score':scoreTrain}).plot(grid = True)

plt.xlabel('Max depth')

plt.ylabel('Score')

plt.show()
cabin_cls = DecisionTreeClassifier(max_depth = 7)

cabin_cls.fit(X_train,y_train)

print('Train accuracy score:',accuracy_score(y_true=y_train, y_pred=cabin_cls.predict(X_train)))

print('Test accuracy score',accuracy_score(y_true=y_test, y_pred=cabin_cls.predict(X_test)))
# Add missing cabin data by Cabin classification model



toPrdicData = dataset[['Fare','Cabin','Ticket','Pclass_1','Pclass_2','Pclass_3']].copy()

none_change_data = toPrdicData[pd.isnull(toPrdicData['Cabin']) == False]

toPrdicData = toPrdicData[pd.isnull(toPrdicData['Cabin']) == True]

toPrdicData['Is_PC_Ticket'] = toPrdicData['Ticket'].str.contains('PC')

toPrdicData.drop('Ticket',inplace = True, axis = 1)

toPrdicX = toPrdicData.drop('Cabin',axis = 1)

toSet = pd.DataFrame(cabin_cls.predict(toPrdicX),index = toPrdicX.index)

toSet.rename(columns ={0 :'Cabin'}, inplace=True)

cabin_prdict_data = pd.concat([toSet,none_change_data],sort=False).sort_index()

dataset['Cabin'] = cabin_prdict_data['Cabin']



cabin_mapper = {'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7}

dataset['Cabin'] = dataset['Cabin'].map(cabin_mapper)

dataset['Cabin'].fillna(0, inplace = True)
survived_df = dataset[dataset['Survived'] == True]

survived_df_f = dataset[dataset['Survived'] == False]



survived_df.groupby('Cabin').size().plot.barh(title = "survived title count")

plt.show()

survived_df_f.groupby('Cabin').size().plot.barh(title = "survived title count")



cabin_mapper = {0:0,1:0,2:0,3:0,4:0,5:1,6:1,7:1}

dataset['Cabin'] = dataset['Cabin'].map(cabin_mapper)
# Fare



train['FareQ'] = pd.qcut(train['Fare'], 5)

fare_quarter = train[['FareQ', 'Survived']].groupby(['FareQ'], as_index=False).mean().sort_values(by='FareQ', ascending=True)

fare_quarter.set_index('FareQ', inplace = True)

fare_quarter.plot.bar()

labels = ['a', 'b', 'c','d','e']



# Change Fare by quarters



def get_fare_by_quarters(fare):

    if fare <= 7.285:

        return 0

    elif fare <= 10.5:

        return 1

    elif fare <= 21.678:

        return 2

    elif fare <= 39.988:

        return 3

    else:

        return 4



dataset['Fare'] = dataset['Fare'].apply(get_fare_by_quarters)
# name 



# Extract all the titles from name

dataset['Name_Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



print(dataset.groupby('Name_Title').size().index)



# Mrage the same title

dataset['Name_Title'].replace('Mlle', 'Miss', inplace = True)    

dataset['Name_Title'].replace('Ms', 'Miss', inplace = True)

dataset['Name_Title'].replace('Mme', 'Mrs', inplace = True)



survived_df = dataset[dataset['Survived'] == True]

not_survived_df = dataset[dataset['Survived'] == False]



survived_df.groupby('Name_Title').size().plot.barh(title = "survived title count")

plt.show()



not_survived_df.groupby('Name_Title').size().plot.barh(title = "not survived title count")

plt.show()



# Get only the important feature and replace with number ()

title_feature_map = {"Miss": 1, "Mrs": 2,"Mr": 3, "Master": 4} 





dataset['Name_Title'] = dataset['Name_Title'].map(title_feature_map)

dataset['Name_Title'].fillna(0, inplace = True)



# change check

print(dataset['Name_Title'].unique())
# fill missing age with midain 

def getAgeIfNeeded(age):

    if pd.isnull(age):

        return dataset['Age'].median()

    return age



dataset['Age'] = dataset['Age'].apply(getAgeIfNeeded)
# SibSp and Parch



dataset['Family_Count'] = dataset['SibSp'] + dataset['Parch']

dataset.groupby('Family_Count').mean()['Survived'].plot.bar(title = 'Family count survived')

plt.show()

#dataset[['Family_Count', 'Survived']].groupby(['Family_Count'], as_index=False).mean().sort_values(by='Survived', ascending=False).plot.bar()



def convertCountToGroups(family_count):

    if family_count == 0:

        return 'alone'

    elif family_count < 4:

        return 'small'

    else:

        return 'big'

    

dataset['Family_Count'] = dataset['Family_Count'].apply(convertCountToGroups)

dataset.groupby('Family_Count').mean()['Survived'].plot.bar(title = 'Family count survived')

plt.show()





dataset.drop(['SibSp','Parch'], axis=1, inplace = True)

dataset = pd.get_dummies(dataset,columns = ['Family_Count'])
dataset = pd.get_dummies(dataset,columns = ['Embarked'])
# remove unneeded feature

dataset.drop(['Ticket','Name','index'], axis=1, inplace = True)



# set back the train and test

train = dataset[dataset['kind'] == 'train'].copy()

test = dataset[dataset['kind'] == 'test'].copy()



train.drop(['kind','PassengerId'],axis = 1,inplace = True)

test.drop('kind',axis = 1,inplace = True)
# train the modle



X, y  = train.drop('Survived', axis = 1), train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=34)



scoreTest = []

scoreTrain = []

for number in range(1,100):

    cls = DecisionTreeClassifier(max_depth = number)

    cls.fit(X_train,y_train)

    scoreTrain.append(round(cls.score(X_train, y_train) * 100, 2))

    scoreTest.append(round(cls.score(X_test, y_test) * 100, 2))

pd.DataFrame({'test score':scoreTest,'train score':scoreTrain}).plot(grid = True)

plt.xlabel('Max depth')

plt.ylabel('Score')

plt.show()



cls = DecisionTreeClassifier(max_depth = 4)

cls.fit(X_train,y_train)

print('train score:',round(cls.score(X_train, y_train) * 100, 2))

print('test score:',round(cls.score(X_test, y_test) * 100, 2))
pd.DataFrame(X_train.columns, index = cls.feature_importances_)
train = dataset[dataset['kind'] == 'train'].copy()



train.drop(['kind','PassengerId','Embarked_C','Embarked_Q','Pclass_2','Cabin'],axis = 1,inplace = True)



X, y  = train.drop('Survived', axis = 1), train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=34)



scoreTest = []

scoreTrain = []

for number in range(1,100):

    cls = DecisionTreeClassifier(max_depth = number)

    cls.fit(X_train,y_train)

    scoreTest.append(round(cls.score(X_train, y_train) * 100, 2))

    scoreTrain.append(round(cls.score(X_test, y_test) * 100, 2))

pd.DataFrame({'test score':scoreTest,'train score':scoreTrain}).plot(grid = True)

plt.xlabel('Max depth')

plt.ylabel('Score')

plt.show()



cls = DecisionTreeClassifier(max_depth = 4)

cls.fit(X_train,y_train)

print('train score:',round(cls.score(X_train, y_train) * 100, 2))

print('test score:',round(cls.score(X_test, y_test) * 100, 2))
test = dataset[dataset['kind'] == 'test'].copy()

X_test = test.drop(['Survived','PassengerId','kind','PassengerId','Embarked_C','Embarked_Q','Pclass_2','Cabin'], axis = 1)

y_prdic = cls.predict(X_test)

submission = pd.DataFrame({

        "PassengerId": test['PassengerId'],

        "Survived": y_prdic

    })

submission.to_csv('submission.csv', index=False)