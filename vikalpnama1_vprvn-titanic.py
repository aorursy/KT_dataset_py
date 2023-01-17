import numpy as np 

import pandas as pd 

from sklearn.neighbors import KNeighborsClassifier



train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



#print  (train.info())

#Now we extract the titles

train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=True)

#10





#train['Title'].unique()



for title, age in train.groupby('Title')['Age'].median().iteritems():

    #print(title, age)

    train.loc[(train['Title']==title) & (train['Age'].isnull()), 'Age'] = age





#Now we remove the elements that do not affect the survivability  of the passenger

needless_elements = ['Ticket', 'Fare', 'Embarked', 'Cabin', 'Name', 'SibSp', 'Parch'] 

train = train.drop(needless_elements, axis = 1)





#train.isnull().sum()

#We have a total of 18 titles. We will sort them into 5 main categories to ease processing

tsort = {'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 

         'Master': 'Master', 'Don': 'Mr', 'Rev': 'Mr',

         'Dr': 'Dr', 'Mme': 'Miss', 'Ms': 'Miss',

         'Major': 'Mr', 'Lady': 'Mrs', 'Sir': 'Mr',

         'Mlle': 'Miss', 'Col': 'Mr', 'Capt': 'Mr',

         'Countess': 'Mrs','Jonkheer': 'Mr',

         'Dona': 'Mrs'}

train['Title'] = train['Title'].map(tsort)

#train['Title'].unique()



#Now we get to Data Cleaning and Data Mapping

#Gender Mapping

train['Sex'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



#Title Mapping

tmap = {"Mr": 1, "Mrs": 2, "Master": 3, "Miss": 4, "Dr": 5}

train['Title'] = train['Title'].map(tmap)

train['Title'] = train['Title'].fillna(0)



#Age Mapping

train.loc[ train['Age'] <= 18, 'Age']       			       = 0

train.loc[(train['Age'] > 18) & (train['Age'] <= 28), 'Age'] = 1

train.loc[(train['Age'] > 28) & (train['Age'] <= 36), 'Age'] = 2

train.loc[(train['Age'] > 36) & (train['Age'] <= 58), 'Age'] = 3

train.loc[ train['Age'] > 58, 'Age']                           = 4



#print (train.info())

#train['Pclass'].unique()







#58



#Now we do the same for TEST

test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=True)







#train['Title'].unique()



for title, age in test.groupby('Title')['Age'].median().iteritems():

    #print(title, age)

    test.loc[(test['Title']==title) & (test['Age'].isnull()), 'Age'] = age

test = test.fillna(3)



#Now we remove the elements that do not affect the survivability  of the passenger

needless_elementsa = ['Ticket', 'Fare', 'Embarked', 'Cabin', 'Name', 'SibSp', 'Parch'] 

test = test.drop(needless_elementsa, axis = 1)







#We have a total of 18 titles. We will sort them into 5 main categories to ease processing

ttsort = {'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 

         'Master': 'Master', 'Don': 'Mr', 'Rev': 'Mr',

         'Dr': 'Dr', 'Mme': 'Miss', 'Ms': 'Miss',

         'Major': 'Mr', 'Lady': 'Mrs', 'Sir': 'Mr',

         'Mlle': 'Miss', 'Col': 'Mr', 'Capt': 'Mr',

         'Countess': 'Mrs','Jonkheer': 'Mr',

         'Dona': 'Mrs'}

test['Title'] = test['Title'].map(ttsort)

#test['Title'].unique()



#Now we get to Data Cleaning and Data Mapping

#Gender Mapping

test['Sex'] = test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



#93

#Title Mapping

ttmap = {"Mr": 1, "Mrs": 2, "Master": 3, "Miss": 4, "Dr": 5}

test['Title'] = test['Title'].map(ttmap)

test['Title'] = test['Title'].fillna(0)



#Age Mapping

test.loc[ test['Age'] <= 18, 'Age']       			       = 0

test.loc[(test['Age'] > 18) & (test['Age'] <= 28), 'Age'] = 1

test.loc[(test['Age'] > 28) & (test['Age'] <= 36), 'Age'] = 2

test.loc[(test['Age'] > 36) & (test['Age'] <= 58), 'Age'] = 3

test.loc[ test['Age'] > 58, 'Age']                           = 4



#print(train.head(10))

#print(test.head(10))

#test['Pclass'].unique()





train_y = train.Survived

predi = ['Pclass', 'Age', 'Sex', 'Title']



train_X = train[predi]



my_model = KNeighborsClassifier()



my_model.fit(train_X, train_y)



#########

test_X = test[predi]



survi = my_model.predict(test_X)

#print(survi)



sub = pd.DataFrame({'PassengerId' : test.PassengerId, 'Survived' : survi})



sub.to_csv('VPRVN_Titanic.csv', index = False)
