import numpy as np 

import pandas as pd 

from matplotlib import pyplot as plt 

import seaborn as sns 
train=pd.read_csv("/kaggle/input/titanic/train.csv")

test=pd.read_csv("/kaggle/input/titanic/test.csv")
train.info()
train.describe(include="all")
train.head()
missin=train.isna().sum().sort_values(ascending=False)

total=len(train)

missin_per=missin/total*100

missin_final_obs=pd.concat([missin,missin_per],axis=1)

missin_final_obs.columns=["total missing","percentage missing"]

missin_final_obs
train.columns.values
women=train[train["Sex"]=="female"]

men=train[train["Sex"]=="male"]

w1=women[women["Survived"]==0].Age.dropna()

w2=women[women["Survived"]==1].Age.dropna()

m1=men[men["Survived"]==0].Age.dropna()

m2=men[men["Survived"]==1].Age.dropna()



fig , axes = plt.subplots(nrows=1,ncols=2,figsize=(10,4))



ax=sns.distplot(w1,bins=18,ax=axes[0],kde=False,label="not Survived")

ax=sns.distplot(w2,bins=18,ax=axes[0],kde=False,label="survived")

ax.legend()

ax.set_title("female survival distribution")



ax=sns.distplot(m1,bins=18,ax=axes[1],label="not survived",kde=False)

ax=sns.distplot(m2,bins=19,ax=axes[1],label="survived",kde=False)

ax.legend()

ax.set_title("male survival distribution")



g=sns.FacetGrid(train,col="Embarked")

g.map(sns.pointplot,'Pclass', 'Survived', 'Sex')
sns.barplot(x='Pclass', y='Survived', data=train)
g=sns.FacetGrid(train,row="Pclass",col="Survived")

g.map(plt.hist,"Age",bins=20)
data=[train,test]

for dataset in data:

    dataset["relatives"]=dataset["SibSp"]+dataset["Parch"]

    dataset.loc[dataset["relatives"]>0,"notrelated"]=0

    dataset.loc[dataset["relatives"]==0,"notrelated"]=1

    dataset["notrelated"]=dataset["notrelated"].astype(int)

    

    

train["notrelated"].value_counts()
g=sns.FacetGrid(train,row="notrelated",col="Sex")

g.map(plt.hist,"Survived",bins=20)
train=train.drop(["PassengerId"],axis=1)
import re

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

data=[train,test]

for dataset in data:

    dataset["Cabin"]=dataset["Cabin"].fillna("U0")

    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

    dataset['Deck'] = dataset['Deck'].map(deck)

    dataset['Deck'] = dataset['Deck'].fillna(0)

    dataset['Deck'] = dataset['Deck'].astype(int)

train=train.drop(["Cabin"],axis=1)

test=test.drop(["Cabin"],axis=1)
data=[train, test]

for dataset in data:

    mean=dataset["Age"].mean()

    std=dataset["Age"].std()

    isnull=dataset["Age"].isna().sum()

    randoms=np.random.randint(mean-std,mean+std,size=isnull)

    age_slice=dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = randoms

    dataset["Age"]=age_slice

    dataset["Age"]=dataset["Age"].astype(int)

print(dataset["Age"].isna().sum())
data=[train,test]

for dataset in data:

    dataset["Embarked"]=dataset["Embarked"].fillna("S")

   
# no more missing values



train.describe()
data=[train,test]

for dataset in data:

    dataset["Fare"]=dataset["Fare"].fillna(0)

    dataset["Fare"]=dataset["Fare"].astype(int)
data = [train, test]

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



for dataset in data:

    # extract titles

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # replace titles with a more common title or as Rare

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\

                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # convert titles into numbers

    dataset['Title'] = dataset['Title'].map(titles)

    # filling NaN with 0, to get safe

    dataset['Title'] = dataset['Title'].fillna(0)

train = train.drop(['Name'], axis=1)

test = test.drop(['Name'], axis=1)
genders={"male":0,"female":1}

data=[train,test]

for dataset in data:

    dataset["Sex"]=dataset["Sex"].map(genders)
train=train.drop(["Ticket"],axis=1)

test=test.drop(["Ticket"],axis=1)
embarks={"S":0,"C":1,"Q":2}

data=[train,test]

for dataset in data:

    dataset["Embarked"]=dataset["Embarked"].map(embarks)
train.info()
data = [train, test]

for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6



# let's see how it's distributed 

train['Age'].value_counts()
data = [train, test]



for dataset in data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3

    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4

    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5

    dataset['Fare'] = dataset['Fare'].astype(int)
train["Fare"].value_counts()
X_train = train.drop("Survived", axis=1)

Y_train = train["Survived"]

X_test  = test.drop("PassengerId", axis=1).copy()
X_train.isna().sum()
from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier
clf=MLPClassifier(hidden_layer_sizes=(1024,512,256))

clf.fit(X_train,Y_train)

Y_prediction = clf.predict(X_test)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)



Y_prediction = random_forest.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': Y_prediction})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")