#Importing pandas and matplotlib

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
#Creating train and test dataframes 

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

#First 5 rows of the training data

train.head()
#First 5 rows of the test data

test.head()
#Merging train and test dataframes into a single dataframe

titanic=pd.concat([train,test],axis=0,ignore_index=True)
#Total number of values in each column

titanic.count()
#Converting the unordered categorical 'Sex'

titanic.Sex=titanic.Sex.map({'male':1,'female':0})
#Total number of passengers and survivors by gender

train.groupby('Sex')['Survived'].agg({'Sex':['count'],'Survived':['sum']})
titanic.loc[train.index].info() #Training set Info
titanic.loc[train.index].describe() #Properties of the training set
#Creating a new feature 'has_Family'

#'has_Family' tells if a passenger is part of a family or not

titanic['has_Family']=((titanic.Parch!=0) | (titanic.SibSp!=0)).map({True:1,False:0})
titanic.has_Family.value_counts()
#Does travelling with family increase the survival chances?

titanic.loc[train.index].groupby('has_Family')['has_Family','Survived'].agg({'has_Family':['count'],'Survived':['sum']})
#Visualizing survival by has_Family

titanic.loc[train.index].Survived.hist(by=titanic.has_Family.map({0:'Without Family',1:'With Family'}),layout=(2,1),sharex=True)

plt.xticks([0,1],['Did not survive','Survived'])
#Visualizing survival by Pclass

train.Survived.hist(by=train.Pclass,layout=(3,1),sharex=True)

plt.xticks([0,1],['Did not survive','Survived'])
titanic.loc[train.index].Survived.hist(by=titanic.Embarked,layout=(3,1),sharex=True)

plt.xticks([0,1],['Did not survive','Survived'])
#Importing colormap

import matplotlib.cm as cm
#Scatterplot to visualize relationship between Age,Pclass and Survival

titanic.loc[train.index].plot(x='Age',y='Survived',c='Pclass',cmap=cm.hot,kind='scatter',figsize=(10,5))

plt.yticks([0,1])

plt.xticks(range(10,101,10))
#Visualizing age distribution

train[train.Survived==1].Age.hist(bins=10,normed=True)
#Embarked has two missing values -> Deleting those two rows

titanic.dropna(subset=['Embarked'],inplace=True)
#Imputing missing Age values with the mean Age

titanic.Age=titanic.Age.fillna(value=train.Age.mean())
#Since Cabin doesn't seem like an important feature, we're dropping it

titanic.drop('Cabin',axis=1)
#Using get_dummmies() method to convert Embarked into variables that can be used as features 

embarked_dummies=pd.get_dummies(titanic.Embarked,prefix='Embarked')
#Adding embarked_dummies to the titanic dataframe

titanic=pd.concat([titanic,embarked_dummies],axis=1)
#Data with the new columns

titanic.head()
#Selecting our features

features=['Pclass','Sex','Age','has_Family','Embarked_C','Embarked_Q','Embarked_S']
#Training set

X_train=titanic[titanic.Survived.notnull()][features]  #Features

y_train=titanic[titanic.Survived.notnull()].Survived  #Response
#Test set

X_test=titanic[titanic.Survived.isnull()][features]
#Using Logistic Regression for classification

from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

logreg.fit(X_train,y_train)
#Prediction

y_pred_class=logreg.predict(X_test)