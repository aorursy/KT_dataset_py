#data analysis
import pandas as pd
import numpy as np
import random as rnd
#visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib  inline
#machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn .ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
combine = [train, test]

print(train.columns.values)

train.head()
train.tail()
train.info()
test.info()
train.describe()
train.describe(include=['O'])
train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
train[['Sex','Survived']].groupby('Sex',as_index=False).mean().sort_values(by='Survived',ascending=False)
train[['SibSp','Survived']].groupby('SibSp',as_index=False).mean().sort_values(by='Survived',ascending=False)
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g=sns.FacetGrid(train,col='Survived')
g.map(plt.hist,'Age',bins=20)
train.head()
g1=sns.FacetGrid(train,col='Survived',row='Pclass',size=2.2,aspect=1.6)
g1.map(plt.hist,'Age',alpha=0.5,bins=20)
g1.add_legend();
g2=sns.FacetGrid(train,row='Embarked',size=2.2,aspect=1.6)
g2.map(sns.pointplot,'Pclass','Survived','Sex',alpha=0.9,palette='deep')
g2.add_legend()
g3=sns.FacetGrid(train,row='Embarked',col='Survived',size=2.2,aspect=1.6)

g3.map(sns.barplot,'Sex','Fare',alpha=0.5,ci=None)
g3.add_legend()
train.head()
print("Before", train.shape, test.shape, combine[0].shape, combine[1].shape)

train=train.drop(['Ticket','Cabin'],axis=1)
test=test.drop(['Ticket','Cabin'],axis=1)
combine = [train, test]

"After", train.shape, test.shape, combine[0].shape, combine[1].shape

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    
dataset['Title']=dataset['Title'].replace('Mlle','Miss')
dataset['Title']=dataset['Title'].replace('Ms','Miss')
dataset['Title']=dataset['Title'].replace('Mme','Mrs')

train[['Title','Survived']].groupby('Title',as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()
train.Title.value_counts()
train=train.drop(['Name','PassengerId'],axis=1)
test=test.drop(['Name'],axis=1)
combine = [train, test]
train.shape,test.shape
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {"female": 1, "male": 0}).astype(int)

train.head()
g=sns.FacetGrid(train,row='Pclass',col='Sex',size=2.2,aspect=1.6)
g.map(plt.hist,'Age',alpha=0.7,bins=20)
g.add_legend()
guess_ages=np.zeros((2,3))
guess_ages
for dataset in combine:
    for i in range(0,2):
        for j in range(0,3):
            guess_df=dataset[(dataset['Sex']==i)&\
                            (dataset['Pclass']==j+1)]['Age'].dropna()
            
            age_guess=guess_df.median()
            guess_ages[i,j]=int(age_guess/0.5+0.5)*0.5
            
    for i in range(0,2):
        for j in range(0,3):
            dataset.loc[(dataset.Age.isnull())&(dataset.Sex==i)&(dataset.Pclass==j+1),\
                       'Age']=guess_ages[i,j]
            
    dataset['Age']=dataset['Age'].astype(int)
    
train.head()

train['AgeBand']=pd.cut(train['Age'],5)
train[['AgeBand','Survived']].groupby(['AgeBand'],as_index=False).mean().sort_values(by='AgeBand',ascending=True)
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train.head()
train=train.drop(['AgeBand'],axis=1)
combine=[train,test]
train.head()
for dataset in combine:
    dataset['FamilySize']=dataset['SibSp']+dataset['Parch']+1
    
train[['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).mean().sort_values(by='Survived',ascending =False)
for dataset in combine:
    dataset['IsAlone']=0
    dataset.loc[dataset['FamilySize']==1,'IsAlone']=1
    
train[['IsAlone','Survived']].groupby(['IsAlone'],as_index=False).mean().sort_values(by='Survived',ascending=False)
train = train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test = test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train, test]

train.head()
for dataset in combine:
    dataset['Age*Class']=dataset.Age*dataset.Pclass

train.loc[:,['Age*Class','Age','Pclass']].head(10)
freq_port=train.Embarked.dropna().mode()[0]
freq_port
for dataset in combine:
    dataset['Embarked']=dataset['Embarked'].fillna(freq_port)

train[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} )

train.head()
test['Fare'].fillna(test['Fare'].dropna().median(),inplace=True)
train['FareBand']=pd.qcut(train['Fare'],4)
train[['FareBand','Survived']].groupby(['FareBand'],as_index=False).mean().sort_values(by='FareBand',ascending =True)
for dataset in combine:
    dataset.loc[dataset['Fare']<=7.91,'Fare']=0
    dataset.loc[(dataset['Fare']>7.91)&(dataset['Fare']<=14.454),'Fare']=1
    dataset.loc[(dataset['Fare']>14.454)&(dataset['Fare']<=31),'Fare']=2
    dataset.loc[dataset['Fare']>31,'Fare']=3
    dataset['Fare']=dataset['Fare'].astype(int)
    
train=train.drop(['FareBand'],axis=1)
combine=[train,test]
train.head()
X_train=train.drop("Survived",axis=1)
Y_train=train["Survived"]
X_test=test.drop("PassengerId",axis=1).copy()
X_train.shape,Y_train.shape,X_test.shape
logreg=LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred=logreg.predict(X_test)
acc_log=round(logreg.score(X_train,Y_train)*100,2)
acc_log
coeff=pd.DataFrame(train.columns.delete(0))
coeff.columns=['Feature']
coeff["Correlation"]=pd.Series(logreg.coef_[0])

coeff.sort_values(by='Correlation',ascending=False)
#Support Vector Machines
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred=svc.predict(X_test)
acc_svc=round(svc.score(X_train,Y_train)*100,2)
acc_svc
#KNN Algorithm
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)
Y_pred=knn.predict(X_test)
acc_knn=round(knn.score(X_train,Y_train)*100,2)
acc_knn
#Gaussian Naive Bayes
gaussian=GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
decision_tree=DecisionTreeClassifier()
decision_tree.fit(X_train,Y_train)
Y_pred=decision_tree.predict(X_test)
acc_decision_tree=round(decision_tree.score(X_train,Y_train)*100,2)
acc_decision_tree
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes',
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest,acc_gaussian, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
