import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
# Loading train and test datasets

Train = pd.read_csv('../input/titanic/train.csv')

Test = pd.read_csv('../input/titanic/test.csv')
#First five row

Train.head()
#First five row

Test.head()
Train.describe()
Test.describe()
Train.info()
Train.shape
#calculate the null values

print('Train Columns with null values : ', Train.isna().sum())

print('Test Columns with null values : ', Test.isna().sum())
data1=pd.concat([Train,Test],axis=0)
# correlation matix

g = sns.heatmap(data1[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True )
Train.Survived.value_counts()
#Survived vs Sex

Train.groupby(Train.Sex).Survived.value_counts().unstack().plot(kind= 'bar')
#Survived vs Pclass

print(Train.groupby(Train.Survived).Pclass.value_counts())

grid = sns.FacetGrid(Train, col='Survived', row='Pclass', size=2, aspect=1.8)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)
#Survived vs Parch

Train.groupby(Train.Parch).Survived.value_counts().unstack().plot(kind= 'bar')
#Survived vs Sibsp

Train.groupby(Train.SibSp).Survived.value_counts().unstack().plot(kind= 'bar')
#Survived vs Embarked

sns.barplot('Embarked', 'Survived', data=Train, color="teal")
train_test = [Train,Test]
for dataset in train_test:

    dataset['Title'] = dataset["Name"].str.extract('([A-Za-z]+)\.', expand=False )
Train.head()
for dataset in train_test:

    dataset['Title'] =dataset['Title'].map({'Mr':0,'Miss':1,'Mrs':2,'Master':0,'Dr':3,'Rev':3,'Mlle':3,'Col':3,'Major':3,'Mme':3,'Ms':3,'Don':3,'Capt':3,'Jonkheer':3,'Countess':3,'Lady':3,'Sir':3})
Train.head()
Train.groupby(Train.Title).Survived.value_counts().unstack().plot.bar() 
print(Train.Embarked.value_counts(dropna=False))
# filling missing values

for dataset in train_test:

    dataset.Embarked.fillna('S', inplace = True)

# I will impute the one missing value for fare with median    

for dataset in train_test:

    dataset['Fare'] = dataset['Fare'].fillna(Train['Fare'].median())
for dataset in train_test:

    dataset["Embarked"]= dataset.Embarked.map({'S':0,'C':1,'Q':2})
ag = Train["Age"].hist(bins=15, color='teal', alpha=0.8)

ag.set(xlabel='Age', ylabel='Count')
Train.groupby(pd.cut(Train.Age, bins=[0,25,50,75,100], labels= [0,1,2,3])).Survived.value_counts().plot(kind= 'bar', stacked = True)
for dataset in train_test:

    dataset['Age'].fillna(dataset.groupby('Title')['Age'].transform('median'),inplace = True)
# we map  age according to age band

for dataset in train_test:

    dataset['Age_Range']=pd.cut(dataset.Age, bins = (0,25,50,75,100), labels = [0,1,2,3])    

    
#family_size = number of family members, people travelling alone will have a value of 1

for dataset in train_test:

    dataset['Family_size'] = dataset['Parch'] + dataset['SibSp']+1
#making it in binary 

for dataset in train_test:

    dataset['Sex'] = dataset['Sex'].map({'male':0,'female':1})
Train.head()
Train.isna().sum()
Test.isna().sum()
Test.Title.fillna(0,inplace=True)
Test.isna().sum()
# drop the rows

for dataset in train_test:

    dataset.drop(['Name','Ticket','Cabin','Age'], axis = 1, inplace = True)
Train.head()
# Machine learning

from sklearn.model_selection import train_test_split #for split the data

from sklearn.metrics import accuracy_score  #for accuracy_score

from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

x = Train.drop(['PassengerId',"Survived"],axis=1)

y= Train["Survived"]

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1000)

X_train.shape,X_test.shape,y_train.shape,y_test.shape
# Kneighbor classifier

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=11)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)

accuracy_knn = round(accuracy_score(pred,y_test)*100,2)

print("The accuracy score of the KNeighborsClassifier : ",accuracy_knn)

kfold = KFold(n_splits=10,random_state=42)

cvs_knn = cross_val_score(knn,x,y,cv =10, scoring='accuracy')

print('The cross_val_score of the KNeighborsClassifier : ',round(cvs_knn.max()*100,2))

# Logistic Regression

from sklearn.linear_model import LogisticRegression

lg =LogisticRegression()

lg.fit(X_train,y_train)

pred = lg.predict(X_test)

accuracy_lr =round(accuracy_score(pred,y_test)*100,2)

print("The accuracy score of the LogisticRegression : ",accuracy_lr)

kfold = KFold(n_splits=10,random_state=22)

cvs_lr = cross_val_score(lg,x,y,cv =10, scoring='accuracy')

print('The cross_val_score of the LogisticRegression : ',round(cvs_lr.mean()*100,2))





#  Gaussian naive bayes

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,y_train)

pred = nb.predict(X_test)

accuracy_nb = round(accuracy_score(pred,y_test)*100,2)

print("The accuracy score of the Gaussian naive bayes classifier : ",accuracy_nb)

kfold = KFold(n_splits=10,random_state=22)

cvs_nb = cross_val_score(nb,x,y,cv =10, scoring='accuracy')

print('The cross_val_score of the Gaussian naive bayes classifier : ',round(cvs_nb.mean()*100,2))

# Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

dc = DecisionTreeClassifier(criterion='gini', 

                             min_samples_split=10,min_samples_leaf=1,

                             max_features='auto')

dc.fit(X_train,y_train)

pred = dc.predict(X_test)

accuracy_dc =round(accuracy_score(pred,y_test)*100,2)

print("The accuracy score of the DecisionTreeClassifier : ",accuracy_dc)

kfold = KFold(n_splits=10,random_state=22)

cvs_dc = cross_val_score(dc,x,y,cv =10, scoring='accuracy')

print('The cross_val_score of the DecisionTreeClassifier : ',round(cvs_dc.mean()*100,2))





# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

rm = RandomForestClassifier(criterion='gini', n_estimators=500,

                             min_samples_split=10,min_samples_leaf=1,

                             max_features='auto',oob_score=True,

                             random_state=1,n_jobs=-1)

rm.fit(X_train,y_train)

pred = rm.predict(X_test)

accuracy_rm=round(accuracy_score(pred,y_test)*100,2)

print("The accuracy score of the RandomForestClassifier : ",accuracy_rm)

kfold = KFold(n_splits=10,random_state=22)

cvs_rm = cross_val_score(rm,x,y,cv =10, scoring='accuracy')

print('The cross_val_score of the RandomForestClassifier : ',round(cvs_rm.mean()*100,2))



# Linear SVC

from sklearn.svm import LinearSVC



svc = LinearSVC()

svc.fit(X_train, y_train)

pred = svc.predict(X_test)

accuracy_svc=round(accuracy_score(pred,y_test)*100,2)

print("The accuracy score of the Linear_svc : ",accuracy_svc)

kfold = KFold(n_splits=10,random_state=22)

cvs_svc = cross_val_score(rm,x,y,cv =10, scoring='accuracy')

print('The cross_val_score of the Linear_svc : ',round(cvs_svc.mean()*100,2))
# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()

gbc.fit(X_train, y_train)

pred = gbc.predict(X_test)

accuracy_gbc=round(accuracy_score(pred,y_test)*100,2)

print("The accuracy score of the RandomForestClassifier : ",accuracy_gbc)

kfold = KFold(n_splits=10,random_state=22)

cvs_gbc = cross_val_score(rm,x,y,cv =10, scoring='accuracy')

print('The cross_val_score of the RandomForestClassifier : ',round(cvs_gbc.mean()*100,2))
models = pd.DataFrame({

    'Model': [ 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes',  

              'Decision Tree','GradientBoostingClassifier','LinearSVC'],

    'Score': [ accuracy_knn, accuracy_lr, 

              accuracy_nb, accuracy_dc,accuracy_rm,accuracy_gbc,accuracy_svc]})

models.sort_values(by='Score',ascending=False)
ids = Test['PassengerId']



pred = dc.predict(Test.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': pred })

output.to_csv('submission.csv', index=False)