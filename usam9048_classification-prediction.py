# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.info()
test.info()
train.columns
train.head()
train.isnull().sum()
test.isnull().sum()
train.Survived.value_counts().plot.pie(explode = [0,0.1], shadow = True, autopct = '%1.1f%%')

sns.countplot('Survived', data = train)
#Types of features

#categorical features



train.groupby(['Sex','Survived'])['Survived'].count()
f,ax = plt.subplots(1,2,figsize = (18,8))

train[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar(ax = ax[0], grid = True)

ax[0].set_title('sex vs survived')

sns.countplot('Sex',hue = 'Survived', data = train, ax= ax[1])
#Age as Factors

print('Oldest age was :', train['Age'].max(),'Years')

print('Youngest Passenger age was:', train['Age'].min(),'Years')

print('Average age was:', train['Age'].mean(),'years')

f,ax= plt.subplots(1,2,figsize= (18,8))

sns.violinplot("Pclass","Age", hue = "Survived", data = train,split = True, ax= ax[0], grid = True)

ax[0].set_title("Pclass and Age vs Survived")

sns.violinplot("Sex", "Age", hue = "Survived", data = train, split = True, ax=ax[1])
#name extract 

train['Initial'] = 0

for i in train:

    train['Initial']= train.Name.str.extract('([A-Za-z]+)\.')
pd.crosstab(train.Initial,train.Sex).T.style.background_gradient(cmap = 'summer_r')
train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
train.groupby('Initial')['Age'].mean()
## Assigning the NaN Values with the Ceil values of the mean ages

train.loc[(train.Age.isnull())&(train.Initial=='Mr'),'Age']=33

train.loc[(train.Age.isnull())&(train.Initial=='Mrs'),'Age']=36

train.loc[(train.Age.isnull())&(train.Initial=='Master'),'Age']=5

train.loc[(train.Age.isnull())&(train.Initial=='Miss'),'Age']=22

train.loc[(train.Age.isnull())&(train.Initial=='Other'),'Age']=46
train.Age.isnull().any() #So no null values left finally 
train['Embarked'].fillna('S',inplace =True)
train.Embarked.isnull().any()
pd.crosstab([train.SibSp],train.Survived).style.background_gradient(cmap='summer_r')


#Fare

print('Highest fare was:',train['Fare'].max())

print('Average fare:',train['Fare'].mean())

print('lowest fare:',train['Fare'].min())
#Observations in a Nutshell for all features:

#Sex: The chance of survival for women is high as compared to men.



#Pclass:There is a visible trend that being a 1st class passenger gives you better chances of survival. The survival rate for Pclass3 is very low. For women, the chance of survival from Pclass1 is almost 1 and is high too for those from Pclass2. Money Wins!!!.



#Age: Children less than 5-10 years do have a high chance of survival. Passengers between age group 15 to 35 died a lot.



#Embarked: This is a very interesting feature. The chances of survival at C looks to be better than even though the majority of Pclass1 passengers got up at S. Passengers at Q were all from Pclass3.



#Parch+SibSp: Having 1-2 siblings,spouse on board or 1-3 Parents shows a greater chance of probablity rather than being alone or having a large family travelling with you
sns.heatmap(train.corr(),annot=True,cmap='Accent',linewidths = 0.2)

fig = plt.gcf()

fig.set_size_inches(10,8)

plt.show()
train['Age_band'] = 0

train.loc[train['Age']<= 16,'Age_band']=0

train.loc[(train['Age']> 16) & (train['Age']<= 32), 'Age_band']= 1

train.loc[(train['Age']> 32) & (train['Age']<= 48), 'Age_band']= 2

train.loc[(train['Age']> 48) & (train['Age']<= 64), 'Age_band']= 3

train.loc[train['Age']> 64,'Age_band']=4

train.head(2)





train['Family_Size'] = 0

train['Family_Size']=train['Parch'] + train['SibSp']

train['Alone']=0

train.loc[train.Family_Size==0,'Alone']=1

train.head(3)





train['Fare_Range'] = pd.qcut(train['Fare'],4)

train.groupby(['Fare_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')
train['Fare_cat']=0

train.loc[train['Fare']<=7.91,'Fare_cat']=0

train.loc[(train['Fare']>7.91) & (train['Fare']<= 14.454),'Fare_cat']=1

train.loc[(train['Fare']>14.454) & (train['Fare']<=31.0), 'Fare_cat']=2

train.loc[train['Fare']>31.0,'Fare_cat']=3
sns.factorplot('Fare_cat', 'Survived', data = train, hue='Sex')

plt.show()

train['Sex'].replace(['male','female'],[0,1],inplace=True)

train['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

train['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)
#Converting String Values into Numeric

train['Sex'].replace(['male','female'],[0,1],inplace = True)

train['Embarked'].replace(['S','C','Q'],[0,1,2],inplace = True)

train['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace = True)
train.head(3)
train.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)

train.head()
sns.heatmap(train.corr(), annot = True, cmap = 'RdYlGn', linewidths = 0.2,annot_kws = {'Size': 20})

fig= plt.gcf()

fig.set_size_inches(18,8)

plt.xticks(fontsize = 14)

plt.yticks(fontsize = 14)

plt.show()
#Part 3 Prediction 

#loading ML packages

from sklearn.linear_model import LogisticRegression #Logistic Regression

from sklearn import svm #Support Vector machine

from sklearn.ensemble import RandomForestClassifier #Random Forest

from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.naive_bayes import GaussianNB #Naive Bayes

from sklearn.tree import DecisionTreeClassifier #Decision Tree

from sklearn.model_selection import train_test_split #data split

from sklearn import metrics #accuracy check

from sklearn.metrics import confusion_matrix #Confusion Matrix
Train,test=train_test_split(train,test_size=0.3,random_state=0,stratify=train['Survived'])

Train_X=Train[Train.columns[1:]]

Train_Y=Train[Train.columns[:1]]

test_X=test[test.columns[1:]]

test_Y=test[test.columns[:1]]

X=train[train.columns[1:]]

Y=train['Survived']
#RADIAL SVM

model = svm.SVC(kernel = 'rbf', C= 1, gamma = 0.1)

model.fit(Train_X,Train_Y)

pred1 = model.predict(test_X)

print('accuracy of rbf:', metrics.accuracy_score(pred1,test_Y))
#Linear SVM

model = svm.SVC(kernel = 'linear',C = 0.1, gamma = 0.1)

model.fit(Train_X,Train_Y)

pred2 = model.predict(test_X)

print('Accuracy of linear svm :',metrics.accuracy_score(pred2,test_Y))
#Logistic Regression

model = LogisticRegression()

model.fit(Train_X,Train_Y)

pred3 = model.predict(test_X)

print('The Accuracy of logistic regression is :', metrics.accuracy_score(pred3,test_Y))
#Decision Tree

model = DecisionTreeClassifier()

model.fit(Train_X,Train_Y)

pred4 = model.predict(test_X)

print("The accuracy of decision  tree classifier", metrics.accuracy_score(pred4,test_Y))
#KNN

model = KNeighborsClassifier()

model.fit(Train_X,Train_Y)

pred5 = model.predict(test_X)

print('The Accuracy of KNN:', metrics.accuracy_score(pred5,test_Y))
#find n neighbors

a_index = list(range(1,11))

a = pd.Series()

x = [0,1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,11)) :

    model = KNeighborsClassifier(i)

    model.fit(Train_X,Train_Y)

    pred6 = model.predict(test_X)

    a=a.append(pd.Series(metrics.accuracy_score(pred6,test_Y)))

plt.plot(a_index,a)

plt.xticks(x)

fig= plt.gcf()

fig.set_size_inches(12,6)

plt.show()

print('the accuracy for different values of n are:', a.values,'with the max value of',a.values.max())

    
#Gaussian Naive Bayes

model = GaussianNB()

model.fit(Train_X,Train_Y)

pred7 = model.predict(test_X)

print('the accuracy of', metrics.accuracy_score(pred7,test_Y))
#Random Forest

model = RandomForestClassifier(n_estimators = 100)

model.fit(Train_X,Train_Y)

pred8 = model.predict(test_X)

print('the accuracy of :',metrics.accuracy_score(pred8,test_Y))




from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts

xyz=[]

accuracy=[]

std=[]

classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest']

models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(n_estimators=100)]

for i in models:

    model = i

    cv_result = cross_val_score(model,X,Y, cv = kfold,scoring = "accuracy")

    cv_result=cv_result

    xyz.append(cv_result.mean())

    std.append(cv_result.std())

    accuracy.append(cv_result)

new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       

new_models_dataframe2



plt.subplots(figsize=(12,6))

box = pd.DataFrame(accuracy, index=[classifiers])

box.T.boxplot()
new_models_dataframe2['CV Mean'].plot.barh(width=0.8)

plt.title('Average CV mean accuracy')

fig = plt.gcf()

fig.set_size_inches(8,5)

plt.show()
#Confusion Matrix

f,ax=plt.subplots(3,3,figsize=(12,10))

y_pred = cross_val_predict(svm.SVC(kernel = 'rbf'),X,Y,cv = 10)

sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,0],annot=True,fmt='2.0f')

ax[0,0].set_title('Matrix for rbf-SVM')

y_pred = cross_val_predict(svm.SVC(kernel = 'linear'),X,Y,cv =10)

sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,1],annot=True,fmt = '2.0f')

ax[0,1].set_title('Matrix for linear SVM')

y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors = 9),X,Y,cv=10)

sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,2],annot=True,fmt = '2.0f')

ax[0,2].set_title('Matrix for KNN')

y_pred = cross_val_predict(LogisticRegression(),X,Y,cv = 10)

sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,0],annot = True,fmt = '2.0f')

ax[1,0].set_title('Matrix for Logistic regression')

y_pred = cross_val_predict(GaussianNB(),X,Y,cv=10)

sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,1],annot = True,fmt = '2.0f')

ax[1,1].set_title('Matrix of Gaussian Naive Bayes')

y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100),X,Y,cv=10)

sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,2],annot=True,fmt='2.0f')

ax[1,2].set_title('Matrix for Random-Forests')

y_pred = cross_val_predict(DecisionTreeClassifier(),X,Y,cv=10)

sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[2,0],annot=True,fmt='2.0f')

ax[2,0].set_title('Matrix for Decision Tree')

plt.subplots_adjust(hspace=0.2,wspace=0.2)

plt.show()

#By looking at all the matrices, we can say that rbf-SVM has a higher chance in correctly 

#predicting dead passengers but NaiveBayes has a higher chance in correctly 

#predicting passengers who survived.
#Hyper parameter tuning

#SVM

from sklearn.model_selection import GridSearchCV

C = [0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]

gamma = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

kernel = ['rbf','linear']

hyper = {'kernel':kernel,'C':C,'gamma':gamma}

gd = GridSearchCV(estimator=svm.SVC(),param_grid=hyper,verbose=True)

gd.fit(X,Y)

print(gd.best_score_)

print(gd.best_estimator_)
#Hyper parameter Tuning 

#Randomforest 

n_estimators = range(100,1000,100)

hyper = {'n_estimators': n_estimators}

gd = GridSearchCV(estimator = RandomForestClassifier(random_state = 0),

                  param_grid=hyper,verbose=True)

gd.fit(X,Y)

print(gd.best_score_)

print(gd.best_estimator_)
#Ensembling

#Voting Classifier

from sklearn.ensemble import VotingClassifier

ensemble_lin_rbf = VotingClassifier(estimators=[('KNN',KNeighborsClassifier(n_neighbors=10)),

                                                ('rbf',svm.SVC(probability = True,kernel = 'rbf',C=0.5,gamma = 0.1)),

                                               ('Rfor',RandomForestClassifier(n_estimators=500,random_state=0)),

                                               ('LR',LogisticRegression(C=0.05)),

                                               ('DT',DecisionTreeClassifier(random_state=0)),

                                               ('NB',GaussianNB()),

                                               ('svm',svm.SVC(probability = True,kernel = 'linear'))],

                                   voting ='soft').fit(Train_X,Train_Y)

print('the accuracy of ensembled model is =',ensemble_lin_rbf.score(test_X,test_Y))

cross= cross_val_score(ensemble_lin_rbf,X,Y,cv=10,scoring='accuracy')

print('the cross validated score is:',cross.mean())
#Bagging

from sklearn.ensemble import BaggingClassifier

model = BaggingClassifier(base_estimator = KNeighborsClassifier(n_neighbors = 3),

                          random_state = 0,n_estimators = 700 )

model.fit(Train_X,Train_Y)

prediction = model.predict(test_X)

print('The accuracy of KNN:',metrics.accuracy_score(prediction,test_Y))

result= cross_val_score(model,X,Y,cv=10,scoring = 'accuracy')

print('The cross validated score is :',cross.mean())

#Bagged DecisionTree

model = BaggingClassifier(base_estimator = DecisionTreeClassifier(),random_state=0,n_estimators = 100)

model.fit(Train_X,Train_Y)

prediction = model.predict(test_X)

print('The accuracy :',metrics.accuracy_score(prediction,test_Y))

result=cross_val_score(model,X,Y,cv=10,scoring='accuracy')

print('The cross validated score is:',cross.mean())
#adaboost

from sklearn.ensemble import AdaBoostClassifier

ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)

result=cross_val_score(ada,X,Y,cv=10,scoring='accuracy')

print('the result of Adaboost is:',result.mean())
#stochastic gradient boosting

from sklearn.ensemble import GradientBoostingClassifier

grad=GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)

result=cross_val_score(grad,X,Y,cv=10,scoring='accuracy')

print('The result of stochastic gradient boosting is:',result.mean())
#xgboost

import xgboost as xg

xgboost=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)

result=cross_val_score(xgboost,X,Y,cv=10,scoring='accuracy')

print('the result using XGBoost is:',result.mean())
#Hyper Parameter tuning for Adaboost

n_estimators=list(range(100,1100,100))

learn_rate=[0.05,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

hyper={'n_estimators':n_estimators,'learning_rate':learn_rate}

gd=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=hyper,verbose=True)

gd.fit(X,Y)

print(gd.best_score_)

print(gd.best_estimator_)
#Confusion matrix of the best model

ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.05)

result=cross_val_predict(ada,X,Y,cv=10)

sns.heatmap(confusion_matrix(Y,result),cmap='winter',annot=True,fmt='2.0f')

plt.show()
f,ax=plt.subplots(2,2,figsize=(15,12))

model=RandomForestClassifier(n_estimators=500,random_state=0)

model.fit(X,Y)

pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,0])

ax[0,0].set_title('Feature importance in Random Forest')

model=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.05)

model.fit(X,Y)

pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,1],color='#ddff11')

ax[0,1].set_title('Feature importance in AdaBoost')

model=GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,random_state=0)

model.fit(X,Y)

pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,0],color='#eee8aa')

ax[1,0].set_title('Feature importance in Gradient Boosting')

model=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)

model.fit(X,Y)

pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,1],color='#FD0F00')

ax[1,1].set_title('Feature importance in XGBoost')

plt.show()
f,ax=plt.subplots(2,2,figsize=(15,12))

model=RandomForestClassifier(n_estimators=500,random_state=0)

model.fit(X,Y)

pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,0])

ax[0,0].set_title('Feature Importance in Random Forests')

model=AdaBoostClassifier(n_estimators=200,learning_rate=0.05,random_state=0)

model.fit(X,Y)

pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,1],color='#ddff11')

ax[0,1].set_title('Feature Importance in AdaBoost')

model=GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,random_state=0)

model.fit(X,Y)

pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,0],cmap='RdYlGn_r')

ax[1,0].set_title('Feature Importance in Gradient Boosting')

model=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)

model.fit(X,Y)

pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,1],color='#FD0F00')

ax[1,1].set_title('Feature Importance in XgBoost')

plt.show()