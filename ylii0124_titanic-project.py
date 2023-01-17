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



# Any results you write to the current directory are saved as output.
#Import libraries

#System functions and parameters

import sys 



#Data processing and analysing

import pandas as pd



#Data computing

import numpy as np



#Data visulization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



import seaborn as sns

%matplotlib inline



#Scientific computing

import scipy as sp

from scipy import stats



#Machine learning algrithms

import sklearn

from sklearn import svm, tree, linear_model, neural_network, ensemble

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn import feature_selection, model_selection, metrics

from sklearn.model_selection import GridSearchCV



#Interative dataframe in Jupyter

import IPython

from IPython import display



#Generating random number

import random



#Ignore the warinings 

import warnings

warnings.filterwarnings('ignore')

#Read the train set

dataTrain = pd.read_csv('../input/titanic/train.csv')

dataTrainC = dataTrain.copy(deep = True)



#Read the test set

dataTest = pd.read_csv('../input/titanic/test.csv')

dataTestC = dataTest.copy(deep = True)

#Insert a Survived column in test set

dataTestC.insert(1,"Survived", np.nan, True)



#Combine the data set

dataFull = pd.concat([dataTrainC,dataTestC], ignore_index =True)

dataFullT = dataFull.copy(deep = True)



#View the data

dataFull.info()

dataFull.sample(10)

#Find the missing values in data

print(dataTrainC.isnull().sum())

print(dataTestC.isnull().sum())



dataFull.describe(include = 'all')
#Fill the missing value in dataset

#Fill fare with median value

dataFull['Fare'].fillna(dataFull['Fare'].median(),inplace = True)



#Fill embarked with the mode

dataFull['Embarked'].fillna(dataFull['Embarked'].mode()[0],inplace = True)

dataTrain['Embarked'].fillna(dataTrain['Embarked'].mode()[0],inplace = True)



#Remove the less correlated variables

dataFull.drop(['PassengerId','Ticket','Cabin'],axis = 1, inplace=True)

dataTrain.drop(['PassengerId','Ticket','Cabin'],axis = 1, inplace=True)
#Combine some variables into a Familysize variable

dataFull['Familysize'] = dataFull['SibSp'] + dataFull['Parch'] + 1



#Get the title of the passengers

dataFull['Title'] = dataFull['Name'].str.split(', ',expand=True)[1].str.split('. ',expand=True)[0]

dataTrain['Title'] = dataTrain['Name'].str.split(', ',expand=True)[1].str.split('. ',expand=True)[0]



#Replace the rare titles

limit = 9

otherName = dataFull['Title'].value_counts()< limit

dataFull['Title'] = dataFull['Title'].apply(lambda x: 'Others'if otherName.loc[x] == True else x)

limit = 9

otherName = dataTrain['Title'].value_counts()< limit

dataTrain['Title'] = dataTrain['Title'].apply(lambda x: 'Others'if otherName.loc[x] == True else x)



dataFullT = dataFull.copy(deep=True)

dataTrainC = dataTrain.copy(deep = True)
#Code the categorical data

label = LabelEncoder()

dataTrainC['Sex'] = label.fit_transform(dataTrainC['Sex'])

dataTrainC['Embarked'] = label.fit_transform(dataTrainC['Embarked'])

dataTrainC['Title'] = label.fit_transform(dataTrainC['Title'])





#Convert the categorical data into dummy

dataC = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Familysize','Title']

dataFullT = pd.get_dummies(dataFullT[dataC])

dataFullT.head()



#Fill the age with Random Forest

dataPreAge = dataFullT.copy(deep=True)

dataPreAge.drop(['Survived'],axis=1,inplace = True)

AgeY = ['Age']

dataPreAgeX = dataPreAge.copy(deep=True)

dataPreAgeX.drop(['Age'],axis=1,inplace=True)

AgeX = dataFull



xV = ['Pclass','SibSp','Parch','Familysize','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S','Title_Master','Title_Miss','Title_Mr','Title_Mrs','Title_Others']

regressor = ensemble.RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)

regressor.fit(dataFullT[dataFullT['Age'].isnull()==False][xV], dataFullT[dataFullT['Age'].isnull()==False]['Age'])

y_pred = regressor.predict(dataFullT[dataFullT['Age'].isnull()][xV])

y_Age = y_pred.tolist()



index_NaN_age = list(dataFullT["Age"][dataFullT["Age"].isnull()].index)



yIdx = 0 

for i in index_NaN_age:

    dataFullT['Age'].iloc[i] = y_Age[yIdx]

    yIdx +=1
#Split the data back to train and test

testLabel = dataFullT['Survived'].isnull()

dataTestNew = dataFullT.loc[testLabel==True]

dataTrainNew = dataFullT.loc[testLabel==False]



#Drop the outliers in numeric variables

zFs = np.abs(stats.zscore(dataTrainNew['Familysize']))

dataTrainNew=dataTrainNew[zFs < 4]



zFa = np.abs(stats.zscore(dataTrainNew['Fare']))

dataTrainNew=dataTrainNew[zFa < 4]



#Split the train set into test and train 

y = dataTrainNew['Survived']

dataTrainX = dataTrainNew.copy(deep=True)

dataTrainX.drop(['Survived'],axis = 1, inplace=True)

X_train, X_test, y_train, y_test = model_selection.train_test_split(dataTrainX, y, test_size=0.3, random_state=42)
#Visualize the numeric variables

plt.figure(figsize=[16,12])



plt.subplot(231)

plt.boxplot(dataTrainNew['Age'],showmeans=True,meanline=True)

plt.title('Age Boxplot')

plt.ylabel('Age (Years)')



plt.subplot(232)

plt.boxplot(dataTrainNew['Fare'],showmeans=True,meanline=True)

plt.title('Fare Boxplot')

plt.ylabel('Fare($)')



plt.subplot(233)

plt.boxplot(dataTrainNew['Familysize'],showmeans=True,meanline=True)

plt.title('Familysize Boxplot')

plt.ylabel('Familysize')



plt.subplot(234)

plt.hist(x = [dataTrainNew[dataTrainNew['Survived']==1]['Age'], dataTrainNew[dataTrainNew['Survived']==0]['Age']], 

         stacked=True, color=['b','c'],label=['Survived','Dead'])

plt.title('Age Histogram')

plt.xlabel('Age (Years)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(235)

plt.hist(x = [dataTrainNew[dataTrainNew['Survived']==1]['Fare'], dataTrainNew[dataTrainNew['Survived']==0]['Fare']],

        stacked=True, color=['b','c'],label=['Survived','Dead'])

plt.title('Fare Histogram')

plt.xlabel('Fare($)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(236)

plt.hist(x = [dataTrainNew[dataTrainNew['Survived']==1]['Familysize'], dataTrainNew[dataTrainNew['Survived']==0]['Familysize']],

        stacked=True, color=['b','c'], label=['Survived','Dead'])

plt.title('Familysize Histogram')

plt.xlabel('Familysize')

plt.ylabel('# of Passengers')

plt.legend()



#Visualize the categorical variables

plt.figure(figsize=[16,6])



plt.subplot(141)

sns.barplot(x = 'Pclass',y = 'Survived', data=dataTrainNew)

plt.title('Pclass Barplot')



plt.subplot(142)

sns.barplot(x = 'Embarked',y = 'Survived', data=dataTrain)

plt.title('Embarked Barplot')



plt.subplot(143)

sns.barplot(x = 'Title',y ='Survived',data = dataFull)

plt.title('Title Barplot')



plt.subplot(144)

sns.barplot(x = 'Sex',y= 'Survived',data = dataTrain)

plt.title('Sex Barplot')
#Compute the correlation plot

plt.subplots(figsize =(14, 12))

cor = sns.heatmap(dataTrainC.corr(), annot=True)

plt.title('Correlation of variables', y=1.05, size=15)
#Normalize the variables

dataTrainNew['Age'] = (dataTrainNew['Age']-dataTrainNew['Age'].mean())/(dataTrainNew['Age'].max()-dataTrainNew['Age'].min())

dataTrainNew['Fare'] = (dataTrainNew['Fare']-dataTrainNew['Fare'].mean())/(dataTrainNew['Fare'].max()-dataTrainNew['Fare'].min())

dataTrainNew['Familysize'] = (dataTrainNew['Familysize']-dataTrainNew['Familysize'].mean())/(dataTrainNew['Familysize'].max()-dataTrainNew['Familysize'].min())





dataTestNew['Age'] = (dataTestNew['Age']-dataTestNew['Age'].mean())/(dataTestNew['Age'].max()-dataTestNew['Age'].min())

dataTestNew['Fare'] = (dataTestNew['Fare']-dataTestNew['Fare'].mean())/(dataTestNew['Fare'].max()-dataTestNew['Fare'].min())

dataTestNew['Familysize'] = (dataTestNew['Familysize']-dataTestNew['Familysize'].mean())/(dataTestNew['Familysize'].max()-dataTestNew['Familysize'].min())

#Model selection 

#List the models

algorithmSet = [

        ensemble.AdaBoostClassifier(),

        ensemble.RandomForestClassifier(),

        ensemble.GradientBoostingClassifier(),

        

        linear_model.LogisticRegressionCV(),

        

        svm.SVC(probability=True),

        svm.LinearSVC(),

        

        tree.DecisionTreeClassifier(),

        

        neural_network.MLPClassifier(hidden_layer_sizes=(50,50,50)),

]



#Split the dataset for cross validation

cv = model_selection.ShuffleSplit(n_splits=10, test_size = 0.3, train_size=0.65, random_state=0)



#Create the dataframe to compare the models

tabelCol = ['AL names','AL parameters','Train Accuracy Mean', 'Test Accuracy Mean','Running Time']

compareTable = pd.DataFrame(columns = tabelCol)



#create table to compare MLA predictions

MLA_predict = dataTrainNew['Survived']



#Compute a for loop to run the algorithms 

row = 1

for alg in algorithmSet:

    

    #Store the name and parameters

    algName = alg.__class__.__name__

    compareTable.loc[row,'AL names'] = algName

    compareTable.loc[row,'AL parameters'] = str(alg.get_params())

    

    #Store the score of the models

    cv_scores = model_selection.cross_validate(alg,dataTrainX,y, cv = cv,return_train_score=True)

    compareTable.loc[row,'Train Accuracy Mean'] = cv_scores['train_score'].mean()

    compareTable.loc[row,'Test Accuracy Mean'] = cv_scores['test_score'].mean()

    compareTable.loc[row,'Running Time'] = cv_scores['fit_time'].mean()

    

    #Save the prediction of models

    alg.fit(dataTrainX,y)

    MLA_predict = alg.predict(dataTrainX)

    

    row += 1

    



#Sort the table to get the top models

compareTable.sort_values(by = ['Test Accuracy Mean'],ascending = False, inplace = True)

compareTable

#Original parameters

gbm0 = ensemble.GradientBoostingClassifier(learning_rate=0.1,min_samples_split=2,n_estimators=100,

                                  min_samples_leaf=1,max_depth=3, subsample=1,random_state=10)

gbm0.fit(dataTrainX,y)

GB_pred = gbm0.predict(dataTrainX)

GB_predprob = gbm0.predict_proba(dataTrainX)[:,1]

print ('Accuracy:',metrics.accuracy_score(y.values, GB_pred))

print ("AUC Score (Train):", metrics.roc_auc_score(y, GB_predprob))
#Changing n_estimators

para1 = {'n_estimators':range(60,160,10)}

gsearch1 = GridSearchCV(estimator=ensemble.GradientBoostingClassifier(learning_rate=0.1,min_samples_split=2,

                                                                     min_samples_leaf=1,max_depth=3, subsample=1,random_state=10),

                       param_grid=para1,scoring='roc_auc',iid=False,cv = cv)

gsearch1.fit(dataTrainX,y)

gsearch1.best_params_, gsearch1.best_score_

#n_estimators = 110
#Changing max_depth and min_samples_split

para2 = {'max_depth':range(3,11,2),'min_samples_split':range(2,100,20)}

gsearch2 = GridSearchCV(estimator=ensemble.GradientBoostingClassifier(learning_rate = 0.1, n_estimators = 110,

                                                                     min_samples_leaf=1, subsample=1,random_state=10),

                       param_grid=para2,scoring='roc_auc',iid=False,cv = cv)

gsearch2.fit(dataTrainX,y)

gsearch2.best_params_, gsearch2.best_score_

#max_depth = 7
#Changing max_depth and min_samples_split

para3 = {'min_samples_leaf':range(2,100,20),'min_samples_split':range(2,100,20)}

gsearch3 = GridSearchCV(estimator=ensemble.GradientBoostingClassifier(learning_rate = 0.1, n_estimators = 110,

                                                                     max_depth=7, subsample=1,random_state=10),

                       param_grid=para3,scoring='roc_auc',iid=False,cv = cv)

gsearch3.fit(dataTrainX,y)

gsearch3.best_params_, gsearch3.best_score_

#min_samples_leaf = 42, min_samples_split=2
#Changing subsample

para4 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9,1]}

gsearch4 = GridSearchCV(estimator=ensemble.GradientBoostingClassifier(learning_rate = 0.1, n_estimators = 110,min_samples_leaf=42,

                                                                     max_depth=7, min_samples_split=2,random_state=10),

                       param_grid=para4,scoring='roc_auc',iid=False,cv = cv)

gsearch4.fit(dataTrainX,y)

gsearch4.best_params_, gsearch4.best_score_





#subsample = 1
#Final score

gbm1 = ensemble.GradientBoostingClassifier(learning_rate=0.1,min_samples_split=2,n_estimators=110,

                                  min_samples_leaf=42,max_depth=7, subsample=1,random_state=10)

gbm1.fit(dataTrainX,y)

gbm_scores = model_selection.cross_validate(gbm1,dataTrainX,y, cv = cv,return_train_score=True)

print(gbm_scores['train_score'].mean())

print(gbm_scores['test_score'].mean())
#Get the original parameters

rf0 = ensemble.RandomForestClassifier()

rf0.fit(dataTrainX,y)

str(rf0.get_params())
#Original score

rf0 = ensemble.RandomForestClassifier(n_estimators=10,min_samples_split=2,min_samples_leaf=1,random_state=10)

rf0.fit(dataTrainX,y)

RF_pred = rf0.predict(dataTrainX)

RF_predprob = rf0.predict_proba(dataTrainX)[:,1]

print ('Accuracy:',metrics.accuracy_score(y.values, RF_pred))

print ("AUC Score (Train):", metrics.roc_auc_score(y, RF_predprob))
#Changing n_estimators

rfPara1 = {'n_estimators':range(10,120,10)}

rfGsearch1 = GridSearchCV(estimator=ensemble.RandomForestClassifier(min_samples_split=2,min_samples_leaf=1,random_state=10),

                         param_grid=rfPara1,scoring='roc_auc',cv=cv)

rfGsearch1.fit(dataTrainX,y)

rfGsearch1.best_params_, rfGsearch1.best_score_

#n_estimators = 100
#Changing max_depth & min_samples_split

rfPara2 = {'max_depth':range(2,14,2),'min_samples_split':range(2,100,20)}

rfGsearch2 = GridSearchCV(estimator=ensemble.RandomForestClassifier(n_estimators=100,min_samples_leaf=1,random_state=10),

                         param_grid=rfPara2,scoring='roc_auc',cv=cv)

rfGsearch2.fit(dataTrainX,y)

rfGsearch2.best_params_, rfGsearch2.best_score_

#max_depth=12
#Changing min_samples_leaf & min_samples_split

rfPara3 = {'min_samples_leaf':range(10,100,10),'min_samples_split':range(2,100,20)}

rfGsearch3 = GridSearchCV(estimator=ensemble.RandomForestClassifier(n_estimators=100,max_depth=12,random_state=10),

                         param_grid=rfPara3,scoring='roc_auc',cv=cv)

rfGsearch3.fit(dataTrainX,y)

rfGsearch3.best_params_, rfGsearch3.best_score_

#minleaf=10, minsplit=2
#Changing max_features

rfPara4 = {'max_features':range(2,10,2)}

rfGsearch4 = GridSearchCV(estimator=ensemble.RandomForestClassifier(n_estimators=100,max_depth=12,random_state=10,

                                                                   min_samples_leaf=10,min_samples_split=2),

                         param_grid=rfPara4,scoring='roc_auc',cv=cv)

rfGsearch4.fit(dataTrainX,y)

rfGsearch4.best_params_, rfGsearch4.best_score_

#max_features = 8
#Final score

rf1 = ensemble.RandomForestClassifier(n_estimators=100,max_depth=12,random_state=10,

                                          max_features=8,min_samples_leaf=10,min_samples_split=2)

rf1.fit(dataTrainX,y)

rf_scores = model_selection.cross_validate(rf1,dataTrainX,y, cv = cv,return_train_score=True)

print(rf_scores['train_score'].mean())

print(rf_scores['test_score'].mean())
#Compare the voting models

#soft voting

voteList = [

    ('lr',linear_model.LogisticRegressionCV()),

    ('rfc',rf1),

    ('gbc',gbm1)

]



votingE = ensemble.VotingClassifier(estimators=voteList,voting='soft')

votingE.fit(dataTrainX,y)

VC_scores = model_selection.cross_validate(votingE,dataTrainX,y, cv = cv,return_train_score=True)

print(VC_scores['train_score'].mean())

print(VC_scores['test_score'].mean())



#hard voting

voteList = [

    ('lr',linear_model.LogisticRegressionCV()),

    ('rfc',rf1),

    ('gbc',gbm1)

]



votingH = ensemble.VotingClassifier(estimators=voteList,voting='hard')

votingH.fit(dataTrainX,y)

VH_scores = model_selection.cross_validate(votingH,dataTrainX,y, cv = cv,return_train_score=True)

print(VH_scores['train_score'].mean())

print(VH_scores['test_score'].mean())
#Choose the soft vote model to predict the whole test set

dataTestX = dataTestNew.copy(deep=True)

dataResult = dataTestNew.copy(deep=True)

dataTestX.drop(['Survived'],axis=1,inplace = True)

dataResult['Survived'] = votingE.predict(dataTestX)

dataResult.reset_index(drop=True,inplace=True)
#Output the final csv

submit = pd.concat([dataTest['PassengerId'],dataResult['Survived']],axis=1)

submit.head()

submit.to_csv('softVoting prediction.csv',index=False)