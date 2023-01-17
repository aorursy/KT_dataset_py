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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
#Percentage of female passengers who survived

women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
#Percentage of male passengers who survived

men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
#Data Modelling Libraries

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC

from sklearn.ensemble import AdaBoostClassifier

from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression



#Others 

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing

from sklearn import decomposition

from sklearn.feature_selection import RFECV

from sklearn.feature_selection import SelectKBest, f_classif

import seaborn as sns

import matplotlib.pyplot as plt



print("Setup completed")
train_data.info()
train_data.describe()
def find_missing_data(data):

    Total = data.isnull().sum().sort_values(ascending = False)

    Percentage = (data.isnull().sum()/data.isnull().count()).sort_values(ascending = False)

    

    return pd.concat([Total,Percentage] , axis = 1 , keys = ['Total' , 'Percent'])
find_missing_data(train_data)
train_data.head(8)
train_data.columns.values
#Features: Age and Sex

fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(12,4))

womem = train_data[train_data['Sex'] == 'female']

men = train_data[train_data['Sex'] == 'male']



#Womem ax

ax = sns.distplot(womem[womem['Survived'] == 1].Age.dropna(),label="survived",ax=axes[0],kde=False)

ax = sns.distplot(womem[womem['Survived'] == 0].Age.dropna(),label = "not_survived", ax = axes[0], kde =False)



ax.legend()

ax.set_title('Female')

#Men ax

ax = sns.distplot(men[men['Survived']==1].Age.dropna(),label = "survived", ax = axes[1], kde = False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(),label = "not_survived", ax = axes[1], kde = False)

ax.legend()

_ = ax.set_title('Male')
#Features: Embarked, Pclass and Sex

FacetGrid = sns.FacetGrid(train_data, row='Embarked', aspect=1.6)

FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )

FacetGrid.add_legend()
#Pclass

sns.barplot(x='Pclass',y='Survived',data=train_data)
grid = sns.FacetGrid(train_data,col='Survived',row='Pclass')

grid.map(plt.hist,'Age')

grid.add_legend()
#SibSp and Parch

data = [train_data,test_data]

for dataset in data:

    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']

    dataset.loc[dataset['relatives'] > 0,'not_alone'] = 0

    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1

    dataset['not_alone'] = dataset['not_alone'].astype(int)

    

train_data['not_alone'].value_counts()
sns.factorplot('relatives','Survived',data=train_data,aspect=2.5)
train_data = train_data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

test_data = test_data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
train_data
test_data
#Null values in feature "Age"

data = [train_data,test_data]

for dataset in data:

    mean = train_data["Age"].mean()

    std = test_data["Age"].std()

    is_null = dataset["Age"].isnull().sum()

    rand_age = np.random.randint(mean-std,mean+std,size = is_null)

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = X["Age"].astype(int)

train_data["Age"].isnull().sum()
#Embarked missing values

train_data["Embarked"].describe()
common_value = 'S'

data = [train_data,test_data]

for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
train_data.info()
#Fare feature

data = [train_data,test_data]

for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)
train_data.info()
test_data.info()
#Convert "Sex" feature into numeric

genders = {"male": 0,"female": 1}

data = [train_data,test_data]

for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(genders)
#Convert "Embarked" feature into numeric

ports = {"S": 0 ,"C": 1,"Q": 2}

data = [train_data,test_data]

for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(ports)
#Age

data = [train_data,test_data]

for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[dataset['Age'] <= 11,'Age'] = 0

    dataset.loc[(dataset['Age'] > 11 ) & (dataset['Age'] <= 18),'Age'] = 1

    dataset.loc[(dataset['Age'] > 18 ) & (dataset['Age'] <= 22),'Age'] = 2

    dataset.loc[(dataset['Age'] > 22 ) & (dataset['Age'] <= 27),'Age'] = 3

    dataset.loc[(dataset['Age'] > 27 ) & (dataset['Age'] <= 33),'Age'] = 4

    dataset.loc[(dataset['Age'] > 33 ) & (dataset['Age'] <= 40),'Age'] = 5

    dataset.loc[(dataset['Age'] > 40 ) & (dataset['Age'] <= 66),'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6

train_data['Age'].value_counts()
#Fare

data = [train_data,test_data]

for dataset in data:

    dataset.loc[dataset['Fare'] <= 7.91,'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare'] = 3

    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare'] = 4

    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5

    dataset['Fare'] = dataset['Fare'].astype(int)
train_data
test_data
X = train_data.drop("Survived",axis=1)

Y = train_data[['Survived']]
X
Y
#Turning the dataframe into arrays

X_train = X.values

Y_train = Y.values.reshape(-1)
X_train
Y_train
# Model training and testing

def fit_model(name,model,X_train,Y_train):

    

    scores = cross_val_score(model,X_train,Y_train,scoring='accuracy')

    hit_rate = np.mean(scores)



    msg = "hit rate of {} is: {}".format(name,hit_rate)

    print(msg)

    return hit_rate
##DecisionTree training

decisionTree_model = DecisionTreeClassifier(random_state=0)

decisionTree_result = fit_model("DecisionTree",decisionTree_model,X_train,Y_train)
#RandomForest training

randomForest_model = RandomForestClassifier(random_state=0)

randomForest_result = fit_model("RandomForest",randomForest_model,X_train,Y_train)
#OnevsRest Training

OnevsRest_model = OneVsRestClassifier(LinearSVC(random_state=0))

OnevsRest_result = fit_model("OnevsRest",OnevsRest_model,X_train,Y_train)
#ONEVSONE training

OnevsOne_model = OneVsOneClassifier(LinearSVC(random_state=0))

OnevsOne_result = fit_model("OnevsOne",OnevsOne_model,X_train,Y_train)
#KNN training

Knn_model = KNeighborsClassifier(n_neighbors=2)

Knn_result = fit_model("Knn",Knn_model,X_train,Y_train)
#MULTINOMIALNB training

MultinomialNB_model = MultinomialNB()

MultinomialNB_result = fit_model("MultinomialNB",MultinomialNB_model,

                                X_train,Y_train)
# AdaBOOST training

AdaBoost_model = AdaBoostClassifier()

AdaBoost_result = fit_model("AdaBoost",AdaBoost_model,

                                X_train,Y_train)
results = pd.DataFrame({

    'Model': ['DecisionTree', 'RandomForest', 'OnevsRest', 

              'ONEVSONE', 'KNN', 'MULTINOMIALNB', 

              'AdaBOOST'],

    'Score': [decisionTree_result, randomForest_result, OnevsRest_result, 

              OnevsOne_result, Knn_result, MultinomialNB_result, 

              AdaBoost_result]})

result_df = results.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Score')

result_df
#Using gridSearchCV for the Random Forest Model

RandomForest_parameters = [

    {'n_estimators':[1,10,100],'criterion':['gini','entropy'],'max_depth':[1,2,3,4,5,6,7,8,9,10]} 

    ]

RF = RandomForestClassifier()
clf = GridSearchCV(estimator=RF, param_grid=RandomForest_parameters, n_jobs=-1)
clf.fit(X_train,Y_train)
print('Best score for data1:', clf.best_score_)

print('Best estimators:',clf.best_estimator_.n_estimators) 

print('Best Criterion:',clf.best_estimator_.criterion)

print('Best max_depth:',clf.best_estimator_.max_depth)
Random_Forest = RandomForestClassifier(n_estimators=10,criterion='gini',max_depth=2)

Random_Forest.fit(X_train,Y_train)

Random_Forest.score(X_train,Y_train)
#Confusion Matrix

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

predictions = cross_val_predict(Random_Forest,X_train,Y_train,cv=3)

confusion_matrix(Y_train,predictions)
#Precision and Recall 

from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(Y_train, predictions))

print("Recall:",recall_score(Y_train, predictions))
#f1Score

from sklearn.metrics import f1_score

f1_score(Y_train, predictions)
X_test = test_data.values

X_test
Y_predicitions = Random_Forest.predict(X_test)

Y_predicitions
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': Y_predicitions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")