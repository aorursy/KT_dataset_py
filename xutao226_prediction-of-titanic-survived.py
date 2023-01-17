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
#1.import package
import types
import numpy as np
import seaborn as sns
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.cross_validation import KFold,train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.linear_model import LogisticRegression,Perceptron
from sklearn.tree import DecisionTreeClassifier
train_data = pd.read_csv(r"../input/train.csv")
test_data = pd.read_csv(r"../input/test.csv")
print("training data's shape:",train_data.shape)
print(train_data.head())
print("test data's shape:",test_data.shape)
print(test_data.head())
#3.exploratory Data Analysis
#3.1 show data info
print("--------------training data information---------------")
print(train_data.info())
print("--------------  test data information  ---------------")
print(test_data.info())
print(train_data['Survived'].value_counts())
#3.2 show survived info
train_data['Survived'].astype(int).plot.hist()
#Age with Survived
print("range of Age ",train_data["Age"].max(),train_data["Age"].min())
mean_Age = train_data["Age"].mean()
std_Age = train_data["Age"].std()
null_count_Age_train = train_data["Age"].isnull().sum()
null_count_Age_test = test_data["Age"].isnull().sum()
train_data["Age"][np.isnan(train_data["Age"])] = np.random.randint(mean_Age - std_Age,mean_Age+ std_Age,null_count_Age_train)
test_data["Age"][np.isnan(test_data["Age"])] = np.random.randint(mean_Age - std_Age,mean_Age+ std_Age,null_count_Age_test)
#feature importance
grid = sns.FacetGrid(train_data,hue='Survived',size=4,palette='seismic')
grid.map(plt.scatter, "PassengerId", "Age")
grid.add_legend()
grid
#Ticket with Survived,
null_count_Ticket_train = train_data["Ticket"].isnull().sum()
null_count_Ticket_test = test_data["Ticket"].isnull().sum()
#Fare with Survived
print("range of Fare ",train_data["Fare"].max(),train_data["Fare"].min())
mean_Fare = train_data["Fare"].mean()
std_Fare = train_data["Fare"].std()
null_count_Fare_train = train_data["Fare"].isnull().sum()
null_count_Fare_test = test_data["Fare"].isnull().sum()

if null_count_Fare_train != 0:
    train_data["Fare"][np.isnan(train_data["Fare"])] = np.random.randint(mean_Fare - std_Fare,mean_Fare+ std_Fare,null_count_Fare_train)
if null_count_Fare_test != 0:
    test_data["Fare"][np.isnan(test_data["Fare"])] = np.random.randint(mean_Fare - std_Fare,mean_Fare+ std_Fare,null_count_Fare_test)
# sns.countplot(x='Survived',hue='Fare',data  = train_data,order=[1,0],ax=axis3)
grid = sns.FacetGrid(train_data,hue='Survived',size=5,palette='seismic')
grid.map(plt.scatter, "PassengerId", "Fare")
grid.add_legend()
grid
##discreate values
fig,axis = plt.subplots(figsize=(8,8))
#Pclass with Survived
null_count_Pclass_train = train_data['Pclass'].isnull().sum()
null_count_Pclass_test = test_data['Pclass'].isnull().sum()
print("num of Pclass's null value: ",null_count_Pclass_train)
print("num of Pclass's null value: ",null_count_Pclass_test)
mean_Pclass = train_data['Pclass'].mean()

train_data['Pclass'][np.isnan(train_data['Pclass'])] = int(mean_Pclass) + 1
test_data['Pclass'][np.isnan(test_data['Pclass'])] = int(mean_Pclass) + 1
sns.countplot(x='Survived',hue='Pclass',data  = train_data,order=[1,0],ax=axis)
#Sex with Survived
fig,axis = plt.subplots(figsize=(8,8))
null_count_Sex_train = train_data['Sex'].isnull().sum()
null_count_Sex_test = test_data['Sex'].isnull().sum()
print("num of Sex's train null value: ",null_count_Sex_train)
print("num of Sex's test null value: ",null_count_Sex_test)

# train_data['Sex'][np.isnan(train_data['Sex'])] = "male"
# test_data['Sex'][np.isnan(test_data['Sex'])] = "male"
sns.countplot(x='Survived',hue='Sex',data  = train_data,order=[1,0],ax=axis)
sex_map = {"male":0,"female":1}
train_data['Sex'] = train_data['Sex'].map(sex_map)
test_data['Sex'] = test_data['Sex'].map(sex_map)
#SibSp with Survived
fig,axis = plt.subplots(figsize=(8,8))
null_count_SibSp_train = train_data['SibSp'].isnull().sum()
null_count_SibSp_test = test_data['SibSp'].isnull().sum()
print("num of SibSp's train null value: ",null_count_SibSp_train)
print("num of SibSp's test null value: ",null_count_SibSp_test)
mean_SibSp = train_data['SibSp'].mean()
train_data['SibSp'][np.isnan(train_data['SibSp'])] = int(mean_SibSp) + 1
test_data['SibSp'][np.isnan(test_data['SibSp'])] = int(mean_SibSp) + 1
sns.countplot(x='Survived',hue='SibSp',data  = train_data,order=[1,0],ax=axis)
#Parch with Survived
fig,axis = plt.subplots(figsize=(8,8))
null_count_Parch_train = train_data['Parch'].isnull().sum()
null_count_Parch_test = test_data['Parch'].isnull().sum()
print("num of Parch's train null value: ",null_count_Parch_train)
print("num of Parch's test null value: ",null_count_Parch_test)
mean_Parch = train_data['Parch'].mean()

train_data['Parch'][np.isnan(train_data['Parch'])] = int(mean_Parch) + 1
test_data['Parch'][np.isnan(test_data['Parch'])] = int(mean_Parch) + 1

sns.countplot(x='Survived',hue='Parch',data  = train_data,order=[1,0],ax=axis)
#Cabin with Survived
null_count_Cabin_train = train_data['Cabin'].isnull().sum()
null_count_Cabin_test = test_data['Cabin'].isnull().sum()
print("num of Cabin's train null value: ",null_count_Cabin_train)
print("num of Cabin's test null value: ",null_count_Cabin_test)
#Embarked with Survived
fig,axis = plt.subplots(figsize=(8,8))
train_data['Embarked'] = train_data['Embarked'].fillna('S')
test_data['Embarked'] = test_data['Embarked'].fillna('S')
sns.countplot(x='Survived',hue='Embarked',data = train_data,order=[0,1],ax=axis)
embkarked_map = {"S":0,"C":1,"Q":3}
train_data['Embarked'] = train_data['Embarked'].map(embkarked_map)
test_data['Embarked'] = test_data['Embarked'].map(embkarked_map)
# 3.5 drop data
train_data = train_data.drop(['PassengerId'],axis = 1)
train_data = train_data.drop(['Name'],axis = 1)
train_data = train_data.drop(['Ticket'],axis = 1)
train_data = train_data.drop(['Cabin'],axis = 1)

# test_data = test_data.drop(['PassengerId'],axis = 1)
test_data = test_data.drop(['Name'],axis = 1)
test_data = test_data.drop(['Ticket'],axis = 1)
test_data = test_data.drop(['Cabin'],axis = 1)
# 3.6 show correlation between features exclude survived column.
fig,ax = plt.subplots(figsize=(12,12))
cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
sns.heatmap(train_data.corr(),cmap=cmap,annot=True,linewidths=.5,fmt='.1f',ax=ax)
#4.1 split training data and validation data
Y_train_data = train_data['Survived']
X_train_data = train_data.drop('Survived',axis=1)
X_train,X_val ,y_train,y_val= train_test_split(X_train_data,Y_train_data,test_size=0.3,random_state=0)
#4.2 select a model:randomforest
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
importance_df = pd.DataFrame(clf.feature_importances_,columns=['feature_Temp'],index=X_train.columns)
importance_df.sort_values(by=['feature_Temp'],ascending=True, inplace=True)
print('feature importance rank:\n',importance_df)
importance_df.plot(kind='barh',stacked = True).get_figure()
#4.3 validate
y_pred = clf.predict(X_val)
acc = accuracy_score(y_pred,y_val)
print("validation accuracy: ",acc)
#4.4 test and submission
test_data_dropId = test_data.drop(['PassengerId'],axis=1)
predict_test = clf.predict(test_data_dropId)
submission = pd.DataFrame({
    "PassengerId":test_data['PassengerId'],
    'Survived':predict_test
})
submission.to_csv('submission.csv',index=False)