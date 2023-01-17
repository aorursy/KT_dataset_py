

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train["Flag"]='train'

test["Flag"]='test'

data=pd.concat([train,test],axis=0)
columns_to_drop=["PassengerId","Name","Ticket"]

data.drop(columns_to_drop,axis=1,inplace=True)



#However we need to look into Fare,SibSp,Parch,Cabin features as well
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,8))

plt.subplot2grid((3,4),(0,0))

train["Survived"].value_counts().plot(kind="bar")
def val_count_plot(feature,i,j):

    plt.subplot2grid((2,2),(i,j))

    train[feature].value_counts(normalize=True).plot(kind='bar')



val_count_plot('Survived',0,0)

val_count_plot('Sex',0,1)

val_count_plot('Embarked',1,0)

val_count_plot('Pclass',1,1)



plt.subplot2grid((2,2),(0,0))

train[(train['Sex']=='male')&(train['Pclass']==1)]['Survived'].value_counts(normalize=True).plot(kind='bar')

plt.title('Survived male')



# SO it seems the male survived less as compared to females.
data.head()
print(data.isnull().values.any())
data.apply(lambda x: sum(x.isnull()))
#Age 

#Since age seems to be an important feature because we need to decide whether 

# the passsenger survived or not. And age is an important feature.



# I am replacing age with mean of ages.



data["Age"].fillna(data["Age"].mean(),inplace=True)



#Since two value is missing in embarked I am replacing it with mode

data["Embarked"].fillna(data["Embarked"].mode,inplace=True)

print(data.apply(lambda x:sum(x.isnull())))



# but in fare I am not thinking of applying mean because the fare will depend on passenger class,

# So any passenger with higher class will be having a higher fare.





#Introduction to Ensembling/Stacking in Python kernel has many interesting steps for data exploration .



# I have considered the idea of creating another field from fare from this kernel using qcut

data['Fare'] = data['Fare'].fillna(train['Fare'].median())

data['CategoricalFare'] = pd.qcut(train['Fare'], 4)



# This divides into 4 ranges as can be seen 

print(data['CategoricalFare'].head(10))



# I can map the fares into one of these ranges





# Next only Cabin is the feature that is left 



# if the cabin is present I am mapping that to 1 else 0



data["Has_Cabin"]=data["Cabin"].apply(lambda x: 1 if type(x) == float else 0)

print(data["Has_Cabin"].head(5))
data.drop(["CategoricalFare","Cabin"],axis=1,inplace=True)
print(data.dtypes)# helpd in identifying the types of data in the dataset, the object is data type is categorical

# to get the names of the categorical columns in the dataset we can use

data.select_dtypes(include='object').columns # so flag , sex, embarked are the categorical data.

# to convert them into numerical data I am using mapping method

# 1.MAPPING METHOD

#dict={"male":0,1:"female"}

#data["Sex"].map(dict)

#print(data["Sex"].value_counts())

# similiarly for column Embarked the values can be mapped
#2. using the Label Encoder

# this method will transform non-numerical labels to numerical labels.

# To underatnd this consider a category say region:- which may be east/west/north/south.

#Label encoder will assign 4 different numerical values to each one of the regions.

# this is a disadvantage esp in case of nominal categorical variables where there are no levels.
#from sklearn.preprocessing import LabelEncoder

#label_enc=LabelEncoder()

#data['Sex_l_enc']=label_enc.fit_transform(data['Sex'])

#data['Sex_l_enc'].value_counts()

# so the male and female category has been encoded.
data=pd.get_dummies(data,columns=['Sex'])

data.drop(['Embarked'],axis=1,inplace=True)

# so now we have our dataset , next we can separate it back to training data and test data

# we have a flag column which will help in this.

train_new = data[data.Flag=='train']

test_new=data[data.Flag=='test']
y=train['Survived']# output data

train_new.drop(['Flag','Survived'],axis=1,inplace=True)

test_new.drop(['Flag','Survived'],axis=1,inplace=True)

print(train_new.head(10))

print(train_new.columns)
X=train_new

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

from sklearn import tree

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=7)



decision_model=DecisionTreeClassifier()

decision_model.fit(X_train,y_train)

dt_predict= decision_model.predict(X_test)

# accuracy 

print("Accuracy of decision tree :",metrics.accuracy_score(y_test,dt_predict))

# using LOgistic Regression 

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)

logistic_predict = logmodel.predict(X_test)

print("Accuracy of logistic regression is:", metrics.accuracy_score(y_test,logistic_predict))



# using Random Forests 

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier()  

random_forest.fit(X_train, y_train)  

randomf_predict = random_forest.predict(X_test)  

print("Accuracy of RandomForest is:", metrics.accuracy_score(y_test,randomf_predict))



# using XGboost 

from xgboost import XGBClassifier



XGB_model = XGBClassifier()

XGB_model.fit(X_train, y_train)

XGB_predict= XGB_model.predict(X_test)

print("Accuracy of XGB is:", metrics.accuracy_score(y_test,XGB_predict))
