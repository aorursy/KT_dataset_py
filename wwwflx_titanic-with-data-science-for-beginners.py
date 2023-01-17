import os

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df =  pd.read_csv("/kaggle/input/titanic/train.csv",index_col=0)

df.head(10)
len(df.index)
f,ax=plt.subplots(1,3,figsize=(20,10))



df['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True, ax=ax[0])

df['Survived'].loc[df['Sex']=='male'].value_counts(sort=False).plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True, ax=ax[1])

df['Survived'].loc[df['Sex']=='female'].value_counts(sort=False).plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True, ax=ax[2])

ax[0].set_title('Overall Survived')

ax[1].set_title('Men Survived')

ax[2].set_title('Female Survived')



ax[0].set_ylabel('')

ax[1].set_ylabel('')

ax[2].set_ylabel('')



labels = ['dead', 'survived'] 

plt.legend(labels)
plt.figure(figsize=(20,10))

sns.heatmap(df.isnull(), yticklabels=False, cbar=False)
df.isnull().sum(axis = 0)
df.loc[df["Survived"] == 0,"Survived"] = -1

df.loc[df["Survived"] == 1,"Survived"] = 0

df.loc[df["Survived"] == -1,"Survived"] = 1
df.head()
pd.crosstab(df.Sex,df.Survived).apply(lambda r: r/r.sum(), axis=1)
df.loc[df["Sex"] == "male","Sex"] = 1

df.loc[df["Sex"] == "female","Sex"] = 0

df.head()
df['Embarked'].value_counts() 
df['Embarked'] = df.Embarked.fillna('S')
pd.crosstab(df.Embarked,df.Survived).apply(lambda r: r/r.sum(), axis=1)
df['Embarked'] = df['Embarked'].map( {'S': 2, 'Q': 1, 'C': 0} ).astype(int)

df.head()
sns.set(style="whitegrid")



g = sns.catplot(x="Embarked", y="Survived", hue="Sex", data=df, height=6, kind="bar", palette="Set1")

g.despine(left=True)

g.set_ylabels("death probability")
#df.loc[df["Name"].str.find('Mr.') >= 0, "Type"] = 0

#df.loc[df["Name"].str.find('Mrs.') >= 0, "Type"] = 1

#df.loc[df["Name"].str.find('Miss.') >= 0, "Type"] = 2

#df.loc[df["Name"].str.find('Master.') >= 0, "Type"] = 3

#df.loc[df["Name"].str.find('Don.') >= 0, "Type"] = 4

#df.loc[df["Name"].str.find('Rev.') >= 0, "Type"] = 4

#df.loc[df["Name"].str.find('Dr.') >= 0, "Type"] = 4

#df['Type'] = df.Type.fillna(4)
#df['Type'].value_counts()
#pd.crosstab(df.Type,df.Survived).apply(lambda r: r/r.sum(), axis=1)
#del df['Type']

df.loc[df["Name"].str.find('Mr.') >= 0, "Type"] = 4

df.loc[df["Name"].str.find('Mrs.') >= 0, "Type"] = 0

df.loc[df["Name"].str.find('Miss.') >= 0, "Type"] = 1

df.loc[df["Name"].str.find('Master.') >= 0, "Type"] = 2

df.loc[df["Name"].str.find('Don.') >= 0, "Type"] = 3

df.loc[df["Name"].str.find('Rev.') >= 0, "Type"] = 3

df.loc[df["Name"].str.find('Dr.') >= 0, "Type"] = 3

df['Type'] = df.Type.fillna(3)
pd.crosstab(df.Type,df.Survived).apply(lambda r: r/r.sum(), axis=1)
df['Type'] = df['Type'].astype(int)
sns.set(style="whitegrid")



g = sns.catplot(x="Type", y="Survived", hue="Sex", data=df, height=6, kind="bar", palette="Set1")

g.despine(left=True)

g.set_ylabels("death probability")
df.head()
plt.figure(figsize=(20,5))

sns.boxplot(y="Survived", x="Fare", data=df, palette="Set2",  orient="h");
f,ax=plt.subplots(1,2,figsize=(20,10))

sns.distplot(df[df['Survived']==0].Fare,ax=ax[0])

sns.distplot(df[df['Survived']==1].Fare,ax=ax[1])

df.loc[df["Fare"] < 10,"Fare"] = 5

df.loc[(df["Fare"] >= 10) & (df["Fare"] < 25),"Fare"] = 3

df.loc[(df["Fare"] >= 25) & (df["Fare"] < 50),"Fare"] = 4

df.loc[(df["Fare"] >= 50) & (df["Fare"] < 100),"Fare"] = 2

df.loc[(df["Fare"] >= 100), 'Fare'] = 0

df.head()
f,ax=plt.subplots(1,2,figsize=(15,7))

sns.distplot(df[df['Survived']==0].Fare,ax=ax[0])

sns.distplot(df[df['Survived']==1].Fare,ax=ax[1])
pd.crosstab(df.Fare,df.Survived).apply(lambda r: r/r.sum(), axis=1)
df['Fare'] = df['Fare'].astype(int)
df['Age'].isnull().sum()
meanAge = df['Age'].mean()

print(meanAge)
df['Age'] = df.Age.fillna(meanAge)

df['Age'] = df['Age'].astype('int64')

df.head()
df.loc[df["Age"] < 15,"Age"] = 0

df.loc[(df["Age"] >= 15) & (df["Age"] < 30),"Age"] = 3

df.loc[(df["Age"] >= 30) & (df["Age"] < 45),"Age"] = 1

df.loc[(df["Age"] >= 45) & (df["Age"] < 60),"Age"] = 2

df.loc[(df["Age"] >= 60), 'Age'] = 4

df.head()
f,ax=plt.subplots(1,2,figsize=(15,7))

sns.distplot(df[df['Survived']==0].Age,ax=ax[0])

sns.distplot(df[df['Survived']==1].Age,ax=ax[1])
pd.crosstab(df.Age,df.Survived).apply(lambda r: r/r.sum(), axis=1)
df.Cabin.value_counts()
df.loc[df["Cabin"].str.find('A') >= 0, "Deck"] = 5

df.loc[df["Cabin"].str.find('B') >= 0, "Deck"] = 3

df.loc[df["Cabin"].str.find('C') >= 0, "Deck"] = 4

df.loc[df["Cabin"].str.find('D') >= 0, "Deck"] = 1

df.loc[df["Cabin"].str.find('E') >= 0, "Deck"] = 2

df.loc[df["Cabin"].str.find('F') >= 0, "Deck"] = 0

df.loc[df["Cabin"].str.find('G') >= 0, "Deck"] = 6
df['Deck'] = df.Deck.fillna(6)
df['Deck'] = df['Deck'].astype(int)
sns.set(style="whitegrid")



g = sns.catplot(x="Deck", y="Survived", hue="Sex", data=df, height=6, kind="bar", palette="Set1")

g.despine(left=True)

g.set_ylabels("death probability")
pd.crosstab(df.Deck,df.Survived).apply(lambda r: r/r.sum(), axis=1)
df.head()
del df['Name']

del df['Ticket']

del df['Parch']

del df['SibSp']

del df['Cabin']
df.head()
sns.heatmap(df.corr(), annot = True)
df.dtypes
y=df["Survived"]

x=df.iloc[:,1:]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=4130)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



rdclass=RandomForestClassifier()

rdclass.fit(X_train,y_train)

ypred=rdclass.predict(X_test)



accRFC = accuracy_score(y_test,ypred)

print(accRFC)
from sklearn import svm

clf = svm.SVR()

clf.fit(X_train,y_train)



ypred = clf.predict(X_test)

accSVM_D = accuracy_score(y_test, ypred.round())

print(accSVM_D)
val = []

euc = []



for c in range(1,50):

    for eps in range(0,10):

        clf = svm.SVR(C=c, epsilon=(eps/10))

        clf.fit(X_train,y_train)



        ypred = clf.predict(X_test)

        euc.append([c,(eps/10)])

        val.append([accuracy_score(y_test, ypred.round()),c,eps])



accSVM_A = max(val)

accSVM_A = accSVM_A[0]

print(accSVM_A)
max(val)
clf = svm.SVR(C=49, epsilon=(4/10))

clf.fit(X_train,y_train)



ypred = clf.predict(X_test)

accuracy_score(y_test, ypred.round())
ypred = pd.DataFrame(ypred)

ypred = pd.Series.round(ypred)

ypred = ypred.astype('int64')

accuracy_score(y_test, ypred.round())
import numpy as np

x_ntest = np.array(X_test)

y_ntest = np.array(y_test)

x_ntrain = np.array(X_train)

y_ntrain = np.array(y_train)
print(x_ntrain.shape, y_ntrain.shape)
from xgboost import XGBClassifier



model = XGBClassifier()



# fit the model with the training data

model.fit(x_ntrain,y_ntrain)



# predict the target on the train dataset

predict_train = model.predict(x_ntrain)



# Accuray Score on train dataset

accuracy_train = accuracy_score(y_ntrain,predict_train)

print('accuracy_score on train dataset : ', accuracy_train)



# predict the target on the test dataset

predict_test = model.predict(x_ntest)



# Accuracy Score on test dataset

accuracy_test = accuracy_score(y_ntest,predict_test)

print('\naccuracy_score on test dataset : ', accuracy_test)
"""

import xgboost as xgb

from sklearn.model_selection import GridSearchCV



# make datasets

dtrain = xgb.DMatrix(x_ntrain, label=y_train)

dtest = xgb.DMatrix(x_ntest)



# set up

param = {'max_depth':8, 'eta':0.1, 'objective':'binary:hinge' }

num_round = 10



# fit the model with the training data

bst = xgb.train(param, dtrain, num_round)



# make prediction

preds = bst.predict(dtest)

preds = preds.astype('int64')



# Accuracy Score on test dataset

accuracy_test = accuracy_score(y_test,preds)

print('accuracy_score on test dataset : ', accuracy_test)

"""
from tensorflow import keras

import tensorflow as tf

from keras.utils import to_categorical
modeltf = keras.Sequential([

    keras.layers.Reshape(target_shape=(1,), input_shape=(7,)),

    keras.layers.Dense(units= 7, activation='relu'),

    keras.layers.Dense(units= 14, activation='relu'),

    keras.layers.Dense(units= 7, activation='relu'),

    keras.layers.Dense(units= 1, activation='softmax')

])



modeltf.compile(optimizer='adam', loss=tf.losses.categorical_crossentropy, metrics=['accuracy'])

modeltf.summary()
df =  pd.read_csv("/kaggle/input/titanic/test.csv", index_col=0)



df.loc[df["Sex"] == "male","Sex"] = 1

df.loc[df["Sex"] == "female","Sex"] = 0



df['Embarked'] = df.Embarked.fillna('S')

df['Embarked'] = df['Embarked'].map( {'S': 2, 'Q': 1, 'C': 0} ).astype(int)



df.loc[df["Name"].str.find('Mr.') >= 0, "Type"] = 4

df.loc[df["Name"].str.find('Mrs.') >= 0, "Type"] = 0

df.loc[df["Name"].str.find('Miss.') >= 0, "Type"] = 1

df.loc[df["Name"].str.find('Master.') >= 0, "Type"] = 2

df.loc[df["Name"].str.find('Don.') >= 0, "Type"] = 3

df.loc[df["Name"].str.find('Rev.') >= 0, "Type"] = 3

df.loc[df["Name"].str.find('Dr.') >= 0, "Type"] = 3

df['Type'] = df.Type.fillna(3)

df['Type'] = df['Type'].astype(int)



df.loc[df["Fare"] < 10,"Fare"] = 5

df.loc[(df["Fare"] >= 10) & (df["Fare"] < 25),"Fare"] = 3

df.loc[(df["Fare"] >= 25) & (df["Fare"] < 50),"Fare"] = 4

df.loc[(df["Fare"] >= 50) & (df["Fare"] < 100),"Fare"] = 2

df.loc[(df["Fare"] >= 100), 'Fare'] = 0

df['Fare'] = df.Type.fillna(3)

df['Fare'] = df['Fare'].astype(int)



df['Age'] = df.Age.fillna(meanAge)

df['Age'] = df['Age'].astype('int64')



df.loc[df["Age"] < 15,"Age"] = 0

df.loc[(df["Age"] >= 15) & (df["Age"] < 30),"Age"] = 3

df.loc[(df["Age"] >= 30) & (df["Age"] < 45),"Age"] = 1

df.loc[(df["Age"] >= 45) & (df["Age"] < 60),"Age"] = 2

df.loc[(df["Age"] >= 60), 'Age'] = 4



df.loc[df["Cabin"].str.find('A') >= 0, "Deck"] = 5

df.loc[df["Cabin"].str.find('B') >= 0, "Deck"] = 3

df.loc[df["Cabin"].str.find('C') >= 0, "Deck"] = 4

df.loc[df["Cabin"].str.find('D') >= 0, "Deck"] = 1

df.loc[df["Cabin"].str.find('E') >= 0, "Deck"] = 2

df.loc[df["Cabin"].str.find('F') >= 0, "Deck"] = 0

df.loc[df["Cabin"].str.find('G') >= 0, "Deck"] = 6

df['Deck'] = df.Deck.fillna(6)

df['Deck'] = df['Deck'].astype(int)



del df['Name']

del df['Ticket']

del df['Parch']

del df['SibSp']

del df['Cabin']



df.head()

data = np.array(df)
dfinal = model.predict(data)   ##XGBoost
preds = dfinal

preds = preds.astype('int64')

preds
preds[preds == 1] = 2

preds[preds == 0] = 1

preds[preds == 2] = 0

preds


my_submission = pd.DataFrame({'PassengerId': df.index, 'Survived': preds})





my_submission.to_csv('submission.csv', index=False)

my_submission.head(20)