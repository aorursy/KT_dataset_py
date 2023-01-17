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
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
train.info()
test.info()
train.head()
test.head()
train.describe()
train.groupby('Survived').mean()
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(x='Pclass',data=train)
sns.countplot(x='Pclass',hue='Survived',data=train)
sns.countplot(x='Sex',hue='Survived',data=train)
sns.countplot(x='Survived',hue='Sex',data=train)
g=plt.figure()

train[train.Survived==0].Age.hist(color='r',alpha=0.5)

train[train.Survived==1].Age.hist(color='g',alpha=0.5)

g.legend(labels='01',loc=0)
sns.distplot(train[train.Survived==0].Age.dropna(), hist=True)

sns.distplot(train[train.Survived==1].Age.dropna(), hist=True)
g=train.groupby(['SibSp','Survived'])

df=pd.DataFrame(g.count()['PassengerId'])

print(df)

sns.countplot(x='SibSp',hue='Survived',data=train)
g=train.groupby(['Parch','Survived'])

df=pd.DataFrame(g.count()['PassengerId'])

print(df)

sns.countplot(x='Parch',hue='Survived',data=train)
plt.figure(figsize=(15,10))

sns.distplot(train[train.Survived==0].Fare.dropna(), hist=True)

sns.distplot(train[train.Survived==1].Fare.dropna(), hist=True)
sns.countplot(x='Embarked',hue='Survived',data=train)
plt.figure(figsize=(10,10))

sns.heatmap(train.drop('PassengerId',axis=1).corr(),square=True,annot=True,cmap='YlGnBu')

plt.title('Correlation between features')
#Selecting Features.

selected_features=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

X_train=train[selected_features]#training Features

X_test=test[selected_features]#testing Features

y_train=train['Survived']#labels
X_train['Embarked'].fillna('S',inplace=True)

X_train['Age'].fillna(X_train['Age'].median(),inplace=True)

X_test['Age'].fillna(X_test['Age'].median(),inplace=True)

X_test['Fare'].fillna(X_test['Fare'].mean(),inplace=True)

from sklearn.feature_extraction import DictVectorizer

dict_vec=DictVectorizer(sparse=False)

X_train=dict_vec.fit_transform(X_train.to_dict(orient='record'))

dict_vec.feature_names_

X_test=dict_vec.transform(X_test.to_dict(orient='record'))
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

dtc=DecisionTreeClassifier(criterion='entropy')

rfc=RandomForestClassifier()

gbc=GradientBoostingClassifier()

xgbc=XGBClassifier()
from sklearn.model_selection import cross_val_score

print('DecisionTree:',cross_val_score(dtc,X_train,y_train,cv=5).mean())

print('RandomForest:',cross_val_score(rfc,X_train,y_train,cv=5).mean())

print('GradientBoosting:',cross_val_score(gbc,X_train,y_train,cv=5).mean())

print('XGBClassifier:',cross_val_score(xgbc,X_train,y_train,cv=5).mean())

gbc.fit(X_train,y_train)
gbc_y_predict=gbc.predict(X_test)

print(gbc_y_predict)
gbc_submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':gbc_y_predict})

gbc_submission.to_csv('gbc_submission_b.csv',index=False)
'''

import tensorflow as tf

from tensorflow import keras

X_train=np.asarray(X_train).reshape(891,10)

X_test=np.asarray(X_test).reshape(418,10)

y_train=np.asarray(y_train)

model = keras.Sequential([

    keras.layers.Dense(20,activation=tf.nn.relu,input_shape=(10,)),    

    keras.layers.Dense(30, activation=tf.nn.selu),

    keras.layers.Dense(2, activation=tf.nn.softmax)

])



model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.005),

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.fit(X_train, y_train, epochs=4000)

predictions = model.predict(X_test)



y_predictions=[]

for i in range(0,test.shape[0]):

    y_predictions.append(np.argmax(predictions[i]))



y_predictions_df=np.asarray(y_predictions)

y_predictions_df_submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':y_predictions_df})

y_predictions_df_submission.to_csv('C:/Users/h206765/Desktop/datasets/titanic/tf_submission_nn.csv',index=False)



'''
'''

X_train=train.drop(['PassengerId','Ticket','Survived'],axis=1)

X_test=test.drop(['PassengerId','Ticket'],axis=1)

y_train=train['Survived']



X_train['Embarked'].fillna('S',inplace=True)

X_train.loc[X_train.Cabin.notna(),'Cabin']='Yes'

X_train['Cabin'].fillna('NO',inplace=True)

X_test['Fare'].fillna(X_test['Fare'].mean(),inplace=True)

X_test.loc[X_train.Age.notna(),'Cabin']='Yes'

X_test['Cabin'].fillna('NO',inplace=True)



Xtitle=[]

for n in list(X_train['Name']):

    Xtitle.append(n.split(',')[1].split('.')[0])

XTitle=pd.Series(Xtitle)

ttitle=[]

for tn in list(X_test['Name']):

    ttitle.append(tn.split(',')[1].split('.')[0])

tTitle=pd.Series(ttitle)



X_train['Name']=XTitle

X_test['Name']=tTitle

def agepredict1(axtrain,aytrain,aytest):

    dict_vec0=DictVectorizer(sparse=False)

    axtrain_s=dict_vec0.fit_transform(axtrain.to_dict(orient='record'))

    aytest_s=dict_vec0.transform(aytest.to_dict(orient='record'))   

    dtc=tre.ExtraTreeRegressor()

    dtc.fit(axtrain_s,aytrain)

    age_pred=dtc.predict(aytest_s)

    return age_pred

    

def agepredict(axtrain,aytrain,aytest):

    dict_vec0=DictVectorizer(sparse=False)

    axtrain_s=dict_vec0.fit_transform(axtrain.to_dict(orient='record'))

    aytest_s=dict_vec0.transform(aytest.to_dict(orient='record'))

    ss_X=StandardScaler()

    axtrain_s=ss_X.fit_transform(axtrain_s)

    aytest_s=ss_X.transform(aytest_s)

    lr=SGDRegressor()

    lr.fit(axtrain_s,aytrain)

    age_pred=lr.predict(aytest_s)

    return age_pred



avr=X_train['Age'].median()

xtr1=X_train[X_train.Age.notna()].drop('Age',axis=1)

xtr2=X_test[X_test.Age.notna()].drop('Age',axis=1)

xtr=xtr1.append(xtr2)



ytr1=X_train[X_train.Age.notna()].Age

ytr2=X_test[X_test.Age.notna()].Age

ytr=ytr1.append(ytr2)

xte=X_train[X_train.Age.isna()].drop('Age',axis=1)

X_train.loc[X_train.Age.isna(),'Age']=agepredict(xtr,ytr,xte)

X_test.loc[X_test.Age.isna(),'Age']=agepredict(xtr,ytr,X_test[X_test.Age.isna()].drop('Age',axis=1))



'''