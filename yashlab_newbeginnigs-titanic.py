import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
comb_df = pd.concat([train,test])
comb_df.isnull().sum()
comb_df['Cabin'] = comb_df['Cabin'].fillna('DECK_CREW')



# We assume that they embarked at Southampton itself (It was the starting point.)

comb_df['Embarked'] = comb_df['Embarked'].fillna('S')
pd.crosstab(train.Pclass,train.Survived).div(pd.crosstab(train.Pclass,train.Survived).sum(1),axis=0).plot.bar(stacked=True).grid()
pd.crosstab(train.Sex,train.Survived).div(pd.crosstab(train.Sex,train.Survived).sum(1),axis=0).plot.bar(stacked=True).grid()
gen_clas_surv = pd.crosstab([train['Sex'],train['Pclass']],train.Survived)



gen_clas_surv.div(gen_clas_surv.sum(1),axis=0).plot.bar()
sns.heatmap(train.corr(),annot=True)
comb_df.sample(5)
# 4. Name

# Possible extractions: Salutation, First Name, Last Name, Nicknames
comb_df['Last_Name'] = comb_df.Name.str.split(',').apply(lambda x: x[0])
comb_df['Salut'] = comb_df.Name.str.split(',').apply(lambda x: x[1]).apply(str.split).apply(lambda x: x[0])
# Let's check if we got anything meanigful

comb_df['Salut'].value_counts()
comb_df.loc[comb_df['Salut'] == 'Master.','Boy-o-Girl'] = 'Boy'

comb_df.loc[comb_df['Salut'].isin(['Miss.','Ms.','Mme.','Mlle']),'Boy-o-Girl'] = 'Girl'
comb_df.loc[comb_df['Salut'] == 'Mr.','Boy-o-Girl']='Man'

comb_df['Boy-o-Girl'].fillna('Women',inplace=True)
comb_df

pd.crosstab(comb_df['Boy-o-Girl'],comb_df['Survived'])
# create feature of upper class

comb_df['Salut'].unique()
novelty = ['Don.', 'Rev.', 'Dr.','Jonkheer.', 'Dona.','Sir']

military = ['Major.','Col.','Capt.']
comb_df.loc[comb_df['Salut'].isin(novelty),'Social_Class'] = 'Novelty'

comb_df.loc[comb_df['Salut'].isin(military),'Social_Class']= 'Military'

comb_df['Social_Class'].fillna('Common',inplace=True)
comb_df.isnull().sum()
comb_df.groupby(['Boy-o-Girl'])['Age'].mean()['Boy']
comb_df['Age'] = comb_df.apply(

    lambda row: comb_df.groupby(['Boy-o-Girl'])['Age'].median()[row['Boy-o-Girl']] if np.isnan(row['Age']) else row['Age'],

    axis=1

)

    

comb_df
train_df  = comb_df[~comb_df.Survived.isnull()]

test_df = comb_df[comb_df.Survived.isnull()]
from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(train_df.drop('Survived',axis=1),

                                               train_df['Survived'],

                                               test_size=0.2,

                                               random_state =123)
comb_df.sample(5)
model_cols = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Boy-o-Girl','Social_Class','Last_Name','Cabin']
cat_cols = ['Sex','Embarked','Boy-o-Girl','Social_Class','Last_Name','Cabin']
from catboost import CatBoostClassifier

from lightgbm import LGBMClassifier
cb = CatBoostClassifier()
cb.fit(X_train[model_cols],y=y_train,eval_set=(X_val[model_cols],y_val),cat_features=cat_cols)
from sklearn.metrics import accuracy_score
accuracy_score(y_val,cb.predict(X_val[model_cols]))
submit1 = pd.DataFrame()

submit1['PassengerId']=test_df.PassengerId

submit1['Survived']=cb.predict(test_df.drop(['PassengerId','Name'],axis=1)[model_cols])
submit1.to_csv('feature_submit.csv',index=False)