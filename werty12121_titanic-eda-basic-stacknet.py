# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

np.random.RandomState(seed=1)



import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

train_Id=train['PassengerId']

train=train.drop(['PassengerId'],axis=1)

test_Id=test['PassengerId']

test=test.drop(['PassengerId'],axis=1)

y=train['Survived']

train.info()
train.head()
sns.countplot(x='Survived',hue='Sex',data=train)
sns.barplot(y='Survived',x='Sex',data=train)
sns.barplot(y='Survived',x='Pclass',hue='Sex',data=train)
f, ax = plt.subplots(figsize=(60, 10))

sns.barplot(y='Survived',x='Age',data=train)
sns.barplot(y='Survived',x='SibSp',data=train)
sns.barplot(y='Survived',x='Parch',data=train)
f, ax = plt.subplots(figsize=(60, 10))

sns.barplot(y='Survived',x='Fare',data=train)
sns.barplot(y='Survived',x='Embarked',data=train)
alldata=pd.concat([train,test])

alldata.head()
alldata.isna().sum()
alldata['Fare']=alldata['Fare'].fillna(alldata['Fare'].mode().values[0])

alldata['Age']=alldata['Age'].fillna(alldata['Age'].median())

alldata['Embarked']=alldata['Embarked'].fillna(alldata['Embarked'].mode().values[0])

alldata['Surename']=alldata.Name.apply(lambda x: x.split(',')[0])

alldata['Title']=alldata.Name.apply(lambda x: x.split(',')[1].split('.')[0])

alldata['SurrnameFreq']=alldata.Surename.apply(lambda x: alldata.groupby('Surename').count().Age[x])

alldata['Deck']=alldata.Cabin.apply(lambda x: str(x)[0])

alldata['Family_Size']=alldata['SibSp']+alldata['Parch']

alldata['Age*Class']=alldata['Age']*alldata['Pclass']

alldata['Fare_Per_Person']=alldata['Fare']/(alldata['Family_Size']+1)

alldata['FareBin'] = pd.qcut(alldata['Fare'], 4)

alldata['AgeBin'] = pd.cut(alldata['Age'].astype(int), 5)

alldata=alldata.drop(['Cabin','Name','Ticket','Surename'],axis=1)
f, ax = plt.subplots(figsize=(30, 10))

sns.countplot(alldata[:len(train)].Title)
f, ax = plt.subplots(figsize=(30, 10))

sns.barplot(y='Survived',x='Title',data=alldata[:len(train)])
alldata.isna().sum()
# alldata['Age']=alldata['Age'].astype('category')
alldata=pd.get_dummies(alldata)

alldata.info()
num_feat=alldata.dtypes[alldata.dtypes!="object"].index

from sklearn.preprocessing import StandardScaler

scX = StandardScaler()

alldata[num_feat] = scX.fit_transform(alldata[num_feat].values)

# alldata['Fare_Per_Person'] = scX.fit_transform(np.array([alldata['Fare_Per_Person'].values]).T)

# alldata['Family_Size'] = scX.fit_transform(np.array([alldata['Family_Size'].values]).T)

# alldata['Age*Class'] = scX.fit_transform(np.array([alldata['Age*Class'].values]).T)
alldata.head()
alldata.shape
from sklearn.model_selection import train_test_split,KFold,cross_val_score
alldata=alldata.drop(['Survived'],axis=1)

train=alldata[:len(train)]

test=alldata[len(train):]

from sklearn.metrics import accuracy_score

kfolds = KFold(n_splits=5, shuffle=True,random_state=1)

def acc_cv(model):

    acc= cross_val_score(model, train.values, y.values, scoring="accuracy", cv = kfolds.get_n_splits(train.values))

    return acc
X_train, X_test, y_train, y_test = train_test_split(train.values, y.values, test_size=0.4,random_state=100)

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split,GridSearchCV



import xgboost as xgb

import lightgbm as lgb

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import BaggingClassifier,VotingClassifier,GradientBoostingClassifier,AdaBoostClassifier,RandomForestClassifier

from sklearn.svm import LinearSVC

class StackNet(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models,meta_final_model, meta_models1=None, meta_models2=None,add_prev_out=True, n_folds=10):

        self.base_models = base_models

        self.meta_models1 = meta_models1

        self.meta_models2 = meta_models2

        self.meta_final_model=meta_final_model

        self.n_folds = n_folds

        self.add_prev_out=add_prev_out

   

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        if self.meta_models1!=None:

            self.meta_models1_ = [list() for x in self.meta_models1]

        if self.meta_models2!=None:

            self.meta_models2_ = [list() for x in self.meta_models2]

        self.meta_final_model_=clone(self.meta_final_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

       

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X[train_index], y[train_index])

                y_pred = instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

        if self.add_prev_out:

            out_of_fold_predictions=np.hstack((out_of_fold_predictions,X))

        if self.meta_models1!=None:

            out_of_fold_predictions2 = np.zeros((X.shape[0],len(self.meta_models1)))

       

        if self.meta_models1!=None:

            for i, model in enumerate(self.meta_models1):



                for train_index, holdout_index in kfold.split(out_of_fold_predictions, y):

                    instance = clone(model)

                    self.meta_models1_[i].append(instance)

                    instance.fit(out_of_fold_predictions[train_index], y[train_index])

                    y_pred = instance.predict(out_of_fold_predictions[holdout_index])

                    out_of_fold_predictions2[holdout_index, i] = y_pred                           

            if self.add_prev_out:

                out_of_fold_predictions2=np.hstack((out_of_fold_predictions2,X))

        else:

            out_of_fold_predictions2=out_of_fold_predictions

            if self.meta_models2!=None:

                out_of_fold_predictions3 = np.zeros((X.shape[0],len(self.meta_models2)))

        

        

        if self.meta_models2!=None:

            for i, model in enumerate(self.meta_models2):



                for train_index, holdout_index in kfold.split(out_of_fold_predictions2, y):

                    instance = clone(model)

                    self.meta_models2_[i].append(instance)

                    instance.fit(out_of_fold_predictions2[train_index], y[train_index])

                    y_pred = instance.predict(out_of_fold_predictions2[holdout_index])

                    out_of_fold_predictions3[holdout_index, i] = y_pred                           

            if self.add_prev_out:

                out_of_fold_predictions3=np.hstack((out_of_fold_predictions3,X))         

        else:

            out_of_fold_predictions3=out_of_fold_predictions2

                                            

        self.meta_final_model_.fit(out_of_fold_predictions3, y)

        return self

   

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_])

        if self.add_prev_out:

            meta_features=np.hstack((meta_features,X))

            

        if self.meta_models1!=None:   

            meta_features2 = np.column_stack([

                np.column_stack([model.predict(meta_features) for model in meta_models1_]).mean(axis=1)

                for meta_models1_ in self.meta_models1_ ])

            if self.add_prev_out:

                meta_features2=np.hstack((meta_features2,X))

        else:

            meta_features2=meta_features

            

        if self.meta_models2!=None:  

            meta_features3 = np.column_stack([

                np.column_stack([model.predict(meta_features2) for model in meta_models2_]).mean(axis=1)

                for meta_models2_ in self.meta_models2_ ])

            if self.add_prev_out:

                meta_features3=np.hstack((meta_features3,X))

        else:

            meta_features3=meta_features2

            

            

        return self.meta_final_model_.predict(meta_features3)


lr=LogisticRegression(random_state=1)

# print(lr.__class__.__name__,acc_cv(lr).mean(),acc_cv(lr).std())



xgbm=xgb.XGBClassifier(objective='binary:hinge',random_state=1)

# print(xgbm.__class__.__name__,acc_cv(xgbm).mean(),acc_cv(xgbm).std())



lgbmm=lgb.LGBMClassifier(objective='huber',random_state=1)

# print(lgbmm.__class__.__name__,acc_cv(lgbmm).mean(),acc_cv(lgbmm).std())



gbc=GradientBoostingClassifier(random_state=1)

# print(gbc.__class__.__name__,acc_cv(gbc).mean(),acc_cv(gbc).std())



adc=AdaBoostClassifier(random_state=1)

# print(adc.__class__.__name__,acc_cv(adc).mean(),acc_cv(adc).std())



rf=RandomForestClassifier(random_state=1,n_jobs=-1,n_estimators=100)

# print(rf.__class__.__name__,acc_cv(rf).mean(),acc_cv(rf).std())



bc=BaggingClassifier(xgb.XGBClassifier(objective='binary:hinge',random_state=1),n_estimators =3,random_state=1)

# print(acc_cv(bc).mean(),acc_cv(bc).std())

bc.fit(X_train,y_train)

print(bc.__class__.__name__,accuracy_score(bc.predict(X_test),y_test))



sn=StackNet((lr,gbc,adc,rf,bc,lgbmm,xgbm),xgbm)

# print(acc_cv(sn).mean(),acc_cv(sn).std())

sn.fit(X_train,y_train)

print(sn.__class__.__name__,accuracy_score(sn.predict(X_test),y_test))


sn.fit(train.values,y.values)



submission = pd.read_csv('../input/gender_submission.csv')

submission['Survived'] = sn.predict(test.values)

submission.to_csv('submission.csv', index=False)