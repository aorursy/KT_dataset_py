import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test= pd.read_csv('../input/test.csv')

gender_submission = pd.read_csv('../input/gender_submission.csv')

y_train = train['Survived']

train_df = train.drop(['Survived'],axis=1)

n_train = len(train_df.index)

n_test = len(test.index)

all_data =pd.concat([train_df,test],axis=0)

all_data = all_data.reset_index(drop = True)

y_train = y_train.to_frame()
all_data.isnull().sum()
all_data['Pclass'].value_counts()
all_data['Initial']=0

for i in all_data:

    all_data['Initial']=all_data.Name.str.extract('([A-Za-z]+)\.') 

all_data.head()

all_data['Initial'].unique()
all_data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Mr'],inplace=True)
all_data.head()
all_data.groupby('Initial')['Age'].mean()
all_data.loc[(all_data.Age.isnull())&(all_data.Initial=='Master'),'Age'] = 5

all_data.loc[(all_data.Age.isnull())&(all_data.Initial=='Miss'),'Age'] = 22

all_data.loc[(all_data.Age.isnull())&(all_data.Initial=='Mr'),'Age'] = 33

all_data.loc[(all_data.Age.isnull())&(all_data.Initial=='Mrs'),'Age'] = 37

all_data.loc[(all_data.Age.isnull())&(all_data.Initial=='Other'),'Age'] = 45
all_data.isnull().sum()
all_data['Embarked'].fillna(all_data['Embarked'].mode()[0],inplace = True)

all_data.isnull().sum()

all_data
all_data['Age_Band']=0

all_data.loc[all_data['Age']<=16,'Age_Band'] = 0

all_data.loc[(all_data['Age']>16)&(all_data['Age']<=32),'Age_Band'] =1

all_data.loc[(all_data['Age']>32)&(all_data['Age']<=48),'Age_Band'] =2

all_data.loc[(all_data['Age']>48)&(all_data['Age']<=64),'Age_Band'] =3

all_data.loc[all_data['Age']>46,'Age_Band'] =4

all_data

all_data['Family_Size'] = 0

all_data['Family_Size'] = all_data['SibSp'] + all_data['Parch']

all_data.head()
all_data['Alone']=0

all_data.loc[all_data['Family_Size']==0,'Alone']=1

all_data
all_data['Fare'].fillna(all_data[all_data['Pclass']==1]['Fare'].mean(),inplace = True)
all_data.isnull().sum()
all_data['Fare_Range'] = pd.qcut(all_data['Fare'],4)

all_data['Fare_Range'].unique()
all_data['Fare_Cat'] = 0

all_data.loc[(all_data['Fare']>-0.001)&(all_data['Fare']<=7.896),'Fare_Cat']=0

all_data.loc[(all_data['Fare']>7.896)&(all_data['Fare']<=14.454),'Fare_Cat']=1

all_data.loc[(all_data['Fare']>14.454)&(all_data['Fare']<=31.275),'Fare_Cat']=2

all_data.loc[(all_data['Fare']>31.275)&(all_data['Fare']<=512.329),'Fare_Cat']=3
all_data['Has_Cabin'] = all_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

all_data
all_data= all_data.drop(['PassengerId','Name','Ticket','Fare','Fare_Range','Age'],axis=1)

all_data.head()
all_data= pd.get_dummies(all_data)

all_data.head()
X_train = all_data[:n_train]

X_test = all_data[n_train:]

X=X_train

y=y_train

test=X_test
#importing all the required ML packages

from sklearn.linear_model import LogisticRegression #logistic regression

from sklearn import svm #support vector Machine

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier #Random Forest

from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.naive_bayes import GaussianNB #Naive bayes

from sklearn.tree import DecisionTreeClassifier #Decision Tree

from sklearn.model_selection import train_test_split,cross_val_score, StratifiedKFold#training and testing data split

from sklearn.metrics import accuracy_score,roc_auc_score #accuracy measure

from xgboost import XGBClassifier ,plot_importance

import xgboost as xgb

from sklearn.svm import SVC

def train_model(classifier):

    '''This function is used to train and print the accuracy of our models'''

    folds = StratifiedKFold(n_splits=5, random_state=42)

    accuracy = np.mean(cross_val_score(classifier, X, y, scoring="accuracy", cv=folds, n_jobs=-1))

    print('CV Accuracy:', accuracy)

    return accuracy
# Lists that keep track cross val means and algorithm names

cv_means = []

alg_list = []
model_svm=SVC(kernel='rbf',C=1,gamma=0.1,random_state=42,probability=True)

model_svm_acc =train_model(model_svm)

cv_means.append(model_svm_acc)

alg_list.append("SVM")

model_svm.fit(X,y)
model_rbf=SVC(kernel='rbf',C=1,gamma=0.1,probability=True)

model_rbf_acc =train_model(model_rbf)

cv_means.append(model_rbf_acc)

alg_list.append("RBF")

model_rbf.fit(X,y)
model_lr=LogisticRegression(penalty='l2',C=0.4,max_iter=100,random_state=42)

model_lr_acc =train_model(model_lr)

cv_means.append(model_lr_acc)

alg_list.append("LR")

model_lr.fit(X,y)
model_dt=DecisionTreeClassifier(max_depth=8,min_samples_split =9,

                                           min_samples_leaf =3,random_state=42)

model_dt_acc =train_model(model_lr)

cv_means.append(model_dt_acc)

alg_list.append("Decision Tree")

model_dt.fit(X,y)
model_knn=KNeighborsClassifier(weights='uniform',n_neighbors=9,p=1,leaf_size=90)

model_knn_acc =train_model(model_knn)

cv_means.append(model_knn_acc)

alg_list.append("KNN")

model_knn.fit(X,y)
model_rfc=RandomForestClassifier(n_estimators=300, max_depth=25, 

                                min_samples_split=2, min_samples_leaf=2,

                                max_features="log2", random_state=42) 

model_rfc_acc =train_model(model_rfc)

cv_means.append(model_rfc_acc)

alg_list.append("RandomForestClassifier")

model_rfc.fit(X,y)
model_nb=GaussianNB() 

model_nb_acc =train_model(model_nb)

cv_means.append(model_nb_acc)

alg_list.append("GaussianNB")

model_nb.fit(X,y)
model_grad=GradientBoostingClassifier(n_estimators=600,random_state=42,learning_rate=0.1)

model_grad_acc =train_model(model_grad)

cv_means.append(model_grad_acc)

alg_list.append("GradientBoostingClassifier")

model_grad.fit(X,y)
model_ada=AdaBoostClassifier(n_estimators=200,random_state=42,learning_rate=0.1)

model_ada_acc =train_model(model_ada)

cv_means.append(model_ada_acc)

alg_list.append("AdaBoost")

model_ada.fit(X,y)
from lightgbm import LGBMClassifier

# Initialize the model

model_lgbm = LGBMClassifier(num_leaves=31, learning_rate=0.1, 

                      n_estimators=64, random_state=42, n_jobs=-1)

# Validate the model

model_lgbm_acc = train_model(model_lgbm)

cv_means.append(model_lgbm_acc)

alg_list.append("LGBM")

model_lgbm.fit(X, y)
model_xgb_1 = XGBClassifier(learning_rate =0.1, n_estimators=80, max_depth=5,

                                                  min_child_weight=3, gamma=0, subsample=1.0, 

                                                  colsample_bytree=0.9,reg_alpha = 0.011,

                                                  objective= 'binary:logistic', 

                                                  scale_pos_weight=1,seed=27, nthread = -1)

model_xgb_acc_1 =train_model(model_xgb_1)

cv_means.append(model_xgb_acc_1)

alg_list.append("XGboost")

model_xgb_1.fit(X,y)
# Create a performance DF with score and Algorithm name

performance_df = pd.DataFrame({"Algorithms": alg_list, "CrossValMeans":cv_means})



# Plot the performace of all models

g = sns.barplot("CrossValMeans","Algorithms", data = performance_df.sort_values(by="CrossValMeans",ascending=False),

                palette="Set3",orient = "h")

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = [model_svm, model_rbf,model_ada,model_grad,model_lgbm]
from vecstack import stacking

S_train, S_test = stacking(models,

                           X_train, y_train, X_test,

                           regression=False,

                           mode='oof_pred_bag',

                           n_folds=5,

                           save_dir=None,

                           needs_proba=False,

                           random_state=42,

                           stratified=True,

                           shuffle=True,

                           verbose=2

                          )
model_xgb_2 = XGBClassifier(learning_rate =0.1, n_estimators=80, max_depth=5,

                                                  min_child_weight=3, gamma=0, subsample=1.0, 

                                                  colsample_bytree=0.9,reg_alpha = 0.011,

                                                  objective= 'binary:logistic', 

                                                  scale_pos_weight=1,seed=27, nthread = -1)

model_xgb_2.fit(S_train, y_train)

stacked_pred = model_xgb_2.predict(S_test)

print('Final prediction score: ', accuracy_score(y_test, stacked_pred))
y1_pred_L1 = models[0].predict(test)

y2_pred_L1 = models[1].predict(test)

y3_pred_L1 = models[2].predict(test)

y4_pred_L1 = models[3].predict(test)

y5_pred_L1 = models[4].predict(test)

S_test_L1 = np.c_[y1_pred_L1, y2_pred_L1, y3_pred_L1,y4_pred_L1,y5_pred_L1]

test_stacked_pred = model_xgb_2.predict(S_test_L1)
from sklearn.ensemble import VotingClassifier

folds = StratifiedKFold(n_splits=5, random_state=42)

ensemble_lin_rbf=VotingClassifier(estimators=[('KNN',model_knn),

                                               ('XGBoost',model_xgb_1),

                                              ('RBF',model_rbf),

                                              ('svm',model_svm),

                                              ('grad',model_grad),

                                             

                                        ], voting='soft')

ensemble_lin_rbf.fit(X,y)

cv_result_ensemble_lin_rbf = np.mean(cross_val_score(ensemble_lin_rbf,X,y,

                                                     cv = folds,scoring = "accuracy",n_jobs=-1))

print('The cv score of ensemble_lin_rbf is ' ,cv_result_ensemble_lin_rbf)
#model_xgb.fit(X_train.values,y_train.values)

#y_pre = model_xgb.predict(X_test.values)

gender_submission['Survived'] = test_stacked_pred

gender_submission.to_csv('submission.csv',index=False)