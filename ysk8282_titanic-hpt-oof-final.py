import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('seaborn')

sns.set(style='white', context='notebook', palette='deep', font_scale=1.5)

import missingno as msno



#ignore warnings

import warnings

warnings.filterwarnings('ignore')



%matplotlib inline



pd.set_option('display.width', 1000)

pd.set_option('display.max_columns', 100)

pd.set_option('display.float_format', lambda x: '%.2f' % x)



colormap = plt.cm.RdBu
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
test_df = pd.DataFrame(test_data)

test_df.set_index('PassengerId', inplace=True)

train_df = pd.DataFrame(train_data)

train_df.set_index('PassengerId', inplace=True)



test_index = test_df.index

train_index = train_df.index



train_df.head(5)
# test_df
y_train_df = train_df.pop('Survived')



tmp = train_df.copy()

# tmp

all_df = tmp.append(test_df)

all_df.head()
# NaN

all_df['Cabin'].fillna('X', inplace=True)

all_df.loc[all_df['Cabin']!='X', 'Cabin'] = 'O'



all_df['Embarked'].fillna('S', inplace=True)



# new feature

all_df['FamilySize'] = all_df['SibSp'] + all_df['Parch'] + 1 



# NaN

all_df['Fare'].fillna(all_df['Fare'].median(), inplace=True)
# new feature

import re

arr_name = all_df['Name'].values

title=[]

for i, full_name in enumerate(arr_name):

    n = re.findall(r'([A-Za-z]+)\.', full_name)

    if len(n) > 1:

        n.pop()

#         print(i, n)    # 자꾸 2개로 나오는 놈 처리.

    title.extend(n)



arr_title = np.array(title)

all_df['Title'] = arr_title
all_df['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'], 

                          ['Miss','Miss','Miss','Other','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mrs'],

                          inplace=True)

all_df['Title'].unique()
del all_df['Name']
# labeling

all_df['Sex'].replace({'male':0, 'female':1}, inplace=True)
tmp = all_df.copy()

tmp['Title'].replace({'Mr':0, 'Mrs':1, 'Miss':2, 'Master':3, 'Other':4}, inplace=True)

g = sns.heatmap(tmp[["Age","Sex","SibSp","Parch","Pclass", "Title"]].corr(),cmap=colormap,annot=True)
# NaN

# 성별과 나이와의 연관성은 적어보이며, 나머지 SibSp, Parch, Pclass, Title와 연관성이 높아보임

all_df['Age'].fillna(all_df.groupby(['Pclass', 'Title', 'SibSp', 'Parch'])['Age'].transform('mean'), inplace=True)        

# all_df['Age'].fillna(all_df.groupby(['Pclass','SibSp', 'Parch'])['Age'].transform('mean'), inplace=True)         

all_df['Age'].fillna(all_df['Age'].median(), inplace=True)



# all_df['Age'].fillna(all_df.groupby(['Pclass'])['Age'].transform('mean'), inplace=True) 
# new feature

tmp = all_df['Ticket'].values.tolist()

t_label = list(map(lambda x: x.replace(".","").replace("/","").strip().split(' ')[0] if not x[0].isdigit() else 'X', tmp))

# t_label = list(map(lambda x: x.replace(".","").replace("/","").strip().split(' ')[0] if not x[0].isdigit() else x[0], tmp))

# t_label = list(map(lambda x: 0 if not x[0].isdigit() else 1, tmp))

# t_label = list(map(lambda x: 'STR' if not x[0].isdigit() else 'NUMBER', tmp))



label_arr = np.array(t_label).reshape(-1,1)

label_arr
all_df['Ticket'] = label_arr
# continuous value



# 0보다 큰 값에 log

all_df['Fare'] = all_df['Fare'].map(lambda i: np.log(i) if i>0 else 0)



# Rescaling

from sklearn import preprocessing



# std_scaler = preprocessing.StandardScaler().fit(all_df['Fare'].values.reshape(-1,1))

# all_df['Fare'] = std_scaler.transform(all_df['Fare'].values.reshape(-1,1))



# mm_scaler = preprocessing.MinMaxScaler().fit(all_df['Fare'].values.reshape(-1,1))

# all_df['Fare'] = mm_scaler.transform(all_df['Fare'].values.reshape(-1,1))



mm_scaler = preprocessing.MinMaxScaler().fit(all_df[['Age','Fare']])

all_df[['Age','Fare']] = mm_scaler.transform(all_df[['Age','Fare']])
# feature 분석용

from collections import defaultdict



label_list =sorted(list(set(t_label)), key = lambda n: n.lower())



d = defaultdict(object) # default dict 생성

d = defaultdict(lambda : 0) # default 값을 0으로 설정

i=1

for label in label_list:

    d[label] = i

    i = i+1
# feature 분석용

f = all_df[all_df.index.isin(train_index)].copy()

f['Title'].replace({'Mr':0, 'Mrs':1, 'Miss':2, 'Master':3, 'Other':4}, inplace=True)

f['Embarked'].replace({'C':0, 'Q':1, 'S':2}, inplace=True)

f['Cabin'].replace({'X':0, 'O':1}, inplace=True)

f['Survived'] = y_train_df

f['Ticket'] = f['Ticket'].map(d)
# Cabin, Title, Embarked, Ticket는 one-hot incoding



# all_df = pd.get_dummies(all_df)

all_df = all_df.merge(pd.get_dummies(all_df['Cabin'], prefix='Cabin'),

                     left_index=True, right_index=True)

del all_df['Cabin']



all_df = all_df.merge(pd.get_dummies(all_df['Embarked'], prefix='Embarked'),

                     left_index=True, right_index=True)

del all_df['Embarked']



all_df = all_df.merge(pd.get_dummies(all_df['Title'], prefix='Title'),

                     left_index=True, right_index=True)

del all_df['Title']



all_df = all_df.merge(pd.get_dummies(all_df['Ticket'], prefix='Ticket'),

                     left_index=True, right_index=True)

del all_df['Ticket']
all_df.head()
# Pearson Correlation

# feature 간 상관관계

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(f.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
# 모델 평가

from sklearn import metrics 

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error



# training set을 쉽게 나눠주는 함수

from sklearn.model_selection import train_test_split  



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

import xgboost

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.tree import DecisionTreeClassifier



from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, cross_validate, KFold
train_df = all_df[all_df.index.isin(train_index)]   # 모델 학습용 데이터

test_df = all_df[all_df.index.isin(test_index)] # 예측해야하는 데이터
def HPT(model, kf, grid, X, y):



    gs = GridSearchCV(model, grid, cv=kf, scoring="accuracy", n_jobs= 4, verbose = 1)



    gs.fit(X,y)

    

    return gs
kfold = KFold(n_splits=10)



# SVC, ADA, RF, GB, ETC



print("SVC Parameters tuning")

SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1, 1],

                  'C': [1, 10, 50, 100, 200, 300]}

svc_es = HPT(SVMC, kfold, svc_param_grid, train_df, y_train_df)

print()



print("ADA Parameters tuning") 

DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "algorithm" : ["SAMME","SAMME.R"],

              "n_estimators" :[1,2],

              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

ada_es = HPT(adaDTC, kfold, ada_param_grid, train_df, y_train_df)

print()



print("RF Parameters tuning")

RFC = RandomForestClassifier()

rf_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10],

#                  "max_features": [0.05, 0.15, 0.5],

#                  "max_features": [0.05, 0.25, 1.0],

#                  "max_features": [0.05, 0.15, 0.45],

#                  "max_features": [0.05, 0.15, 0.25, 0.45, 0.5, 1.0],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,200,300],

              "criterion": ["gini"]}

rf_es = HPT(RFC, kfold, rf_param_grid, train_df, y_train_df)

print()



print("GB Parameters tuning") 

GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.1, 0.3] 

              }

gb_es = HPT(GBC, kfold, gb_param_grid, train_df, y_train_df)

print()



print("ETC Parameters tuning") 

ExtC = ExtraTreesClassifier()

ex_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10],

#                  "max_features": [0.05, 0.15, 0.5],

#                  "max_features": [0.05, 0.25, 1.0],

#                  "max_features": [0.05, 0.15, 0.45],

#                  "max_features": [0.05, 0.15, 0.25, 0.45, 0.5, 1.0],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,200,300],

              "criterion": ["gini"]}

etc_es = HPT(ExtC, kfold, ex_param_grid, train_df, y_train_df)

print()
# lr_best_score = lr_es.best_score_

# print("LR BEST: ", lr_best_score)



# xgb_best_score = xgb_es.best_score_

# print("XGB BEST: ", xgb_best_score)



svc_best_score = svc_es.best_score_

print("SVC BEST: ", svc_best_score)



ada_best_score = ada_es.best_score_

print("ADA BEST: ", ada_best_score)



rf_best_score = rf_es.best_score_

print("RF BEST: ", rf_best_score)



gb_best_score = gb_es.best_score_

print("GB BEST: ", gb_best_score)



etc_best_score = etc_es.best_score_

print("ETC BEST: ", etc_best_score)
# lr_best = lr_es.best_estimator_

# xgb_best = xgb_es.best_estimator_

svc_best = svc_es.best_estimator_

ada_best = ada_es.best_estimator_

rf_best = rf_es.best_estimator_

gb_best = gb_es.best_estimator_

etc_best = etc_es.best_estimator_
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, 

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    

    """Generate a simple plot of the test and training learning curve"""

    

    plt.figure()

    plt.title(title)

    

    if ylim is not None:

        plt.ylim(*ylim)

        

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
g = plot_learning_curve(svc_best,"SVC learning curves",train_df,y_train_df,cv=kfold)

g = plot_learning_curve(ada_best,"AdaBoost learning curves",train_df,y_train_df,cv=kfold)

g = plot_learning_curve(rf_best,"RandomForest learning curves",train_df,y_train_df,cv=kfold)

g = plot_learning_curve(gb_best,"GradientBoosting learning curves",train_df,y_train_df,cv=kfold)

g = plot_learning_curve(etc_best,"ExtraTrees learning curves",train_df,y_train_df,cv=kfold)
# RF, ETC, ADA, GB



rf_best.fit(train_df.values, y_train_df.ravel())

rf_features = rf_best.feature_importances_



etc_best.fit(train_df.values, y_train_df.ravel())

etc_features = etc_best.feature_importances_



ada_best.fit(train_df.values, y_train_df.ravel())

ada_features = ada_best.feature_importances_



gb_best.fit(train_df.values, y_train_df.ravel())

gb_features = gb_best.feature_importances_



# train_df.columns[indices].values[:7]

# rf_features[indices][:7]

# std[indices][:7]

fig = plt.figure(figsize=(40,15))

plt.title("Feature importance")



std = np.std([tree.feature_importances_ for tree in rf_best.estimators_], axis=0)

indices = np.argsort(rf_features)[::-1]

fig.add_subplot(4,1,1)

plt.bar(train_df.columns[indices].values[:7], rf_features[indices][:7],

        color="r", yerr=std[indices][:7], align="center")

plt.legend(['RandomForest'])



std = np.std([tree.feature_importances_ for tree in etc_best.estimators_], axis=0)

indices = np.argsort(etc_features)[::-1]

fig.add_subplot(4,1,2)

plt.bar(train_df.columns[indices].values[:7], etc_features[indices][:7],

        color="g", yerr=std[indices][:7], align="center")

plt.legend(['ExtraTrees'])



std = np.std([tree.feature_importances_ for tree in ada_best.estimators_], axis=0)

indices = np.argsort(ada_features)[::-1]

fig.add_subplot(4,1,3)

plt.bar(train_df.columns[indices].values[:7], ada_features[indices][:7],

        color="b", yerr=std[indices][:7], align="center")

plt.legend(['AdaBoost'])



std = np.std([tree.feature_importances_ for tree in gb_best.estimators_.flatten().tolist()], axis=0)

indices = np.argsort(gb_features)[::-1]

fig.add_subplot(4,1,4)

plt.bar(train_df.columns[indices].values[:7], gb_features[indices][:7],

        color="black", yerr=std[indices][:7], align="center")

plt.legend(['GradientBoosting'])



plt.show()    


def get_oof(clf, x_train, y_train, x_test):

    ntrain = X.shape[0]

    ntest = x_test.shape[0]

    NFOLDS = 5

    kf = KFold(n_splits=NFOLDS)

    

    oof_train = np.zeros((ntrain,))    # 검증 결과 저장

    

    oof_test_skf = np.empty((NFOLDS, ntest))    # iteration마다 모델로 예측한 결과 저장

    oof_test = np.zeros((ntest,))    # 결과 평균 값으로 저장.



    for i, (train_index, test_index) in enumerate(kf.split(x_train)):

        # train set

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        

        # validation set

        x_val = x_train[test_index]



        clf.fit(x_tr, y_tr)    # 예측 모델 생성



        oof_train[test_index] = clf.predict(x_val)    # 검증 결과

        oof_test_skf[i, :] = clf.predict(x_test)    # 한 KFlod에 대한 예측 결과 저장.



    oof_test[:] = oof_test_skf.mean(axis=0)   # 예측 결과 평균

    

    # new feature set

    # 학습셋x와 예측셋x

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
# LR, xgb, SVC, ADA, RF, GB, ET

X, y, test = train_df.values, y_train_df.ravel(), test_df.values

ntrain = X.shape[0]

ntest = test_df.shape[0]



# lr_oof_train, lr_oof_test = get_oof(lr_best, X, y, test)

# xgb_oof_train, xgb_oof_test = get_oof(xgb_best, X, y, test)

svc_oof_train, svc_oof_test = get_oof(svc_best, X, y, test)

ada_oof_train, ada_oof_test = get_oof(ada_best, X, y, test)

rf_oof_train, rf_oof_test = get_oof(rf_best, X, y, test)

gb_oof_train, gb_oof_test = get_oof(gb_best, X, y, test)

etc_oof_train, etc_oof_test = get_oof(etc_best, X, y, test)
# First-level output as new features

base_predictions_train = pd.DataFrame( {#'LR': lr_oof_train.ravel(),

                                        #'XGB': xgb_oof_train.ravel(),

                                        'SVC': svc_oof_train.ravel(),

                                        'AdaBoost': ada_oof_train.ravel(),

                                        'RandomForest': rf_oof_train.ravel(),

                                        'GradientBoost': gb_oof_train.ravel(),

                                        'ExtraTrees': etc_oof_train.ravel()

                                       })

base_predictions_train.head()
import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



data = [go.Heatmap(

        z= base_predictions_train.astype(float).corr().values ,

        x=base_predictions_train.columns.values,

        y= base_predictions_train.columns.values,

          colorscale='Viridis',

            showscale=True,

            reversescale = True)]

py.iplot(data, filename='labelled-heatmap')
x_train = np.concatenate((#lr_oof_train,

                          #xgb_oof_train,

                          svc_oof_train,

                          ada_oof_train,

                          rf_oof_train,

                          gb_oof_train,

                          etc_oof_train

                         ), axis=1)

y_train = y

x_test = np.concatenate((#lr_oof_test,

                         #xgb_oof_test,

                         svc_oof_test,

                         ada_oof_test,

                         rf_oof_test,

                         gb_oof_test,

                         etc_oof_test

                        ), axis=1)
kf = KFold(n_splits=10, shuffle=False)



print("Xgboost Parameters tuning")

XGB = xgboost.XGBClassifier()

xgbm_param_grid = {'learning_rate': [.01, .03, .05, .1, .25],

                   "n_estimators" :[300, 500],

                   "max_depth" : [1,2,4,6,8,10],

#                    "min_child_weight" : [2,3],

#                    'gamma': [0.5, 1],

                   #"objective" : ['binary:logistic']

                   'seed': [0]

                  }

xgb_es = HPT(XGB, kf, xgbm_param_grid, x_train, y_train)



xgb_best = xgb_es.best_estimator_

xgb_cv = cross_validate(xgb_best, x_train, y_train, cv=kf)



print("Xgboost Training core mean: {:.2f}". format(xgb_cv['train_score'].mean()*100)) 

print("Xgboost Test score mean: {:.2f}". format(xgb_cv['test_score'].mean()*100))
print("# LogisticRegression Parameters tuning")

LR = LogisticRegression()

lr_param_grid={"C":np.logspace(-3,3,7), 

               #"penalty":["l1","l2"], # l1 lasso l2 ridge, 

            'random_state': [0],

            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

              'fit_intercept': [True,False]}   

lr_es = HPT(LR, kf, lr_param_grid, x_train, y_train)



lr_best = lr_es.best_estimator_

lr_best_cv = cross_validate(lr_best, x_train, y_train, cv=kf)



print("LR_best Training core mean: {:.2f}". format(lr_best_cv['train_score'].mean()*100)) 

print("LR_best Test score mean: {:.2f}". format(lr_best_cv['test_score'].mean()*100))
g = plot_learning_curve(xgb_best,"Xgboost learning curves",x_train,y_train,cv=kf)

g = plot_learning_curve(lr_best,"LR_best learning curves",x_train,y_train,cv=kf)
print("SVC Parameters tuning")

SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1, 1],

                  'C': [1, 10, 50, 100, 200, 300]}

svc_es = HPT(SVMC, kf, svc_param_grid, x_train, y_train)



svc_best = svc_es.best_estimator_

svc_cv = cross_validate(svc_best, x_train, y_train, cv=kf)



print("SVC Training core mean: {:.2f}". format(svc_cv['train_score'].mean()*100)) 

print("SVC Test score mean: {:.2f}". format(svc_cv['test_score'].mean()*100))

print('-'*60)



print("ADA Parameters tuning") 

DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "algorithm" : ["SAMME","SAMME.R"],

              "n_estimators" :[1,2],

              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

ada_es = HPT(adaDTC, kf, ada_param_grid, x_train, y_train)



ada_best = ada_es.best_estimator_

ada_cv = cross_validate(ada_best, x_train, y_train, cv=kf)



print("ADA Training core mean: {:.2f}". format(ada_cv['train_score'].mean()*100)) 

print("ADA Test score mean: {:.2f}". format(ada_cv['test_score'].mean()*100))

print('-'*60)





print("RF Parameters tuning")

RFC = RandomForestClassifier()

rf_param_grid = {"max_depth": [None],

#               "max_features": [1, 3],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,200,300],

              "criterion": ["gini"]}

rf_es = HPT(RFC, kf, rf_param_grid, x_train, y_train)



rf_best = rf_es.best_estimator_

rf_cv = cross_validate(rf_best, x_train, y_train, cv=kf)



print("RF Training core mean: {:.2f}". format(rf_cv['train_score'].mean()*100)) 

print("RF Test score mean: {:.2f}". format(rf_cv['test_score'].mean()*100))

print('-'*60)





print("GB Parameters tuning") 

GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.1, 0.3] 

              }

gb_es = HPT(GBC, kf, gb_param_grid, x_train, y_train)



gb_best = gb_es.best_estimator_

gb_cv = cross_validate(gb_best, x_train, y_train, cv=kf)



print("GB Training core mean: {:.2f}". format(gb_cv['train_score'].mean()*100)) 

print("GB Test score mean: {:.2f}". format(gb_cv['test_score'].mean()*100))

print('-'*60)





print("ETC Parameters tuning") 

ExtC = ExtraTreesClassifier()

ex_param_grid = {"max_depth": [None],

#               "max_features": [1, 3],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,200,300],

              "criterion": ["gini"]}

etc_es = HPT(ExtC, kf, ex_param_grid, x_train, y_train)



etc_best = etc_es.best_estimator_

etc_cv = cross_validate(etc_best, x_train, y_train, cv=kf)



print("ETC Training core mean: {:.2f}". format(etc_cv['train_score'].mean()*100)) 

print("ETC Test score mean: {:.2f}". format(etc_cv['test_score'].mean()*100))

print('-'*60)

g = plot_learning_curve(svc_best,"SVC learning curves",x_train,y_train,cv=kf)

g = plot_learning_curve(ada_best,"AdaBoost learning curves",x_train,y_train,cv=kf)

g = plot_learning_curve(rf_best,"RandomForest learning curves",x_train,y_train,cv=kf)

g = plot_learning_curve(gb_best,"GradientBoosting learning curves",x_train,y_train,cv=kf)

g = plot_learning_curve(etc_best,"ExtraTrees learning curves",x_train,y_train,cv=kf)
# 예측



# xgb_best.fit(x_train, y_train)

# pre_xgb = xgb_best.predict(x_test)



# gb_best.fit(x_train, y_train)

# pre_gb = gb_best.predict(x_test)



# svc_best.fit(x_train, y_train)

# pre_svc = svc_best.predict(x_test)



# rf_best.fit(x_train, y_train)

# pre_rf = rf_best.predict(x_test)



# lr_best.fit(x_train, y_train)

# pre_lr = lr_best.predict(x_test)



# etc_best.fit(x_train, y_train)

# pre_etc = etc_best.predict(x_test)



# ada_best.fit(x_train, y_train)

# pre_ada = ada_best.predict(x_test)
# kaggle 제출용으로 결과 데이터 변환

# 예측해야하는 데이터(test_df)의 인덱스와 예측 결과를 각 각 세로모양으로 바꿔준 후 합침.



# result_for_kaggle1 = np.concatenate((test_index.values.reshape(-1,1),

#                                     pre_xgb.reshape(-1,1)),

#                                    axis=1)



# result_for_kaggle2 = np.concatenate((test_index.values.reshape(-1,1),

#                                     pre_gb.reshape(-1,1)),

#                                    axis=1)



# result_for_kaggle3 = np.concatenate((test_index.values.reshape(-1,1),

#                                     pre_svc.reshape(-1,1)),

#                                    axis=1)



# result_for_kaggle4 = np.concatenate((test_index.values.reshape(-1,1),

#                                     pre_rf.reshape(-1,1)),

#                                    axis=1)



# result_for_kaggle5 = np.concatenate((test_index.values.reshape(-1,1),

#                                     pre_lr.reshape(-1,1)),

#                                    axis=1)



# csv 형태로 저장하기 위해 다시 DataFrame으로,,

# df_for_kaggle = pd.DataFrame(result_for_kaggle1, columns=["PassengerId","Survived"])

# df_for_kaggle.to_csv('./titanic_ES_XGB.csv', index=False)



# df_for_kaggle = pd.DataFrame(result_for_kaggle2, columns=["PassengerId","Survived"])

# df_for_kaggle.to_csv('./titanic_ES_GB.csv', index=False)



# df_for_kaggle = pd.DataFrame(result_for_kaggle3, columns=["PassengerId","Survived"])

# df_for_kaggle.to_csv('./titanic_ES_SVC.csv', index=False)



# df_for_kaggle = pd.DataFrame(result_for_kaggle4, columns=["PassengerId","Survived"])

# df_for_kaggle.to_csv('./titanic_ES_RF.csv', index=False)



# df_for_kaggle = pd.DataFrame(result_for_kaggle4, columns=["PassengerId","Survived"])

# df_for_kaggle.to_csv('./titanic_ES_LR.csv', index=False)
xgb_best.fit(x_train, y_train)

pre_xgb = xgb_best.predict(x_test)



gb_best.fit(x_train, y_train)

pre_gb = gb_best.predict(x_test)



svc_best.fit(x_train, y_train)

pre_svc = svc_best.predict(x_test)



rf_best.fit(x_train, y_train)

pre_rf = rf_best.predict(x_test)



lr_best.fit(x_train, y_train)

pre_lr = lr_best.predict(x_test)



# 예측 평균

predictions = np.concatenate((pre_xgb.reshape(-1,1), 

                              pre_gb.reshape(-1,1), 

                              pre_svc.reshape(-1,1), 

                              pre_rf.reshape(-1,1), 

                              pre_lr.reshape(-1,1)), axis=1)

pres_mean = (predictions.mean(axis=1).reshape(-1,1)+0.4).round()
# 예측 평균

predictions = np.concatenate((pre_xgb.reshape(-1,1), 

                              pre_gb.reshape(-1,1), 

                              pre_svc.reshape(-1,1), 

                              pre_rf.reshape(-1,1), 

                              pre_lr.reshape(-1,1)

#                               pre_etc.reshape(-1,1),

#                               pre_ada.reshape(-1,1)

                             ),

                             

                             axis=1)

pres_mean = (predictions.mean(axis=1).reshape(-1,1)+0.4).round()
result_for_kaggle = np.concatenate((test_index.values.reshape(-1,1),

                                    pres_mean.astype(int)),

                                   axis=1)
# result_for_kaggle
df_for_kaggle = pd.DataFrame(result_for_kaggle, columns=["PassengerId","Survived"])

df_for_kaggle.to_csv('./titanic_final.csv', index=False)