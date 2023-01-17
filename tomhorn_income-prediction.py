import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import sklearn as sk

import cufflinks as cf

import plotly.offline

import pandas_profiling

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,roc_auc_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

import cufflinks as cf

import warnings





cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)

warnings.filterwarnings('ignore')
AdultDf = pd.read_csv('../input/adult-census-income/adult.csv')

AdultDf.head()
profile = pandas_profiling.ProfileReport(AdultDf, title='Pandas Profiling Report for training dataset', html={'style':{'full_width':True}})
profile.to_notebook_iframe()
AdultDf = AdultDf.replace('?', np.nan)

AdultDf = AdultDf.dropna()



AdultDf = AdultDf[AdultDf.age != 90]

AdultDf = AdultDf[AdultDf['capital.gain'] != 99999]



AdultDf.head()
IncomeDist = pd.crosstab(AdultDf.age, AdultDf.income)

IncomeDist.iplot(kind ='bar', barmode = 'stack', xTitle = 'Age', yTitle = 'Num of Individuals', title = 'Distribution of Income between Ages', theme = 'white')
AdultDf.iplot(kind ='histogram', column ='education.num', color ='orange', xTitle = 'Years Spent in Education', yTitle = 'Num of Individuals', title = 'Distribution of Income Levels', theme = 'white', bargap = 0.05)
EduIncome = pd.crosstab(AdultDf.education, AdultDf.income)

EducationLevels = {"Preschool":0, "1st-4th":1, "5th-6th":2, "7th-8th":3, "9th":4, "10th":5, "11th":6, "12th":7, "HS-grad":8, "Some-college":9, "Assoc-voc":10, "Assoc-acdm":11, "Bachelors":12, "Masters":13, "Prof-school":14, "Doctorate":15}

EduIncome = EduIncome.sort_values(by=['education'], key=lambda x: x.map(EducationLevels))

EduIncome.iplot(kind = 'bar', barmode = 'stack', xTitle = 'Education Levels', yTitle = 'Num of Individuals', title = 'Distibution of Education Levels', theme = 'white')
IncomeDist = pd.crosstab(AdultDf['relationship'], AdultDf.income)

IncomeDist.iplot(kind ='bar', barmode = 'stack', xTitle = 'Relationship Status', yTitle = 'Num of Individuals', title = 'Distribution of Income between Relationship Status', theme = 'white')
MarAge = pd.crosstab(AdultDf.age, AdultDf.relationship)

MarAge.iplot(kind = 'bar', barmode = 'stack', xTitle = 'Age', yTitle = 'Num of Individuals', title = 'Distribution of Relationships between Ages', theme = 'white')
SexIncome = pd.crosstab(AdultDf.sex, AdultDf.income)

SexIncome.iplot(kind = 'bar', barmode = 'stack', xTitle = 'Sex', yTitle = 'Num of Individuals', title = 'Distribution of Income between Sex', theme = 'white')
OccuInc = pd.crosstab(AdultDf.occupation , AdultDf.income)

OccuInc = OccuInc.sort_values('<=50K', ascending = False)

OccuInc.iplot(kind = 'bar', barmode = 'stack', theme ='white', xTitle = 'Occupation', yTitle = 'Num of Individuals', title = 'Distribution of Income between Occupations')
AdultDfTarget = AdultDf.copy()

target = AdultDf['income'].unique()

Ints = {name: n for n, name in enumerate(target)}

AdultDf['target'] = AdultDf['income'].replace(Ints)

AdultDf.head()
HeadDrop = ['capital.gain','capital.loss','education','fnlwgt','income']

AdultDf.drop(HeadDrop, inplace = True, axis = 1)

AdultDf.head()
AdultDf['native.country'] = AdultDf['native.country'].replace(['Mexico', 'Greece', 'Vietnam', 'China', 'Taiwan',

       'India', 'Philippines', 'Trinadad&Tobago', 'Canada', 'South',

       'Holand-Netherlands', 'Puerto-Rico', 'Poland', 'Iran', 'England',

       'Germany', 'Italy', 'Japan', 'Hong', 'Honduras', 'Cuba', 'Ireland',

       'Cambodia', 'Peru', 'Nicaragua', 'Dominican-Republic', 'Haiti',

       'Hungary', 'Columbia', 'Guatemala', 'El-Salvador', 'Jamaica',

       'Ecuador', 'France', 'Yugoslavia', 'Portugal', 'Laos', 'Thailand',

       'Outlying-US(Guam-USVI-etc)', 'Scotland'], 'NonUS')



AdultDf['native.country'].unique()
categorical_columns = AdultDf.select_dtypes(exclude=np.number).columns

BinarisedDf = pd.get_dummies(data=AdultDf, prefix=categorical_columns, drop_first=True)



AdultDf = BinarisedDf

AdultDf.head()
X = AdultDf.drop('target', axis=1)

y = AdultDf['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)



DtClas = DecisionTreeClassifier()

DtClas.fit(X_train, y_train)



DtPred = DtClas.predict(X_test)



print(classification_report(y_test, DtPred))

print("Accuracy :",accuracy_score(y_test, DtPred))

LogReg = LogisticRegression(C = 0.05, max_iter = 1000)

LrMod = LogReg.fit(X_train, y_train)

LrPred = LrMod.predict(X_test)



print(classification_report(y_test, LrPred))

print("Accuracy :",accuracy_score(y_test, LrPred))
HypeParamLogReg = {'penalty':['l1', 'l2', 'elasticnet'], 'C':[0.001, 0.01, 0.1, 1, 10, 100], 'l1_ratio':[0.001, 0.01, 0.1]}

LogRegGrid = GridSearchCV(LogisticRegression(), param_grid=HypeParamLogReg, verbose=3)

LogRegGrid.fit(X, y)
print("Best Params : ",LogRegGrid.best_params_)

print("Accuracy : ",LogRegGrid.best_score_)
LogRegAdj = LogisticRegression(C = 0.1, l1_ratio = 0.001, penalty = 'l2', max_iter = 1000)

LrModAdj = LogRegAdj.fit(X_train, y_train)

LrPredAdj = LrModAdj.predict(X_test)



print(classification_report(y_test, LrPredAdj))

print("Accuracy :",accuracy_score(y_test, LrPredAdj))
RandomForest = RandomForestClassifier(n_estimators=500,max_features=5)

RandomForest.fit(X_train, y_train)

RfPred = RandomForest.predict(X_test)



print(classification_report(y_test, RfPred))

print("Accuracy :",accuracy_score(y_test, RfPred))
HypeParamRf = {'criterion':['gini', 'entropy'], 'max_depth':[2, 5, 8, 11], 'n_estimators':[200, 300, 400, 500]}

RfGrid = GridSearchCV(RandomForestClassifier(), param_grid=HypeParamRf, verbose=3)



RfGrid.fit(X, y)
print("Best Params : ",RfGrid.best_params_)

print("Accuracy : ",RfGrid.best_score_)
GradBoost = XGBClassifier(learning_rate = 0.35, n_estimator = 200)

GbMod = GradBoost.fit(X_train, y_train)

GbPred = GbMod.predict(X_test)
print(classification_report(y_test, GbPred))

print("Accuracy :",accuracy_score(y_test, GbPred))
classifiers = [GradBoost,RandomForest,LogReg]



folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=11)



scores_dict = {}



for train_index, valid_index in folds.split(X_train, y_train):

    X_train_fold, X_valid_fold = X.iloc[train_index], X.iloc[valid_index]

    y_train_fold, y_valid_fold = y.iloc[train_index], y.iloc[valid_index]

    

    for classifier in classifiers:

        name = classifier.__class__.__name__

        classifier.fit(X_train_fold, y_train_fold)

        training_predictions = classifier.predict_proba(X_valid_fold)

        scores = roc_auc_score(y_valid_fold, training_predictions[:, 1])

        if name in scores_dict:

            scores_dict[name] += scores

        else:

            scores_dict[name] = scores





for classifier in scores_dict:

    scores_dict[classifier] = scores_dict[classifier]/folds.n_splits
print("Accuracy :",scores_dict)