import pandas as pd # for data processing and analysis modeled

import matplotlib   # for scientific and visualization

import numpy as np  # for scientific computing

import scipy as sp  # for scientific computing and mathematics Functions

import IPython 

from IPython import display #  printing of dataframes in Jupyter notebook

import sklearn      # for machine learning algorithms



import seaborn as sns

import matplotlib.pylab as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4





#misc libraries

import random

import time



#ignore warnings

import warnings

warnings.filterwarnings('ignore')
data_raw = pd.read_csv('../input/titanic/train.csv') # this is the data for training and our Evaluation



data_val = pd.read_csv('../input/titanic/test.csv') # this is provided by Kaggle and to be used to Submit the final Predictions



# make a copy for future usage to check on data.

data_train = data_raw.copy(deep = True)

data_test = data_val.copy(deep = True)



print (data_train.info())

print ("#"*50)

print (data_test.info())

print ("#"*50)



"""

Combine both Test and Train Datasets for doing analysis on Categorical values (Classes) that may be present 

only in Test but not in Training Dataset

"""

data_combine = [data_train, data_test]

data_train.head(15)
data_train.describe(include=['O'])
data_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
data_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
data_train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
data_train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
plt.hist(x = [data_train[data_train['Survived']==1]['Age'], data_train[data_train['Survived']==0]['Age']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age (Years)')

plt.ylabel('# of Passengers')

plt.legend()
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')

grid = sns.FacetGrid(data_train, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
# grid = sns.FacetGrid(train_df, col='Embarked')

grid = sns.FacetGrid(data_train, row='Embarked', height=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(data_train, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
print("Before", data_train.shape, data_test.shape, data_combine[0].shape, data_combine[1].shape)



data_train = data_train.drop(['Ticket', 'Cabin'], axis=1)

data_test = data_test.drop(['Ticket', 'Cabin'], axis=1)

data_combine = [data_train, data_test]



print("After", data_train.shape, data_test.shape, data_combine[0].shape, data_combine[1].shape)
for dataset in data_combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(data_train['Title'], data_train['Sex'])
for dataset in data_combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

data_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data_combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



data_train.head()
data_train = data_train.drop(['Name', 'PassengerId'], axis=1)

data_test = data_test.drop(['Name'], axis=1)

data_combine = [data_train, data_test]

data_train.shape, data_test.shape
for dataset in data_combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



data_train.head()
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')

grid = sns.FacetGrid(data_train, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
guess_ages = np.zeros((2,3))

guess_ages
for dataset in data_combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



data_train.head()
data_train['AgeBand'] = pd.cut(data_train['Age'], 5)

data_train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in data_combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

data_train.head()
#AgeBand feature can be removed

data_train = data_train.drop(['AgeBand'], axis=1)

data_combine = [data_train, data_test]

data_train.head()
for dataset in data_combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



data_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# another new feature called IsAlone. 



for dataset in data_combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



data_train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
# with IsAlone field with good correlation with Survived Feature, can drop Parch, Sibsp and Familysize Features

data_train = data_train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

data_test = data_test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

data_combine = [data_train, data_test]



data_train.head()
# Another New feature combining Pclass and Age.

for dataset in data_combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



data_train.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
freq_port = data_train.Embarked.dropna().mode()[0]



for dataset in data_combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

data_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in data_combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



data_train.head()
data_test['Fare'].fillna(data_test['Fare'].dropna().median(), inplace=True)

data_test.head()
# Fare has continous numeric data and hence to be converted to category range to make it categorical

data_train['FareBand'] = pd.qcut(data_train['Fare'], 4)

data_train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
# Convert the Fare feature to ordinal values based on the FareBand

for dataset in data_combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



data_train = data_train.drop(['FareBand'], axis=1)

data_combine = [data_train, data_test]
print (data_train.head(10))

print ("#"*75)

print (data_test.head(10))
import scipy.stats as stats

import statsmodels.api as sm

from statsmodels.formula.api import ols



def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(8, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        data_train.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.6 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=12)



correlation_heatmap(data_train)
#set correlation above 0.75 and see true/false values

abs(data_train.corr())> 0.50
sns.heatmap(data_train.corr(), center=0);
X_train = data_train.drop("Survived", axis=1)

Y_train = data_train["Survived"]

#X_test  = data_test.drop("PassengerId", axis=1).copy()

test_Titanic  = data_test.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, test_Titanic.shape
# machine learning

import xgboost as xgb 

from xgboost.sklearn import XGBClassifier

from sklearn import metrics   #Additional scklearn functions

from sklearn import model_selection

from sklearn.model_selection import GridSearchCV, cross_val_score   #Perforing grid search and Cross Validation Score
"""

# starting of the Parameter setting should be with finding the right number of trees (n_estimators). 

# Use below to get the incremental addition of trees by XGBoost and Validation error stops reducing at a time and 

that would be our n_estimators

"""

eval_set = [(X_train, Y_train)]

eval_metric = ["auc","error"]

xgb_eval = XGBClassifier(n_estimators=1000)

xgb_eval.fit(X_train, Y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=True)





#Parameters list can be found here as well: https://xgboost.readthedocs.io/en/latest/parameter.html

"""

submission_xgb = pd.DataFrame({

        "PassengerId": data_test["PassengerId"],

        "Survived": Y_pred_xgb

    })



#submission.to_csv("submission_TitanicSurvived_pred.csv", index=False)



print('Validation Data Distribution: \n', submission_xgb['Survived'].value_counts(normalize = True))

submission_xgb.sample(5)

"""
"""

This is a function to accept Alogithm, Train and Test Datasets. Train and Test are inputs coz, in case we have 

different dataset with different Features (independed variables), we can still use this function in a common way.



Remain inputs to function have default values. If required, alternative values can be passed. 



Returns score so that i build a Dataframe of the Parameter Trail and its respective Accurac Score, 

after each call to this function

"""



def modelfit(alg, train,test,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    

    if useTrainCV:

        xgb_param = alg.get_xgb_params()

        xgtrain = xgb.DMatrix(train, label=test)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, metrics='auc', 

                          early_stopping_rounds=early_stopping_rounds)   #, show_progress=False)

        

    alg.set_params(n_estimators=cvresult.shape[0])

    

    #Fit the algorithm on the data

    alg.fit(train, test,eval_metric='rmse')



    #Predict training set:

    dtrain_predictions = alg.predict(train)

    dtrain_predprob = alg.predict_proba(train)[:,1]

        

    #Print model report:

    print ("\nModel Report")

    print ("Accuracy : %.4g" % metrics.accuracy_score(test, dtrain_predictions))

    print ("AUC Score (Train): %f" % metrics.roc_auc_score(test, dtrain_predprob))

    print ("RMSE (Train): %f" % metrics.mean_squared_error(test, dtrain_predprob))

        

#    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)

    feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)

    feat_imp.plot(kind='bar', title='Feature Importances')

    plt.ylabel('Feature Importance Score')

    return (metrics.accuracy_score(test, dtrain_predictions), metrics.roc_auc_score(test, dtrain_predprob), 

            metrics.mean_squared_error(test, dtrain_predprob))
#Choose all Predictors (Independent Variables except the Dependent Variable)

predictors = [x for x in X_train.columns if x not in ['Survived']]

xgb1 = XGBClassifier(

 learning_rate =0.1,

 n_estimators=742,

 max_depth=5,

 min_child_weight=1,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27,

 verbosity=1)





xgb1_score, xgb1_auc, xgb1_rmse = modelfit(xgb1, X_train, Y_train, predictors)
param_test1 = {

 'max_depth':range(4,5,6),

 'min_child_weight':range(1,4,6)

# 'max_depth':range(3,10,2),

# 'min_child_weight':range(1,6,2)



}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=742, max_depth=5,

 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 

 param_grid = param_test1, scoring='roc_auc',n_jobs=-1,iid=False, cv=5)

gsearch1.fit(X_train,Y_train)

gsearch1.score, gsearch1.best_params_, gsearch1.best_score_
param_test2 = {

 'max_depth':[4,5,6],

 'min_child_weight':[4,5,6,10,12]

}

gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=742, max_depth=9,

 min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch2.fit(X_train,Y_train)

gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_
#we got 6 as optimum value for min_child_weight but we haven’t tried values more than 10. We can do that as follow:



param_test2b = {

 'min_child_weight':[4,5,6,8,10,12]

}

gsearch2b = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=742, max_depth=4,

 min_child_weight=10, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test2b, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch2b.fit(X_train,Y_train)

gsearch2b.cv_results_, gsearch2b.best_params_, gsearch2b.best_score_
gsearch2_score, gsearch2_auc, gsearch2_rmse = modelfit(gsearch2b.best_estimator_, X_train, Y_train, predictors) 

gsearch2b.cv_results_, gsearch2b.best_params_, gsearch2b.best_score_
# best params for max_Depth and min_child_weight are 4 and 4 respectively. This is my findings above. 

param_test3 = {

 'gamma':[i/10.0 for i in range(0,5)]

}

gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=743, max_depth=4,

 min_child_weight=10, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch3.fit(X_train,Y_train)

gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_
# best params for max_Depth and min_child_weight are 4 and 6 respectively. This is the example i am trying the Tunning learnt from

# analyticsvidya



param_test3 = {

 'gamma':[i/10.0 for i in range(0,5)]

}

gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=743, max_depth=4,

 min_child_weight=10, gamma=0.4, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch3.fit(X_train,Y_train)

gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_
#sorted(sklearn.metrics.SCORERS.keys())
xgb2 = XGBClassifier(

 learning_rate =0.1,

 n_estimators=743,

 max_depth=4,

 min_child_weight=10,

 gamma=0.4,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)

xgb2_score, xgb2_auc, xgb2_rmse = modelfit(xgb2, X_train, Y_train, predictors)
param_test4 = {

 'subsample':[i/10.0 for i in range(6,10)],

 'colsample_bytree':[i/10.0 for i in range(6,10)]

}

gsearch4 = GridSearchCV(estimator = XGBClassifier(

 learning_rate =0.1,

 n_estimators=743,

 max_depth=4,

 min_child_weight=10,

 gamma=0.4,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27),param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)



gsearch4.fit(X_train,Y_train)

gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_
param_test4a = {

 'n_estimators':[1000, 500, 200, 180, 175],

 'subsample':[i/10.0 for i in range(6,10)],

 'colsample_bytree':[i/10.0 for i in range(6,10)]

}

gsearch4a = GridSearchCV(estimator = XGBClassifier(

 learning_rate =0.1,

 n_estimators=743,

 max_depth=4,

 min_child_weight=10,

 gamma=0.4,

 subsample=0.9,

 colsample_bytree=0.9,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27),param_grid = param_test4a, scoring='roc_auc',n_jobs=4,iid=False, cv=5)



gsearch4a.fit(X_train,Y_train)

gsearch4a.cv_results_, gsearch4a.best_params_, gsearch4a.best_score_
#Choose all Predictors (Independent Variables except the Dependent Variable)

#predictors = [x for x in X_train.columns if x not in ['Survived']]

xgb3 = XGBClassifier(

 learning_rate =0.1,

 n_estimators=743,

 max_depth=4,

 min_child_weight=10,

 gamma=0.4,

 subsample=0.9,

 colsample_bytree=0.9,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)





xgb3_score, xgb3_auc, xgb3_rmse = modelfit(xgb3, X_train, Y_train, predictors)
param_test5 = {

 'subsample':[i/100.0 for i in range(75,90,5)],

 'colsample_bytree':[i/100.0 for i in range(75,90,5)]

}

gsearch5 = GridSearchCV(estimator = XGBClassifier(

 learning_rate =0.1,

 n_estimators=743,

 max_depth=4,

 min_child_weight=10,

 gamma=0.4,

 subsample=0.9,

 colsample_bytree=0.9,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27),param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)



gsearch5.fit(X_train,Y_train)

gsearch5.cv_results_, gsearch5.best_params_, gsearch5.best_score_
#Choose all Predictors (Independent Variables except the Dependent Variable)

#predictors = [x for x in X_train.columns if x not in ['Survived']]

xgb4 = XGBClassifier(

 learning_rate =0.1,

 n_estimators=7430,

 max_depth=4,

 min_child_weight=10,

 gamma=0.4,

 subsample=0.8,

 colsample_bytree=0.75,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)





xgb4_score, xgb4_auc, xgb4_rmse = modelfit(xgb4, X_train, Y_train, predictors)
param_test6 = {

'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]

}

gsearch6 = GridSearchCV(estimator = XGBClassifier(

 learning_rate =0.1,

 n_estimators=743,

 max_depth=4,

 min_child_weight=10,

 gamma=0.4,

 subsample=0.8,

 colsample_bytree=0.75,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27),param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)



gsearch6.fit(X_train,Y_train)

gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_
param_test7 = {

'reg_alpha':[0, 0.01, 0.05, 0.1, 0.5, 0.75, 1, 1.1,1.25, 1.5]

}

gsearch7 = GridSearchCV(estimator = XGBClassifier(

 learning_rate =0.1,

 n_estimators=743,

 max_depth=4,

 min_child_weight=10,

 gamma=0.4,

 subsample=0.8,

 colsample_bytree=0.75,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27),param_grid = param_test7, scoring='roc_auc',n_jobs=4,iid=False, cv=5)



gsearch7.fit(X_train,Y_train)

gsearch7.cv_results_, gsearch7.best_params_, gsearch7.best_score_
#Choose all Predictors (Independent Variables except the Dependent Variable)

#predictors = [x for x in X_train.columns if x not in ['Survived']]

xgb5 = XGBClassifier(

 learning_rate =0.1,

 n_estimators=743,

 max_depth=4,

 min_child_weight=10,

 gamma=0.4,

 subsample=0.8,

 colsample_bytree=0.75,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27,

reg_alpha=0.1)





xgb5_score,xgb5_auc, xgb5_rmse  = modelfit(xgb5, X_train, Y_train, predictors)
param_test8 = {

'learning_rate': [0.01, 0.05, 0.1, 0.25, 0.5],

'n_estimators': [1800,750, 500, 180, 90, 50 ]

}

gsearch8 = GridSearchCV(estimator = XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=4,

 min_child_weight=10,

 gamma=0.4,

 subsample=0.8,

 colsample_bytree=0.75,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27,

 reg_alpha=0.1),param_grid = param_test8, scoring='roc_auc',n_jobs=4,iid=False, cv=5)



gsearch8.fit(X_train,Y_train)

gsearch8.cv_results_, gsearch8.best_params_, gsearch8.best_score_
xgb6 = XGBClassifier(

 learning_rate =0.05,

 n_estimators=1800,

 max_depth=4,

 min_child_weight=10,

 gamma=0.4,

 subsample=0.8,

 colsample_bytree=0.75,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27,

reg_alpha=0.1)



xgb6_score, xgb6_auc, xgb6_rmse = modelfit(xgb6, X_train, Y_train, predictors)
def timer(start_time=None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)

        tmin, tsec = divmod(temp_sec, 60)

        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

#import xgboost



param_test9 = {    



 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],

 "min_child_weight" : [ 1, 3, 5, 7 ],

 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],

 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]

}





"""

'n_estimators': [1800,750, 500, 180, 90, 50 ],

'learning_rate': [0.01, 0.05, 0.1, 0.25, 0.5],

'colsample_bytree':[0.3, 0.4, 0.5 , 0.7, 0.8, 0.9],

'gamma':[0.0, 0.1, 0.2 , 0.3, 0.4],

'min_child_weight':[1,4,5,6,8,10,12],

'max_depth':[3, 4, 5, 6, 8, 10, 12, 15],

'subsample':[i/100.0 for i in range(75,90,5)],    

'reg_alpha':[0, 0.01, 0.05, 0.1, 0.5, 0.75, 1, 1.1,1.25, 1.5],

"""









xgb9=xgb.XGBClassifier()



rsearch9 = RandomizedSearchCV(xgb9, param_distributions=param_test9, 

                              n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)



from datetime import datetime

# Here we go

start_time = timer(None) # timing starts from this point for "start_time" variable

rsearch9.fit(X_train,Y_train)

timer(start_time)

rsearch9.best_estimator_, rsearch9.best_params_
xgbrs_score, xgbrs_auc, xgbrs_rmse = modelfit(XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

               colsample_bynode=1, colsample_bytree=0.5, gamma=0.2,

               learning_rate=0.05, max_delta_step=0, max_depth=4,

               min_child_weight=3, missing=None, n_estimators=750, n_jobs=1,

               nthread=None, objective='binary:logistic', random_state=0,

               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

               silent=None, subsample=1, verbosity=1), X_train, Y_train, predictors)
#Choose all Predictors (Independent Variables except the Dependent Variable)

predictors = [x for x in X_train.columns if x not in ['Survived']]

xgb1 = XGBClassifier(

 learning_rate =0.1,

 n_estimators=143,

 max_depth=5,

 min_child_weight=1,

 gamma=0.0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)





modelfit(xgb1, X_train, Y_train, predictors)
xgb7_score, xgb7_auc, xgb7_rmse = modelfit(XGBClassifier(

 learning_rate =0.1,

 n_estimators=756,

 max_depth=5,

 min_child_weight=1,

 gamma=0.0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27,

 silent=None, 

 verbosity=1), X_train, Y_train, predictors)
params = pd.DataFrame({

    'Model': ['XGB1', 'XGB2', 'XGB3', 'XGB4', 'XGB5', 'XGB6', 'GSearch2',  'XGB7'],

    'Score': [xgb1_score, xgb2_score, xgb3_score, xgb4_score, xgb5_score, xgb6_score, gsearch2_score, xgb7_score],

    'AUC': [xgb1_auc, xgb2_auc, xgb3_auc, xgb4_auc, xgb5_auc, xgb6_auc, gsearch2_auc, xgb7_auc],    

    'RMSE': [xgb1_rmse, xgb2_rmse, xgb3_rmse, xgb4_rmse, xgb5_rmse, xgb6_rmse, gsearch2_rmse, xgb7_rmse]        

})

params.sort_values(by='RMSE', ascending=True)
xgb_final = XGBClassifier(

 learning_rate =0.1,

 n_estimators=756,

 max_depth=5,

 min_child_weight=1,

 gamma=0.0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27,

 silent=None, 

 verbosity=1)



xgb_final.fit(X_train, Y_train)

Y_pred_final=xgb_final.predict(test_Titanic)

xgb_final.score(X_train, Y_train)

acc_xgb_final = round(xgb_final.score(X_train, Y_train) * 100, 2)

acc_xgb_final
Y_train.shape