# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
titanic = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
titanic.head()
# data loading

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

df = train.append(test, ignore_index=True)

passenger_id = df[891:].PassengerId

df.head()
def description(df):

    print(f'Dataset Shape:{df.shape}')

    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values   

    summary['Uniques'] = df.nunique().values

    return summary

print('Data Description:')

description(df)
sns.countplot('Survived', data=df, palette='Set2')

plt.ylabel('Number of survivors')

plt.title('Distribution of survivors');
sns.boxplot(x="Survived", y="Age", data=df, palette='rainbow');
sns.violinplot('Pclass','Age', hue='Survived',

               data=df,palette="Set2", split=True,scale="count");
sns.swarmplot(x='Embarked', y='Fare', data=df);
ax = sns.violinplot(x="Sex", y="Age", data=df, inner=None)

ax = sns.swarmplot(x="Sex", y="Age", data=df,

                   color="white", edgecolor="gray")
sns.catplot(x="Pclass", y="Fare",

            hue="Survived", col="Sex",

            data=df, kind="swarm");
sns.jointplot("Age", "Pclass", data=df,

                  kind="kde", space=0, color="g");
plt.figure(figsize=(17,10))

matrix = np.triu(df.corr())

sns.heatmap(df.corr(), annot=True, mask=matrix,cmap= 'coolwarm');
df = df.drop(['PassengerId','Name','Ticket','Cabin','Fare'], axis=1)

df.head()
# Remember what needs to be done

description(df)
from sklearn.preprocessing import LabelEncoder

labelEnc = LabelEncoder()

df.Sex=labelEnc.fit_transform(df.Sex)
df['Age'] = df.Age.fillna(df.Age.mean())
df.Embarked.value_counts()
df['Embarked'] = df.Embarked.fillna('S')
Embarked = pd.get_dummies(df.Embarked , prefix='Embarked' )

Embarked.head()
Pclass = pd.get_dummies(df.Pclass, prefix='Pclass')

SibSp = pd.get_dummies(df.SibSp, prefix='SibSp')

Parch = pd.get_dummies(df.Parch, prefix='Parch')

df_new = pd.concat([df, Embarked, Pclass, SibSp, Parch], axis=1)
df_new = df_new.drop(['Pclass', 'SibSp','Parch', 'Embarked'], axis=1)

description(df_new)
df_new.shape
X_train = df_new.drop(['Survived'],axis=1)[ 0:891]

y_train= df_new.Survived[ 0:891]

X_test = df_new.drop(['Survived'],axis=1)[891:]

y_test = titanic.Survived
# Preprocessing, modelling and evaluating

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix, roc_auc_score

from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold

from xgboost import XGBClassifier

import xgboost as xgb



## Hyperopt modules

from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING

from functools import partial



import os

import gc
from sklearn.model_selection import KFold,TimeSeriesSplit

from sklearn.metrics import roc_auc_score

from xgboost import plot_importance

from sklearn.metrics import make_scorer



import time

def objective(params):

    time1 = time.time()

    params = {

        'max_depth': int(params['max_depth']),

        'gamma': "{:.3f}".format(params['gamma']),

        'subsample': "{:.2f}".format(params['subsample']),

        'reg_alpha': "{:.3f}".format(params['reg_alpha']),

        'reg_lambda': "{:.3f}".format(params['reg_lambda']),

        'learning_rate': "{:.3f}".format(params['learning_rate']),

        'num_leaves': '{:.3f}'.format(params['num_leaves']),

        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),

        'min_child_samples': '{:.3f}'.format(params['min_child_samples']),

        'feature_fraction': '{:.3f}'.format(params['feature_fraction']),

        'bagging_fraction': '{:.3f}'.format(params['bagging_fraction'])

    }



    print("\n############## New Run ################")

    print(f"params = {params}")

    FOLDS = 7

    count=1

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)



    tss = TimeSeriesSplit(n_splits=FOLDS)

    y_preds = np.zeros(titanic.shape[0])

    y_oof = np.zeros(X_train.shape[0])

    score_mean = 0

    for tr_idx, val_idx in tss.split(X_train, y_train):

        clf = xgb.XGBClassifier(

            n_estimators=600, random_state=4, verbose=True, 

           # tree_method='gpu_hist', 

            **params

        )



        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]

        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        

        clf.fit(X_tr, y_tr)

        #y_pred_train = clf.predict_proba(X_vl)[:,1]

        #print(y_pred_train)

        score = make_scorer(roc_auc_score, needs_proba=True)(clf, X_vl, y_vl)

        # plt.show()

        score_mean += score

        print(f'{count} CV - score: {round(score, 4)}')

        count += 1

    time2 = time.time() - time1

    print(f"Total Time Run: {round(time2 / 60,2)}")

    gc.collect()

    print(f'Mean ROC_AUC: {score_mean / FOLDS}')

    del X_tr, X_vl, y_tr, y_vl, clf, score

    return -(score_mean / FOLDS)





space = {

    # The maximum depth of a tree, same as GBM.

    # Used to control over-fitting as higher depth will allow model 

    # to learn relations very specific to a particular sample.

    # Should be tuned using CV.

    # Typical values: 3-10

    'max_depth': hp.quniform('max_depth', 7, 23, 1),

    

    # reg_alpha: L1 regularization term. L1 regularization encourages sparsity 

    # (meaning pulling weights to 0). It can be more useful when the objective

    # is logistic regression since you might need help with feature selection.

    'reg_alpha':  hp.uniform('reg_alpha', 0.01, 0.4),

    

    # reg_lambda: L2 regularization term. L2 encourages smaller weights, this

    # approach can be more useful in tree-models where zeroing 

    # features might not make much sense.

    'reg_lambda': hp.uniform('reg_lambda', 0.01, .4),

    

    # eta: Analogous to learning rate in GBM

    # Makes the model more robust by shrinking the weights on each step

    # Typical final values to be used: 0.01-0.2

    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),

    

    # colsample_bytree: Similar to max_features in GBM. Denotes the 

    # fraction of columns to be randomly samples for each tree.

    # Typical values: 0.5-1

    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, .9),

    

    # A node is split only when the resulting split gives a positive

    # reduction in the loss function. Gamma specifies the 

    # minimum loss reduction required to make a split.

    # Makes the algorithm conservative. The values can vary depending on the loss function and should be tuned.

    'gamma': hp.uniform('gamma', 0.01, .7),

    

    # more increases accuracy, but may lead to overfitting.

    # num_leaves: the number of leaf nodes to use. Having a large number 

    # of leaves will improve accuracy, but will also lead to overfitting.

    'num_leaves': hp.choice('num_leaves', list(range(20, 250, 10))),

    

    # specifies the minimum samples per leaf node.

    # the minimum number of samples (data) to group into a leaf. 

    # The parameter can greatly assist with overfitting: larger sample

    # sizes per leaf will reduce overfitting (but may lead to under-fitting).

    'min_child_samples': hp.choice('min_child_samples', list(range(100, 250, 10))),

    

    # subsample: represents a fraction of the rows (observations) to be 

    # considered when building each subtree. Tianqi Chen and Carlos Guestrin

    # in their paper A Scalable Tree Boosting System recommend 

    'subsample': hp.choice('subsample', [0.2, 0.4, 0.5, 0.6, 0.7, .8, .9]),

    

    # randomly select a fraction of the features.

    # feature_fraction: controls the subsampling of features used

    # for training (as opposed to subsampling the actual training data in 

    # the case of bagging). Smaller fractions reduce overfitting.

    'feature_fraction': hp.uniform('feature_fraction', 0.4, .8),

    

    # randomly bag or subsample training data.

    'bagging_fraction': hp.uniform('bagging_fraction', 0.4, .9)

    

    # bagging_fraction and bagging_freq: enables bagging (subsampling) 

    # of the training data. Both values need to be set for bagging to be used.

    # The frequency controls how often (iteration) bagging is used. Smaller

    # fractions and frequencies reduce overfitting.

}
# Set algoritm parameters

best = fmin(fn=objective,

            space=space,

            algo=tpe.suggest,

            max_evals=27)



# Print best parameters

best_params = space_eval(space, best)
print("BEST PARAMS: ", best_params)



best_params['max_depth'] = int(best_params['max_depth'])
clf = xgb.XGBClassifier(

    n_estimators=300,

    **best_params

    #tree_method='gpu_hist'

)



clf.fit(X_train, y_train)



predict = clf.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test, predict)
feature_important = clf.get_booster().get_score(importance_type="weight")

keys = list(feature_important.keys())

values = list(feature_important.values())



data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)



# Top 10 features

data.head(20)
titanic['Survived'] = predict

titanic.to_csv('titanicpred.csv', index = False)
titanic