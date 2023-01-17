import pandas as pd 

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)





import gc

from datetime import datetime 

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from catboost import CatBoostClassifier

from sklearn import svm

import lightgbm as lgb

from lightgbm import LGBMClassifier

import xgboost as xgb



pd.set_option('display.max_columns', 100)

MAX_ROUNDS = 1000 #lgb iterations

EARLY_STOP = 50 #lgb early stop 

OPT_ROUNDS = 1000  #To be adjusted based on best validation rounds

VERBOSE_EVAL = 50 #Print out metric result



IS_LOCAL = False



import os



if(IS_LOCAL):

    PATH="../input/credit-card-fraud-detection"

else:

    PATH="../input"

print(os.listdir(PATH))
data_df=pd.read_csv('../input/creditcardfraud/creditcard.csv')
data_df.head(20)
data_df.shape
data_df.describe()
data_df.isnull().sum().max()
sns.countplot(x='Class',data=data_df)

plt.ylabel('Transaction')

plt.title('Credit Card Fraud - Class unbalance')

plt.legend()

sns.kdeplot(data=data_df.loc[data_df['Class']==0]['Time'],label='No Fraud')

sns.kdeplot(data=data_df.loc[data_df['Class']==1]['Time'],label='Fraud')

plt.xlabel('Time')

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))

s = sns.boxplot(ax = ax1, x="Class", y="Amount", hue="Class",data=data_df, palette="PRGn",showfliers=True)

s = sns.boxplot(ax = ax2, x="Class", y="Amount", hue="Class",data=data_df, palette="PRGn",showfliers=False)
sns.scatterplot(x='Time',y='Amount',data=data_df.loc[data_df['Class']==1])
plt.figure(figsize=(13,13))

sns.heatmap(data_df.corr(),linewidths=.1,cmap='Reds')

plt.title('Credit Card Transactions Correlation Matrix')
s = sns.lmplot(x='V20', y='Amount',data=data_df, hue='Class', fit_reg=True,scatter_kws={'s':2})

s = sns.lmplot(x='V7', y='Amount',data=data_df, hue='Class', fit_reg=True,scatter_kws={'s':2})

plt.show()
s = sns.lmplot(x='V2', y='Amount',data=data_df, hue='Class', fit_reg=True,scatter_kws={'s':2})

s = sns.lmplot(x='V5', y='Amount',data=data_df, hue='Class', fit_reg=True,scatter_kws={'s':2})

plt.show()
var = data_df.columns.values



i = 0

t0 = data_df.loc[data_df['Class'] == 0]

t1 = data_df.loc[data_df['Class'] == 1]



sns.set_style('whitegrid')

plt.figure()

fig, ax = plt.subplots(8,4,figsize=(16,28))



for feature in var:

    i += 1

    plt.subplot(8,4,i)

    sns.kdeplot(t0[feature], bw=0.5,label="Class = 0")

    sns.kdeplot(t1[feature], bw=0.5,label="Class = 1")

    plt.xlabel(feature, fontsize=12)

    locs, labels = plt.xticks()

    plt.tick_params(axis='both', which='major', labelsize=12)

plt.show()
target = 'Class'

predictors = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',\

       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',\

       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',\

       'Amount']
train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=2, shuffle=True)

train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=2, shuffle=True)
clf = RandomForestClassifier(n_jobs=4, 

                             random_state=2,

                             criterion='gini',

                             n_estimators=100,

                             verbose=False)
clf.fit(train_df[predictors], train_df[target].values)

preds = clf.predict(valid_df[predictors])

tmp=pd.DataFrame({'Features':predictors,'Importance':clf.feature_importances_})

tmp.sort_values(by='Importance',ascending=False,inplace=True)

sns.barplot(x='Features',y='Importance',data=tmp)

plt.xticks(rotation=90)
cm = pd.crosstab(valid_df[target].values, preds, rownames=['Actual'], colnames=['Predicted'])

fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))

sns.heatmap(cm,xticklabels=['Not Fraud', 'Fraud'],yticklabels=['Not Fraud', 'Fraud'],annot=True,ax=ax1,linewidths=.2,linecolor="Darkblue", cmap="Blues")

plt.title('Confusion Matrix', fontsize=14)

plt.show()
roc_auc_score(valid_df[target].values, preds)

clf = AdaBoostClassifier(random_state=2,

                         algorithm='SAMME.R',

                         learning_rate=0.8,

                             n_estimators=100)
clf.fit(train_df[predictors], train_df[target].values)

preds = clf.predict(valid_df[predictors])

tmp=pd.DataFrame({'Features':predictors,'Importance':clf.feature_importances_})

tmp.sort_values(by='Importance',ascending=False,inplace=True)

sns.barplot(x='Features',y='Importance',data=tmp)

plt.xticks(rotation=90)
cm = pd.crosstab(valid_df[target].values, preds, rownames=['Actual'], colnames=['Predicted'])

fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))

sns.heatmap(cm, 

            xticklabels=['Not Fraud', 'Fraud'],

            yticklabels=['Not Fraud', 'Fraud'],

            annot=True,ax=ax1,

            linewidths=.2,linecolor="Darkblue", cmap="Blues")

plt.title('Confusion Matrix', fontsize=14)

plt.show()
roc_auc_score(valid_df[target].values, preds)
clf = CatBoostClassifier(iterations=500,

                             learning_rate=0.02,

                             depth=12,

                             eval_metric='AUC',

                             random_seed = 2,

                             bagging_temperature = 0.2,

                             od_type='Iter',

                             metric_period = 50,

                             od_wait=100)

clf.fit(train_df[predictors], train_df[target].values,verbose=True)

preds = clf.predict(valid_df[predictors])

tmp=pd.DataFrame({'Features':predictors,'Importance':clf.feature_importances_})

tmp.sort_values(by='Importance',ascending=False,inplace=True)

sns.barplot(x='Features',y='Importance',data=tmp)

plt.xticks(rotation=90)
cm = pd.crosstab(valid_df[target].values, preds, rownames=['Actual'], colnames=['Predicted'])

fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))

sns.heatmap(cm, 

            xticklabels=['Not Fraud', 'Fraud'],

            yticklabels=['Not Fraud', 'Fraud'],

            annot=True,ax=ax1,

            linewidths=.2,linecolor="Darkblue", cmap="Blues")

plt.title('Confusion Matrix', fontsize=14)

plt.show()
roc_auc_score(valid_df[target].values, preds)
kf = KFold(n_splits = 5, random_state = 2, shuffle = True)



# Create arrays and dataframes to store results

oof_preds = np.zeros(train_df.shape[0])

test_preds = np.zeros(test_df.shape[0])

feature_importance_df = pd.DataFrame()

n_fold = 0

for train_idx, valid_idx in kf.split(train_df):

    train_x, train_y = train_df[predictors].iloc[train_idx],train_df[target].iloc[train_idx]

    valid_x, valid_y = train_df[predictors].iloc[valid_idx],train_df[target].iloc[valid_idx]

    

    evals_results = {}

    model =  LGBMClassifier(

                  nthread=-1,

                  n_estimators=2000,

                  learning_rate=0.01,

                  num_leaves=80,

                  colsample_bytree=0.98,

                  subsample=0.78,

                  reg_alpha=0.04,

                  reg_lambda=0.073,

                  subsample_for_bin=50,

                  boosting_type='gbdt',

                  is_unbalance=False,

                  min_split_gain=0.025,

                  min_child_weight=40,

                  min_child_samples=510,

                  objective='binary',

                  metric='auc',

                  silent=-1,

                  verbose=-1,

                  feval=None)

    model.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 

                eval_metric= 'auc', verbose= VERBOSE_EVAL, early_stopping_rounds= EARLY_STOP)

    

    oof_preds[valid_idx] = model.predict_proba(valid_x, num_iteration=model.best_iteration_)[:, 1]

    test_preds += model.predict_proba(test_df[predictors], num_iteration=model.best_iteration_)[:, 1] / kf.n_splits

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = predictors

    fold_importance_df["importance"] = clf.feature_importances_

    fold_importance_df["fold"] = n_fold + 1

    

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))

    del model, train_x, train_y, valid_x, valid_y

    gc.collect()

    n_fold = n_fold + 1

train_auc_score = roc_auc_score(train_df[target], oof_preds)

print('Full AUC score %.6f' % train_auc_score)