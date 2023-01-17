# Data filter

finish_data = 2014 # no leaks

#finish_data = 2019 # all data



# LGB

lgb_num_leaves_max = 200

lgb_in_leaf = 10

lgb_lr = 0.001

lgb_bagging = 7



# XGB

xgb_max_depth = 7

xgb_min_child_weight = 75

xgb_lr = 0.0004

xgb_num_boost_round_max = 3000



# Set weight of models

w_lgb = 0.4

w_xgb = 0.5

w_logreg = 1 - w_lgb - w_xgb

w_logreg
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import eli5



from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.utils import shuffle

from sklearn.model_selection import GridSearchCV, KFold, train_test_split

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix

import lightgbm as lgb

import xgboost as xgb



import gc



import warnings

warnings.filterwarnings("ignore")
tourney_result = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WNCAATourneyCompactResults.csv')

if finish_data == 2014:

    tourney_result = tourney_result[tourney_result['Season'] < 2015]

tourney_seed = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WNCAATourneySeeds.csv')

if finish_data == 2014:

    tourney_seed = tourney_seed[tourney_seed['Season'] < 2015]

tourney_result = tourney_result.drop(['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], axis=1)

tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

tourney_result.rename(columns={'Seed':'WSeed'}, inplace=True)

tourney_result = tourney_result.drop('TeamID', axis=1)

tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

tourney_result.rename(columns={'Seed':'LSeed'}, inplace=True)

tourney_result = tourney_result.drop('TeamID', axis=1)
tourney_seed
def get_seed(x):

    return int(x[1:3])



tourney_result['WSeed'] = tourney_result['WSeed'].map(lambda x: get_seed(x))

tourney_result['LSeed'] = tourney_result['LSeed'].map(lambda x: get_seed(x))
season_result = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WRegularSeasonCompactResults.csv')

if finish_data == 2014:

    season_result = season_result[season_result['Season'] < 2015]

season_win_result = season_result[['Season', 'WTeamID', 'WScore']]

season_lose_result = season_result[['Season', 'LTeamID', 'LScore']]

season_win_result.rename(columns={'WTeamID':'TeamID', 'WScore':'Score'}, inplace=True)

season_lose_result.rename(columns={'LTeamID':'TeamID', 'LScore':'Score'}, inplace=True)

season_result = pd.concat((season_win_result, season_lose_result)).reset_index(drop=True)

season_score = season_result.groupby(['Season', 'TeamID'])['Score'].sum().reset_index()
tourney_result = pd.merge(tourney_result, season_score, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

tourney_result.rename(columns={'Score':'WScoreT'}, inplace=True)

tourney_result = tourney_result.drop('TeamID', axis=1)

tourney_result = pd.merge(tourney_result, season_score, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

tourney_result.rename(columns={'Score':'LScoreT'}, inplace=True)

tourney_result = tourney_result.drop('TeamID', axis=1)
tourney_win_result = tourney_result.drop(['Season', 'WTeamID', 'LTeamID'], axis=1)

tourney_win_result.rename(columns={'WSeed':'Seed1', 'LSeed':'Seed2', 'WScoreT':'ScoreT1', 'LScoreT':'ScoreT2'}, inplace=True)
tourney_lose_result = tourney_win_result.copy()

tourney_lose_result['Seed1'] = tourney_win_result['Seed2']

tourney_lose_result['Seed2'] = tourney_win_result['Seed1']

tourney_lose_result['ScoreT1'] = tourney_win_result['ScoreT2']

tourney_lose_result['ScoreT2'] = tourney_win_result['ScoreT1']
tourney_win_result['Seed_diff'] = tourney_win_result['Seed1'] - tourney_win_result['Seed2']

tourney_win_result['ScoreT_diff'] = tourney_win_result['ScoreT1'] - tourney_win_result['ScoreT2']

tourney_lose_result['Seed_diff'] = tourney_lose_result['Seed1'] - tourney_lose_result['Seed2']

tourney_lose_result['ScoreT_diff'] = tourney_lose_result['ScoreT1'] - tourney_lose_result['ScoreT2']
tourney_win_result['result'] = 1

tourney_lose_result['result'] = 0

train_df = pd.concat((tourney_win_result, tourney_lose_result)).reset_index(drop=True)

train_df
season_result
tourney_result
test_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WSampleSubmissionStage1_2020.csv')

sub = test_df.copy()
test_df
test_df['Season'] = test_df['ID'].map(lambda x: int(x[:4]))

test_df['WTeamID'] = test_df['ID'].map(lambda x: int(x[5:9]))

test_df['LTeamID'] = test_df['ID'].map(lambda x: int(x[10:14]))
tourney_seed = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WNCAATourneySeeds.csv')

if finish_data == 2014:

    tourney_seed = tourney_seed[tourney_seed['Season'] > 2014]
season_result = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WRegularSeasonCompactResults.csv')

if finish_data == 2014:

    season_result = season_result[season_result['Season'] > 2014]

season_win_result = season_result[['Season', 'WTeamID', 'WScore']]

season_lose_result = season_result[['Season', 'LTeamID', 'LScore']]

season_win_result.rename(columns={'WTeamID':'TeamID', 'WScore':'Score'}, inplace=True)

season_lose_result.rename(columns={'LTeamID':'TeamID', 'LScore':'Score'}, inplace=True)

season_result = pd.concat((season_win_result, season_lose_result)).reset_index(drop=True)

season_score = season_result.groupby(['Season', 'TeamID'])['Score'].sum().reset_index()
test_df = pd.merge(test_df, tourney_seed, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

test_df.rename(columns={'Seed':'Seed1'}, inplace=True)

test_df = test_df.drop('TeamID', axis=1)

test_df = pd.merge(test_df, tourney_seed, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

test_df.rename(columns={'Seed':'Seed2'}, inplace=True)

test_df = test_df.drop('TeamID', axis=1)

test_df = pd.merge(test_df, season_score, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

test_df.rename(columns={'Score':'ScoreT1'}, inplace=True)

test_df = test_df.drop('TeamID', axis=1)

test_df = pd.merge(test_df, season_score, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

test_df.rename(columns={'Score':'ScoreT2'}, inplace=True)

test_df = test_df.drop('TeamID', axis=1)
test_df['Seed1'] = test_df['Seed1'].map(lambda x: get_seed(x))

test_df['Seed2'] = test_df['Seed2'].map(lambda x: get_seed(x))

test_df['Seed_diff'] = test_df['Seed1'] - test_df['Seed2']

test_df['ScoreT_diff'] = test_df['ScoreT1'] - test_df['ScoreT2']

test_df = test_df.drop(['ID', 'Pred', 'Season', 'WTeamID', 'LTeamID'], axis=1)

test_df
X = train_df.drop('result', axis=1)

y = train_df.result
print(sum(y), np.mean(y), np.std(y))
X
y
params_lgb = {'num_leaves': lgb_num_leaves_max,

              'min_data_in_leaf': lgb_in_leaf,

              'objective': 'binary',

              'max_depth': -1,

              'learning_rate': lgb_lr,

              "boosting_type": "gbdt",

              "bagging_seed": lgb_bagging,

              "metric": 'logloss',

              "verbosity": -1,

              'random_state': 42,

             }
NFOLDS = 5

folds = KFold(n_splits=NFOLDS)



columns = X.columns

splits = folds.split(X, y)

y_preds_lgb = np.zeros(test_df.shape[0])

y_train_lgb = np.zeros(X.shape[0])

y_oof = np.zeros(X.shape[0])



feature_importances = pd.DataFrame()

feature_importances['feature'] = columns

  

for fold_n, (train_index, valid_index) in enumerate(splits):

    print('Fold:',fold_n+1)

    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    dtrain = lgb.Dataset(X_train, label=y_train)

    dvalid = lgb.Dataset(X_valid, label=y_valid)



    clf = lgb.train(params_lgb, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200)

    

    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()

    

    y_pred_valid = clf.predict(X_valid)

    y_oof[valid_index] = y_pred_valid

    

    y_train_lgb += clf.predict(X) / NFOLDS

    y_preds_lgb += clf.predict(test_df) / NFOLDS

    

    del X_train, X_valid, y_train, y_valid

    gc.collect()
print(min(y_preds_lgb),max(y_preds_lgb))

print(min(y_train_lgb),max(y_train_lgb))
params_xgb = {'max_depth': xgb_max_depth,

              'objective': 'binary:logistic',

              'min_child_weight': xgb_min_child_weight,

              'learning_rate': xgb_lr,

              'eta'      : 0.3,

              'subsample': 0.8,

              'eval_metric': 'logloss',

              'colsample_bylevel': 1

              }
# Thanks to https://www.kaggle.com/khoongweihao/ncaam2020-xgboost-lightgbm-k-fold-baseline

NFOLDS = 5

folds = KFold(n_splits=NFOLDS)



columns = X.columns

splits = folds.split(X, y)



y_preds_xgb = np.zeros(test_df.shape[0])

y_train_xgb = np.zeros(X.shape[0])

y_oof_xgb = np.zeros(X.shape[0])



train_df_set = xgb.DMatrix(X)

test_set = xgb.DMatrix(test_df)

  

for fold_n, (train_index, valid_index) in enumerate(splits):

    print('Fold:',fold_n+1)

    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    train_set = xgb.DMatrix(X_train, y_train)

    val_set = xgb.DMatrix(X_valid, y_valid)

    

    clf = xgb.train(params_xgb, train_set, num_boost_round=xgb_num_boost_round_max, evals=[(train_set, 'train'), (val_set, 'val')], verbose_eval=100)

    

    y_train_xgb += clf.predict(train_df_set) / NFOLDS

    y_preds_xgb += clf.predict(test_set) / NFOLDS

    

    del X_train, X_valid, y_train, y_valid

    gc.collect()
test_df.head()
%%time

# Standardization for regression models

scaler = StandardScaler()

train_log = pd.DataFrame(

    scaler.fit_transform(X),

    columns=X.columns,

    index=X.index

)

test_log = pd.DataFrame(

    scaler.transform(test_df),

    columns=test_df.columns,

    index=test_df.index

)
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(train_log, y)

coeff_logreg = pd.DataFrame(train_log.columns.delete(0))

coeff_logreg.columns = ['feature']

coeff_logreg["score_logreg"] = pd.Series(logreg.coef_[0])

coeff_logreg.sort_values(by='score_logreg', ascending=False)
# Eli5 visualization

eli5.show_weights(logreg)
y_logreg_train = logreg.predict(train_log)

y_logreg_pred = logreg.predict(test_log)
# Showing Confusion Matrix

# Thanks to https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud

def plot_cm(y_true, y_pred, title, figsize=(7,6)):

    y_pred = y_pred.round().astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

    cm_sum = np.sum(cm, axis=1, keepdims=True)

    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape

    for i in range(nrows):

        for j in range(ncols):

            c = cm[i, j]

            p = cm_perc[i, j]

            if i == j:

                s = cm_sum[i]

                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

            elif c == 0:

                annot[i, j] = ''

            else:

                annot[i, j] = '%.1f%%\n%d' % (p, c)

    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))

    cm.index.name = 'Actual'

    cm.columns.name = 'Predicted'

    fig, ax = plt.subplots(figsize=figsize)

    plt.title(title)

    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
# Showing Confusion Matrix for LGB model

plot_cm(y, y_train_lgb, 'Confusion matrix for LGB model')
# Showing Confusion Matrix for XGB model

plot_cm(y, y_train_xgb, 'Confusion matrix for XGB model')
# Showing Confusion Matrix for Logistic Regression

plot_cm(y, y_logreg_train, 'Confusion matrix for Logistic Regression')
y_preds = w_lgb*y_preds_lgb + w_xgb*y_preds_xgb + w_logreg*y_logreg_pred
# Showing Confusion Matrix for Merging solution

y_train_preds = w_lgb*y_train_lgb + w_xgb*y_train_xgb + w_logreg*y_logreg_train

plot_cm(y, y_train_preds, 'Confusion matrix for Merging solution')
sub['Pred'] = y_preds

sub.head()
sub.info()
sub['Pred'].hist()
sub.to_csv('submission.csv', index=False)