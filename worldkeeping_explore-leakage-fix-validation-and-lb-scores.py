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
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.metrics import roc_auc_score, log_loss

import lightgbm as lgb

import string
path = '../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/'

regular_result_brief=pd.read_csv(path+'WDataFiles_Stage1/WRegularSeasonCompactResults.csv')

tour_result_brief=pd.read_csv(path+'WDataFiles_Stage1/WNCAATourneyCompactResults.csv')

tour_seeds=pd.read_csv(path+'WDataFiles_Stage1/WNCAATourneySeeds.csv')

seasons=pd.read_csv(path+'WDataFiles_Stage1/WSeasons.csv')

sub_test=pd.read_csv(path+'WSampleSubmissionStage1_2020.csv')
regular_result_brief.drop(['WLoc','NumOT'],axis=1,inplace=True)

tour_result_brief.drop(['WLoc','NumOT'],axis=1,inplace=True)

rwin=regular_result_brief[['Season', 'DayNum', 'WTeamID', 'WScore']]

rwin.columns=['Season', 'DayNum', 'TeamID', 'Score']

rlos=regular_result_brief[['Season', 'DayNum', 'LTeamID', 'LScore']]

rlos.columns=['Season', 'DayNum', 'TeamID', 'Score']

regular_score=pd.concat((rwin,rlos)).reset_index(drop=True)

regular_score=regular_score.groupby(['Season','TeamID'],as_index=False).mean()

regular_score.head(7)
tour_seeds['Seed']=tour_seeds['Seed'].apply(lambda x: int(x[1:]))

def myMerge(indf):

    df=indf[['Season', 'WTeamID', 'LTeamID']]

    df=df.merge(tour_seeds,left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

    df.rename(columns={'Seed':'WSeed'}, inplace=True)

    df = df.drop('TeamID', axis=1)

    df = df.merge(tour_seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

    df.rename(columns={'Seed':'LSeed'}, inplace=True)

    df = df.drop('TeamID', axis=1)

    df = pd.merge(df, regular_score, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

    df.rename(columns={'Score':'WScore','DayNum':'WDayNum'}, inplace=True)

    df = df.drop(['TeamID'], axis=1)

    df = pd.merge(df, regular_score, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

    df.rename(columns={'Score':'LScore','DayNum':'LDayNum'}, inplace=True)

    df = df.drop(['TeamID'], axis=1)

    return df

tour_merge=myMerge(tour_result_brief)

tour_merge.head(7)
tour_win=tour_merge.copy()

tour_los=tour_merge.copy()

tour_win.columns=[c.replace('W','T1_').replace('L','T2_') for c in tour_win.columns]

tour_los.columns=[c.replace('W','T2_').replace('L','T1_') for c in tour_los.columns]

tour_win['Pred']=1

tour_los['Pred']=0

tour_all=pd.concat((tour_win, tour_los),sort=True).reset_index(drop=True)

tour_all
sub_test['Season'] = sub_test['ID'].map(lambda x: int(x[:4]))

sub_test['WTeamID'] = sub_test['ID'].map(lambda x: int(x[5:9]))

sub_test['LTeamID'] = sub_test['ID'].map(lambda x: int(x[10:14]))

test=myMerge(sub_test)

test.columns=[c.replace('W','T1_').replace('L','T2_') for c in test.columns]

test.head(7)
feature_dungeon=['ID', 'Pred', 'T1_TeamID', 'T2_TeamID']

features=[c for c in tour_all.columns if c not in feature_dungeon]

target=tour_all.Pred

train=tour_all[features]

test=test[features]
print('train shape: {}, test shape: {}'.format(train.shape,test.shape))
params = {

        'learning_rate': 0.01,

        'feature_fraction': 0.5,

        'min_data_in_leaf' : 12,

        'max_depth': 8,

        'objective': 'binary',

        'metric': 'binary_logloss',

        'n_jobs': -1,

        'feature_fraction_seed': 42,

        'bagging_seed': 42,

        'boosting_type': 'gbdt',

        'verbosity': -1,

        'is_unbalance': False,

        'boost_from_average': False}

traintion = np.zeros(len(train))

validation = np.zeros(len(train))

predictions = np.zeros(len(test))

feature_importance_df = pd.DataFrame()

folds = KFold(n_splits=100, shuffle=True, random_state=1990)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train)):

    print("fold n°{}".format(fold_))

    train_x=train.iloc[trn_idx].reset_index(drop=True)

    valid_x=train.iloc[val_idx].reset_index(drop=True)

    target_train=target.iloc[trn_idx].reset_index(drop=True)

    target_valid=target.iloc[val_idx].reset_index(drop=True)

    trn_data = lgb.Dataset(train_x,

                           label=target_train,

                          )

    val_data = lgb.Dataset(valid_x,

                           label=target_valid,

                          )



    num_round = 1000000

    clf = lgb.train(params,

                    trn_data,

                    num_round,

                    valid_sets = [trn_data, val_data],

                    verbose_eval=False,

                    early_stopping_rounds = 1000)

    traintion[trn_idx] += clf.predict(train_x, num_iteration=clf.best_iteration)/(folds.n_splits-1)

    validation[val_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = features

    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    predictions += clf.predict(test, num_iteration=clf.best_iteration) / folds.n_splits

print("Train AUC score: {:<8.5f}".format(log_loss(target,traintion)))

print("Valid AUC score: {:<8.5f}".format(log_loss(target,validation)))