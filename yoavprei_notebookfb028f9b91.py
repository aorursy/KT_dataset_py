import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

from matplotlib.patches import Circle, Rectangle, Arc

from sklearn import preprocessing

from xgboost.sklearn import XGBClassifier

from xgboost import plot_importance

import xgboost as xgb

from sklearn.cross_validation import train_test_split, cross_val_score

import heapq

%matplotlib inline
kobe_df=pd.read_csv('../input/data.csv')

sample_submission_df=pd.read_csv('../input/sample_submission.csv')
# calculate seconds remaining to cobine the minutes and seconds remaining variables

kobe_df.loc[:,'calculated_seconds_remaining'] = kobe_df.loc[:,'seconds_remaining'] + 60*kobe_df.loc[:,'minutes_remaining']

# add home game feature

kobe_df.loc[:, 'home'] = 1

kobe_df.loc[kobe_df.matchup.str.contains("@"), 'home'] = 0
#game month

#w['female'] = w['female'].map({'female': 1, 'male': 0})

def gamemonth(lst):

    lst = pd.to_datetime(lst)

    dict_month={'1': 'Jan', '2': 'Feb', '3': 'Mar',

                '4': 'Apr', '5': 'May', '6': 'Jun',

                '7': 'Jul', '8': 'Aug', '9': 'Sep',

                '10': 'Oct', '11': 'Nov', '12': 'Dec'}

    

    splitseries=lst.apply(lambda row :str(row.month))

    

    newseries=splitseries.map(dict_month)

    

    return newseries

    

gamemonth(kobe_df['game_date']).isnull().values.any() #false

kobe_df['gamemonth']=gamemonth(kobe_df['game_date'])
#if it is a back to back game 

#(df_all['game_date'][100]-df_all['game_date'][1]).days #int

def btb(lst): #0,1

    lst = pd.to_datetime(lst)

    btb_lst=[0]

    flag=0

    for i in range(1,len(lst)):

        if (lst[i]-lst[i-1]).days==1:

            btb_lst.append(1)

            flag=1

        elif (lst[i]-lst[i-1]).days==0:

            btb_lst.append(flag)

        else:

            flag=0

            btb_lst.append(flag)

    return btb_lst



kobe_df['btb']=btb(kobe_df['game_date'])
#if last shot was made

def lastshot(lst):

    last=[0]

    for i in range(1,len(lst)):

        

        if lst[i-1]==0:

            flag=0

            last.append(0)

        elif lst[i-1]==1:

            flag=1

            last.append(1)

        else:

            last.append('unknown') #due to the random test data

    return last



kobe_df['last_shot_flag']=lastshot(kobe_df['shot_made_flag'])
#add column secondsToPeriodEnd

kobe_df['secondsToPeriodEnd'] = 60*kobe_df['minutes_remaining']+kobe_df['seconds_remaining']

#add column secondsFromPeriodEnd

kobe_df['secondsFromGameStart'] = kobe_df['period'].astype(int)*12*60 - kobe_df['secondsToPeriodEnd']
#kobe_df.drop(['game_event_id', 'game_id', 'lat', 'lon', 'team_id', 'team_name', 'matchup'], axis=1, inplace=True)

kobe_df.sort_values('game_date', inplace=True, kind='mergesort')

kobe_df = kobe_df.reset_index(drop=True)

mask = kobe_df['shot_made_flag'].isnull()

value_action_type_list = kobe_df.action_type.value_counts()

for index,count in enumerate(value_action_type_list):

    kobe_df.loc[kobe_df.action_type==value_action_type_list.index[index],'type'] = kobe_df.action_type

kobe_df.loc[kobe_df.action_type.isin(value_action_type_list.index[value_action_type_list<20]),'type'] = kobe_df.combined_shot_type

#kobe_df.drop(['action_type', 'combined_shot_type'], axis=1, inplace=True)



kobe_df.loc[kobe_df.shot_distance >= 45, 'shot_distance'] = 45

kobe_df.loc[kobe_df.calculated_seconds_remaining >= 3, 'last_shot_in_period'] = 0

kobe_df.loc[kobe_df.calculated_seconds_remaining < 3, 'last_shot_in_period'] = 1



#data = pd.get_dummies(kobe_df['type'],prefix="action_type")

data=pd.DataFrame()



# features=["period", "shot_type", "shot_zone_area", "shot_zone_basic", "season",

#            "shot_zone_range", "opponent", "shot_distance", "minutes_remaining", "gamemonth", "last_shot_flag"]

# features=["period", "shot_type", "shot_zone_area", "shot_zone_basic", "season", 'action_type',

#            "opponent", "gamemonth", 'btb', "last_shot_flag", "home", 'combined_shot_type', "playoffs"]

ctg_features=['combined_shot_type','shot_type','shot_zone_area', 'action_type', 'shot_zone_basic', 'playoffs', 'period'

          ,'opponent','season','home','btb','gamemonth','last_shot_flag', 'shot_id']

# for f in features:

#     data = pd.concat([data, pd.get_dummies(kobe_df[f], prefix=f),], axis=1)

# df_ctg = df_all.loc[:, lambda df: ctg_feature]

encoded_ctg=pd.get_dummies(kobe_df[ctg_features]).astype(np.int16)

# "last_3_shot_distance"

num_features=["shot_id", 'loc_x', 'loc_y', 'shot_distance', 'seconds_remaining','secondsToPeriodEnd','secondsFromGameStart']

data = kobe_df.loc[:, lambda df: num_features]

# data = pd.concat([data, kobe_df[["game_date","game_id","shot_id", "home", "playoffs", "last_shot_in_period", 'btb', "shot_distance"]]], axis=1)



# #f1

# last_3_action_type_Slam_Dunk_Shot = data.groupby('game_id').apply(lambda x: sliding(x, window=3, col="action_type_Slam Dunk Shot", func=np.mean))

# last_3_action_type_Slam_Dunk_Shot.index = last_3_action_type_Slam_Dunk_Shot.index.get_level_values(1)

# last_3_action_type_Slam_Dunk_Shot[np.isnan(last_3_action_type_Slam_Dunk_Shot)] = last_3_action_type_Slam_Dunk_Shot.mean()

# data.loc[:,'last_3_action_type_Slam_Dunk_Shot']=last_3_action_type_Slam_Dunk_Shot

# #f2

# last_3_action_type_Running_Jump_Shot = data.groupby('game_id').apply(lambda x: sliding(x, window=3, col="action_type_Running Jump Shot", func=np.mean))

# last_3_action_type_Running_Jump_Shot.index = last_3_action_type_Running_Jump_Shot.index.get_level_values(1)

# last_3_action_type_Running_Jump_Shot[np.isnan(last_3_action_type_Running_Jump_Shot)] = last_3_action_type_Running_Jump_Shot.mean()

# data.loc[:,'last_3_action_type_Running Jump Shot']=last_3_action_type_Running_Jump_Shot

# #f3

# last_3_action_type_Layup_Shot = data.groupby('game_id').apply(lambda x: sliding(x, window=3, col="action_type_Layup Shot", func=np.mean))

# last_3_action_type_Layup_Shot.index = last_3_action_type_Layup_Shot.index.get_level_values(1)

# last_3_action_type_Layup_Shot[np.isnan(last_3_action_type_Layup_Shot)] = last_3_action_type_Layup_Shot.mean()

# data.loc[:,'last_3_action_type_Layup_Shot']=last_3_action_type_Layup_Shot

# data.loc[:,'end_of_carrier']=0

# data.loc[(data["season_2015-16"]==1) | (data["season_2013-14"]==1) | (data["season_2014-15"]==1),'end_of_carrier']=1

# data.loc[:,'start_of_carrier']=0

# data.loc[(data["season_1996-97"]==1) | (data["season_1997-98"]==1),'start_of_carrier']=1

# data.loc[:,'medium_of_carrier']=1

# data.loc[(data["start_of_carrier"]==1) | (data["end_of_carrier"]==1),'medium_of_carrier']=0



#data.drop(['game_id', 'shot_id', 'game_date'], axis=1, inplace=True)

#data.drop(['game_id', 'game_date'], axis=1, inplace=True)

# data.drop(['season_1996-97', 'season_1997-98', 'season_1998-99',

#        'season_1999-00', 'season_2000-01', 'season_2001-02',

#        'season_2002-03', 'season_2003-04', 'season_2004-05',

#        'season_2005-06', 'season_2006-07', 'season_2007-08',

#        'season_2008-09', 'season_2009-10', 'season_2010-11',

#        'season_2011-12', 'season_2012-13', 'season_2013-14',

#        'season_2014-15', 'season_2015-16'], axis=1, inplace=True)

data = pd.merge(encoded_ctg, data,on='shot_id')

X = data[~mask]

y = kobe_df.shot_made_flag[~mask]
X_dtrain, X_deval, y_dtrain, y_deval = train_test_split(X, y, random_state=2046, test_size=0.15)

prior = 0.4

dtrain = xgb.DMatrix(X_dtrain, y_dtrain)

deval = xgb.DMatrix(X_deval, y_deval)

watchlist = [(deval, 'eval')]

params = {

    'booster': 'gbtree',

    'objective': 'binary:logistic',

    'colsample_bytree': 0.8,

    'eta': 0.1,

    'max_depth': 3,

    'seed': 2017,

    'silent': 1,

   # 'gamma':0.005,

    'subsample':0.8,

     'base_score': prior,

    'eval_metric': 'logloss'

}



clf = xgb.train(params, dtrain, 200, watchlist, early_stopping_rounds=50)