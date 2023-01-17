%reload_ext autoreload

%autoreload 2

%matplotlib inline



import numpy as np

import pandas as pd

import os

from tqdm import tqdm_notebook

from collections import Counter

import warnings 

warnings.filterwarnings("ignore")

import optuna

import multiprocessing

from joblib import Parallel, delayed

from typing import Any

import gc

import re

import random

pd.set_option('display.max_columns', None)

# pd.set_option('display.max_rows', None)

target = 'accuracy_group_target'
def seed_everything(seed=1234): 

    random.seed(seed) 

    os.environ['PYTHONHASHSEED'] = str(seed) 

    np.random.seed(seed)
seed_everything(2020)
DATA_PATH = '/kaggle/input/data-science-bowl-2019/'

def read_data():

    print('Reading train.csv file....')

    train = pd.read_csv(DATA_PATH + 'train.csv')

    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))



    print('Reading test.csv file....')

    test = pd.read_csv(DATA_PATH + 'test.csv')

    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))



    print('Reading train_labels.csv file....')

    train_labels = pd.read_csv(DATA_PATH + 'train_labels.csv')

    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))



    print('Reading specs.csv file....')

    specs = pd.read_csv(DATA_PATH + 'specs.csv')

    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))



    print('Reading sample_submission.csv file....')

    sample_submission = pd.read_csv(DATA_PATH + 'sample_submission.csv')

    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))

    return train, test, train_labels, specs, sample_submission



train, test, train_labels, specs, sample_submission = read_data()

keep_id = train[(train.type == "Assessment")  & ((train.event_code==4100) | (train.event_code==4110))][['installation_id']].drop_duplicates()

train_target = pd.merge(train, keep_id, on="installation_id", how="inner")

train_target['flag'] = 1

test['flag'] = 0

df = pd.concat([train_target, test])
print(df.shape)



del train, test, train_labels, train_target

gc.collect()
def get_unique(df):

#     title_event_codes = df.title_event_code.unique()

    titles = df.title.unique()

    event_codes = df.event_code.unique()

    event_ids = df.event_id.unique()

    worlds = df.world.unique()

    types = df.type.unique()

    asses_titles = ['Mushroom Sorter (Assessment)', 'Bird Measurer (Assessment)',

       'Chest Sorter (Assessment)', 'Cauldron Filler (Assessment)',

       'Cart Balancer (Assessment)']

    event_data = ['description', 'round_number', 'shell_size', 'location', 'nest', 'rocket', 'exit_type', 'target_water_level', 'crystal_id', 'height', 'buckets_placed', 'dinosaur_count', 'round_target', 'tutorial_step', 'dinosaurs', 'tape_length', 'game_time', 'media_type', 'buglength', 'weight', 'holding_shell', 'castles_placed', 'sand', 'has_water', 'crystals', 'end_position', 'movie_id', 'event_count', 'cloud', 'previous_jars', 'bottle', 'duration', 'starting_weights', 'event_code', 'table_weights', 'level', 'growth', 'stumps', 'scale_weights', 'resources', 'hat', 'time_played', 'bottles', 'containers', 'cauldron', 'dinosaur', 'target_containers', 'hats_placed', 'bowls', 'options', 'pillars', 'caterpillar', 'current_containers', 'holes', 'jar', 'scale_contents', 'container_type', 'bucket', 'bug', 'target_bucket', 'bowl_id', 'hats', 'group', 'toy', 'stage_number', 'bird_height', 'misses', 'house', 'target_weight', 'dinosaurs_placed', 'molds', 'scale_weight', 'bug_length', 'houses', 'buckets', 'prompt', 'right', 'object_type', 'position', 'session_duration', 'target_size', 'jar_filled', 'total_duration', 'item_type', 'caterpillars', 'layout', 'dinosaur_weight', 'animal', 'hole_position', 'correct', 'target_distances', 'gate', 'dwell_time', 'object', 'weights', 'flowers', 'round_prompt', 'source', 'max_position', 'chests', 'version', 'side', 'round', 'cloud_size', 'shells', 'flower', 'mode', 'distance', 'total_bowls', 'size', 'identifier', 'launched', 'toy_earned', 'diet', 'has_toy', 'animals', 'water_level', 'left', 'filled', 'coordinates', 'destination', 'total_containers']

    win_codes = {t:4100 for t in titles}

    win_codes['Bird Measurer (Assessment)'] = 4110

    df.timestamp = pd.to_datetime(df.timestamp)

    return titles, event_codes, event_ids, worlds, types, asses_titles, event_data, win_codes

titles, event_codes, event_ids, worlds, types, asses_titles, event_data, win_codes = get_unique(df)

asses_titles = ['Mushroom Sorter (Assessment)', 'Bird Measurer (Assessment)',

'Chest Sorter (Assessment)', 'Cauldron Filler (Assessment)',

'Cart Balancer (Assessment)']

win_codes = {t:4100 for t in titles}

win_codes['Bird Measurer (Assessment)'] = 4110

df.timestamp = pd.to_datetime(df.timestamp)

df = df.merge(specs, how='left', on='event_id', suffixes=('','_y'))
compiled_session = []

def update_counters(counter: dict, col: str, session):

    num_of_session_count = Counter(session[col])

    for k in num_of_session_count.keys():

        x = k

        counter[x] += num_of_session_count[k]

    return counter

for i, session in tqdm_notebook(df.groupby('game_session', sort=False), total=203912):

    features = {c: 0 for c in list(worlds) + list(types) + list(titles) + list(event_codes) + list(event_ids)}

    session_type = session['type'].iloc[0]

    session_title = session['title'].iloc[0]

    features = update_counters(features, "event_id", session)

    features = update_counters(features, "world", session)

    features = update_counters(features, "type", session)

    features = update_counters(features, "title", session)

    features = update_counters(features, "event_code", session)

    features['installation_id'] = session['installation_id'].iloc[0]

    features['game_session'] = session['game_session'].iloc[0]

    features['session_title'] = session_title

    features['flag'] = session['flag'].iloc[0]

    features['session_type'] = session_type

    features['world'] = session['world'].iloc[0]

    features['event_count'] = session['event_count'].iloc[-1]

    features['session_count'] = 1

    features['var_event_id'] =  session.event_id.nunique()

    features['var_title'] = session.title.nunique()

    features[session_type] = 1

    features['start_time'] = session['timestamp'].iloc[0]

    features['end_time'] = session['timestamp'].iloc[-1]

    features['duration'] =  (session['timestamp'].iloc[-1] - session['timestamp'].iloc[0]).seconds

    features['game_time'] = session['game_time'].iloc[-1]

    features['0'] = 0

    features['1'] = 0

    features['2'] = 0

    features['3'] = 0

    features['num_click'] = session['info'].str.contains('click').sum()

    if (session_type == 'Assessment') & (len(session) > 1):

        all_attempts = session.query(f'event_code == {win_codes[session_title]}')

        true_attempts = all_attempts['event_data'].str.contains('true').sum()

        false_attempts = all_attempts['event_data'].str.contains('false').sum()

        features['num_incorrect'] = false_attempts

        features['num_correct'] = true_attempts

        if (true_attempts+false_attempts)>0:

            accuracy = true_attempts/(true_attempts+false_attempts)

            features['accuracy'] = accuracy

            if accuracy == 0:

                features['accuracy_group'] = 0

            elif accuracy == 1:

                features['accuracy_group'] = 3

            elif accuracy == 0.5:

                features['accuracy_group'] = 2

            else:

                features['accuracy_group'] = 1

            features[str(features['accuracy_group'])] = 1

    if (session_type == 'Game') & (len(session) > 1):

        true_attempts = session['info'].str.contains('Correct').sum()

        false_attempts = session['info'].str.contains('Incorrect').sum()

        play_times = session['info'].str.contains('again').sum()

        features['game_num_incorrect'] = false_attempts

        features['game_num_correct'] = true_attempts

        features['game_play_again'] = play_times

    compiled_session.append(features)
del df

gc.collect()
compiled_df = pd.DataFrame(compiled_session)

del compiled_session, df

gc.collect()
clip_lengh = {

    'Welcome to Lost Lagoon!':19,

    'Tree Top City - Level 1':17,

    'Ordering Spheres':61,

    'Costume Box':61,

    '12 Monkeys':109,

    'Tree Top City - Level 2':25,

    "Pirate's Tale":80,

    'Treasure Map':156,

    'Tree Top City - Level 3':26,

    'Rulers':126,

    'Magma Peak - Level 1':20,

    'Slop Problem':60,

    'Magma Peak - Level 2':22,

    'Crystal Caves - Level 1':18,

    'Balancing Act':72,

    'Lifting Heavy Things':118,

    'Crystal Caves - Level 2':24,

    'Honey Cake':142,

    'Crystal Caves - Level 3':19,

    'Heavy, Heavier, Heaviest':61

}

compiled_df['clip_time'] = compiled_df['session_title'].map(clip_lengh)

compiled_df['clip_time'].fillna(0, inplace=True)

compiled_df['game_time'] = compiled_df['game_time']/1000

compiled_df['game_time'] = compiled_df['game_time'] + compiled_df['clip_time']

compiled_df.drop(['clip_time'], axis=1, inplace=True)
def block2feature(sample_id):

    installation_id = sample_id['installation_id'].values[0]

    sample_id.drop(columns=['installation_id','game_session','world'], inplace=True, axis=1)

    ## find the user has previous assessment or not

    idx = list(sample_id[sample_id.accuracy.notnull()].index)

    b = sample_id.index[0]

    ## if train and have more than two assessment : len(idx) > 1

    ## if train and only have one assessment :len(idx) = 1

    ## if test and have previous assessment: len(idx) > 0

    ## if test and no previous assessment: len(idx) = 0 

    if sample_id['flag'].values[0] == 0:

        idx.append(sample_id.index[-1])

    for e in idx:

        one_block = sample_id.loc[b:e-1]

        features = {}

        drop_cols = ['session_title','session_type','start_time', 'end_time', 'flag']

        for col in one_block.columns.drop(drop_cols):

            features[str(col)+'_sum'] = one_block[col].sum()

#             features[str(col)+'_mean'] = one_block[col].mean()

#             features[str(col)+'_max'] = one_block[col].max()

#             features[str(col)+'_min'] = one_block[col].min()

#             features[str(col)+'_std'] = one_block[col].std()

#             if len(one_block[col].mode()) != 0:

#                 features[str(col)+'_mode'] = one_block[col].mode()[0]

#             features[str(col)+'_skew'] = one_block[col].skew()

        features['accuracy_target'] = sample_id['accuracy'].loc[e]

        features['accuracy_group_target'] = sample_id['accuracy_group'].loc[e]

        features['installation_id'] = installation_id

        features['start_time'] = sample_id['start_time'].loc[e]

        features['session_title'] = sample_id['session_title'].loc[e]

        features['flag'] = sample_id['flag'].values[0]

        feature_df.append(features)

    return feature_df
myList = []

for i, sample_id in compiled_df.groupby('installation_id', sort=False):

    myList.append(sample_id)

inputs = tqdm_notebook(myList)

feature_df = []

feature_df = Parallel(n_jobs=8)(delayed(block2feature)(sample_id) for sample_id in inputs)

feature_df = [l for f in feature_df for l in f]

feature_df = pd.DataFrame(feature_df)

feature_df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in feature_df.columns]
feature_df['hour'] = feature_df['start_time'].dt.hour

feature_df['weekday'] = feature_df['start_time'].dt.weekday

feature_df['is_weekday'] = feature_df['weekday'] < 5

feature_df.drop(['start_time', 'accuracy_target'], axis=1, inplace=True)



# category

title2no = {t:n for n, t in enumerate(titles)}

cat_features = ['session_title']

for c in cat_features:

    feature_df[c] = feature_df[c].map(title2no)
del compiled_df

gc.collect()
features = feature_df.columns.drop(['accuracy_group_target','installation_id'])

reduce_train_org = feature_df[feature_df.accuracy_group_target.notnull()]

reduce_test = feature_df[feature_df.accuracy_group_target.isnull()]

reduce_train_org.fillna(-1, inplace=True)

reduce_test.fillna(-1, inplace=True)
import time

import lightgbm as lgb

from math import sqrt

from sklearn.metrics import mean_squared_error



def get_feature_importances(data,features, cat_features, target, shuffle, seed=None):

    # Shuffle target if required

    y = data[target].copy()

    if shuffle:

        # Here you could as well use a binomial distribution

        y = data[target].copy().sample(frac=1.0)

    

    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest

    dtrain = lgb.Dataset(data[features], y, free_raw_data=False, silent=False, categorical_feature=cat_features)



    params = {

            'n_estimators':1000,

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': 'rmse',

            'subsample': 0.75,

            'subsample_freq': 12,

            'learning_rate': 0.03341868192252964,

            'feature_fraction': 0.9219472462181388,

            'max_depth': 13,

            'lambda_l1': 0.8355562498835661,  

            'lambda_l2': 0.09460962025087172,

            'bagging_seed':seed

            }

    

    

    # Fit the model

    clf = lgb.train(params=params, train_set=dtrain,verbose_eval=500)



    # Get feature importances

    imp_df = pd.DataFrame()

    imp_df["feature"] = list(features)

    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')

    imp_df["importance_split"] = clf.feature_importance(importance_type='split')

    imp_df['trn_score'] = sqrt(mean_squared_error(y, clf.predict(data[features])))

    

    return imp_df
# Get the actual importance, i.e. without shuffling

actual_imp_df1 = get_feature_importances(data=reduce_train_org,features=features,cat_features=cat_features, target='accuracy_group_target', shuffle=False, seed=42)

actual_imp_df2 = get_feature_importances(data=reduce_train_org,features=features,cat_features=cat_features, target='accuracy_group_target', shuffle=False, seed=88)

actual_imp_df3 = get_feature_importances(data=reduce_train_org,features=features,cat_features=cat_features, target='accuracy_group_target', shuffle=False, seed=999)

actual_imp_df4 = get_feature_importances(data=reduce_train_org,features=features,cat_features=cat_features, target='accuracy_group_target', shuffle=False, seed=2020)

actual_imp_df5 = get_feature_importances(data=reduce_train_org,features=features,cat_features=cat_features, target='accuracy_group_target', shuffle=False, seed=1000)

actual_imp_df = actual_imp_df1.copy()

actual_imp_df['importance_gain'] = (actual_imp_df1['importance_gain'] + actual_imp_df2['importance_gain'] +actual_imp_df3['importance_gain']+actual_imp_df4['importance_gain']+actual_imp_df5['importance_gain'])/5
null_imp_df = pd.DataFrame()

nb_runs = 30

import time

start = time.time()

dsp = ''

for i in range(nb_runs):

    # Get current run importances

    imp_df = get_feature_importances(data=reduce_train_org,features=features,cat_features=cat_features, target='accuracy_group_target', shuffle=True)

    imp_df['run'] = i + 1 

    # Concat the latest importances with the old ones

    null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)

    # Erase previous message

    for l in range(len(dsp)):

        print('\b', end='', flush=True)

    # Display current run and time used

    spent = (time.time() - start) / 60

    dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)

    print(dsp, end='', flush=True) 
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns

%matplotlib inline



feature_scores = []

for _f in actual_imp_df['feature'].unique():

    f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values

    f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()

    # act_importance should be much bigger than null importance

    gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero

    f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values

    f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()

    split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero

    feature_scores.append((_f, split_score, gain_score))



scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])



plt.figure(figsize=(20, 20))

gs = gridspec.GridSpec(1, 2)

# Plot Gain importances

ax = plt.subplot(gs[0, 1])

sns.barplot(x='gain_score', y='feature', data=scores_df.sort_values('gain_score', ascending=False).iloc[0:100], ax=ax)

ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)

plt.tight_layout()



pd.set_option('max_rows',2000)

new_list = scores_df.sort_values(by=['gain_score'],ascending=False).reset_index(drop=True)

new_list.head(2000)



for item in new_list['feature']:

    #print (item) 

    print ('"' + str(item) +  '",')   
reduce_valid = pd.DataFrame()

for i, row in reduce_train_org.groupby('installation_id', sort=False):

    reduce_valid = reduce_valid.append(row.sample(1))

reduce_train = reduce_train_org.drop(reduce_valid.index)
features = new_list.loc[new_list.gain_score >= 0.05, 'feature'].values
params = {

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': 'rmse',

            'subsample': 0.75,

            'subsample_freq': 12,

            'learning_rate': 0.03341868192252964,

            'feature_fraction': 0.9219472462181388,

            'max_depth': 13,

            'lambda_l1': 0.8355562498835661,  

            'lambda_l2': 0.09460962025087172,

            }
target = 'accuracy_group_target'



train_x, train_y = reduce_train[features], reduce_train[target]

valid_x, valid_y = reduce_valid[features], reduce_valid[target]

print ('train_x shape:',train_x.shape)

print ('valid_x shape:',valid_x.shape)

dtrain = lgb.Dataset(train_x, label=train_y,categorical_feature=cat_features)

dval = lgb.Dataset(valid_x, label=valid_y, reference=dtrain,categorical_feature=cat_features) 

bst = lgb.train(params, dtrain, num_boost_round=50000,categorical_feature = cat_features,

    valid_sets=[dval,dtrain], verbose_eval=500,early_stopping_rounds=300)

valid_pred = bst.predict(valid_x, num_iteration=bst.best_iteration)
from sklearn.metrics import cohen_kappa_score

dist = Counter(reduce_train_org['accuracy_group_target'])

for k in dist:

    dist[k] /= len(reduce_train_org)

reduce_train_org['accuracy_group_target'].hist()

acum = 0

bound = {}

for i in range(3):

    acum += dist[i]

    bound[i] = np.percentile(valid_pred, acum * 100)

print(bound)



def classify(x):

    if x <= bound[0]:

        return 0

    elif x <= bound[1]:

        return 1

    elif x <= bound[2]:

        return 2

    else:

        return 3

valid_pred = np.array(list(map(classify, valid_pred))).reshape(valid_y.shape)

print(cohen_kappa_score(valid_y, valid_pred, weights='quadratic'))
test_pred = bst.predict(reduce_test[features], num_iteration=bst.best_iteration)

dist = Counter(reduce_valid['accuracy_group_target'])

for k in dist:

    dist[k] /= len(reduce_valid)

reduce_valid['accuracy_group_target'].hist()

acum = 0

bound = {}

for i in range(3):

    acum += dist[i]

    bound[i] = np.percentile(test_pred, acum * 100)

print(bound)



def classify(x):

    if x <= bound[0]:

        return 0

    elif x <= bound[1]:

        return 1

    elif x <= bound[2]:

        return 2

    else:

        return 3

test_pred = np.array(list(map(classify, test_pred)))

sample_submission['accuracy_group'] = test_pred.astype(int)

sample_submission.to_csv('submission.csv', index=False)

sample_submission['accuracy_group'].value_counts(normalize=True)