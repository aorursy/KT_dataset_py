%matplotlib inline

import os

import copy

import random



random.seed(69)



import time

from collections import Counter

from typing import List, Any

from itertools import product

from collections import defaultdict

import datetime

import json

import gc

from numba import jit

import warnings

import re

import eli5

import shap

from IPython.display import HTML

import altair as alt

import networkx as nx



from joblib import Parallel, delayed



warnings.filterwarnings("ignore")



from functools import partial

import numpy as np

np.random.seed(69)



import pandas as pd

import seaborn as sns

import scipy as sp

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

from tqdm import tqdm

pd.set_option('max_rows', 500)



from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit, RepeatedStratifiedKFold

from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import cohen_kappa_score, mean_squared_error

from sklearn import linear_model

from category_encoders.ordinal import OrdinalEncoder



import lightgbm as lgb

import xgboost as xgb

import catboost as cat

from catboost import CatBoostRegressor, CatBoostClassifier



from bayes_opt import BayesianOptimization
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:80% !important; }</style>"))
# global flags here:

BOOTSTRAP_MODE = True # set True if check code error quickly.
def to_cyclic(value, max, amp=1):

    return (np.sin(np.pi * value / max) * amp)
def read_data():

    print('Reading train.csv file....')

    train = pd.read_csv('../input/data-science-bowl-2019/train.csv')

    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))



    print('Reading test.csv file....')

    test = pd.read_csv('../input/data-science-bowl-2019/test.csv')

    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))



    print('Reading train_labels.csv file....')

    train_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')

    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))



    print('Reading specs.csv file....')

    specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')

    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))



    print('Reading sample_submission.csv file....')

    sample_submission = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')

    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))

    return train, test, train_labels, specs, sample_submission
def encode_title(train, test, train_labels):

    # encode title

    # 여기서는 title과 event_code를 _를 사이에 두고 붙여 버린다.

    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))

    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))

    all_title_event_code = sorted(list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique())))



    # make a list with all the unique 'titles' from the train and test set

    # title들을 모아 숫자값으로 변경

    list_of_user_activities = sorted(list(set(train['title'].unique()).union(set(test['title'].unique()))))



    # make a list with all the unique 'event_code' from the train and test set

    list_of_event_code = sorted(list(set(train['event_code'].unique()).union(set(test['event_code'].unique()))))

    list_of_event_id = sorted(list(set(train['event_id'].unique()).union(set(test['event_id'].unique()))))



    # make a list with all the unique worlds from the train and test set

    list_of_worlds = sorted(list(set(train['world'].unique()).union(set(test['world'].unique()))))

    # create a dictionary numerating the titles

    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))

    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))

    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))

    assess_titles = sorted(list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index))))

    # replace the text titles with the number titles from the dict



    # 문자열을 위에서 생성한 dictionary에 해당하는 숫자로 바꾼다.

    train['title'] = train['title'].map(activities_map)

    test['title'] = test['title'].map(activities_map)

    train['world'] = train['world'].map(activities_world)

    test['world'] = test['world'].map(activities_world)

    train_labels['title'] = train_labels['title'].map(activities_map)

    

    # win_code 생성(Bird Measurer (Assessment)만 4110, 그 외에는 4100)

    win_code = dict(zip(activities_map.values(), (4100 * np.ones(len(activities_map))).astype('int')))

    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest

    win_code[activities_map['Bird Measurer (Assessment)']] = 4110



    # convert text into datetime

    train['timestamp'] = pd.to_datetime(train['timestamp'])

    test['timestamp'] = pd.to_datetime(test['timestamp'])

    

    train['hour'] = train['timestamp'].dt.hour

    test['hour'] = test['timestamp'].dt.hour    

    

    event_data = {"train_labels":train_labels, "win_code":win_code, "list_of_user_activities":list_of_user_activities, "list_of_event_code":list_of_event_code,

                 "activities_labels":activities_labels, "assess_titles":assess_titles, "list_of_event_id":list_of_event_id, "all_title_event_code":all_title_event_code,

                 "activities_map":activities_map}    



    return train, test, event_data
clip_time = {'Welcome to Lost Lagoon!':19,'Tree Top City - Level 1':17,'Ordering Spheres':61, 'Costume Box':61,

        '12 Monkeys':109,'Tree Top City - Level 2':25, 'Pirate\'s Tale':80, 'Treasure Map':156,'Tree Top City - Level 3':26,

        'Rulers':126, 'Magma Peak - Level 1':20, 'Slop Problem':60, 'Magma Peak - Level 2':22, 'Crystal Caves - Level 1':18,

        'Balancing Act':72, 'Lifting Heavy Things':118,'Crystal Caves - Level 2':24, 'Honey Cake':142, 'Crystal Caves - Level 3':19,

        'Heavy, Heavier, Heaviest':61}
def cnt_miss(df):

    cnt = 0

    for e in range(len(df)):

        x = df['event_data'].iloc[e]

        y = json.loads(x)['misses']

        cnt += y

    return cnt
def get_data(user_sample, test_set=False):

    '''

    The user_sample is a DataFrame from train or test where the only one 

    installation_id is filtered

    And the test_set parameter is related with the labels processing, that is only requered

    if test_set=False

    

    args:

        user_sample --- DataFrameGroupBy object group by 'installation_id'

        test_set -- on test data, only last game session need to be predicted.

        

    return:

        array of assessments(by game session).

    '''

    # Constants and parameters declaration

    last_activity = 0

    

    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

    game_time_dict = {'Clip_gametime':0, 'Game_gametime':0, 'Activity_gametime':0, 'Assessment_gametime':0}

    Assessment_mean_event_count = 0

    Game_mean_event_count = 0

    Activity_mean_event_count = 0

    mean_game_round = 0

    mean_game_duration = 0 

    mean_game_level = 0

    accumulated_game_miss = 0

    

    # new features: time spent in each activity

    last_session_time_sec = 0

    accuracy_groups = {0:0, 1:0, 2:0, 3:0}

    all_assessments = []

    accumulated_accuracy_group = 0

    accumulated_accuracy = 0

    accumulated_correct_attempts = 0 

    accumulated_uncorrect_attempts = 0

    accumulated_actions = 0

    counter = 0

    time_first_activity = float(user_sample['timestamp'].values[0])    

    durations = []

    clip_durations = []

    Activity_durations = []

    Game_durations = []

    

    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}

    event_code_count: Dict[str, int] = {ev: 0 for ev in list_of_event_code}

    event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}

    title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()} 

    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}

        

    # last features

    sessions_count = 0

    

    # itarates through each session of one instalation_id

    for i, session in user_sample.groupby('game_session', sort=False):

        # i = game_session_id

        # session is a DataFrame that contain only one game_session

        

        # get some sessions information

        session_type = session['type'].iloc[0]

        session_title = session['title'].iloc[0]

        session_title_text = activities_labels[session_title]

        

        if session_type == 'Clip':

            clip_durations.append((clip_time[activities_labels[session_title]]))

        

        if session_type == 'Activity':

            Activity_durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

            Activity_mean_event_count = (Activity_mean_event_count + session['event_count'].iloc[-1])/2.0

        

        if session_type == 'Game':

            Game_durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

            Game_mean_event_count = (Game_mean_event_count + session['event_count'].iloc[-1])/2.0

            

            game_s = session[session.event_code == 2030]   

            misses_cnt = cnt_miss(game_s)

            accumulated_game_miss += misses_cnt

            

            try:

                game_round = json.loads(session['event_data'].iloc[-1])["round"]

                mean_game_round =  (mean_game_round + game_round)/2.0

            except:

                pass



            try:

                game_duration = json.loads(session['event_data'].iloc[-1])["duration"]

                mean_game_duration = (mean_game_duration + game_duration) /2.0

            except:

                pass

            

            try:

                game_level = json.loads(session['event_data'].iloc[-1])["level"]

                mean_game_level = (mean_game_level + game_level) /2.0

            except:

                pass

            

        # for each assessment, and only this kind off session, the features below are processed

        # and a register are generated

        if (session_type == 'Assessment') & (test_set or len(session)>1):

            # search for event_code 4100, that represents the assessments trial

            all_attempts = session.query(f'event_code == {win_code[session_title]}')

            # then, check the numbers of wins and the number of losses

            true_attempts = all_attempts['event_data'].str.contains('true').sum()

            false_attempts = all_attempts['event_data'].str.contains('false').sum()

            # copy a dict to use as feature template, it's initialized with some itens: 

            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

            features = user_activities_count.copy()

            features.update(last_accuracy_title.copy())

            features.update(event_code_count.copy())

            features.update(event_id_count.copy())

            features.update(title_count.copy())

            features.update(title_event_code_count.copy())

            features.update(last_accuracy_title.copy())

            features['installation_session_count'] = sessions_count

            features['hour'] = session['hour'].iloc[-1]

            features['Assessment_mean_event_count'] = Assessment_mean_event_count

            features['Game_mean_event_count'] = Game_mean_event_count

            features['Activity_mean_event_count'] = Activity_mean_event_count

            features['mean_game_round'] = mean_game_round

            features['mean_game_duration'] = mean_game_duration

            features['mean_game_level'] = mean_game_level

            features['accumulated_game_miss'] = accumulated_game_miss

            

            variety_features = [('var_event_code', event_code_count),

                                ('var_event_id', event_id_count),

                                ('var_title', title_count),

                                ('var_title_event_code', title_event_code_count)]

            

            for name, dict_counts in variety_features:

                arr = np.array(list(dict_counts.values()))

                features[name] = np.count_nonzero(arr)

                 

            # get installation_id for aggregated features

            features['installation_id'] = session['installation_id'].iloc[-1]

            # add title as feature, remembering that title represents the name of the game

            features['session_title'] = session['title'].iloc[0]

            # the 4 lines below add the feature of the history of the trials of this player

            # this is based on the all time attempts so far, at the moment of this assessment

            features['accumulated_correct_attempts'] = accumulated_correct_attempts

            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts

            accumulated_correct_attempts += true_attempts

            accumulated_uncorrect_attempts += false_attempts

            

            # the time spent in the app so far

            if durations == []:

                features['duration_mean'] = 0

                features['duration_std'] = 0

            else:

                features['duration_mean'] = np.mean(durations)

                features['duration_std'] = np.std(durations)

                

            if clip_durations == []:

                features['Clip_duration_mean'] = 0

                features['Clip_duration_std'] = 0

            else:

                features['Clip_duration_mean'] = np.mean(clip_durations)

                features['Clip_duration_std'] = np.std(clip_durations)

                

            if Activity_durations == []:

                features['Activity_duration_mean'] = 0

                features['Activity_duration_std'] = 0

            else:

                features['Activity_duration_mean'] = np.mean(Activity_durations)

                features['Activity_duration_std'] = np.std(Activity_durations)

                

            if Game_durations == []:

                features['Game_duration_mean'] = 0

                features['Game_duration_std'] = 0

            else:

                features['Game_duration_mean'] = np.mean(Game_durations)

                features['Game_duration_std'] = np.std(Game_durations)

            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

            Assessment_mean_event_count = (Assessment_mean_event_count + session['event_count'].iloc[-1])/2.0

            # the accurace is the all time wins divided by the all time attempts

            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0

            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0

            accumulated_accuracy += accuracy

            last_accuracy_title['acc_' + session_title_text] = accuracy

            # a feature of the current accuracy categorized

            # it is a counter of how many times this player was in each accuracy group

            if accuracy == 0:

                features['accuracy_group'] = 0

            elif accuracy == 1:

                features['accuracy_group'] = 3

            elif accuracy == 0.5:

                features['accuracy_group'] = 2

            else:

                features['accuracy_group'] = 1

            features.update(accuracy_groups)

            accuracy_groups[features['accuracy_group']] += 1

            # mean of the all accuracy groups of this player

            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0

            accumulated_accuracy_group += features['accuracy_group']

            # how many actions the player has done so far, it is initialized as 0 and updated some lines below

            features['accumulated_actions'] = accumulated_actions

            

            # there are some conditions to allow this features to be inserted in the datasets

            # if it's a test set, all sessions belong to the final dataset

            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')

            # that means, must exist an event_code 4100 or 4110

            if test_set:

                all_assessments.append(features)

            elif true_attempts+false_attempts > 0:

                all_assessments.append(features)



            counter += 1



        sessions_count += 1

        

        # this piece counts how many actions was made in each event_code so far

        def update_counters(counter: dict, col: str):

            num_of_session_count = Counter(session[col])

            for k in num_of_session_count.keys():

                x = k

                if col == 'title':

                    x = activities_labels[k]

                counter[x] += num_of_session_count[k]

            return counter

            

        game_time_dict[session_type+'_gametime'] = (game_time_dict[session_type+'_gametime'] + (session['game_time'].iloc[-1]/1000.0))/2.0

        event_code_count = update_counters(event_code_count, "event_code")

        event_id_count = update_counters(event_id_count, "event_id")

        title_count = update_counters(title_count, 'title')

        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')



        # counts how many actions the player has done so far, used in the feature of the same name

        accumulated_actions += len(session)

        if last_activity != session_type:

            user_activities_count[session_type] += 1

            last_activitiy = session_type 



    # if it't the test_set, only the last assessment must be predicted, the previous are scraped

    if test_set:

        return all_assessments[-1]

    # in the train_set, all assessments goes to the dataset

    return all_assessments
def get_session_data(df, is_test=False, cut_last=False):

    compiled_df = []

    total_cnt = df.installation_id.unique().shape[0]

    for i, (ins_id, user_sample) in tqdm(enumerate(df.groupby('installation_id', sort = False)), total = total_cnt):

        session_data = get_data(user_sample, test_set=is_test)        

        if type(session_data) is list:

            if cut_last:

                session_data = session_data[:-1]

            compiled_df += session_data

        else:

            compiled_df.append(session_data)



    reduce_df = pd.DataFrame(compiled_df)    

    categoricals = ['session_title']

    return reduce_df, categoricals

    
def preprocess(df):

    df['installation_session_count'] = df.groupby(['installation_id'])['Clip'].transform('count') #installation_id 마다 이루어진 session의 수

    df['installation_duration_mean'] = df.groupby(['installation_id'])['duration_mean'].transform('mean')        

    df['installation_title_nunique'] = df.groupby(['installation_id'])['session_title'].transform('nunique')



    df['sum_event_code_count'] = df[[2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021, 

                                    4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 4050, 2010, 2020, 4070, 2025, 2030, 4080, 2035, 

                                    2040, 4090, 4220, 4095]].sum(axis = 1)



    df['installation_event_code_count_mean'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('mean') # or 'std'

    

    

    # cyclic categorical values conversion.

    df['hour'] = to_cyclic(df['hour'], 24, 4)    

    

    # remove special characters from column names

    df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df.columns]

        

    return df
# read data

try:

    del train

    del test

    gc.collect()

except:

    pass



train, test, train_labels, specs, sample_submission = read_data()



# get usefull dict with maping encode

train, test, event_data = encode_title(train, test, train_labels)



train_labels = event_data['train_labels']

win_code = event_data['win_code']

list_of_user_activities = event_data['list_of_user_activities']

list_of_event_code = event_data['list_of_event_code']

activities_labels = event_data['activities_labels']

assess_titles = event_data['assess_titles']

list_of_event_id = event_data['list_of_event_id']

all_title_event_code = event_data['all_title_event_code']



reduce_train, categoricals = get_session_data(train, is_test=False)

reduce_train = preprocess(reduce_train)



reduce_test, categoricals = get_session_data(test, is_test = True)

reduce_test = preprocess(reduce_test)



del train

del test
reduce_train.head()
reduce_test.head()
cols_to_drop = ['game_session', 'installation_id', 'timestamp', 'accuracy_group', 'timestampDate']

all_features = [x for x in reduce_train.columns if x not in cols_to_drop]

cat_features = ['session_title']
feat_remove = []

def get_duplicated_features(df, features, threshold = 0.995):

    counter = 0

    to_remove = []

    for feat_a in features:

        for feat_b in features:

            if feat_a != feat_b and feat_a not in to_remove and feat_b not in to_remove:

                c = np.corrcoef(df[feat_a], df[feat_b])[0][1] # (2x2) table by (x,y)

                if c > threshold:

                    counter += 1

                    to_remove.append(feat_b)

                    #print('{}: FEAT_A: {} FEAT_B: {} - Correlation: {}'.format(counter, feat_a, feat_b, c))

    print(f'{counter} duplicated features found.')

    return to_remove



#feat_remove = get_duplicated_features(reduce_train, all_features)
def stract_hists(feature, train=reduce_train, test=reduce_test, adjust=False, plot=False):

    n_bins = 10

    train_data = train[feature]

    test_data = test[feature]

    if adjust:

        test_data *= train_data.mean() / test_data.mean()

    perc_90 = np.percentile(train_data, 95)

    train_data = np.clip(train_data, 0, perc_90)

    test_data = np.clip(test_data, 0, perc_90)

    train_hist = np.histogram(train_data, bins=n_bins)[0] / len(train_data)

    test_hist = np.histogram(test_data, bins=n_bins)[0] / len(test_data)

    mse = mean_squared_error(train_hist, test_hist)

    if plot:

        print(mse)

        plt.bar(range(n_bins), train_hist, color='blue', alpha=0.5)

        plt.bar(range(n_bins), test_hist, color='red', alpha=0.5)

        plt.show()

    return mse



feat_exclude = []

features_except = ['accuracy_group', 'installation_id', 'accuracy_group', 'session_title']

def get_different_dist_features(features_except = features_except):

    to_exclude = []

    ajusted_test = reduce_test.copy()

    for feature in ajusted_test.columns:

        if feature not in features_except:

            data = reduce_train[feature]

            train_mean = data.mean()

            data = ajusted_test[feature] 

            test_mean = data.mean()

            try:

                error = stract_hists(feature, adjust=True)

                ajust_factor = train_mean / test_mean

                if ajust_factor > 10 or ajust_factor < 0.1:# or error > 0.01:

                    to_exclude.append(feature)

                    #print(feature, train_mean, test_mean, error)

                else:

                    ajusted_test[feature] *= ajust_factor

            except:

                to_exclude.append(feature)

                #print(feature, train_mean, test_mean)

    return to_exclude



#feat_exclude = get_different_dist_features()

print(f"{len(feat_exclude)} features to exclude : \n{feat_exclude}")
all_features = [x for x in reduce_train.columns if x not in cols_to_drop + feat_remove + feat_exclude]
def eval_qwk_lgb_regr(y_true, y_pred):

    """

    Fast cappa eval function for lgb.

    """

    dist = Counter(reduce_train['accuracy_group'])

    for k in dist:

        dist[k] /= len(reduce_train)

    #reduce_train['accuracy_group'].hist()

    

    acum = 0

    bound = {}

    for i in range(3):

        acum += dist[i]

        bound[i] = np.percentile(y_pred, acum * 100)



    def classify(x):

        if x <= bound[0]:

            return 0

        elif x <= bound[1]:

            return 1

        elif x <= bound[2]:

            return 2

        else:

            return 3



    y_pred = np.array(list(map(classify, y_pred))).reshape(y_true.shape)



    return 'cappa', cohen_kappa_score(y_true, y_pred, weights='quadratic'), True
def make_model_data(df, is_train = True):

    _X = df[all_features]    

    _X = _X.reset_index(drop=True)

    _y = df['accuracy_group']

    _y = _y.reset_index(drop=True)

    if is_train:

        return _X, _y

    else:

        return _X, None
#make data for train

X, y = make_model_data(reduce_train, is_train = True)
# predictions dictionary : model_name : (score, prediction)

predictions = {}

models = []
class lgb_model():

    def __init__(self, X, y, categoricals, n_folds=2):

        self.X = X

        self.y = y

        self.categoricals = categoricals

        self.n_folds = n_folds

        self.models = []

        print(f'input shape={X.shape} label shape={y.shape}')

        

    def make_params(self, learning_rate, max_depth, lambda_l1, lambda_l2, 

                    bagging_fraction, bagging_freq, colsample_bytree, subsample_freq, feature_fraction):

        params = {'n_estimators':5000,

                    'boosting_type': 'gbdt',

                    'objective': 'regression',

                    'metric': 'rmse',

                    #'eval_metric': 'cappa',

                    'subsample' : 1.0,

                    'subsample_freq' : subsample_freq,

                    'feature_fraction' : feature_fraction,

                    'n_jobs': -1,

                    'seed': 42,                    

                    'learning_rate': learning_rate,

                    'max_depth': int(max_depth),

                    'lambda_l1': lambda_l1,

                    'lambda_l2': lambda_l2,

                    'bagging_fraction' : bagging_fraction,

                    'bagging_freq': int(bagging_freq),

                    'colsample_bytree': colsample_bytree,

                    'early_stopping_rounds': 100,

                    'verbose' : 0

                 }

        return params



    def opt_func(self, learning_rate, max_depth, lambda_l1, lambda_l2, 

                     bagging_fraction, bagging_freq, colsample_bytree, subsample_freq, feature_fraction):

        params = self.make_params(learning_rate, max_depth, lambda_l1, lambda_l2, 

                     bagging_fraction, bagging_freq, colsample_bytree, subsample_freq, feature_fraction)

        

        folds = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=345)

        oof = np.zeros(len(self.X))

        self.models = []

        for fold, (train_idx, test_idx) in enumerate(folds.split(self.X, self.y)):

            #print(f'running fold {fold}')

            X_train, y_train = self.X.iloc[train_idx], self.y.iloc[train_idx]

            X_valid, y_valid = self.X.iloc[test_idx], self.y.iloc[test_idx]

            

            model = lgb.LGBMRegressor()

            model.set_params(**params)

            eval_set = [(X_valid, y_valid)]

            eval_names = ['valid']

            model.fit(X=X_train, y=y_train, eval_set=eval_set, eval_names=eval_names, eval_metric=eval_qwk_lgb_regr,

                      verbose=0, categorical_feature='auto')            



            pr = model.predict(self.X.loc[test_idx]).reshape(len(test_idx))

            oof[test_idx] = pr[:]

            self.models.append(model)



        _, score, _ = eval_qwk_lgb_regr(y, oof)        

        return score

    

    

    def predict(self, X):

        if len(self.models) == 0:

            return        

        preds = []

        for m in self.models:

            pred = m.predict(X).reshape(X.shape[0])

            preds.append(pred)

        preds = np.array(preds)

        final_pred = np.mean(preds, axis=0)

        print("final_pred.shape = ", final_pred.shape)

            

        return final_pred

    

def make_opt_hyperparams_lgb():

    params = {'learning_rate' : (0.02, 0.3),

                   'max_depth': (13, 20),

                   'lambda_l1': (1, 12),

                   'lambda_l2': (1, 12),

            'bagging_fraction': (0.7, 1.0),

                'bagging_freq': (1, 10),

            'colsample_bytree': (0.6, 1.0),

              'subsample_freq': (1, 10),

            'feature_fraction': (0.6, 1.0)

             }

    

    n_folds = (2 if BOOTSTRAP_MODE else 4)

    modelWrapper = lgb_model(X, y, categoricals, n_folds = n_folds)

    lgbBO = BayesianOptimization(modelWrapper.opt_func, params, random_state=1030)



    with warnings.catch_warnings():

        warnings.filterwarnings('ignore')

        init_points = (2 if BOOTSTRAP_MODE else 16)

        n_iter = (2 if BOOTSTRAP_MODE else 16)

        lgbBO.maximize(init_points = init_points, n_iter = n_iter, acq='ucb', xi=0.0, alpha=1e-6)

        return lgbBO



def make_pred_lgb():

    # run bayesian optimization

    optimizer = make_opt_hyperparams_lgb()

    

    # get best hyperparameter from optimizer result

    params = optimizer.max['params'] # use best parameter from optimization result

    

    # make model with best hyperparameters and build model

    n_folds = (2 if BOOTSTRAP_MODE else 6)

    modelWrapper = lgb_model(X, y, categoricals, n_folds=n_folds)

    score = modelWrapper.opt_func(**params) # train with optimized hyperparameters

    

    # predict with reduce_test

    X_test, _ = make_model_data(reduce_test, is_train = False)

    pred = modelWrapper.predict(X_test)

    

    # for OptimizedRounder

    pred_train = modelWrapper.predict(X)



    print("lgb model score : {}".format(score))

    print("pred.shape : ", pred.shape)

    predictions["lgb"] = (score, pred)

    

    models.extend(modelWrapper.models)

    

    #print("test pred:", pred)

    

    return pred_train



#pred_train = make_pred_lgb()
models = []



optimized_params_list = [

    {'learning_rate' : 0.0657,

     'max_depth': 17,

     'lambda_l1': 8,

     'lambda_l2': 9,

     'bagging_fraction': 0.92,

     'bagging_freq': 9.2,

     'colsample_bytree': 0.96,

     'subsample_freq' : 5.6,

     'feature_fraction' : 0.686

     },

    {'learning_rate' : 0.05789,

     'max_depth': 16,

     'lambda_l1': 9.2,

     'lambda_l2': 9.768,

     'bagging_fraction': 0.7364,

     'bagging_freq': 1.02,

     'colsample_bytree': 0.7993,

     'subsample_freq' : 1.0,

     'feature_fraction' : 0.9

     },

    {'learning_rate' : 0.023,

     'max_depth': 19,

     'lambda_l1': 5.0,

     'lambda_l2': 11.0,

     'bagging_fraction': 0.9465,

     'bagging_freq': 1.4,

     'colsample_bytree': 0.94,

     'subsample_freq' : 7.0,

     'feature_fraction' : 0.657

     },    

    {'learning_rate' : 0.05399,

     'max_depth': 14,

     'lambda_l1': 7.812,

     'lambda_l2': 8.155,

     'bagging_fraction': 0.9331,

     'bagging_freq': 8.897,

     'colsample_bytree': 0.738,

     'subsample_freq' : 1.0,

     'feature_fraction' : 0.9

     },

    {'learning_rate' : 0.04307,

     'max_depth': 16,

     'lambda_l1': 9.917,

     'lambda_l2': 9.89,

     'bagging_fraction': 0.9523,

     'bagging_freq': 1.042,

     'colsample_bytree':0.7772,

     'subsample_freq' : 1.0,

     'feature_fraction' : 0.9

     },

    {'learning_rate' : 0.05789,

     'max_depth': 16,

     'lambda_l1': 9.2,

     'lambda_l2': 9.768,

     'bagging_fraction': 0.7364,

     'bagging_freq': 1.02,

     'colsample_bytree':0.7993,

     'subsample_freq' : 1.0,

     'feature_fraction' : 0.9

     },

    {'learning_rate' : 0.0269,

     'max_depth': 16,

     'lambda_l1': 9.78,

     'lambda_l2': 9.691,

     'bagging_fraction': 0.9205,

     'bagging_freq': 1.023,

     'colsample_bytree':0.7992,

     'subsample_freq' : 1.0,

     'feature_fraction' : 0.9

     },

    {'learning_rate' : 0.04897,

     'max_depth': 20,

     'lambda_l1': 12,

     'lambda_l2': 5.3,

     'bagging_fraction': 0.9131,

     'bagging_freq': 6.63,

     'colsample_bytree':0.6854,

     'subsample_freq' : 7.79,

     'feature_fraction' : 0.7599

     },

    {'learning_rate' : 0.028,

     'max_depth': 20,

     'lambda_l1': 11.66,

     'lambda_l2': 11,

     'bagging_fraction': 0.9723,

     'bagging_freq': 3.987,

     'colsample_bytree':0.726,

     'subsample_freq' : 6.748,

     'feature_fraction' : 0.61

     },

    {'learning_rate' : 0.031,

     'max_depth': 17,

     'lambda_l1': 5.5,

     'lambda_l2': 5.2,

     'bagging_fraction': 0.9413,

     'bagging_freq': 2.771,

     'colsample_bytree':0.6012,

     'subsample_freq' : 2.7,

     'feature_fraction' : 0.6168

     }

]



n_folds = 6

X_test, _ = make_model_data(reduce_test, is_train = False)

tmp_modelWrapper = None

for i, params in enumerate(optimized_params_list):

    name = f'lgb_model_{i}'

    print("running model : {}".format(name))

    modelWrapper = lgb_model(X, y, categoricals, n_folds=n_folds)

    tmp_modelWrapper = modelWrapper

    score = modelWrapper.opt_func(**params) # train with optimized hyperparameters

    print(f'score : {score}')

    

    # predict with reduce_test

    pred = modelWrapper.predict(X_test)

    models.extend(modelWrapper.models)

    

    predictions[name] = (score, pred)

def show_pred_dist_by_accuracy_group():

    X, y = make_model_data(reduce_train, is_train = True)

    py = tmp_modelWrapper.predict(X)

    X['accuracy_group_label'] = y

    X['accuracy_group_pred'] = py



    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(20,8))

    plt.rcParams.update({'font.size': 16})



    ax.set_xlim(-1, 4)

    ax.set_title("dist by accuray_group")

    for i in range(4):

        subdf = X[X['accuracy_group_label'] == i]

        print(i, " = ", subdf['accuracy_group_pred'].min(), " ~ ", subdf['accuracy_group_pred'].max())

        subdf['accuracy_group_pred'].plot.kde(ax=ax, label=str(i), legend=True)



show_pred_dist_by_accuracy_group()
# blend data



total_accuracy = 0.0

final_pred = None

for p in predictions:    

    total_accuracy += predictions[p][0]



print(f"mean accuracy : {total_accuracy / len(predictions)}")

    

final_pred = np.zeros(predictions[p][1].shape)

# print(predictions[p][1].shape)

# print(final_pred.shape)



for p in predictions:

    accuracy = predictions[p][0]

    #print(f'{p} current acc:{accuracy}     total acc sum:{total_accuracy}')

    final_pred += predictions[p][1]

    

final_pred = final_pred / len(predictions)



print("blended final_pred.shape : ", final_pred.shape)
def submit_with_ag_variance(final_pred):    

    dist = Counter(reduce_train['accuracy_group'])



    # change 'accuracy_group' counts to percentage

    for k in dist:

        dist[k] /= len(reduce_train)



    # get percentile from final_pred using train's accuracy_group variance.

    acum = 0

    bound = {}

    for i in range(3):

        acum += dist[i]

        bound[i] = np.percentile(final_pred, acum * 100)



    print("bounds : ", bound)

    def classify(x):

        if x <= bound[0]:

            return 0

        elif x <= bound[1]:

            return 1

        elif x <= bound[2]:

            return 2

        else:

            return 3



    # get final prediction

    final_pred = np.array(list(map(classify, final_pred)))

    plt.hist(final_pred)



    # make submit.

    sample_submission['accuracy_group'] = final_pred.astype(int)

    sample_submission.to_csv('submission.csv', index=False)

    sample_submission['accuracy_group'].value_counts(normalize=True)
submit_with_ag_variance(final_pred)