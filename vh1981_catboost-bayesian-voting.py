%matplotlib inline

import os

import copy

import random

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

    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))

    

    # make a list with all the unique 'titles' from the train and test set

    # title들을 모아 숫자값으로 변경

    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))

    

    # make a list with all the unique 'event_code' from the train and test set

    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))    

    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))

    

    # make a list with all the unique worlds from the train and test set

    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))

    

    # create a dictionary numerating the titles

    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))

    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))

    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))

    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))

    

    # replace the text titles with the number titles from the dict

    # 문자열을 위에서 생성한 dictionary에 해당하는 숫자로 바꾼다.

    train['title'] = train['title'].map(activities_map)

    test['title'] = test['title'].map(activities_map)

    train['world'] = train['world'].map(activities_world)

    test['world'] = test['world'].map(activities_world)

    train_labels['title'] = train_labels['title'].map(activities_map)

    

    # win_code 생성(Bird Measurer (Assessment)만 4110, 그 외에는 4100)

    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))    

    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest

    win_code[activities_map['Bird Measurer (Assessment)']] = 4110

    

    # convert text into datetime

    # 시간 문자열을 실제 시간 데이터 타입으로 변경

    train['timestamp'] = pd.to_datetime(train['timestamp'])

    test['timestamp'] = pd.to_datetime(test['timestamp'])    

    

    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code
def get_data(user_sample, test_set=False):

    '''

    The user_sample is a DataFrame from train or test where the only one 

    installation_id is filtered

    And the test_set parameter is related with the labels processing, that is only requered

    if test_set=False

    '''

    # Constants and parameters declaration

    last_activity = 0

    

    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

    

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

    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}

    event_code_count: Dict[str, int] = {ev: 0 for ev in list_of_event_code}

    event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}

    title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()} 

    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}

    

    # itarates through each session of one instalation_id

    for i, session in user_sample.groupby('game_session', sort=False):

        # i = game_session_id

        # session is a DataFrame that contain only one game_session

        

        # get some sessions information

        session_type = session['type'].iloc[0]

        session_title = session['title'].iloc[0]

        session_title_text = activities_labels[session_title]

                    

            

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

            else:

                features['duration_mean'] = np.mean(durations)

            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

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

        

        # this piece counts how many actions was made in each event_code so far

        def update_counters(counter: dict, col: str):

                num_of_session_count = Counter(session[col])

                for k in num_of_session_count.keys():

                    x = k

                    if col == 'title':

                        x = activities_labels[k]

                    counter[x] += num_of_session_count[k]

                return counter

            

        event_code_count = update_counters(event_code_count, "event_code")

        event_id_count = update_counters(event_id_count, "event_id")

        title_count = update_counters(title_count, 'title')

        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')



        # counts how many actions the player has done so far, used in the feature of the same name

        accumulated_actions += len(session)

        if last_activity != session_type:

            user_activities_count[session_type] += 1

            last_activitiy = session_type 

    

    # test_set인 경우 마지막 assessment만 예측하면 하므로 마지막 데이터만 리턴한다.

    if test_set:

        return all_assessments[-1]

    # in the train_set, all assessments goes to the dataset

    return all_assessments
def get_train_and_test(train, test):

    """

    event 데이터를 session에 대한 데이터로 생성해서 DataFrame으로 생성한다.

    """

    compiled_train = []

    compiled_test = []

    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = 17000):

        compiled_train += get_data(user_sample)

    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):

        test_data = get_data(user_sample, test_set = True)

        compiled_test.append(test_data)

    reduce_train = pd.DataFrame(compiled_train)

    reduce_test = pd.DataFrame(compiled_test)

    categoricals = ['session_title']

    return reduce_train, reduce_test, categoricals
def preprocess(reduce_train, reduce_test):

    """

    get_train_and_test()의 DataFrame에 통계 열을 추가함.

    """

    for df in [reduce_train, reduce_test]:

        df['installation_session_count'] = df.groupby(['installation_id'])['Clip'].transform('count') #installation_id 마다 이루어진 session의 수

        df['installation_duration_mean'] = df.groupby(['installation_id'])['duration_mean'].transform('mean')        

        df['installation_title_nunique'] = df.groupby(['installation_id'])['session_title'].transform('nunique')

        

        df['sum_event_code_count'] = df[[2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021, 

                                        4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 4050, 2010, 2020, 4070, 2025, 2030, 4080, 2035, 

                                        2040, 4090, 4220, 4095]].sum(axis = 1)

        

        df['installation_event_code_count_mean'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('mean') # or 'std'

        

    return reduce_train, reduce_test
# read data

train, test, train_labels, specs, sample_submission = read_data()

# get usefull dict with maping encode

train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code = encode_title(train, test, train_labels)



# tranform function to get the train and test set

reduce_train, reduce_test, categoricals = get_train_and_test(train, test)



# call feature engineering function

reduce_train, reduce_test = preprocess(reduce_train, reduce_test)



del train

del test
reduce_train.head()
reduce_test.head()
from sklearn.base import BaseEstimator, TransformerMixin

@jit

def qwk(a1, a2):

    """

    Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168



    :param a1:

    :param a2:

    :param max_rat:

    :return:

    """

    max_rat = 3

    a1 = np.asarray(a1, dtype=int)

    a2 = np.asarray(a2, dtype=int)



    hist1 = np.zeros((max_rat + 1, ))

    hist2 = np.zeros((max_rat + 1, ))



    o = 0

    for k in range(a1.shape[0]):

        i, j = a1[k], a2[k]

        hist1[i] += 1

        hist2[j] += 1

        o +=  (i - j) * (i - j)



    e = 0

    for i in range(max_rat + 1):

        for j in range(max_rat + 1):

            e += hist1[i] * hist2[j] * (i - j) * (i - j)



    e = e / a1.shape[0]



    return 1 - o / e





def eval_qwk_lgb(y_true, y_pred):

    """

    Fast cappa eval function for lgb.

    """



    y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)

    return 'cappa', qwk(y_true, y_pred), True





def eval_qwk_lgb_regr(y_true, y_pred):

    """

    Fast cappa eval function for lgb.

    """

    y_pred[y_pred <= 1.12232214] = 0

    y_pred[np.where(np.logical_and(y_pred > 1.12232214, y_pred <= 1.73925866))] = 1

    y_pred[np.where(np.logical_and(y_pred > 1.73925866, y_pred <= 2.22506454))] = 2

    y_pred[y_pred > 2.22506454] = 3



    # y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)



    return 'cappa', qwk(y_true, y_pred), True





    

def eval_qwk_xgb(y_pred, y_true):

    """

    Fast cappa eval function for xgb.

    """

    # print('y_true', y_true)

    # print('y_pred', y_pred)

    y_true = y_true.get_label()

    y_pred = y_pred.argmax(axis=1)

    return 'cappa', -qwk(y_true, y_pred)





class MainTransformer(BaseEstimator, TransformerMixin):



    def __init__(self, convert_cyclical: bool = False, create_interactions: bool = False, n_interactions: int = 20):

        """

        Main transformer for the data. Can be used for processing on the whole data.



        :param convert_cyclical: convert cyclical features into continuous

        :param create_interactions: create interactions between features

        """



        self.convert_cyclical = convert_cyclical

        self.create_interactions = create_interactions

        self.feats_for_interaction = None

        self.n_interactions = n_interactions



    def fit(self, X, y=None):



        if self.create_interactions:

            self.feats_for_interaction = [col for col in X.columns if 'sum' in col

                                          or 'mean' in col or 'max' in col or 'std' in col

                                          or 'attempt' in col]

            self.feats_for_interaction1 = np.random.choice(self.feats_for_interaction, self.n_interactions)

            self.feats_for_interaction2 = np.random.choice(self.feats_for_interaction, self.n_interactions)



        return self



    def transform(self, X, y=None):

        data = copy.deepcopy(X)

        if self.create_interactions:

            for col1 in self.feats_for_interaction1:

                for col2 in self.feats_for_interaction2:

                    data[f'{col1}_int_{col2}'] = data[col1] * data[col2]



        if self.convert_cyclical:

            data['timestampHour'] = np.sin(2 * np.pi * data['timestampHour'] / 23.0)

            data['timestampMonth'] = np.sin(2 * np.pi * data['timestampMonth'] / 23.0)

            data['timestampWeek'] = np.sin(2 * np.pi * data['timestampWeek'] / 23.0)

            data['timestampMinute'] = np.sin(2 * np.pi * data['timestampMinute'] / 23.0)



#         data['installation_session_count'] = data.groupby(['installation_id'])['Clip'].transform('count')

#         data['installation_duration_mean'] = data.groupby(['installation_id'])['duration_mean'].transform('mean')

#         data['installation_title_nunique'] = data.groupby(['installation_id'])['session_title'].transform('nunique')



#         data['sum_event_code_count'] = data[['2000', '3010', '3110', '4070', '4090', '4030', '4035', '4021', '4020', '4010', '2080', '2083', '2040', '2020', '2030', '3021', '3121', '2050', '3020', '3120', '2060', '2070', '4031', '4025', '5000', '5010', '2081', '2025', '4022', '2035', '4040', '4100', '2010', '4110', '4045', '4095', '4220', '2075', '4230', '4235', '4080', '4050']].sum(axis=1)



        # data['installation_event_code_count_mean'] = data.groupby(['installation_id'])['sum_event_code_count'].transform('mean')



        return data



    def fit_transform(self, X, y=None, **fit_params):

        data = copy.deepcopy(X)

        self.fit(data)

        return self.transform(data)





class FeatureTransformer(BaseEstimator, TransformerMixin):



    def __init__(self, main_cat_features: list = None, num_cols: list = None):

        """



        :param main_cat_features:

        :param num_cols:

        """

        self.main_cat_features = main_cat_features

        self.num_cols = num_cols



    def fit(self, X, y=None):



#         self.num_cols = [col for col in X.columns if 'sum' in col or 'mean' in col or 'max' in col or 'std' in col

#                          or 'attempt' in col]

        



        return self



    def transform(self, X, y=None):

        data = copy.deepcopy(X)

#         for col in self.num_cols:

#             data[f'{col}_to_mean'] = data[col] / data.groupby('installation_id')[col].transform('mean')

#             data[f'{col}_to_std'] = data[col] / data.groupby('installation_id')[col].transform('std')



        return data



    def fit_transform(self, X, y=None, **fit_params):

        data = copy.deepcopy(X)

        self.fit(data)

        return self.transform(data)
cols_to_drop = ['game_session', 'installation_id', 'timestamp', 'accuracy_group', 'timestampDate']

all_features = [x for x in reduce_train.columns if x not in cols_to_drop]

cat_features = ['session_title']
def make_model_data(df, is_train = True):

    _X = df[all_features]    

    _X.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in _X.columns]

    

    if is_train:

        return _X, df['accuracy_group']

    else:

        return _X, None
# FIXME: need to refactoring. (merge with above code block...)

_all_features = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in all_features]
#make data for train

X, y = make_model_data(reduce_train, is_train = True)
def func_catboost(bagging_temperature, depth, learning_rate, border_count, verbose=0, bo_use = True, NFOLDS = 2):    

    # split train/test    

    folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=345)

    oof = np.zeros(len(X))

    models = []

    

    for fold, (train_idx, test_idx) in enumerate(folds.split(X, y)):

        if verbose:

            print(f"training on fold {fold + 1}")

            

        params = {'loss_function' : 'MultiClass',

                  'eval_metric' : "WKappa",

                  'task_type' : "CPU",

                  'verbose' : 1,

                  'bagging_temperature' : bagging_temperature,

                  'learning_rate' : learning_rate,

                  'depth' : int(depth),

                  'border_count' : int(border_count)}        

        

        clf = CatBoostClassifier(**params)

        

        args = (X.loc[train_idx, _all_features], y.loc[train_idx])

        kwargs = {

            'verbose': 0,

            'eval_set': (X.loc[test_idx, _all_features], y.loc[test_idx]),                

        }

        kw = kwargs.copy()

        kw.update({

            'use_best_model': True,

            'cat_features': cat_features,

        })

        clf.fit(*args, **kw)

        

        if verbose:

            print(f"training on fold {fold + 1} finished... eval model")

        

        pr = clf.predict(X.loc[test_idx, _all_features]).reshape(len(test_idx))

        oof[test_idx] = pr[:]

        models.append(clf)



    if bo_use:

        return qwk(y, oof)

    else:

        return qwk(y, oof), models
# make parameters with bayesian optimization

def bo_catboost(X, y):

    params = {'bagging_temperature': (0, 20),

              'depth': (5, 10) ,

              "learning_rate" : (0.001, 0.1) , 

              'border_count': (1, 20)}

    

    catBO = BayesianOptimization(func_catboost,

                                 params,

                                 random_state=0)

    

    with warnings.catch_warnings():

        warnings.filterwarnings('ignore')

        catBO.maximize(init_points = 10, n_iter = 16)

    return catBO
# optimize parameters

optimizer = bo_catboost(X, y)
# make models

params = optimizer.max['params'] # use best parameter from optimization result

score, models = func_catboost(**params, verbose=1, bo_use=False, NFOLDS=6)



# save models to file

for i, m in enumerate(models):

    fname = f'pretrained_models_{i}'

    m.save_model(fname)
X, _ = make_model_data(reduce_test, is_train = False)
from collections import Counter 



# make test input data

X, _ = make_model_data(reduce_test, is_train = False)



# make prediction with test data

predictions = []

for model in models:

    p = np.array(model.predict(X))    

    if len(p.shape) == 2:

        p = p.reshape([p.shape[0],])

    predictions.append(p)

    

predictions = np.array(predictions)

p2 = np.transpose(predictions)



# make final prediction with voting.

final_pred = []

for i in range(p2.shape[0]):    

    cur = p2[i]

    voted_result = Counter(cur).most_common(1)[0][0]

    final_pred.append(voted_result)

print("final result shape : ", np.array(final_pred).shape)

preds = np.array(final_pred)

gc.collect()
sample_submission['accuracy_group'] = preds.astype(int)

sample_submission['accuracy_group'].value_counts(normalize=True)
sample_submission.to_csv('submission.csv', index=False)