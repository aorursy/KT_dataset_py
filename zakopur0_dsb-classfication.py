import numpy as np

import pandas as pd

import os

import copy

import matplotlib.pyplot as plt

import lightgbm as lgb

import time

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit, RepeatedStratifiedKFold

import datetime

from sklearn.metrics import accuracy_score,f1_score,roc_auc_score

from joblib import Parallel, delayed

from statistics import mean

from numba import jit

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

import random

import logging

from collections import Counter

from tqdm import tqdm_notebook as tqdm

import gc

pd.set_option('display.max_columns', 5000)

random.seed(127)

np.random.seed(127)
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
def read_data():

    print('Reading train.csv file....')

    train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))



    print('Reading test.csv file....')

    test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')

    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))



    print('Reading train_labels.csv file....')

    train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))



    print('Reading specs.csv file....')

    specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')

    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))



    print('Reading sample_submission.csv file....')

    sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')

    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))

    return train, test, train_labels, specs, sample_submission
train, test, train_labels, specs, sample_submission = read_data()
def encode_title(train, test, train_labels):

    # encode title

    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))

    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))

    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))

    # make a list with all the unique 'titles' from the train and test set

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

    train['title'] = train['title'].map(activities_map)

    test['title'] = test['title'].map(activities_map)

    train['world'] = train['world'].map(activities_world)

    test['world'] = test['world'].map(activities_world)

    train_labels['title'] = train_labels['title'].map(activities_map)

    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))

    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest

    win_code[activities_map['Bird Measurer (Assessment)']] = 4110

    # convert text into datetime

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

                        

    # if it't the test_set, only the last assessment must be predicted, the previous are scraped

    if test_set:

        return all_assessments[-1]

    # in the train_set, all assessments goes to the dataset

    return all_assessments



def get_train_and_test(train, test):

    compiled_train = []

    compiled_test = []

    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = 17000):

        compiled_train += get_data(user_sample)

    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):

        test_data = get_data(user_sample, test_set = True)

        compiled_test.append(test_data)

    reduce_train = pd.DataFrame(compiled_train)

    reduce_test = pd.DataFrame(compiled_test)

    return reduce_train, reduce_test
train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code = encode_title(train, test, train_labels)
reduce_train, reduce_test = get_train_and_test(train, test)
del train, test

gc.collect()
session_title1 = reduce_train['session_title'].value_counts().index[0]

session_title2 = reduce_train['session_title'].value_counts().index[1]

session_title3 = reduce_train['session_title'].value_counts().index[2]

session_title4 = reduce_train['session_title'].value_counts().index[3]

session_title5 = reduce_train['session_title'].value_counts().index[4]



reduce_train['session_title'] = reduce_train['session_title'].replace({session_title1:0,session_title2:1,session_title3:2,session_title4:3,session_title5:4})

reduce_test['session_title'] = reduce_test['session_title'].replace({session_title1:0,session_title2:1,session_title3:2,session_title4:3,session_title5:4})
for col in reduce_train.columns:

    if type(col) != str:

        reduce_train = reduce_train.rename(columns={col:str(col)})

        reduce_test = reduce_test.rename(columns={col:str(col)})



col_order = sorted(reduce_train.columns)

reduce_train = reduce_train.ix[:,col_order]

reduce_test = reduce_test.ix[:,col_order]
reduce_train.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in reduce_train.columns]

reduce_test.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in reduce_test.columns]
cols_to_drop = ['game_session', 'installation_id', 'timestamp', 'accuracy_group', 'timestampDate']

target_enc_cols = ['session_title','Game']

categoricals = ['session_title']
clparams   = {'n_estimators':2000,

            'boosting_type': 'gbdt',

            'objective': 'binary',

            'metric': 'auc',

            'subsample': 0.75,

            'subsample_freq': 1,

            'learning_rate': 0.04,

            'feature_fraction': 0.9,

            'max_depth': 15,

            'lambda_l1': 1,  

            'lambda_l2': 1,

            'verbose': 100,

            'early_stopping_rounds': 100, 

            'bagging_fraction_seed': 127,

            'feature_fraction_seed': 127,

            'data_random_seed': 127,

            'seed':127

            }
n_fold = 5

folds = GroupKFold(n_splits=n_fold)

X = reduce_train.copy()

cl_y = reduce_train['accuracy_group'].copy()

cl_y.loc[cl_y<=1]=0

cl_y.loc[cl_y>=2]=1

cols_to_drop = ['installation_id','accuracy_group']

cl_oof = np.zeros(len(reduce_train))

models = []

for fold_n, (train_index, valid_index) in enumerate(folds.split(X, cl_y, X['installation_id'])):

    print('Fold {} started at {}'.format(fold_n+1,time.ctime()))

    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

    y_train, y_valid = cl_y.iloc[train_index], cl_y.iloc[valid_index]

    

    X_train = X_train.drop(cols_to_drop,axis=1)

    X_valid = X_valid.drop(cols_to_drop,axis=1)

    

    trn_data = lgb.Dataset(X_train,label=y_train)

    val_data = lgb.Dataset(X_valid,label=y_valid)

    

    cl_lgb_model = lgb.train(clparams,

                        trn_data,

                        valid_sets=[trn_data,val_data],

                        verbose_eval=100,

                        categorical_feature = categoricals

                        )

    pred = cl_lgb_model.predict(X_valid)

    models.append(cl_lgb_model)

    cl_oof[valid_index] = pred

print('oof auc:',roc_auc_score(cl_y,cl_oof))
from functools import partial

import scipy as sp

class OptimizedRounder(object):

    """

    An optimizer for rounding thresholds

    to maximize Quadratic Weighted Kappa (QWK) score

    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved

    """

    def __init__(self):

        self.coef_ = 0



    def _kappa_loss(self, coef, X, y):

        """

        Get loss according to

        using current coefficients

        

        :param coef: A list of coefficients that will be used for rounding

        :param X: The raw predictions

        :param y: The ground truth labels

        """

        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])



        return -qwk(y, X_p)



    def fit(self, X, y,random_flg=False):

        """

        Optimize rounding thresholds

        

        :param X: The raw predictions

        :param y: The ground truth labels

        """

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        # [1.09830188 1.67317237 2.17390658]

        if random_flg:

            initial_coef = [np.random.uniform(0.5,0.6), np.random.uniform(0.6,0.7), np.random.uniform(0.8,0.9)]

        else:

            initial_coef = [0.5, 1.5, 2.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')



    def predict(self, X, coef):

        """

        Make predictions with specified thresholds

        

        :param X: The raw predictions

        :param coef: A list of coefficients that will be used for rounding

        """

        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])





    def coefficients(self):

        """

        Return the optimized coefficients

        """

        return self.coef_['x']
y = reduce_train['accuracy_group']
best_score = 0

for i in range(100):

    optR = OptimizedRounder()

    optR.fit(cl_oof, y,random_flg=True)

    coefficients = optR.coefficients()

    opt_preds1 = optR.predict(cl_oof, coefficients)

    score = qwk(y, opt_preds1)

    if score > best_score:

        best_score = score

        best_coefficients = coefficients

print(best_score)

print(best_coefficients)
oof = pd.cut(cl_oof, [-np.inf] + list(np.sort(best_coefficients)) + [np.inf], labels = [0, 1, 2, 3])

qwk(y,oof)
def cl_predict(test,models):

    all_ans = np.zeros((len(test)))

    cols_to_drop = ['installation_id','accuracy_group']

    test_copy = test.drop(cols_to_drop,axis=1)

    for model in models:

        ans = model.predict(test_copy)

        all_ans += ans

        

    return all_ans/n_fold
pred = cl_predict(reduce_test,models)
f_pred = pd.cut(pred, [-np.inf] + list(np.sort(best_coefficients)) + [np.inf], labels = [0, 1, 2, 3])
sample_submission['accuracy_group'] = f_pred.astype(int)

sample_submission['accuracy_group'].value_counts(normalize=True)
sample_submission.to_csv('submission.csv', index=False)