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
def seed_everything(seed=127):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

seed_everything()
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

    train['type_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['type'], train['event_code']))

    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))

    test['type_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['type'], test['event_code']))

    all_title_event_code =sorted(list(set(train["title_event_code"].unique()).union(set(test["title_event_code"].unique()))))

    type_event_code = sorted(list(set(train["type_event_code"].unique()).union(set(test["type_event_code"].unique()))))

    # make a list with all the unique 'titles' from the train and test set

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

    Clip_titles = sorted(list(set(train[train['type'] == 'Clip']['title'].value_counts().index).union(set(test[test['type'] == 'Clip']['title'].value_counts().index))))

    Activity_titles = sorted(list(set(train[train['type'] == 'Activity']['title'].value_counts().index).union(set(test[test['type'] == 'Clip']['title'].value_counts().index))))

    Game_titles = sorted(list(set(train[train['type'] == 'Game']['title'].value_counts().index).union(set(test[test['type'] == 'Clip']['title'].value_counts().index))))

    noClip_titles = assess_titles+Game_titles+Activity_titles



    

    



    split_list_type_event_code = lambda data,select_type: list(map(lambda x, y: str(x) + '_' + str(y), data[data.title.isin(select_type)]['title'], data[data.title.isin(select_type)]['event_code']))

    train_Clip_title_event_code = split_list_type_event_code(train, Clip_titles)

    train_Activity_title_event_code = split_list_type_event_code(train, Activity_titles)

    train_Assessment_title_event_code = split_list_type_event_code(train, assess_titles)

    train_Game_title_event_code = split_list_type_event_code(train, Game_titles)

    

    test_Clip_title_event_code = split_list_type_event_code(test, Clip_titles)

    test_Activity_title_event_code = split_list_type_event_code(test, Activity_titles)

    test_Assessment_title_event_code = split_list_type_event_code(test, assess_titles)

    test_Game_title_event_code = split_list_type_event_code(test, Game_titles)

    

    Clip_title_event_code = sorted(list(set(train_Clip_title_event_code).union(set(test_Clip_title_event_code))))

    Activity_title_event_code = sorted(list(set(train_Activity_title_event_code).union(set(test_Assessment_title_event_code))))

    Assessment_title_event_code = sorted(list(set(train_Assessment_title_event_code).union(set(test_Assessment_title_event_code))))

    Game_title_event_code = sorted(list(set(train_Game_title_event_code).union(set(test_Game_title_event_code))))

    noClip_title_event_code= Activity_title_event_code+Assessment_title_event_code+Game_title_event_code

    

    # replace the text titles with the number titles from the dict

    train['title'] = train['title'].map(activities_map)

    test['title'] = test['title'].map(activities_map)

    train['world'] = train['world'].map(activities_world)

    test['world'] = test['world'].map(activities_world)

    train_labels['title'] = train_labels['title'].map(activities_map)

    

    title_to_type = list([train[train['title']==i].iloc[0]['type'] for i in range(len(list_of_user_activities))])

    title_type_map = dict(zip(list_of_user_activities, title_to_type))

    

    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))

    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest

    win_code[activities_map['Bird Measurer (Assessment)']] = 4110

    # convert text into datetime

    train['timestamp'] = pd.to_datetime(train['timestamp'])

    test['timestamp'] = pd.to_datetime(test['timestamp'])

    

    

    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code, activities_map, title_type_map, title_to_type, noClip_title_event_code, Activity_title_event_code, Assessment_title_event_code, Game_title_event_code, Clip_titles, type_event_code







train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code, activities_map, title_type_map, title_to_type, noClip_title_event_code, Activity_title_event_code, Assessment_title_event_code, Game_title_event_code, Clip_titles, type_event_code= encode_title(train, test, train_labels)
train = train[train.installation_id.isin(train_labels.installation_id.unique())]
Clip_title_list=[]

Activity_title_list=[]

Assessment_title_list=[]

Game_title_list=[]

Clip_title_name_list=[]

Activity_title_name_list=[]

Assessment_title_name_list=[]

Game_title_name_list=[]







for index, (title,event_type) in enumerate(title_type_map.items()):

    if event_type=='Activity':

        Activity_title_list.append(index)

        Activity_title_name_list.append(title)

    elif event_type=='Assessment':

        Assessment_title_list.append(index)

        Assessment_title_name_list.append(title)

    elif event_type=='Game':

        Game_title_list.append(index)

        Game_title_name_list.append(title)

    else:

        Clip_title_list.append(index)

        Clip_title_name_list.append(title)

print(f'{Activity_title_list},\n{Assessment_title_list},\n{Game_title_list},\n{Clip_title_list}')

Assessment_session_title_list = [activities_map[name] for name in ['Bird Measurer (Assessment)','Cart Balancer (Assessment)','Cauldron Filler (Assessment)','Chest Sorter (Assessment)','Mushroom Sorter (Assessment)']]
positive_word='Right|Great|Good|Nice|Amazing|WOW|Thumb|Cool|right|great|good|nice|amazing|Wow|thumb|cool|You did'

negative_word='Try|try|close|check|again|too'
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

    world_count = {f'world_count_{x}':0 for x in range(4)} 

    Clip_title_count = {f'Clip_count_{x}':0 for x in Clip_titles}

    Assessment_title_count = {f'Assessment_count_{x}':0 for x in assess_titles}

    N=44

    # new features: time spent in each activity

    last_session_time_sec = 0

    accuracy_groups = {0:0, 1:0, 2:0, 3:0}

    accuracy_groups_title={f'accuracy_groups_{title}':{f'0_{title}':0, f'1_{title}':0,f'2_{title}':0,f'3_{title}':0} for title in assess_titles}

    all_assessments = []

    accumulated_accuracy_group = 0

    accumulated_accuracy_group_title=[0]*N

    accumulated_accuracy = 0

    accumulated_correct_attempts = 0 

    accumulated_uncorrect_attempts = 0

    accumulated_actions = 0

    counter = 0

    time_first_activity = float(user_sample['timestamp'].values[0])

    durations = []

    

    positive_attemps = 0

    negative_attemps = 0

    accumulated_positive_attemps=0

    accumulated_negative_attemps=0

    positive_attemps_mean = 0

    negative_attemps_mean = 0

    accumulated_word_accuracy = 0

    

    

    durations_list = [[] for i in range(3)]

    duration_assess_title = {'duration_' + title: [] for title in assess_titles}

    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}

    accumulated_correct_attempts_title = {f'accumulated_correct_attempts_{title}': -1 for title in assess_titles}

    accumulated_uncorrect_attempts_title = {f'accumulated_uncorrect_attempts_{title}': -1 for title in assess_titles}

    

    

    event_code_count: Dict[str, int] = {ev: 0 for ev in list_of_event_code}

    event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}

    title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()} 

#     title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}

    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in noClip_title_event_code}

    

    accumulated_accuracy_list=[0]*N

    accumulated_correct_attempts_list = [0]*N

    accumulated_uncorrect_attempts_list = [0]*N

    true_attempts_list = [0]*N

    all_attempts_list = [0]*N

    false_attempts_list = [0]*N

    counter_list = [0]*N

    accuracy_list = [0]*N

    

    word_accumulated_accuracy_list=[0]*N

    word_accumulated_postive_attempts_list = [0]*N

    word_accumulated_negative_attempts_list = [0]*N

    word_positive_attempts_list = [0]*N

    word_negative_attempts_list = [0]*N

    word_accumulated_error_ratio_list = [0]*N

    word_error_ratio_list = [0]*N

    

    

    need_acc_list =[]

    need_acc_list.extend(Activity_title_list)

    need_acc_list.extend(Game_title_list)

#     need_acc_list.extend(Assessment_session_title_list)

    type_event_code_count = {x: 0 for x in type_event_code}

#     type_event_code_count = {f'{x}_{y}':0 for x,y in type_event_code}

#     print(type_event_code_count)

    type_event_code_count_save_0=dict.fromkeys(type_event_code_count,0)

    def count_type_ec_from_title_ec(type_eve_count, title_eve_count):

        type_eve_count=type_event_code_count_save_0.copy()

#         print(type_eve_count)

        for ((t,eve),v) in [((k.split('_')[0],k.split('_')[1]),v) for k,v in title_eve_count.items()]:

            type_eve_count[f'{title_type_map[t]}_{eve}']+=v

        return type_eve_count

        

    # itarates through each session of one instalation_id

    for i, session in user_sample.groupby('game_session', sort=False):

        # i = game_session_id

        # session is a DataFrame that contain only one game_session

        

        # get some sessions information

        session_type = session['type'].iloc[0]

        session_title = session['title'].iloc[0]

        session_title_text = activities_labels[session_title]

        world_type = session['world'].iloc[0]

        world_one_hot = {f'world_onehot_{x}':0 for x in range(4)} 

        world_one_hot[f'world_onehot_{world_type}']=1

        world_count[f'world_count_{world_type}']+=1

        

        

        if (session_type != 'Assessment') & (test_set or len(session)>1):

            if session_type=='Activity' or session_type=='Game':

                

                all_attempts_list[session_title] = session.query(f'event_code == {win_code[session_title]}')

                # then, check the numbers of wins and the number of losses

                true_attempts_title = all_attempts_list[session_title]['event_data'].str.contains('true').sum()

                false_attempts_title = all_attempts_list[session_title]['event_data'].str.contains('false').sum()

                true_attempts_list[session_title]+=true_attempts_title

                false_attempts_list[session_title]+=false_attempts_title

                counter_list[session_title]+=1

                accumulated_correct_attempts_list[session_title]+=true_attempts_title

                accumulated_uncorrect_attempts_list[session_title]+=false_attempts_title

                

                positive = session['event_data'].str.contains(positive_word).sum()

                negative = session['event_data'].str.contains(negative_word).sum()

                positive_attemps+= positive

                accumulated_positive_attemps+=positive

                negative_attemps+= negative

                accumulated_negative_attemps+=negative

                word_accumulated_postive_attempts_list[session_title]+=positive

                word_accumulated_negative_attempts_list[session_title] +=negative 

                word_positive_attempts_list[session_title] += positive

                word_negative_attempts_list[session_title] += negative

                

                if session_type=='Activity':

                    durations_list[0].append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

                elif session_type=='Game':

                    durations_list[1].append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

            else:

                durations_list[2].append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)



            

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

#             features.update(event_code_count.copy())

#             features.update(event_id_count.copy())

            features.update(title_count.copy())

            features.update(title_event_code_count.copy())

#             features.update(type_event_code_count.copy())

            features.update(last_accuracy_title.copy())

            features.update(Clip_title_count.copy())

            features.update(world_count.copy())

            features.update(world_one_hot.copy())

            if user_activities_count['Assessment']==0:

                features['first_play']=1

            else:

                features['first_play']=0

            if Assessment_title_count[f'Assessment_count_{session_title_text}']==0:

                for t in assess_titles:

                    features[f'{t}_first_play']=0

                features[f'{session_title_text}_first_play']=1

            else:

                for t in assess_titles:

                    features[f'{t}_first_play']=0

            features.update(Assessment_title_count.copy())

            Assessment_title_count[f'Assessment_count_{session_title_text}']+=1

            features['accumulated_positive_attemps']=accumulated_positive_attemps

            features['accumulated_negative_attemps']=accumulated_negative_attemps

            features['positive_attemps']=positive_attemps

            features['negative_attemps']=negative_attemps

            features['accumulated_negative_erorr_ratio'] = accumulated_negative_attemps/(accumulated_positive_attemps+accumulated_negative_attemps) if (accumulated_positive_attemps+accumulated_negative_attemps)>0 else 0

            features['negative_erorr_ratio'] = negative_attemps/(positive_attemps+negative_attemps) if (positive_attemps+negative_attemps)>0 else 0

            # clear for add the recently attemps

            positive_attemps=0

            negative_attemps=0

            

            # get installation_id for aggregated features

            features['installation_id'] = session['installation_id'].iloc[-1]

            # add title as feature, remembering that title represents the name of the game

            features['session_title'] = session['title'].iloc[0]

            # the 4 lines below add the feature of the history of the trials of this player

            # this is based on the all time attempts so far, at the moment of this assessment

            features['accumulated_correct_attempts'] = accumulated_correct_attempts

            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts

            features.update(accumulated_correct_attempts_title.copy())

            features.update(accumulated_uncorrect_attempts_title.copy())

            

            accumulated_correct_attempts += true_attempts 

            accumulated_uncorrect_attempts += false_attempts

            accumulated_correct_attempts_title[f'accumulated_correct_attempts_{session_title_text}']+=true_attempts

            accumulated_uncorrect_attempts_title[f'accumulated_uncorrect_attempts_{session_title_text}']+=false_attempts

#             if durations_list == []:

#                 features['duration_mean'] = 0

#             else:

#                 features['duration_mean'] = np.mean(durations)

            

#             [sum(t) if t!=[] 0 for t in duration_list[0]]

            features['Activity_duration_total'],features['Game_duration_total'],features['Clip_duration_total'] = [sum(l) if l!=[] else 0 for l in durations_list]

            features['Activity_duration_mean'],features['Game_duration_mean'],features['Clip_duration_mean'] = [features[f'{i}_duration_total']/user_activities_count[i] if user_activities_count[i]>0 else 0 for i in ['Activity','Game','Clip']]

            # the time spent in the app so far

            # the time spent in the app so far

            if durations == []:

                features['duration_mean'] = -1

                features['duration_sum'] = -1

                features['duration_std'] = -1

            else:

                features['duration_mean'] = np.mean(durations)

                features['duration_sum'] = sum(durations)

                features['duration_std'] = np.std(durations)

#             for t in assess_titles:

#                 if duration_assess_title[f'duration_{t}']==[]:

#                     features[f'duration_{t}_mean'] = -1

#                     features[f'duration_{t}_sum'] = -1

#                 else:

#                     features[f'duration_{t}_mean'] = np.mean(duration_assess_title[f'duration_{t}'])

#                     features[f'duration_{t}_sum'] = sum(duration_assess_title[f'duration_{t}'])

#                     features[f'duration_{t}_std'] = np.std(duration_assess_title[f'duration_{t}'])

                

            duration_assess_title[f'duration_{session_title_text}'].append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)    

            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

            

            



            

            # the accurace is the all time wins divided by the all time attempts

            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0

#             for t in assess_titles:

#                 t_label = activities_map[t]

            features[f'accumulated_accuracy_{session_title_text}'] = accumulated_accuracy_list[session_title]/counter_list[session_title] if counter_list[session_title] > 0 else 0

#             features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else -1

            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0

            accumulated_accuracy += accuracy

            accumulated_accuracy_list[session_title]+=accuracy

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

#             for t in assess_titles:

            features.update(accuracy_groups_title[f'accuracy_groups_{session_title_text}'])

            accuracy_groups[features['accuracy_group']] += 1

#             accuracy_groups_title[f'accuracy_groups_{session_title_text}'][features['accuracy_group']]+=1

            accuracy_groups_title[f'accuracy_groups_{session_title_text}'][str(features['accuracy_group'])+'_'+session_title_text]+=1

            # mean of the all accuracy groups of this player

            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0

#             for t in assess_titles:

#                 features[f'accumulated_accuracy_group_{t}'] = accumulated_accuracy_group_title[activities_map[t]]/counter_list[activities_map[t]] if counter_list[activities_map[t]] > 0 else 0

            features[f'accumulated_accuracy_group_{session_title_text}'] = accumulated_accuracy_group_title[session_title]/counter_list[session_title] if counter_list[session_title] > 0 else 0

            accumulated_accuracy_group += features['accuracy_group']

            accumulated_accuracy_group_title[session_title]+=features['accuracy_group']

            # how many actions the player has done so far, it is initialized as 0 and updated some lines below

            features['accumulated_actions'] = accumulated_actions

            

            # there are some conditions to allow this features to be inserted in the datasets

            # if it's a test set, all sessions belong to the final dataset

            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')

            # that means, must exist an event_code 4100 or 4110



            for i in need_acc_list:

#                 features[f'accumulated_correct_attempts_{activities_labels[i]}']=accumulated_correct_attempts_list[i]

#                 features[f'accumulated_uncorrect_attempts_{activities_labels[i]}'] = accumulated_uncorrect_attempts_list[i]

# #                 accumulated_correct_attempts_list[i] += true_attempts_list[i] 

# #                 accumulated_uncorrect_attempts_list[i] += false_attempts_list[i]

#                 accumulated_all_attempts=accumulated_correct_attempts_list[i]+accumulated_uncorrect_attempts_list[i]

#                 features[f'accumulated_accuracy_{activities_labels[i]}'] = accumulated_correct_attempts_list[i]/accumulated_all_attempts if accumulated_all_attempts > 0 else 0

#                 accuracy_list[i] = true_attempts_list[i]/(true_attempts_list[i]+false_attempts_list[i]) if (true_attempts_list[i]+false_attempts_list[i]) != 0 else 0

#                 features[f'accuracy_{activities_labels[i]}']=accuracy_list[i]

#                 true_attempts_list[session_title]=0

#                 false_attempts_list[session_title]=0

                

                

                features[f'word_accumulated_postive_attempts_{activities_labels[i]}'] = word_accumulated_postive_attempts_list[i]

                features[f'word_accumulated_negative_attempts_{activities_labels[i]}'] =word_accumulated_negative_attempts_list[i]

                word_accumulated_all_attempts=word_accumulated_postive_attempts_list[i]+word_accumulated_negative_attempts_list[i]

                word_accumulated_error_ratio_list[i]=word_accumulated_negative_attempts_list[i]/word_accumulated_all_attempts if word_accumulated_all_attempts > 0 else 0

                features[f'word_accumulated_error_ratio_{activities_labels[i]}'] = word_accumulated_error_ratio_list[i]

                word_error_ratio_list[i] = word_negative_attempts_list[i]/(word_positive_attempts_list[i]+word_negative_attempts_list[i]) if (word_positive_attempts_list[i]+word_negative_attempts_list[i]) != 0 else 0

                features[f'word_error_ratio_{activities_labels[i]}']=word_error_ratio_list[i]

                

                word_positive_attempts_list[i] = 0

                word_negative_attempts_list[i] = 0

                

#                 features[f'accumulated_accuracy_{activities_labels[i]}'] = accumulated_accuracy_list[i]/counter_list[i] if counter_list[i] > 0 else -1

#                 accuracy_list[i] = true_attempts_list[i]/(true_attempts_list[i]+false_attempts_list[i]) if (true_attempts_list[i]+false_attempts_list[i]) != 0 else -1

#                 accumulated_accuracy_list[i] += accuracy_list[i]





            all_attempts_list[session_title]+= true_attempts+false_attempts

            # then, check the numbers of wins and the number of losses

            true_attempts_list[session_title]+= true_attempts

            false_attempts_list[session_title]+= false_attempts

            accumulated_correct_attempts_list[session_title]+=true_attempts

            accumulated_uncorrect_attempts_list[session_title]+=false_attempts

            

            

            if test_set:

                all_assessments.append(features)

            elif true_attempts+false_attempts > 0:

                all_assessments.append(features)

                

            counter += 1

            counter_list[session_title]+=1

        

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

        if session_type!='Clip':

            title_event_code_count = update_counters(title_event_code_count, 'title_event_code')

            type_event_code_count = count_type_ec_from_title_ec(type_event_code_count, title_event_code_count)

        else:

            Clip_title_count[f'Clip_count_{session_title_text}']+=1



#         if session_type == 'Game' or session_type == 'Activity':





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

    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = 3614):

        compiled_train += get_data(user_sample)

    reduce_train = pd.DataFrame(compiled_train)

    del train

    gc.collect()

    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):

        test_data = get_data(user_sample, test_set = True)

        compiled_test.append(test_data)

    reduce_test = pd.DataFrame(compiled_test)

    del test

    categoricals = ['session_title']

    gc.collect()

    return reduce_train, reduce_test, categoricals





# tranform function to get the train and test set

reduce_train, reduce_test, categoricals = get_train_and_test(train, test)
# del train, test

# gc.collect()
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
# clparams   = {'n_estimators':2000,

#             'boosting_type': 'gbdt',

#             'objective': 'binary',

#             'metric': 'auc',

#             'subsample': 0.75,

#             'subsample_freq': 1,

#             'learning_rate': 0.04,

#             'feature_fraction': 0.9,

#             'max_depth': 15,

#             'lambda_l1': 1,  

#             'lambda_l2': 1,

#             'is_unbalanced':True,

#             'verbose': 100,

#             'early_stopping_rounds': 100, 

#             'bagging_fraction_seed': 127,

#             'feature_fraction_seed': 127,

#             'data_random_seed': 127,

#             'seed':127

#             }
# n_fold = 5

# folds = GroupKFold(n_splits=n_fold)

# X = reduce_train.copy()

# cl_y = reduce_train['accuracy_group'].copy()

# cl_y.loc[cl_y>0]=1

# cols_to_drop = ['installation_id','accuracy_group']

# cl_oof = np.zeros(len(reduce_train))

# models = []

# for fold_n, (train_index, valid_index) in enumerate(folds.split(X, cl_y, X['installation_id'])):

#     print('Fold {} started at {}'.format(fold_n+1,time.ctime()))

#     X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

#     y_train, y_valid = cl_y.iloc[train_index], cl_y.iloc[valid_index]

    

#     X_train = X_train.drop(cols_to_drop,axis=1)

#     X_valid = X_valid.drop(cols_to_drop,axis=1)

    

#     trn_data = lgb.Dataset(X_train,label=y_train)

#     val_data = lgb.Dataset(X_valid,label=y_valid)

    

#     cl_lgb_model = lgb.train(clparams,

#                         trn_data,

#                         valid_sets=[trn_data,val_data],

#                         verbose_eval=100,

#                         categorical_feature = categoricals

#                         )

#     pred = cl_lgb_model.predict(X_valid)

#     models.append(cl_lgb_model)

#     cl_oof[valid_index] = pred

# print('oof auc:',roc_auc_score(cl_y,cl_oof))
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit, RepeatedStratifiedKFold

import time

from sklearn.metrics import accuracy_score,f1_score,roc_auc_score

n_fold = 5

def cls_lgb(reduce_train,clsparams):



    folds = GroupKFold(n_splits=n_fold)

    X = reduce_train.copy()

    cl_y = reduce_train['accuracy_group'].copy()

    cl_y.loc[cl_y>0]=1

    # cl_y.loc[cl_y<=1]=0

    # cl_y.loc[cl_y>=2]=1



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



        cl_lgb_model = lgb.train(clsparams,

                            trn_data,

                            valid_sets=[trn_data,val_data],

                            verbose_eval=100,

                            categorical_feature = categoricals

                            )

        pred = cl_lgb_model.predict(X_valid)

        models.append(cl_lgb_model)

        cl_oof[valid_index] = pred

    auc = roc_auc_score(cl_y,cl_oof)

    print('oof auc:',auc)

    return models,auc,cl_oof
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit, RepeatedStratifiedKFold

import time

from sklearn.metrics import accuracy_score,f1_score,roc_auc_score

n_fold = 5

def cls_lgb_2(reduce_train,clsparams):



    folds = GroupKFold(n_splits=n_fold)

    X = reduce_train.copy()

    cl_y = reduce_train['accuracy_group'].copy()

#     

    cl_y.loc[cl_y<3]=0

    cl_y.loc[cl_y==3]=1

    # cl_y.loc[cl_y<=1]=0

    # cl_y.loc[cl_y>=2]=1



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



        cl_lgb_model = lgb.train(clsparams,

                            trn_data,

                            valid_sets=[trn_data,val_data],

                            verbose_eval=100,

                            categorical_feature = categoricals

                            )

        pred = cl_lgb_model.predict(X_valid)

        models.append(cl_lgb_model)

        cl_oof[valid_index] = pred

    auc = roc_auc_score(cl_y,cl_oof)

    print('oof auc:',auc)

    return models,auc,cl_oof
params={'n_estimators': 2000,

         'boosting_type': 'gbdt',

         'objective': 'binary',

         'metric': 'auc',

         'subsample': 0.75,

         'subsample_freq': 1,

         'learning_rate': 0.02651958317868789,

         'feature_fraction': 0.9054145662538011,

         'max_depth': 7,

         'lambda_l1': 6.390102565986075,

         'lambda_l2': 0.4826006183378617,

         'bagging_fraction': 0.8931783428063896,

         'bagging_freq': 1,

         'is_unbalance':True,

         'colsample_bytree': 0.7062357653698856,

         'verbose': 100,

         'early_stopping_rounds': 100,

         'bagging_fraction_seed': 127,

         'feature_fraction_seed': 127,

         'data_random_seed': 127,

         'seed': 127}
models,auc,cl_oof = cls_lgb(reduce_train,params)
models_2,auc_2,cl_oof_2 = cls_lgb_2(reduce_train,params)
max(cl_oof_2)
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
best_score = 0

for i in range(100):

    optR = OptimizedRounder()

    optR.fit(cl_oof_2, y,random_flg=True)

    coefficients = optR.coefficients()

    opt_preds1 = optR.predict(cl_oof_2, coefficients)

    score = qwk(y, opt_preds1)

    if score > best_score:

        best_score = score

        best_coefficients_2 = coefficients

print(best_score)

print(best_coefficients_2)
oof = pd.cut(cl_oof_2, [-np.inf] + list(np.sort(best_coefficients_2)) + [np.inf], labels = [0, 1, 2, 3])

qwk(y,oof)
cl_oof_merge = 0.6*cl_oof+0.4*cl_oof_2

cl_oof_merge
best_score = 0

for i in range(100):

    optR = OptimizedRounder()

    optR.fit(cl_oof_merge, y,random_flg=True)

    coefficients = optR.coefficients()

    opt_preds1 = optR.predict(cl_oof_merge, coefficients)

    score = qwk(y, opt_preds1)

    if score > best_score:

        best_score = score

        best_coefficients_merge = coefficients

print(best_score)

print(best_coefficients_merge)
oof = pd.cut(cl_oof_merge, [-np.inf] + list(np.sort(best_coefficients_merge)) + [np.inf], labels = [0, 1, 2, 3])

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

pred
pred_2 = cl_predict(reduce_test,models_2)
pred_2
final_pred = 0.6*pred+0.4*pred_2
# f_pred = pd.cut(pred, [-np.inf] + list(np.sort(best_coefficients_merge)) + [np.inf], labels = [0, 1, 2, 3])
# sample_submission['accuracy_group'] = f_pred.astype(int)

# sample_submission['accuracy_group'].value_counts(normalize=True)
# sample_submission.to_csv('submission.csv', index=False)
dist = Counter(reduce_train['accuracy_group'])

for k in dist:

    dist[k] /= len(reduce_train)

reduce_train['accuracy_group'].hist()



acum = 0

bound = {}

for i in range(3):

    acum += dist[i]

    bound[i] = np.percentile(final_pred, acum * 100)

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

    

final_pred = np.array(list(map(classify, final_pred)))



sample_submission['accuracy_group'] = final_pred.astype(int)

sample_submission.to_csv('submission.csv', index=False)

sample_submission['accuracy_group'].value_counts(normalize=True)