# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import xgboost as xgb

from xgboost import XGBClassifier, XGBRegressor

from xgboost import plot_importance

from catboost import CatBoostRegressor

from matplotlib import pyplot

import shap

from bayes_opt import BayesianOptimization

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from time import time

from tqdm import tqdm_notebook as tqdm

from collections import Counter

from scipy import stats

import lightgbm as lgb

from sklearn.metrics import cohen_kappa_score, mean_squared_error

from sklearn.model_selection import KFold, StratifiedKFold

import gc

import json

pd.set_option('display.max_columns', 1000)
def eval_qwk_lgb_regr(y_true, y_pred):

    """

    Fast cappa eval function for lgb.

    """

    dist = Counter(reduce_train['accuracy_group'])

    for k in dist:

        dist[k] /= len(reduce_train)

    reduce_train['accuracy_group'].hist()

    

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
def cohenkappa(ypred, y):

    y = y.get_label().astype("int")

    ypred = ypred.reshape((4, -1)).argmax(axis = 0)

    loss = cohenkappascore(y, y_pred, weights = 'quadratic')

    return "cappa", loss, True
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

    # hour

    train['hour'] = train['timestamp'].dt.hour

    test['hour'] = test['timestamp'].dt.hour

    

    

    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code
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
# this is the function that convert the raw data into processed features

def get_data(user_sample, test_set=False):

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

    

    # news features: time spent in each activity

    time_spent_each_act = {actv: 0 for actv in list_of_user_activities}

    time_spent_each_type = {'Clip_time':0, 'Activity_time':0, 'Assessment_time': 0, 'Game_time': 0}

    event_code_count = {eve: 0 for eve in list_of_event_code}

    event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}

    title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()}

    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}

    

    accuracy_groups = {'0':0, '1':0, '2':0, '3':0}

    all_assessments = []

    accumulated_accuracy_group = 0

    accumulated_accuracy=0

    accumulated_correct_attempts = 0 

    accumulated_uncorrect_attempts = 0 

    accumulated_actions = 0

    counter = 0

    

    nonstudy_time = []

    accumulated_nonstudy_time = 0

    time_first_activity = user_sample.iloc[0, 2]

    last_session_time_sec = user_sample.iloc[0, 2]

#     accumulated_duration = 0

    durations = []

    clip_durations = []

    durations_game = []

    durations_activity = []

    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}

    

    last_game_time_title = {'lgt_' + title: 0 for title in assess_titles}

    ac_game_time_title = {'agt_' + title: 0 for title in assess_titles}

    ac_true_attempts_title = {'ata_' + title: 0 for title in assess_titles}

    ac_false_attempts_title = {'afa_' + title: 0 for title in assess_titles}

    

    accumulated_hour_inday = {str(i)+'_hour':0 for i in range(24)}

    

    sessions_count = 0

    

    # itarates through each session of one instalation_id

    for i, session in user_sample.groupby('game_session', sort=False):

        # i = game_session_id

        # session is a DataFrame that contain only one game_session

        

        #update accumulated non study

        time_first_activity = session.iloc[0, 2]

        accumulated_nonstudy_time += (time_first_activity - last_session_time_sec).total_seconds()

        nonstudy_time.append((time_first_activity - last_session_time_sec).total_seconds())

        last_session_time_sec = session.iloc[-1, 2]

        

        # get some sessions information

        session_type = session['type'].iloc[0]

        session_title = session['title'].iloc[0]

        session_title_text = activities_labels[session_title]

        

        if session_type == 'Clip':

            clip_durations.append((clip_time[activities_labels[session_title]]))

        

        if session_type == 'Activity':

            Activity_mean_event_count = (Activity_mean_event_count + session['event_count'].iloc[-1])/2.0

        

        if session_type == 'Game':

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

        

        # get current session time in seconds

        if session_type != 'Assessment':

            time_spent = int(session['game_time'].iloc[-1] / 1000)

            time_spent_each_act[activities_labels[session_title]] += time_spent

            time_spent_each_type["{}_time".format(session_type)] += time_spent

            if session_type != 'Clip':

                list_hours = list(session['timestamp'].dt.hour.unique())

                for i in list_hours:

                    accumulated_hour_inday['{}_hour'.format(i)] += 1

        

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

            features.update(time_spent_each_act.copy())

            features.update(event_code_count.copy())

            features.update(time_spent_each_type.copy())

            features.update(last_accuracy_title.copy())

            features.update(event_id_count.copy())

            features.update(title_count.copy())

            features.update(title_event_code_count.copy())

            

            features.update(last_game_time_title.copy())

            features.update(ac_game_time_title.copy())

            features.update(ac_true_attempts_title.copy())

            features.update(ac_false_attempts_title.copy())

            

            features.update(accumulated_hour_inday.copy())

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

            

            #calculate spent time of each session

            time_spent = int(session['game_time'].iloc[-1] / 1000)

            time_spent_each_type["{}_time".format(session_type)] += time_spent

            # get installation_id for aggregated features

            features['installation_id'] = session['installation_id'].iloc[-1] #from Andrew

            # add title as feature, remembering that title represents the name of the game

            features['session_title'] = session['title'].iloc[0] 

            # add world as feature, world represents the type of educational goal

            features['world'] = session['world'].iloc[0]

            # the 4 lines below add the feature of the history of the trials of this player

            # this is based on the all time attempts so far, at the moment of this assessment

            features['accumulated_correct_attempts'] = accumulated_correct_attempts

            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts

            accumulated_correct_attempts += true_attempts 

            accumulated_uncorrect_attempts += false_attempts

            # ----------------------------------------------

            ac_true_attempts_title['ata_' + session_title_text] += true_attempts

            ac_false_attempts_title['afa_' + session_title_text] += false_attempts

            

            

            last_game_time_title['lgt_' + session_title_text] = session['game_time'].iloc[-1]

            ac_game_time_title['agt_' + session_title_text] += session['game_time'].iloc[-1]

            # ----------------------------------------------

            

            # accumulated non study time

            features["accumulated_notstudy_time"] = accumulated_nonstudy_time

            features['last_notstudy_time'] = nonstudy_time[-1]

            features['notstudy_time_mean'] = np.mean(nonstudy_time)

            features['notstudy_time_std'] = np.std(nonstudy_time)

            features['notstudy_time_max'] = np.max(nonstudy_time)

            features['notstudy_time_min'] = np.max(nonstudy_time)

            

            # accumulated duration time

            # the time spent in the app so far

            if durations == []:

                features['duration_mean'] = 0

                features['duration_std'] = 0

                features['last_duration'] = 0

                features['duration_max'] = 0

                features["accumulated_duration"] = 0

            else:

                features['duration_mean'] = np.mean(durations)

                features['duration_std'] = np.std(durations)

                features['last_duration'] = durations[-1]

                features['duration_max'] = np.max(durations)

                features["accumulated_duration"] = np.sum(durations)

            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

            Assessment_mean_event_count = (Assessment_mean_event_count + session['event_count'].iloc[-1])/2.0

            

            if clip_durations == []:

                features['Clip_duration_mean'] = 0

                features['Clip_duration_std'] = 0

            else:

                features['Clip_duration_mean'] = np.mean(clip_durations)

                features['Clip_duration_std'] = np.std(clip_durations)

            

            if durations_game == []:

                features['duration_game_mean'] = 0

                features['duration_game_std'] = 0

                features['game_last_duration'] = 0

                features['game_max_duration'] = 0

            else:

                features['duration_game_mean'] = np.mean(durations_game)

                features['duration_game_std'] = np.std(durations_game)

                features['game_last_duration'] = durations_game[-1]

                features['game_max_duration'] = np.max(durations_game)

                

            if durations_activity == []:

                features['duration_activity_mean'] = 0

                features['duration_activity_std'] = 0

                features['game_activity_duration'] = 0

                features['game_activity_max'] = 0

            else:

                features['duration_activity_mean'] = np.mean(durations_activity)

                features['duration_activity_std'] = np.std(durations_activity)

                features['game_activity_duration'] = durations_activity[-1]

                features['game_activity_max'] = np.max(durations_activity)

                

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

            accuracy_groups['{}'.format(features['accuracy_group'])] += 1

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

        

        if session_type == 'Game':

            durations_game.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

            

        if session_type == 'Activity':

            durations_activity.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

        

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

        # this piece counts how many actions was made in each event_code so far

        event_code_count = update_counters(event_code_count, "event_code")

        event_id_count = update_counters(event_id_count, "event_id")

        title_count = update_counters(title_count, 'title')

        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')

        

        # counts how many actions the player has done so far, used in the feature of the same name

        accumulated_actions += len(session)

        if last_activity != session_type:

            user_activities_count[session_type] += 1

            last_activitiy = session_type

    # if test_set=True, only the last assessment must be predicted, the previous are scraped

    if test_set:

        return all_assessments[-1]

    # in train_set, all assessments are kept

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

    categoricals = ['session_title']

    return reduce_train, reduce_test, categoricals
# executing crossing features

from sklearn.preprocessing import PolynomialFeatures

from itertools import combinations



def crossing_features(reduce_train, reduce_test, select_feature):

    # making 2 dimentional cross

#     poly_transformer = PolynomialFeatures(degree = 3)



#     poly_features_train = reduce_train[select_feature].copy()

#     poly_features_train = poly_transformer.fit_transform(poly_features_train)

#     poly_features_train = poly_features_train[:, len(select_feature)+1:]

#     poly_features_train = pd.DataFrame(poly_features_train, 

#                                  columns = poly_transformer.get_feature_names(select_feature)[len(select_feature)+1:])

#     poly_features_test = reduce_test[select_feature].copy()

#     poly_features_test = poly_transformer.fit_transform(poly_features_test)

#     poly_features_test = poly_features_test[:, len(select_feature)+1:]

#     poly_features_test = pd.DataFrame(poly_features_test,

#                                      columns = poly_transformer.get_feature_names(select_feature)[len(select_feature)+1:])

    poly_features_train = {}

    poly_features_test = {}

    for f1, f2 in combinations(select_feature, 2):

        x1 = reduce_train[select_feature].copy()

        x2 = reduce_train[select_feature].copy()

        x1_test = reduce_test[select_feature].copy()

        x2_test = reduce_test[select_feature].copy()

        # sum

        poly_features_train['{}_sum_{}'.format(f1, f2)] = (x1.values + x2.values).reshape(-1,)

        poly_features_test['{}_sum_{}'.format(f1, f2)] = (x1_test.values + x2_test.values).reshape(-1,)

        # subtraction

        poly_features_train['{}_sub_{}'.format(f1, f2)] = (x1.values - x2.values).reshape(-1,)

        poly_features_test['{}_sub_{}'.format(f1, f2)] = (x1_test.values - x2_test.values).reshape(-1,)

        # rate

        poly_features_train['{}_rate_{}'.format(f1, f2)] = (x1.values / x2.values).reshape(-1,)

        poly_features_test['{}_rate_{}'.format(f1, f2)] = (x1_test.values / x2_test.values).reshape(-1,)

        # multiplication

        poly_features_train['{}_mul_{}'.format(f1, f2)] = (x1.values * x2.values).reshape(-1,)

        poly_features_test['{}_mul_{}'.format(f1, f2)] = (x1_test.values * x2_test.values).reshape(-1,)

        # sum of square

        poly_features_train['{}_sumsq_{}'.format(f1, f2)] = (x1.values**2 + x2.values**2).reshape(-1,)

        poly_features_test['{}_sumsq_{}'.format(f1, f2)] = (x1_test.values**2 + x2_test.values**2).reshape(-1,)

        # sub of square

        poly_features_train['{}_subsq_{}'.format(f1, f2)] = (x1.values**2 - x2.values**2).reshape(-1,)

        poly_features_test['{}_subsq_{}'.format(f1, f2)] = (x1_test.values**2 - x2_test.values**2).reshape(-1,)

        # rate of square

        poly_features_train['{}_ratesq_{}'.format(f1, f2)] = (x1.values**2 / x2.values**2).reshape(-1,)

        poly_features_test['{}_ratesq_{}'.format(f1, f2)] = (x1_test.values**2 / x2_test.values**2).reshape(-1,)

    

    poly_features_train = pd.DataFrame(poly_features_train)

    poly_features_test = pd.DataFrame(poly_features_test)

    # merge

    reduce_train = reduce_train.merge(poly_features_train, how = 'left', left_index=True, right_index=True)

    reduce_test = reduce_test.merge(poly_features_test, how = 'left', left_index=True, right_index=True)

    return reduce_train, reduce_test
class Base_Model(object):

    

    def __init__(self, train_df, test_df, features, categoricals=[], n_splits=5, verbose=True,ps={}):

        self.train_df = train_df

        self.test_df = test_df

        self.features = features

        self.n_splits = n_splits

        self.categoricals = categoricals

        self.target = 'accuracy_group'

        self.cv = self.get_cv()

        self.verbose = verbose

        if ps == {}:

            self.params = self.get_params()

        else:

            self.params = self.set_params(ps)

        self.y_pred, self.score, self.model, self.oof_pred = self.fit()

        

    def train_model(self, train_set, val_set):

        raise NotImplementedError

        

    def get_cv(self):

        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        return cv.split(self.train_df, self.train_df[self.target])

    

    def get_params(self):

        raise NotImplementedError

        

    def convert_dataset(self, x_train, y_train, x_val, y_val):

        raise NotImplementedError

        

    def convert_x(self, x):

        return x

        

    def fit(self):

        oof_pred = np.zeros((len(reduce_train), ))

        y_pred = np.zeros((len(reduce_test), ))

        for fold, (train_idx, val_idx) in enumerate(self.cv):

            x_train, x_val = self.train_df[self.features].iloc[train_idx], self.train_df[self.features].iloc[val_idx]

            y_train, y_val = self.train_df[self.target][train_idx], self.train_df[self.target][val_idx]

            train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)

            model = self.train_model(train_set, val_set)

            conv_x_val = self.convert_x(x_val)

            oof_pred[val_idx] = model.predict(conv_x_val).reshape(oof_pred[val_idx].shape)

            x_test = self.convert_x(self.test_df[self.features])

            y_pred += model.predict(x_test).reshape(y_pred.shape) / self.n_splits

            print('Partial score of fold {} is: {}'.format(fold, eval_qwk_lgb_regr(y_val, oof_pred[val_idx])[1]))

        _, loss_score, _ = eval_qwk_lgb_regr(self.train_df[self.target], oof_pred)

        if self.verbose:

            print('Our oof cohen kappa score is: ', loss_score)

        return y_pred, loss_score, model, oof_pred
class Lgb_Model(Base_Model):

    

    def train_model(self, train_set, val_set):

        verbosity = 100 if self.verbose else 0

        return lgb.train(self.params, train_set, valid_sets=[train_set, val_set], verbose_eval=verbosity)

        

    def convert_dataset(self, x_train, y_train, x_val, y_val):

        train_set = lgb.Dataset(x_train, y_train, categorical_feature=self.categoricals)

        val_set = lgb.Dataset(x_val, y_val, categorical_feature=self.categoricals)

        return train_set, val_set

        

    def get_params(self):

        #'bagging_fraction': 0.9311148828727116,

#          'bagging_freq': 3.795689485219932,

#          'colsample_bytree': 0.5809972971939291,

#          'lambda_l1': 9.311879811058626,

#          'lambda_l2': 8.728037766263142,

#          'learning_rate': 0.06375263735223892,

#          'max_depth': 15.73127668358948}

#         params = {'n_estimators':5000,

#                     'boosting_type': 'gbdt',

#                     'objective': 'regression',

#                     'metric': 'rmse',

#                     'subsample': 0.75,

#                     'subsample_freq': 1,

#                     'learning_rate': 0.063753,

#                     'max_depth': 15,

#                     'feature_fraction': 0.9,

#                     'lambda_l1': 9.311879811058626,  

#                     'lambda_l2': 8.728037766263142,

#                     'colsample_bytree': 0.5809972971939291,

#                     'bagging_freq': 4,

#                     'bagging_fraction': 0.9311148828727116,

#                     'early_stopping_rounds': 100

#                     }

        params = {'n_estimators':5000,

                    'boosting_type': 'gbdt',

                    'objective': 'regression',

                    'metric': 'rmse',

                    'subsample': 0.75,

                    'subsample_freq': 1,

                    'learning_rate': 0.01,

                    'feature_fraction': 0.9,

                    'max_depth': 15,

                    'lambda_l1': 1,  

                    'lambda_l2': 1,

                    'early_stopping_rounds': 100

                    }

        return params

    def set_params(self,ps={}):

        params = self.get_params()

        if 'subsample_freq' in ps:

            params['subsample_freq']=int(ps['subsample_freq'])

            params['learning_rate']=ps['learning_rate']

            params['feature_fraction']=ps['feature_fraction']

            params['lambda_l1']=ps['lambda_l1']

            params['lambda_l2']=ps['lambda_l2']

            params['max_depth']=int(ps['max_depth'])

        

        return params
class Xgb_Model(Base_Model):

    

    def train_model(self, train_set, val_set):

        verbosity = 100 if self.verbose else 0

        return xgb.train(self.params, train_set, 

                         num_boost_round=5000, evals=[(train_set, 'train'), (val_set, 'val')], 

                         verbose_eval=verbosity, early_stopping_rounds=100)

        

    def convert_dataset(self, x_train, y_train, x_val, y_val):

        train_set = xgb.DMatrix(x_train, y_train)

        val_set = xgb.DMatrix(x_val, y_val)

        return train_set, val_set

    

    def convert_x(self, x):

        return xgb.DMatrix(x)

        

    def get_params(self):

#         {'bagging_fraction': 0.23283188403210275,

#  'bagging_freq': 4.687926220292777,

#  'colsample_bytree': 0.6709713243168485,

#  'lambda_l2': 13.498999372317197,

#  'learning_rate': 0.04512332788509399,

#  'max_depth': 7.701984127446185}

#         params = {'colsample_bytree': 0.6709713243168485,                 

#             'learning_rate': 0.04512332788509399,

#             'max_depth': 8,

#             'subsample': 1,

#             'objective':'reg:squarederror',

#             #'eval_metric':'rmse',

#             'min_child_weight':3,

#             'reg_lambda': 13.498999372317197,

#             'subsample_freq': 5,

#             'feature_fraction':0.23283188403210275,

#             'gamma':0.25,

#             'n_estimators':5000}

        params = {'colsample_bytree': 0.8,                 

            'learning_rate': 0.01,

            'max_depth': 10,

            'subsample': 1,

            'objective':'reg:squarederror',

            #'eval_metric':'rmse',

            'min_child_weight':3,

            'gamma':0.25,

            'n_estimators':5000}



        return params

    def set_params(self, ps={}):

        params = self.get_params()

        if 'subsample_freq' in ps:

            params['subsample_freq']=int(ps['subsample_freq'])

            params['learning_rate']=ps['learning_rate']

            params['colsample_bytree']=ps['colsample_bytree']

            params['reg_alpha']=ps['reg_alpha']

            params['reg_lambda']=ps['reg_lambda']

            params['max_depth']=int(ps['max_depth'])

        return params
class Catb_Model(Base_Model):

    

    def train_model(self, train_set, val_set):

        verbosity = 100 if self.verbose else 0

        clf = CatBoostRegressor(**self.params)

        clf.fit(train_set['X'], 

                train_set['y'], 

                eval_set=(val_set['X'], val_set['y']),

                verbose=verbosity, 

                cat_features=self.categoricals)

        return clf

        

    def convert_dataset(self, x_train, y_train, x_val, y_val):

        train_set = {'X': x_train, 'y': y_train}

        val_set = {'X': x_val, 'y': y_val}

        return train_set, val_set

        

    def get_params(self):

        params = {'loss_function': 'RMSE',

                   'task_type': "CPU",

                   'iterations': 5000,

                   'od_type': "Iter",

                    'depth': 10,

                  'colsample_bylevel': 0.5, 

                   'early_stopping_rounds': 300,

                    'l2_leaf_reg': 18,

                   'random_seed': 42,

                    'use_best_model': True

                    }

        return params
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder



class Nn_Model(Base_Model):

    

    def __init__(self, train_df, test_df, features, categoricals=[], n_splits=5, verbose=True):

        features = features.copy()

        if len(categoricals) > 0:

            for cat in categoricals:

                enc = OneHotEncoder()

                train_cats = enc.fit_transform(train_df[[cat]])

                test_cats = enc.transform(test_df[[cat]])

                cat_cols = ['{}_{}'.format(cat, str(col)) for col in enc.active_features_]

                features += cat_cols

                train_cats = pd.DataFrame(train_cats.toarray(), columns=cat_cols)

                test_cats = pd.DataFrame(test_cats.toarray(), columns=cat_cols)

                train_df = pd.concat([train_df, train_cats], axis=1)

                test_df = pd.concat([test_df, test_cats], axis=1)

        scalar = MinMaxScaler()

        train_df[features] = scalar.fit_transform(train_df[features])

        test_df[features] = scalar.transform(test_df[features])

        print(train_df[features].shape)

        super().__init__(train_df, test_df, features, categoricals, n_splits, verbose)

        

    def train_model(self, train_set, val_set):

        verbosity = 100 if self.verbose else 0

        model = tf.keras.models.Sequential([

            tf.keras.layers.Input(shape=(train_set['X'].shape[1],)),

            tf.keras.layers.Dense(200, activation='relu'),

            tf.keras.layers.LayerNormalization(),

            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(100, activation='relu'),

            tf.keras.layers.LayerNormalization(),

            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(50, activation='relu'),

            tf.keras.layers.LayerNormalization(),

            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(25, activation='relu'),

            tf.keras.layers.LayerNormalization(),

            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(1, activation='relu')

        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=4e-4), loss='mse')

        print(model.summary())

        save_best = tf.keras.callbacks.ModelCheckpoint('nn_model.w8', save_weights_only=True, save_best_only=True, verbose=1)

        early_stop = tf.keras.callbacks.EarlyStopping(patience=20)

        model.fit(train_set['X'], 

                train_set['y'], 

                validation_data=(val_set['X'], val_set['y']),

                epochs=100,

                 callbacks=[save_best, early_stop])

        model.load_weights('nn_model.w8')

        return model

        

    def convert_dataset(self, x_train, y_train, x_val, y_val):

        train_set = {'X': x_train, 'y': y_train}

        val_set = {'X': x_val, 'y': y_val}

        return train_set, val_set

        

    def get_params(self):

        return None
from random import choice



class Cnn_Model(Base_Model):

    

    def __init__(self, train_df, test_df, features, categoricals=[], n_splits=5, verbose=True):

        features = features.copy()

        if len(categoricals) > 0:

            for cat in categoricals:

                enc = OneHotEncoder()

                train_cats = enc.fit_transform(train_df[[cat]])

                test_cats = enc.transform(test_df[[cat]])

                cat_cols = ['{}_{}'.format(cat, str(col)) for col in enc.active_features_]

                features += cat_cols

                train_cats = pd.DataFrame(train_cats.toarray(), columns=cat_cols)

                test_cats = pd.DataFrame(test_cats.toarray(), columns=cat_cols)

                train_df = pd.concat([train_df, train_cats], axis=1)

                test_df = pd.concat([test_df, test_cats], axis=1)

        scalar = MinMaxScaler()

        train_df[features] = scalar.fit_transform(train_df[features])

        test_df[features] = scalar.transform(test_df[features])

        self.create_feat_2d(features)

        super().__init__(train_df, test_df, features, categoricals, n_splits, verbose)

        

    def create_feat_2d(self, features, n_feats_repeat=50):

        self.n_feats = len(features)

        self.n_feats_repeat = n_feats_repeat

        self.mask = np.zeros((self.n_feats_repeat, self.n_feats), dtype=np.int32)

        for i in range(self.n_feats_repeat):

            l = list(range(self.n_feats))

            for j in range(self.n_feats):

                c = l.pop(choice(range(len(l))))

                self.mask[i, j] = c

        self.mask = tf.convert_to_tensor(self.mask)

        print(self.mask.shape)

       

        

    

    def train_model(self, train_set, val_set):

        verbosity = 100 if self.verbose else 0



        inp = tf.keras.layers.Input(shape=(self.n_feats))

        x = tf.keras.layers.Lambda(lambda x: tf.gather(x, self.mask, axis=1))(inp)

        x = tf.keras.layers.Reshape((self.n_feats_repeat, self.n_feats, 1))(x)

        x = tf.keras.layers.Conv2D(18, (50, 50), strides=50, activation='relu')(x)

        x = tf.keras.layers.Flatten()(x)

        #x = tf.keras.layers.Dense(200, activation='relu')(x)

        #x = tf.keras.layers.LayerNormalization()(x)

        #x = tf.keras.layers.Dropout(0.3)(x)

        x = tf.keras.layers.Dense(100, activation='relu')(x)

        x = tf.keras.layers.LayerNormalization()(x)

        x = tf.keras.layers.Dropout(0.3)(x)

        x = tf.keras.layers.Dense(50, activation='relu')(x)

        x = tf.keras.layers.LayerNormalization()(x)

        x = tf.keras.layers.Dropout(0.3)(x)

        out = tf.keras.layers.Dense(1)(x)

        

        model = tf.keras.Model(inp, out)

    

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

        print(model.summary())

        save_best = tf.keras.callbacks.ModelCheckpoint('nn_model.w8', save_weights_only=True, save_best_only=True, verbose=1)

        early_stop = tf.keras.callbacks.EarlyStopping(patience=20)

        model.fit(train_set['X'], 

                train_set['y'], 

                validation_data=(val_set['X'], val_set['y']),

                epochs=100,

                 callbacks=[save_best, early_stop])

        model.load_weights('nn_model.w8')

        return model

        

    def convert_dataset(self, x_train, y_train, x_val, y_val):

        train_set = {'X': x_train, 'y': y_train}

        val_set = {'X': x_val, 'y': y_val}

        return train_set, val_set

        

    def get_params(self):

        return None
# read data

train, test, train_labels, specs, sample_submission = read_data()

# get usefull dict with maping encode

train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code = encode_title(train, test, train_labels)

# tranform function to get the train and test set

reduce_train, reduce_test, categoricals = get_train_and_test(train, test)
new_valid = []

for ins_id, user_sample in tqdm(test.groupby('installation_id', sort=False), total=test.installation_id.nunique(), desc='Installation_id', position=0):

    new_valid += get_data(user_sample)

valid_df = pd.DataFrame(new_valid)

del new_valid

valid_df.shape
reduce_train = pd.concat([reduce_train, valid_df])

reduce_train.shape
reduce_train = reduce_train.reset_index()
reduce_train = reduce_train.drop(['index'], axis=1)
reduce_train.shape
# #crossing features

# select_feature = ["accumulated_notstudy_time", 'notstudy_time_mean', 'accumulated_uncorrect_attempts', 'duration_mean',\

# 'duration_activity_mean', 'duration_game_mean', 'accumulated_accuracy', 'session_title', 'world', 'game_last_duration',\

#                  'game_activity_duration', 'last_duration', 'last_notstudy_time']

# reduce_train, reduce_test = crossing_features(reduce_train, reduce_test, select_feature)

# reduce_train.shape
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

    msre = mean_squared_error(train_hist, test_hist)

    if plot:

        print(msre)

        plt.bar(range(n_bins), train_hist, color='blue', alpha=0.5)

        plt.bar(range(n_bins), test_hist, color='red', alpha=0.5)

        plt.show()

    return msre

stract_hists('Magma Peak - Level 1_2000', adjust=False, plot=True)
# call feature engineering function

features = reduce_train.loc[(reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns # delete useless columns

features = [x for x in features if x not in ['accuracy_group', 'installation_id']]
# counter = 0

# to_remove = []

# for feat_a in features:

#     for feat_b in features:

#         if feat_a != feat_b and feat_a not in to_remove and feat_b not in to_remove:

#             c = np.corrcoef(reduce_train[feat_a], reduce_train[feat_b])[0][1]

#             if c > 0.995:

#                 counter += 1

#                 to_remove.append(feat_b)

#                 print('{}: FEAT_A: {} FEAT_B: {} - Correlation: {}'.format(counter, feat_a, feat_b, c))
to_exclude = [] 

ajusted_test = reduce_test.copy()

for feature in ajusted_test.columns:

    if feature not in ['accuracy_group', 'installation_id', 'accuracy_group', 'session_title']:

        data = reduce_train[feature]

        train_mean = data.mean()

        data = ajusted_test[feature] 

        test_mean = data.mean()

        try:

            error = stract_hists(feature, adjust=True)

            ajust_factor = train_mean / test_mean

            if ajust_factor > 10 or ajust_factor < 0.1:# or error > 0.01:

                to_exclude.append(feature)

                print(feature, train_mean, test_mean, error)

            else:

                ajusted_test[feature] *= ajust_factor

        except:

            to_exclude.append(feature)

            print(feature, train_mean, test_mean)
# features = [x for x in features if x not in (to_exclude + to_remove + ['index'])]

features = [x for x in features if x not in (to_exclude + ['index'])]

reduce_train[features].shape
new_cols = []

for i in features:

    if isinstance(i, np.int64) or isinstance(i, int):

        new_cols.append(i)

    else:

        new_cols.append(i.replace(', ', '-'))

change_colname_dict = {features[i]:new_cols[i] for i in range(len(features))}
reduce_train = reduce_train.rename(columns = change_colname_dict)

ajusted_test = ajusted_test.rename(columns = change_colname_dict)

features = new_cols.copy()
len(features)
del train, test, train_labels, specs

gc.collect()
def LGB_Beyes(subsample_freq,

                    learning_rate,

                    feature_fraction,

                    max_depth,

                    lambda_l1,

                    lambda_l2):

    params={}

    params['subsample_freq']=subsample_freq

    params['learning_rate']=learning_rate

    params['feature_fraction']=feature_fraction

    params['lambda_l1']=lambda_l1

    params['lambda_l2']=lambda_l2

    params['max_depth']=max_depth

    lgb_model = Lgb_Model(reduce_train, ajusted_test, features, categoricals=categoricals, verbose=False, ps=params)

    print('kappa: ',lgb_model.score)

    return lgb_model.score



bounds_LGB = {

    'subsample_freq': (1, 5),

    'learning_rate': (0.01, 0.1),

    'feature_fraction': (0.5, 1),

    'lambda_l1': (0, 10),

    'lambda_l2': (0, 10),

    'max_depth': (14, 16),

}



LGB_BO = BayesianOptimization(LGB_Beyes, bounds_LGB, random_state=1029)

import warnings

init_points = 6

n_iter = 10

with warnings.catch_warnings():

    warnings.filterwarnings('ignore')

    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
# def XGB_Beyes(subsample_freq,

#                     learning_rate,

#                     colsample_bytree,

#                     max_depth,

#                     reg_lambda,

#                     reg_alpha):

#     params={}

#     params['subsample_freq']=subsample_freq

#     params['learning_rate']=learning_rate

#     params['colsample_bytree']=colsample_bytree

#     params['reg_alpha']=reg_alpha

#     params['reg_lambda']=reg_lambda

#     params['max_depth']=max_depth

#     xgb_model = Xgb_Model(reduce_train, ajusted_test, features, categoricals=categoricals, verbose=False, ps=params)

#     print('kappa: ',xgb_model.score)

#     return xgb_model.score



# bounds_XGB = {

#     'subsample_freq': (1, 10),

#     'learning_rate': (0.01, 0.1),

#     'colsample_bytree': (0.6, 1),

#     'reg_alpha': (0, 5),

#     'reg_lambda': (0, 5),

#     'max_depth': (9, 11),

# }



# XGB_BO = BayesianOptimization(XGB_Beyes, bounds_XGB, random_state=1029)

# import warnings

# init_points = 5

# n_iter = 5

# with warnings.catch_warnings():

#     warnings.filterwarnings('ignore')

#     XGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
lgb_model_5 = Lgb_Model(reduce_train, ajusted_test, features, categoricals=categoricals, ps=LGB_BO.max['params'])

# lgb_model_5 = Lgb_Model(reduce_train, ajusted_test, features, categoricals=categoricals)

# xgb_model = Xgb_Model(reduce_train, ajusted_test, features, categoricals=categoricals, ps=XGB_BO.max['params'])

xgb_model = Xgb_Model(reduce_train, ajusted_test, features, categoricals=categoricals)
# cat_model = Catb_Model(reduce_train, ajusted_test, features, categoricals=categoricals)
#cnn_model = Cnn_Model(reduce_train, ajusted_test, features, categoricals=categoricals)

nn_model = Nn_Model(reduce_train, ajusted_test, new_cols, categoricals=categoricals)
import itertools

all_combinations1 = list(np.linspace(0.40,0.80,30))

all_combinations2 = list(np.linspace(0.0,0.30,30))

all_combinations3 = list(np.linspace(0.0,0.30,30))

# all_combinations4 = list(np.linspace(0.0,0.3,16))

# all_combinations5 = list(np.linspace(0.1,0.25,5))

# all_combinations6 = list(np.linspace(0.05,0.18,5))

print(all_combinations1)

l = [all_combinations1, all_combinations2, all_combinations3]

# all_combinations3

#      , all_combinations4, all_combinations5, all_combinations6]

all_l = list(itertools.product(*l))

# print(all_l)

filtered_combis = [l for l in all_l if l[0] + l[1] + l[2] > 0.98 and \

                   l[0] + l[1] + l[2] < 1.02]

# print(filtered_combis)

print(len(filtered_combis))
best_combi = [] # of the form (i, score)

for i, combi in enumerate(filtered_combis):

    w1 = combi[0]

    w2 = combi[1]

    w3 = combi[2]

#     w4 = combi[3]

#     w5 = combi[4]

#     w6 = combi[5]

    curr_score = 0

    

    pr = w1 * lgb_model_5.oof_pred + w2 * xgb_model.oof_pred + w3 * nn_model.oof_pred

    _, curr_score, _ = eval_qwk_lgb_regr(reduce_train['accuracy_group'], pr)

    

    if len(best_combi) > 0:

        prev_score = best_combi[0][1]

        if curr_score > prev_score:

            print('{}, {}, {}'.format(w1, w2, w3))

            print("score: {}".format(curr_score))

            best_combi[:] = []

            best_combi += [(i, curr_score)]

    else:

        print('{}, {}, {}'.format(w1, w2, w3))

        print("score: {}".format(curr_score))

        best_combi += [(i, curr_score)]

score = best_combi[0][1]

print(score)
final_combi = filtered_combis[best_combi[0][0]]

w1 = final_combi[0]

w2 = final_combi[1]

w3 = final_combi[2]

# w4 = final_combi[3]

# w5 = final_combi[4]

# w6 = final_combi[5]
weights = {'lgb5': w1, 'xgb': w2, 'nn': w3, 'cat': 0}



final_pred = (lgb_model_5.y_pred * weights['lgb5']) + (xgb_model.y_pred * weights['xgb']) + (nn_model.y_pred * weights['nn'])

#                                      + cat_model.y_pred * weights['cat'])

#final_pred = cnn_model.y_pred

print(final_pred.shape)
#pd.DataFrame([(round(a, 2), round(b, 2), round(c, 2), round(d, 2)) for a, b, c, d in zip(lgb_model.y_pred, cat_model.y_pred, xgb_model.y_pred, nn_model.y_pred)], columns=['lgb', 'cat', 'xgb', 'nn']).head(50)
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