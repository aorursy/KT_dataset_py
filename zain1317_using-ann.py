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



import numpy as np

import pandas as pd

import os

import copy

import matplotlib.pyplot as plt

%matplotlib inline

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error

pd.options.display.precision = 15

from collections import defaultdict

import lightgbm as lgb

import xgboost as xgb

import catboost as cat

import time

from collections import Counter

import datetime

from catboost import CatBoostRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit, RepeatedStratifiedKFold

from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix

from sklearn import linear_model

import gc

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

from bayes_opt import BayesianOptimization

import eli5

import shap

from IPython.display import HTML

import json

import altair as alt

from category_encoders.ordinal import OrdinalEncoder

import networkx as nx

import matplotlib.pyplot as plt

%matplotlib inline

from typing import List



import os

import time

import datetime

import json

import gc

from numba import jit



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook



import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn import metrics

from typing import Any

from itertools import product

pd.set_option('max_rows', 500)

import re

from tqdm import tqdm

from joblib import Parallel, delayed
def add_datepart(df: pd.DataFrame, field_name: str,

                 prefix: str = None, drop: bool = True, time: bool = True, date: bool = True):

    """

    Helper function that adds columns relevant to a date in the column `field_name` of `df`.

    from fastai: https://github.com/fastai/fastai/blob/master/fastai/tabular/transform.py#L55

    """

    field = df[field_name]

    prefix = ifnone(prefix, re.sub('[Dd]ate$', '', field_name))

    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Is_month_end', 'Is_month_start']

    if date:

        attr.append('Date')

    if time:

        attr = attr + ['Hour', 'Minute']

    for n in attr:

        df[prefix + n] = getattr(field.dt, n.lower())

    if drop:

        df.drop(field_name, axis=1, inplace=True)

    return df





def ifnone(a: Any, b: Any) -> Any:

    """`a` if `a` is not None, otherwise `b`.

    from fastai: https://github.com/fastai/fastai/blob/master/fastai/core.py#L92"""

    return b if a is None else a
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





class LGBWrapper_regr(object):

    """

    A wrapper for lightgbm model so that we will have a single api for various models.

    """



    def __init__(self):

        self.model = lgb.LGBMRegressor()



    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):

        if params['objective'] == 'regression':

            eval_metric = eval_qwk_lgb_regr

        else:

            eval_metric = 'auc'



        eval_set = [(X_train, y_train)]

        eval_names = ['train']

        self.model = self.model.set_params(**params)



        if X_valid is not None:

            eval_set.append((X_valid, y_valid))

            eval_names.append('valid')



        if X_holdout is not None:

            eval_set.append((X_holdout, y_holdout))

            eval_names.append('holdout')



        if 'cat_cols' in params.keys():

            cat_cols = [col for col in params['cat_cols'] if col in X_train.columns]

            if len(cat_cols) > 0:

                categorical_columns = params['cat_cols']

            else:

                categorical_columns = 'auto'

        else:

            categorical_columns = 'auto'



        self.model.fit(X=X_train, y=y_train,

                       eval_set=eval_set, eval_names=eval_names, eval_metric=eval_metric,

                       verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'],

                       categorical_feature=categorical_columns)



        self.best_score_ = self.model.best_score_

        self.feature_importances_ = self.model.feature_importances_



    def predict(self, X_test):

        return self.model.predict(X_test, num_iteration=self.model.best_iteration_)



    

def eval_qwk_xgb(y_pred, y_true):

    """

    Fast cappa eval function for xgb.

    """

    # print('y_true', y_true)

    # print('y_pred', y_pred)

    y_true = y_true.get_label()

    y_pred = y_pred.argmax(axis=1)

    return 'cappa', -qwk(y_true, y_pred)





class LGBWrapper(object):

    """

    A wrapper for lightgbm model so that we will have a single api for various models.

    """



    def __init__(self):

        self.model = lgb.LGBMClassifier()



    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):



        eval_set = [(X_train, y_train)]

        eval_names = ['train']

        self.model = self.model.set_params(**params)



        if X_valid is not None:

            eval_set.append((X_valid, y_valid))

            eval_names.append('valid')



        if X_holdout is not None:

            eval_set.append((X_holdout, y_holdout))

            eval_names.append('holdout')



        if 'cat_cols' in params.keys():

            cat_cols = [col for col in params['cat_cols'] if col in X_train.columns]

            if len(cat_cols) > 0:

                categorical_columns = params['cat_cols']

            else:

                categorical_columns = 'auto'

        else:

            categorical_columns = 'auto'



        self.model.fit(X=X_train, y=y_train,

                       eval_set=eval_set, eval_names=eval_names, eval_metric=eval_qwk_lgb,

                       verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'],

                       categorical_feature=categorical_columns)



        self.best_score_ = self.model.best_score_

        self.feature_importances_ = self.model.feature_importances_



    def predict_proba(self, X_test):

        if self.model.objective == 'binary':

            return self.model.predict_proba(X_test, num_iteration=self.model.best_iteration_)[:, 1]

        else:

            return self.model.predict_proba(X_test, num_iteration=self.model.best_iteration_)





class CatWrapper(object):

    """

    A wrapper for catboost model so that we will have a single api for various models.

    """



    def __init__(self):

        self.model = cat.CatBoostClassifier()



    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):



        eval_set = [(X_train, y_train)]

        self.model = self.model.set_params(**{k: v for k, v in params.items() if k != 'cat_cols'})



        if X_valid is not None:

            eval_set.append((X_valid, y_valid))



        if X_holdout is not None:

            eval_set.append((X_holdout, y_holdout))



        if 'cat_cols' in params.keys():

            cat_cols = [col for col in params['cat_cols'] if col in X_train.columns]

            if len(cat_cols) > 0:

                categorical_columns = params['cat_cols']

            else:

                categorical_columns = None

        else:

            categorical_columns = None

        

        self.model.fit(X=X_train, y=y_train,

                       eval_set=eval_set,

                       verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'],

                       cat_features=categorical_columns)



        self.best_score_ = self.model.best_score_

        self.feature_importances_ = self.model.feature_importances_



    def predict_proba(self, X_test):

        if 'MultiClass' not in self.model.get_param('loss_function'):

            return self.model.predict_proba(X_test, ntree_end=self.model.best_iteration_)[:, 1]

        else:

            return self.model.predict_proba(X_test, ntree_end=self.model.best_iteration_)





class XGBWrapper(object):

    """

    A wrapper for xgboost model so that we will have a single api for various models.

    """



    def __init__(self):

        self.model = xgb.XGBClassifier()



    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):



        eval_set = [(X_train, y_train)]

        self.model = self.model.set_params(**params)



        if X_valid is not None:

            eval_set.append((X_valid, y_valid))



        if X_holdout is not None:

            eval_set.append((X_holdout, y_holdout))



        self.model.fit(X=X_train, y=y_train,

                       eval_set=eval_set, eval_metric=eval_qwk_xgb,

                       verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'])



        scores = self.model.evals_result()

        self.best_score_ = {k: {m: m_v[-1] for m, m_v in v.items()} for k, v in scores.items()}

        self.best_score_ = {k: {m: n if m != 'cappa' else -n for m, n in v.items()} for k, v in self.best_score_.items()}



        self.feature_importances_ = self.model.feature_importances_



    def predict_proba(self, X_test):

        if self.model.objective == 'binary':

            return self.model.predict_proba(X_test, ntree_limit=self.model.best_iteration)[:, 1]

        else:

            return self.model.predict_proba(X_test, ntree_limit=self.model.best_iteration)









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
class CategoricalTransformer(BaseEstimator, TransformerMixin):



    def __init__(self, cat_cols=None, drop_original: bool = False, encoder=OrdinalEncoder()):

        """

        Categorical transformer. This is a wrapper for categorical encoders.



        :param cat_cols:

        :param drop_original:

        :param encoder:

        """

        self.cat_cols = cat_cols

        self.drop_original = drop_original

        self.encoder = encoder

        self.default_encoder = OrdinalEncoder()



    def fit(self, X, y=None):



        if self.cat_cols is None:

            kinds = np.array([dt.kind for dt in X.dtypes])

            is_cat = kinds == 'O'

            self.cat_cols = list(X.columns[is_cat])

        self.encoder.set_params(cols=self.cat_cols)

        self.default_encoder.set_params(cols=self.cat_cols)



        self.encoder.fit(X[self.cat_cols], y)

        self.default_encoder.fit(X[self.cat_cols], y)



        return self



    def transform(self, X, y=None):

        data = copy.deepcopy(X)

        new_cat_names = [f'{col}_encoded' for col in self.cat_cols]

        encoded_data = self.encoder.transform(data[self.cat_cols])

        if encoded_data.shape[1] == len(self.cat_cols):

            data[new_cat_names] = encoded_data

        else:

            pass



        if self.drop_original:

            data = data.drop(self.cat_cols, axis=1)

        else:

            data[self.cat_cols] = self.default_encoder.transform(data[self.cat_cols])



        return data



    def fit_transform(self, X, y=None, **fit_params):

        data = copy.deepcopy(X)

        self.fit(data)

        return self.transform(data)
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

    categoricals = ['session_title']

    return reduce_train, reduce_test, categoricals



# read data

train, test, train_labels, specs, sample_submission = read_data()

# get usefull dict with maping encode

train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code = encode_title(train, test, train_labels)

# tranform function to get the train and test set

reduce_train, reduce_test, categoricals = get_train_and_test(train, test)
def preprocess(reduce_train, reduce_test):

    for df in [reduce_train, reduce_test]:

        df['installation_session_count'] = df.groupby(['installation_id'])['Clip'].transform('count')

        df['installation_duration_mean'] = df.groupby(['installation_id'])['duration_mean'].transform('mean')

        #df['installation_duration_std'] = df.groupby(['installation_id'])['duration_mean'].transform('std')

        df['installation_title_nunique'] = df.groupby(['installation_id'])['session_title'].transform('nunique')

        

        df['sum_event_code_count'] = df[[2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021, 

                                        4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 4050, 2010, 2020, 4070, 2025, 2030, 4080, 2035, 

                                        2040, 4090, 4220, 4095]].sum(axis = 1)

        

        df['installation_event_code_count_mean'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('mean')

        #df['installation_event_code_count_std'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('std')

        

    features = reduce_train.loc[(reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns # delete useless columns

    features = [x for x in features if x not in ['accuracy_group', 'installation_id']] + ['acc_' + title for title in assess_titles]

   

    return reduce_train, reduce_test, features

# call feature engineering function

reduce_train, reduce_test, features = preprocess(reduce_train, reduce_test)
complete_input = pd.concat((reduce_train, reduce_test))
# cols_to_drop = ['game_session', 'installation_id', 'timestamp', 'accuracy_group', 'timestampDate']
complete_input.drop(['installation_id', 'accuracy_group'], axis = 1, inplace = True)
complete_input = pd.get_dummies(complete_input)
complete_input = complete_input.astype(float)
X_train = complete_input[:reduce_train.shape[0]]

X_test = complete_input[reduce_train.shape[0]:]

y = reduce_train['accuracy_group']

del complete_input
# mt = MainTransformer()

# ft = FeatureTransformer()

# transformers = {'ft': ft}

# regressor_model1 = RegressorModel(model_wrapper=LGBWrapper_regr())

# regressor_model1.fit(X=reduce_train, y=y, folds=folds, params=params, preprocesser=mt, transformers=transformers,

#                     eval_metric='cappa', cols_to_drop=cols_to_drop)

from keras.layers import Dense, BatchNormalization,LeakyReLU

from keras.models import Sequential

from keras.optimizers import adam



num_cols = len(X_train.columns)

model = Sequential()



model.add(Dense(1000, input_shape = (num_cols,), activation = 'relu'))

model.add(Dense(1000, activation = 'relu')) #hidden layer 1

model.add(Dense(1000, activation = 'relu')) #hidden

model.add(Dense(1000, activation = 'relu'))

model.add(Dense(1000, activation = 'relu')) 

# model.add(BatcNormalization())

# model.add(LeakyReLU(alpha=0.1))

model.add(Dense(1000, activation = 'relu'))

# model.add(Dense(2000, activation = 'relu'))

# model.add(BatchNormalization())

model.add(Dense(1000, activation = 'relu')) #hidden layer 3

# model.add(Flatten(1000, activation = 'relu'))

model.add(Dense(1,)) #output layer

model.summary()
model.compile(optimizer = 'adam', loss = 'mse')
# history = model.fit(x,y, epochs = 10)

# history_dict = history.history



model.fit(X_train,y,epochs = 40, batch_size = 126, verbose = 1)
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



    def fit(self, X, y):

        """

        Optimize rounding thresholds

        

        :param X: The raw predictions

        :param y: The ground truth labels

        """

        loss_partial = partial(self._kappa_loss, X=X, y=y)

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
%%time

pr1 = model.predict(X_train)



optR = OptimizedRounder()

optR.fit(pr1.reshape(-1,), y)

coefficients = optR.coefficients()
opt_preds = optR.predict(pr1.reshape(-1, ), coefficients)

qwk(y, opt_preds)
# some coefficients calculated by me.

pr1 = model.predict(X_test)

pr1[pr1 <= 1.12232214] = 0

pr1[np.where(np.logical_and(pr1 > 1.12232214, pr1 <= 1.73925866))] = 1

pr1[np.where(np.logical_and(pr1 > 1.73925866, pr1 <= 2.22506454))] = 2

pr1[pr1 > 2.22506454] = 3
sample_submission['accuracy_group'] = pr1.astype(int)

sample_submission.to_csv('submission.csv', index=False)
sample_submission['accuracy_group'].value_counts(normalize=True)
sample_submission.head()