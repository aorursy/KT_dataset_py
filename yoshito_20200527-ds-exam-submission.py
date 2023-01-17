# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.set_option('display.max_columns' , 100)
train_df = pd.read_csv('/kaggle/input/exam-for-students20200527/train.csv')

sta_info_df = pd.read_csv('/kaggle/input/exam-for-students20200527/station_info.csv')

data_dic_df = pd.read_csv('/kaggle/input/exam-for-students20200527/data_dictionary.csv')

test_df = pd.read_csv('/kaggle/input/exam-for-students20200527/test.csv')

city_info_df = pd.read_csv('/kaggle/input/exam-for-students20200527/city_info.csv')

sample_sub_df = pd.read_csv('/kaggle/input/exam-for-students20200527/sample_submission.csv')
from sklearn.model_selection import *
sample_sub_df.head()
test_df.head()
train_df.head()
# sub_df = sample_sub_df.copy()
# sub_df['TradePrice'] = train_df['TradePrice'].median()
# sub_df.to_csv('submission.csv', index=False)
_train_df = train_df.copy()

_test_df = test_df.copy()



cols = _train_df.columns

for x in cols:

    if _train_df[x].dtype == 'object':

        del _train_df[x]

        del _test_df[x]
_train_df.head()
del _train_df['id'], _test_df['id']
import lightgbm as lgb

import xgboost as xgb

import optuna

import sklearn
def objective(trial):

    param = {

        #'metric': 'binary_logloss',

        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),

        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),

        'num_leaves': trial.suggest_int('num_leaves', 2, 256),

        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),

        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),

        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),

        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),

    }

    

    # train val split

    target_col = 'TradePrice'

    df = _train_df

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(

        df.drop(columns=[target_col]), df[target_col], 

                                                        test_size=0.33, random_state=0)

    

    clf = lgb.LGBMRegressor(**param)

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    #acc = accuracy_score(y_test, preds) # accuracy

    return - np.sqrt( np.mean( (np.log(preds + 1) - np.log(y_test + 1)) ** 2 ) ) # rmsle


# study = optuna.create_study(direction='maximize')

# study.optimize(objective, n_trials=50)



# trial = study.best_trial
# print('rmsle: {}'.format(trial.value))

# print("Best parameters:")

# print(trial.params)
#clf = lgb.LGBMRegressor(**trial.params)

# clf = lgb.LGBMRegressor({'lambda_l1': 1.242223770483025e-08, 'lambda_l2': 6.592418511655523e-08, 'num_leaves': 246, 'feature_fraction': 0.9115515953910815, 'bagging_fraction': 0.9571605409093717, 'bagging_freq': 6, 'min_child_samples': 61})

# clf.fit(_train_df.drop(columns='TradePrice'), _train_df['TradePrice'])
# preds = clf.predict(_test_df)
# sub_df = sample_sub_df.copy()
# sub_df['TradePrice'] = preds
# sub_df.to_csv('submission.csv', index=False)
def add_feat0(df, orig_df):

    for col in df.columns:

        df[col] = df[col].fillna(df[col].median())

    return df
_train_df = add_feat0(_train_df, train_df)

_test_df = add_feat0(_test_df, test_df)
def add_feat1(df, orig_df):

    df['NearestStation'] = orig_df['NearestStation']

    df = pd.merge(df, sta_info_df, how='left', left_on='NearestStation', right_on='Station')

    del df['NearestStation'], df['Station']

    return df
_train_df = add_feat1(_train_df, train_df)

_test_df = add_feat1(_test_df, test_df)
_train_df['TradePrice'] = np.log1p(_train_df['TradePrice'])
def add_feat3(df, orig_df):

    df['Municipality'] = orig_df['Municipality']

    df = pd.merge(df, 

                  city_info_df.rename(columns={'Latitude' : 'city_Latitude', 'Longitude' : 'city_Longitude'}), 

                  how='left', left_on='Municipality', right_on='Municipality')

    del df['Municipality'], df['Prefecture']

    return df
_train_df = add_feat3(_train_df, train_df)

_test_df = add_feat3(_test_df, test_df)
def add_feat4(df, orig_df):

    return pd.concat([df, pd.get_dummies(orig_df['Region'], drop_first=True, prefix='Region')],

                     axis='columns')
_train_df = add_feat4(_train_df, train_df)

_test_df = add_feat4(_test_df, test_df)
def add_feat5(df, orig_df):

    df['TimeToNearestStation'] = orig_df['TimeToNearestStation'].replace({'30-60minutes' : 30, '1H-1H30' : 60, 

                                          '1H30-2H' : 90, '2H-' : 120})

    df['TimeToNearestStation'] = df['TimeToNearestStation'].fillna(df['TimeToNearestStation'].median()).astype(int)

    return df
_train_df = add_feat5(_train_df, train_df)

_test_df = add_feat5(_test_df, test_df)
def add_feat6(df, orig_df):

    dum = pd.get_dummies(orig_df['Type'], drop_first=True, prefix='Type')

    dum.columns = [x.split(',')[0].replace('(', '_').replace(')', '') for x in dum.columns]

    df = pd.concat([df, dum],

                     axis='columns')

    

    return df
_train_df = add_feat6(_train_df, train_df)

_test_df = add_feat6(_test_df, test_df)
def add_feat7(df, orig_df, data):

    col = 'FloorPlan_' + data

    df[col] = 0

    df.loc[orig_df['FloorPlan'].str.contains('[' + data + ']', na=False), col] = 1

    return df
_train_df = add_feat7(_train_df, train_df, data='23456')

_test_df = add_feat7(_test_df, test_df, data='23456')
def add_feat8(df, orig_df):

    for x in ['Northeast', 'East', 'Northwest', 'South', 'Southwest',

       'West', 'Southeast', 'North', 'No facing road']:

        col = 'Direction_' + x

        df[col] = 0

        df.loc[orig_df['Direction'] == x, col] = 1

    return df
_train_df = add_feat8(_train_df, train_df)

_test_df = add_feat8(_test_df, test_df)
def add_feat9_2(df, orig_df):

    df['Structure_num'] = orig_df['Structure'].str.split(',').apply(lambda x : len(x) if type(x) is list else 0)

    return df
_train_df = add_feat9_2(_train_df, train_df)

_test_df = add_feat9_2(_test_df, test_df)
def add_feat10(df, orig_df, _list):

    for x in _list:

        col = 'Classification_' + x

        df[col] = 0

        df.loc[orig_df['Classification'].str.contains(x, na=False), col] = 1

    return df
_train_df = add_feat10(_train_df, train_df, ['City Road', 'Prefectural Road', 'Private Road', 'Road',

       'Ward Road', 'Town Road', 'Village Road',

       'Tokyo Metropolitan Road', 'Access Road', 'National Highway',

       'Hokkaido Prefectural Road', 'Forest Road', 'Agricultural Road',

       'Kyoto/ Osaka Prefectural Road'])

_test_df = add_feat10(_test_df, test_df, ['City Road', 'Prefectural Road', 'Private Road', 'Road',

       'Ward Road', 'Town Road', 'Village Road',

       'Tokyo Metropolitan Road', 'Access Road', 'National Highway',

       'Hokkaido Prefectural Road', 'Forest Road', 'Agricultural Road',

       'Kyoto/ Osaka Prefectural Road'])
def add_feat11(df, orig_df, _list):

    for x in _list:

        col = 'Renovation_' + x

        df[col] = 0

        df.loc[orig_df['Renovation'].str.contains(x, na=False), col] = 1

    return df
_train_df = add_feat11(_train_df, train_df, ['Not yet','Done'])

_test_df = add_feat11(_test_df, test_df, ['Not yet','Done'])
def add_feat12(df, orig_df, _list):

    for x in _list:

        col = 'CityPlanning_' + x

        df[col] = 0

        df.loc[orig_df['CityPlanning'].str.contains(x, na=False), col] = 1

    return df
_train_df = add_feat12(_train_df, train_df, ['Category I Residential Zone', 'Outside City Planning Area',

       'Neighborhood Commercial Zone',

       'Category II Exclusively Medium-high Residential Zone',

       'Category I Exclusively Low-story Residential Zone',

       'Category I Exclusively Medium-high Residential Zone',

       'Urbanization Control Area', 'Category II Residential Zone',

       'Quasi-industrial Zone', 'Quasi-residential Zone',

       'Non-divided City Planning Area', 'Industrial Zone',

       'Commercial Zone', 'Exclusively Industrial Zone',

       'Category II Exclusively Low-story Residential Zone',

       'Quasi-city Planning Area'])



_test_df = add_feat12(_test_df, test_df, ['Category I Residential Zone', 'Outside City Planning Area',

       'Neighborhood Commercial Zone',

       'Category II Exclusively Medium-high Residential Zone',

       'Category I Exclusively Low-story Residential Zone',

       'Category I Exclusively Medium-high Residential Zone',

       'Urbanization Control Area', 'Category II Residential Zone',

       'Quasi-industrial Zone', 'Quasi-residential Zone',

       'Non-divided City Planning Area', 'Industrial Zone',

       'Commercial Zone', 'Exclusively Industrial Zone',

       'Category II Exclusively Low-story Residential Zone',

       'Quasi-city Planning Area'])
split_n = 5
clf = lgb.LGBMRegressor(**{'lambda_l1': 1.242223770483025e-08, 'lambda_l2': 6.592418511655523e-08, 'num_leaves': 246, 'feature_fraction': 0.9115515953910815, 'bagging_fraction': 0.9571605409093717, 'bagging_freq': 6, 'min_child_samples': 61})



skf = KFold(n_splits=split_n)

skf.get_n_splits(_train_df.drop(columns='TradePrice'), _train_df['TradePrice'])



cv_results = cross_validate(clf, _train_df.drop(columns='TradePrice'), _train_df['TradePrice'], 

                            cv=skf, return_train_score=True, return_estimator=True, n_jobs=-1,

                            ) # cross validation
_pred = np.zeros((len(_test_df), split_n))



for i, _clf in enumerate(cv_results['estimator']):

    pred = _clf.predict(_test_df)

    _pred[:, i] = pred

    

_pred = np.maximum(_pred, 0)

#

_pred = np.exp(_pred) - 1 # unnormalization

#

preds = _pred.mean(axis=1)
sub_df = sample_sub_df.copy()

sub_df['TradePrice'] = preds

sub_df.to_csv('submission.csv', index=False)
sub_df.head()