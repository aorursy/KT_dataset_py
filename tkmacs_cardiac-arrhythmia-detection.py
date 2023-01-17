# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



'''

for dirname, _, filenames in os.walk('/kaggle/input/1056lab-cardiac-arrhythmia-detection/normal'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import lightgbm as lgb

from sklearn.model_selection import cross_validate,StratifiedKFold,cross_val_score,train_test_split

from sklearn.metrics import make_scorer,roc_auc_score

import xgboost as xgb

import featuretools as ft
def met_auc(y_test,y_pred):

    return roc_auc_score(y_test,y_pred)
stratifiedkfold = StratifiedKFold(n_splits = 3)

score_func = {'auc':make_scorer(met_auc)}
from glob import glob

from scipy.io import loadmat

idx_ = []  # index

len_ = []  # length

mean_ = []  # mean

std_ = []  # standard deviation

ste_ = []  # standard error

max_ = []  # maximum value

min_ = []  # minimum value

med_ = []

y_ = []

for d in ['normal', 'af']:

    for path in sorted(glob('/kaggle/input/1056lab-cardiac-arrhythmia-detection/' + d +'/*.mat')):

        filename = path.split('/')[-1]  # e.g. B05821.mat

        i = filename.split('.')[0]  # e.g. B05821

        idx_.append(i)

        mat_contents = loadmat(path)

        x = mat_contents['val'][0]

        len_.append(len(x))

        mean_.append(x.mean())

        std_.append(x.std())

        ste_.append(x.std()/np.sqrt(len(x)))

        max_.append(x.max())

        min_.append(x.min())

        med_.append(np.median(x))

        if d == 'normal':

            y_.append(0)

        else:

            y_.append(1)
train_df = pd.DataFrame(columns=['id','length', 'mean', 'standard deviation', 'standard error', 'maximum value', 'minimum value', 'y'])

train_df['id']=idx_

train_df['length'] = len_

train_df['mean'] = mean_

train_df['standard deviation'] = std_

train_df['standard error'] = ste_

train_df['maximum value'] = max_

train_df['minimum value'] = min_

train_df['median'] = med_ 

train_df['y'] = y_
es = ft.EntitySet(id = 'example')

es = es.entity_from_dataframe(entity_id='locations',dataframe=train_df.drop(['y'],axis = 1),index = 'id')
feature_matrix, feature_defs = ft.dfs(entityset=es,

                                       target_entity='locations',

                                       trans_primitives=['add_numeric', 'subtract_numeric','divide_numeric','modulo_numeric','multiply_numeric'],

                                       agg_primitives=[],

                                       max_depth=1,

                                       )
X = feature_matrix.to_numpy()

Y = train_df['y'].to_numpy()

feature_matrix.shape

#feature_defs
model = lgb.LGBMClassifier()

scores = cross_validate(model, X, Y, cv = stratifiedkfold, scoring=score_func)

print('auc:', scores['test_auc'])

print('auc:', scores['test_auc'].mean())
idx_ = []  # index

len_ = []  # length

mean_ = []  # mean

std_ = []  # standard deviation

ste_ = []  # standard error

max_ = []  # maximum value

min_ = []  # minimum value

med_ = []  #median

for path in sorted(glob('/kaggle/input/1056lab-cardiac-arrhythmia-detection/test/*.mat')):

    filename = path.split('/')[-1]  # e.g. B05821.mat

    i = filename.split('.')[0]  # e.g. B05821

    idx_.append(i)

    mat_contents = loadmat(path)

    x = mat_contents['val'][0]

    len_.append(len(x))

    mean_.append(x.mean())

    std_.append(x.std())

    ste_.append(x.std()/np.sqrt(len(x)))

    max_.append(x.max())

    min_.append(x.min())

    med_.append(np.median(x))
test_df = pd.DataFrame( columns=['id','length', 'mean', 'standard deviation', 'standard error', 'maximum value', 'minimum value'])

test_df['id'] = idx_ 

test_df['length'] = len_

test_df['mean'] = mean_

test_df['standard deviation'] = std_

test_df['standard error'] = ste_

test_df['maximum value'] = max_

test_df['minimum value'] = min_

test_df['median'] = med_ 

test_df
es_test = ft.EntitySet(id = 'example')

es_test = es.entity_from_dataframe(entity_id='locations',dataframe=test_df,index = 'id')
feature_matrix_test, feature_defs_test = ft.dfs(entityset=es_test,

                                       target_entity='locations',

                                       trans_primitives=['add_numeric', 'subtract_numeric','divide_numeric','modulo_numeric','multiply_numeric'],

                                       agg_primitives=[],

                                       max_depth=1,

                                       )
X_test = feature_matrix_test.to_numpy()
import optuna

def objective(trial):

    param = {

        'objective': 'binary',

        'metric': 'binary_logloss',

        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),

        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),

        'num_leaves': trial.suggest_int('num_leaves', 2, 256),

        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),

        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),

        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),

        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),

    }

 

    model = lgb.LGBMClassifier()

    scores = cross_validate(model, X, Y, cv = stratifiedkfold, scoring=score_func)

    return scores['test_auc'].mean()
study = optuna.create_study(direction='maximize')

study.optimize(objective, n_trials=100)
model = lgb.LGBMClassifier()

model.fit(X,Y)
p = model.predict_proba(X_test)[:,1]
sample = pd.read_csv('../input/1056lab-cardiac-arrhythmia-detection/sampleSubmission.csv',index_col = 0)

sample['af'] = p
sample.to_csv('predict_lgbm_ft.csv',header = True)