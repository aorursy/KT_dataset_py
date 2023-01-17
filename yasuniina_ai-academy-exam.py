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
import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error

from sklearn.model_selection import KFold

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm

from sklearn.preprocessing import StandardScaler



import itertools

from sklearn.decomposition import PCA

import lightgbm as lgb

import eli5

import math

from sklearn.cluster import MiniBatchKMeans

from lightgbm import LGBMRegressor
train = pd.read_csv('/kaggle/input/exam-for-students20200923/train.csv', index_col=0)

test  = pd.read_csv('/kaggle/input/exam-for-students20200923/test.csv', index_col=0)

country_info = pd.read_csv('/kaggle/input/exam-for-students20200923/country_info.csv')
display(train.shape)

display(test.shape)

display(country_info.shape)
train.head()
df_train = train.copy()

df_test  = test.copy()



y_train = df_train['ConvertedSalary']

y_train = y_train.apply(np.log1p)

X_train = df_train.drop(['ConvertedSalary'], axis=1)

X_test  = df_test

df_train.describe()
#country_cols = ['Country', 'Population', 'Area (sq. mi.)', 'GDP ($ per capita)', 'Literacy (%)']

X_train = pd.merge(X_train, country_info, on='Country', how='left')

X_test  = pd.merge(X_test , country_info, on='Country', how='left')
X_train['Age'].unique()
#X_train = X_train.replace({'Age':{'18 - 24 years old':20, '25 - 34 years old':30, '35 - 44 years old':40, '45 - 54 years old':50,

#                                '55 - 64 years old':60, '65 years or older':70, 'Under 18 years old':10}})

#X_test = X_test.replace({'Age':{'18 - 24 years old':20, '25 - 34 years old':30, '35 - 44 years old':40, '45 - 54 years old':50,

#                                '55 - 64 years old':60, '65 years or older':70, 'Under 18 years old':10}})

#X_train['Age_null'] = X_train['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)

#X_test['Age_null']  = X_test['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)

#X_train['Age'] = X_train['Age'].astype(int)

#X_test['Age']  = Z_test['Age'].astype(int)

#X_train['Age'].unique()
for col in ['Employment', 'YearsCodingProf', 'SalaryType']:

    summary = X_train[col].value_counts()

    X_train[col+'_cnt'] = X_train[col].map(summary)

    X_test[col+'_cnt']  = X_test[col].map(summary)
cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        print(col, X_train[col].nunique(), X_train[col].isnull().sum())
texts = ['DevType',

'CommunicationTools',

'FrameworkWorkedWith',

'AdsActions',

'ErgonomicDevices',

'Gender',

'SexualOrientation',

'RaceEthnicity']
for col in texts:

    col_train = X_train[col].copy().replace(';', '')

    col_test  = X_test[col].copy().replace(';', '')

    col_train.fillna('', inplace=True)

    col_test.fillna('', inplace=True)



    col_tfidf = TfidfVectorizer(max_features=1000, use_idf=True)

    tfidf_matrix_train = pd.DataFrame(col_tfidf.fit_transform(col_train).todense()).add_prefix(f'tfidf_{col}_')

    tfidf_matrix_test  = pd.DataFrame(col_tfidf.transform(col_test).todense()).add_prefix(f'tfidf_{col}_')

    X_train = pd.concat([X_train.reset_index(drop=True), tfidf_matrix_train], axis=1)

    X_test  = pd.concat([X_test.reset_index(drop=True), tfidf_matrix_test], axis=1)
nums = []

for col in X_train.columns:

    if X_train[col].dtype != 'object':

        nums.append(col)

        print(col, X_train[col].nunique(), X_train[col].isnull().sum())
col_tfidf.fit_transform(col_train).tocsr().toarray()
"""

tfidfs = ['tfidf_DevType']

#tfidfs = ['tfidf_DevType',

#'tfidf_CommunicationTools',

#'tfidf_FrameworkWorkedWith',

#'tfidf_AdsActions',

#'tfidf_ErgonomicDevices',

#'tfidf_Gender',

#'tfidf_SexualOrientation',

#'tfidf_RaceEthnicity']



for col in ['DevType']: #, 'DevTypes', 'RaceEthnicity', ''CommunicationTools'']:

    col_train = X_train[col].copy()

    col_test  = X_test[col].copy()

    col_train.fillna('#', inplace=True)

    col_test.fillna('#', inplace=True)



    col_tfidf = TfidfVectorizer(max_features=1000, use_idf=True)

    col_train = col_tfidf.fit_transform(col_train).tocsr()

    col_test  = col_tfidf.transform(col_test).tocsr()



    scores = []

    random_states = [71]

    n_splits = 5

    stack_train = np.zeros(len(X_train))

    stack_test  = np.zeros(len(X_test))

    for r in random_states:

        skf = KFold(n_splits=n_splits, random_state=r, shuffle=True)

        for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(col_train, y_train))):

            X_train_, y_train_ = col_train[train_ix], y_train.values[train_ix]

            X_val, y_val = col_train[test_ix], y_train.values[test_ix]

            clf = LGBMRegressor(

                learning_rate = 0.05,

                n_estimators=9999,

                random_state=r,

            )

            clf.fit(X_train_, y_train_,

                    early_stopping_rounds=50,

                    eval_metric='rmse',

                    eval_set=[(X_val, y_val)])

            y_pred = clf.predict(X_val)

            score = np.sqrt(mean_squared_error(y_val, y_pred))

            scores.append(score)    

            print('CV Score of Fold_%d is %f' % (i, score))

            y_pred = clf.predict(X_val)

            stack_train[test_ix] = y_pred

            stack_test += clf.predict(col_test)

    

    print(np.array(scores).mean())

    X_train['tfidf_' + col] = stack_train

    X_test['tfidf_' + col] = stack_test

"""
new_cats = list(set(cats) - set(texts))

new_cats
scaler = StandardScaler()

scaler.fit(X_train[nums])

X_train[nums] = scaler.transform(X_train[nums])

X_test[nums]  = scaler.transform(X_test[nums])



oe = OrdinalEncoder(cols=new_cats)

oe.fit(X_train[new_cats])

X_train[new_cats] = oe.transform(X_train[new_cats])

X_test[new_cats]  = oe.transform(X_test[new_cats])



X_train[texts] = X_train[texts].apply(lambda x: x.str.count(';'))

X_test[texts]  = X_test[texts].apply(lambda x: x.str.count(';'))



X_train = X_train[nums+new_cats+texts]#+tfidfs]

X_test  = X_test[nums+new_cats+texts]#+tfidfs]



X_train = X_train.replace([np.inf, -np.inf, np.nan], -99999)

X_test  = X_test.replace([np.inf, -np.inf, np.nan], -99999)
drop_cols =['Climate',

'TimeFullyProductive',

'RaceEthnicity',

'AdsActions',

'StackOverflowConsiderMember',

'AIFuture',

'OpenSource',

'Crops (%)',

'Service',

'AdBlocker',

'Phones (per 1000)',

'Hobby',

'StackOverflowHasAccount',

'MilitaryUS',

'Arable (%)',

'Other (%)',

'Birthrate',

'Pop. Density (per sq. mi.)',

'Infant mortality (per 1000 births)',

'JobSearchStatus',

'TimeAfterBootcamp',

'JobEmailPriorities2',

'SurveyTooLong',

'Gender']



X_train = X_train.drop(drop_cols, axis=1)

X_test  = X_test.drop(drop_cols, axis=1)
scores = []

y_pred_test = np.zeros(len(X_test))



random_states = [71, 42]

n_splits = 5

for r in random_states:

    skf = KFold(n_splits=n_splits, random_state=r, shuffle=True)

    for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

        X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

        X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

        clf = LGBMRegressor(

#            max_depth=3,

            learning_rate = 0.05,

#            colsample_bytree=0.7,

#            subsample=0.7,

#            min_split_gain=0,

#            reg_lambda=1,

#            reg_alpha=1,

#            min_child_weight=2,

            n_estimators=9999,

            random_state=r,

#            importance_type='gain'

        )

        clf.fit(X_train_, y_train_,

                early_stopping_rounds=50,

#                verbose=100,

                eval_metric='rmse',

                eval_set=[(X_val, y_val)])

        y_pred = clf.predict(X_val)

        score = np.sqrt(mean_squared_error(y_val, y_pred))

        scores.append(score)    

        print('CV Score of Fold_%d is %f' % (i, score))

        y_pred_test += clf.predict(X_test)



print(np.array(scores).mean()) # 1.2548831993250675

y_pred_test /= n_splits * len(random_states)

y_pred_test = np.exp(y_pred_test)-1
y_pred_test
submission = pd.read_csv('/kaggle/input/exam-for-students20200923/sample_submission.csv', index_col=0)

submission.ConvertedSalary = y_pred_test

submission.to_csv('submission.csv')
eli5.show_weights(clf, feature_names = X_train.columns.tolist(),top=500)
#gyakusuuにしてsubmit

#ageを数字に