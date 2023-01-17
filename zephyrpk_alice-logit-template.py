# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

from scipy.sparse import hstack

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# a helper function for writing predictions to a file

def write_to_submission_file(predicted_labels, out_file,

                             target='target', index_label="session_id"):

    predicted_df = pd.DataFrame(predicted_labels,

                                index = np.arange(1, predicted_labels.shape[0] + 1),

                                columns=[target])

    predicted_df.to_csv(out_file, index_label=index_label)
train_df = pd.read_csv('../input/catch-me-if-you-can/train_sessions.csv',

                       index_col='session_id')

test_df = pd.read_csv('../input/catch-me-if-you-can/test_sessions.csv',

                      index_col='session_id')



# Convert time1, ..., time10 columns to datetime type

times = ['time%s' % i for i in range(1, 11)]

train_df[times] = train_df[times].apply(pd.to_datetime)

test_df[times] = test_df[times].apply(pd.to_datetime)



# Sort the data by time

train_df = train_df.sort_values(by='time1')



# Look at the first rows of the training set

train_df.head()
sites = ['site%s' % i for i in range(1, 11)]

train_df[sites].fillna(0).astype('int').to_csv('train_sessions_text.txt', 

                                               sep=' ', 

                       index=None, header=None)

test_df[sites].fillna(0).astype('int').to_csv('test_sessions_text.txt', 

                                              sep=' ', 

                       index=None, header=None)
!head -5 train_sessions_text.txt
!ls
cv = CountVectorizer()
%%time

with open('train_sessions_text.txt') as inp_train_file:

    X_train = cv.fit_transform(inp_train_file)

with open('test_sessions_text.txt') as inp_test_file:

    X_test = cv.transform(inp_test_file)

print(X_train.shape, X_test.shape)
type(X_train)
y_train = train_df['target'].astype(int)
y_train.head()
logit = LogisticRegression(C = 1, random_state=42)
%%time

cv_scores = cross_val_score(logit, X_train, y_train, cv= 5, scoring='roc_auc')
cv_scores.mean()
%%time

logit.fit(X_train, y_train)
test_pred_logit1 = logit.predict_proba(X_test)[:,1]
write_to_submission_file(test_pred_logit1, 'logit_sub1.txt') ## .908 ROC AUC
!head logit_sub1.txt
def add_time_features(time1_series, X_sparse):

    hour = time1_series.apply(lambda ts: ts.hour)

    morning = ((hour >= 7) & (hour <= 11)).astype('int')

    day = ((hour >= 12) & (hour <= 18)).astype('int')

    evening = ((hour >= 19) & (hour <= 23)).astype('int')

    night = ((hour >= 0) & (hour <= 6)).astype('int')

    X = hstack([X_sparse, morning.values.reshape(-1, 1), 

                day.values.reshape(-1, 1), evening.values.reshape(-1, 1), 

                night.values.reshape(-1, 1)])

    return X
test_df.loc[:, 'time1'].fillna(0).apply(lambda ts: ts.hour).head()
%%time

X_train_with_time = add_time_features(train_df['time1'].fillna(0), X_train)

X_test_with_time = add_time_features(test_df['time1'].fillna(0), X_test)
logit_with_time = LogisticRegression(C = 1, random_state=42)
%%time

cv_scores = cross_val_score(logit_with_time, X_train_with_time, y_train, cv= 5, scoring='roc_auc');
cv_scores.mean()
%%time

logit_with_time.fit(X_train_with_time, y_train)
test_pred_logit2 = logit_with_time.predict_proba(X_test_with_time)[:,1]
write_to_submission_file(test_pred_logit2, 'logit_sub2.txt') ## .93565 ROC AUC
!head logit_sub2.txt