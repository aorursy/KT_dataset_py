# устанавливаю пакеты



import os

import pandas as pd

import numpy as np

import scipy.sparse as sps

from time import time

from scipy.sparse import csr_matrix

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn import preprocessing

from collections import Counter
!cd

!ls
# загружаем данные



PATH_TO_DATA = '../input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/'



train_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_sessions.csv'),

                       index_col='session_id')

test_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_sessions.csv'),

                      index_col='session_id')
# проверяю тренировочные

train_df.head()
# проверяю тестовые

test_df.head()
# проверю тип данных

train_df.info()
# в таймс тип данных надо менять

# сначало сделаю возможность для выбора колонок time и site



times = ['time' + str(i) for i in range(1, 11)]

sites = ['site' + str(i) for i in range(1, 11)]
# теперь поменяю тип данных в time в тренировочном и тестовом сете



train_df[times] = train_df[times].apply(pd.to_datetime)

test_df[times] = test_df[times].apply(pd.to_datetime)
train_df.info()
# функция для создания разряженной матрицы



def create_sparse_matrix(dataframe):

    tmp_arr = np.array(dataframe)

    row = 0

    rows = []

    cols = []

    data = []



    for arr in tmp_arr:

        unique, counts = np.unique(arr, return_counts=True)

        #print(dict(zip(unique, counts)))

        for key, value in dict(zip(unique, counts)).items():

            if key != 0:

                rows.append(row)

                cols.append(key-1)

                data.append(value)

        row = row + 1

        

    return(sps.coo_matrix((data, (rows, cols))))
# выделю y и запольню nan

y = train_df.target

train_df = train_df.fillna(0)

test_df = test_df.fillna(0)
# объеденю две таблицы

train_test_df = pd.concat([train_df.drop('target', axis=1), test_df])
# выделю колонки только с site и конвертирую все значения в int

train_test_df_sites = train_test_df[sites]

train_test_df_sites = train_test_df_sites.astype(int)
# определю длину тренировочного теста

idx_split = y.shape[0]
# попробую получить разряженную матрицу

train_test_sparse = csr_matrix(create_sparse_matrix(train_test_df_sites))
train_test_sparse.shape
#выделю тренировочный разряженный тест и тестовый тест

X_train_sparse = train_test_sparse[:idx_split]

X_test_sparse = train_test_sparse[idx_split:]
X_train_sparse.shape
train_share = int(.1 * X_train_sparse.shape[0])

train_share
#разделю еще тренеривочный тест в пропорции 90 : 10 для первоначального обучения



# это разбиение для любых алгоритмов

train_share = int(.1 * X_train_sparse.shape[0])

X_train, y_train = X_train_sparse[:train_share, :], y[:train_share]

X_valid, y_valid  = X_train_sparse[train_share:, :], y[train_share:]
from sklearn import linear_model

from sklearn.metrics import roc_auc_score
sgd_logit = SGDClassifier(loss='log', n_jobs=-1)

sgd_logit.fit(X_train, y_train)
logit_valid_pred_proba = sgd_logit.predict(X_valid)
roc_auc_score(y_valid, logit_valid_pred_proba)
from sklearn import ensemble

from sklearn import datasets

from sklearn.utils import shuffle

from sklearn.metrics import mean_squared_error
# Fit regression model

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.01, 'loss': 'ls'}

clf = ensemble.GradientBoostingRegressor(**params)



clf.fit(X_train, y_train)

mse = mean_squared_error(y_valid, clf.predict(X_valid))

print("MSE: %.4f" % mse)
roc_auc_score(y_valid, clf.predict(X_valid))
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
time_split = TimeSeriesSplit(n_splits=10)
[(el[0].shape, el[1].shape) for el in time_split.split(X_train_sparse)]
logit = LogisticRegression(C=1, solver='lbfgs')
cv_scores = cross_val_score(logit, X_train_sparse, y, cv=time_split, 

                            scoring='roc_auc', n_jobs=1) 
cv_scores, cv_scores.mean()
logit.fit(X_train_sparse, y)
# сделаю функцию для записи решения в файл



def write_to_submission_file(predicted_labels, out_file,

                             target='target', index_label="session_id"):

    # turn predictions into data frame and save as csv file

    predicted_df = pd.DataFrame(predicted_labels,

                                index = np.arange(1, predicted_labels.shape[0] + 1),

                                columns=[target])

    predicted_df.to_csv(out_file, index_label=index_label)
cv_pred = logit.predict_proba(X_test_sparse)[:, 1]

#write_to_submission_file(cv_pred, "./kaggle_data/time_predictions.csv")
c_values = np.logspace(-2, 2, 10)



logit_grid_searcher = GridSearchCV(estimator=logit, param_grid={'C': c_values},

                                  scoring='roc_auc', n_jobs=1, cv=time_split, verbose=1)
logit_grid_searcher.fit(X_train_sparse, y)

lg_searcher = logit_grid_searcher.predict_proba(X_test_sparse)[:, 1]

#write_to_submission_file(lg_searcher, "./kaggle_data/logit_grid_searche_site.csv")