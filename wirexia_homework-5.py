# Import libraries and set desired options

import pickle

from pathlib2 import Path

import numpy as np

import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt

import seaborn as sns

sns.set()

%config InlineBackend.figure_format = 'retina'
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Read the training and test data sets, change paths if needed

PATH_TO_DATA = Path('../input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/')



times = ['time%s' % i for i in range(1, 11)]

train_df = pd.read_csv(PATH_TO_DATA / 'train_sessions.csv',

                       index_col='session_id', parse_dates=times)

test_df = pd.read_csv(PATH_TO_DATA / 'test_sessions.csv',

                      index_col='session_id', parse_dates=times)



# Sort the data by time

train_df = train_df.sort_values(by='time1')



# Look at the first rows of the training set

train_df.head()
# Change site1, ..., site10 columns type to integer and fill NA-values with zeros

sites = ['site%s' % i for i in range(1, 11)]

train_df[sites] = train_df[sites].fillna(0).astype(np.uint16)

test_df[sites] = test_df[sites].fillna(0).astype(np.uint16)



# Load websites dictionary

with open(PATH_TO_DATA / 'site_dic.pkl', "rb") as input_file:

    site_dict = pickle.load(input_file)



# Create dataframe for the dictionary

sites_dict = pd.DataFrame(list(site_dict.keys()), index=list(site_dict.values()), 

                          columns=['site'])

print(u'Websites total:', sites_dict.shape[0])

sites_dict.head()
# Top websites in the training data set

top_sites = pd.Series(train_df[sites].values.flatten()

                     ).value_counts().sort_values(ascending=False).head(5)

print(top_sites)

sites_dict.loc[top_sites.drop(0).index]
topAlice = train_df[train_df['target']==1]

top_Alice = pd.Series(topAlice[sites].values.flatten()).value_counts().sort_values(ascending=False).head()

sites_dict.loc[top_Alice.index]
# Create a separate dataframe where we will work with timestamps

time_df = pd.DataFrame(index=train_df.index)

time_df['target'] = train_df['target']



# Find sessions' starting and ending

time_df['min'] = train_df[times].min(axis=1)

time_df['max'] = train_df[times].max(axis=1)



# Calculate sessions' duration in seconds

time_df['seconds'] = (time_df['max'] - time_df['min']) / np.timedelta64(1, 's')



time_df.head()
sesionMeanAlice = time_df[time_df['target']==1]['seconds'].mean()

sesionMeanSimple = time_df[time_df['target']==0]['seconds'].mean()

print("Средний сеанс Алисы равен", sesionMeanAlice, "а вот обычных пользователей", sesionMeanSimple)



sesionBelongAlice = len(time_df[time_df['target']==1]['seconds'])

sesionBelongAll = len(time_df['seconds'])

belong = 1 - (sesionBelongAll-sesionBelongAlice)/sesionBelongAll

print("Сеансы Алисы = ", belong, "%")



sesionMinAlice = time_df[time_df['target']==1]['seconds'].min()

sesionMinSimple = time_df[time_df['target']==0]['seconds'].min()

sesionMaxAlice = time_df[time_df['target']==1]['seconds'].max()

sesionMaxSimple = time_df[time_df['target']==0]['seconds'].max()

print("Минимальная продолжительность Алисы =", sesionMinAlice, "а у обычных пользователей =", sesionMinSimple)

print("Максимальная продолжительность Алисы =", sesionMaxAlice, "а у обычных пользователей =", sesionMaxSimple)



sesionStdAlice = time_df[time_df['target']==1]['seconds'].std()

sesionStdSimple = time_df[time_df['target']==0]['seconds'].std()

print("Стандартное отклонение Алисы =", sesionStdAlice, "а у обычных пользователей =", sesionStdSimple)



sesionSortAlice = len(time_df[(time_df['target']==1)&(time_df['seconds']>=40)])

sesionAlice = len(time_df[time_df['target']==1]['seconds'])

session = sesionSortAlice / sesionAlice

print("Соотношение времени сеансов состовляет", session, "по сравнению с 1/4=0,25")                  
# Our target variable

y_train = train_df['target']



# United dataframe of the initial data 

full_df = pd.concat([train_df.drop('target', axis=1), test_df])



# Index to split the training and test data sets

idx_split = train_df.shape[0]
# Dataframe with indices of visited websites in session

full_sites = full_df[sites]

full_sites.head()
# sequence of indices

sites_flatten = full_sites.values.flatten()



# and the matrix we are looking for 

# (make sure you understand which of the `csr_matrix` constructors is used here)

# a further toy example will help you with it

full_sites_sparse = csr_matrix(([1] * sites_flatten.shape[0],

                                sites_flatten,

                                range(0, sites_flatten.shape[0]  + 10, 10)))[:, 1:]
full_sites_sparse.shape
# How much memory does a sparse matrix occupy?

print('{0} elements * {1} bytes = {2} bytes'.format(full_sites_sparse.count_nonzero(), 8, 

                                                    full_sites_sparse.count_nonzero() * 8))

# Or just like this:

print('sparse_matrix_size = {0} bytes'.format(full_sites_sparse.data.nbytes))
# data, create the list of ones, length of which equal to the number of elements in the initial dataframe (9)

# By summing the number of ones in the cell, we get the frequency,

# number of visits to a particular site per session

data = [1] * 9



# To do this, you need to correctly distribute the ones in cells

# Indices - website ids, i.e. columns of a new matrix. We will sum ones up grouping them by sessions (ids)

indices = [1, 0, 0, 1, 3, 1, 2, 3, 4]



# Indices for the division into rows (sessions)

# For example, line 0 is the elements between the indices [0; 3) - the rightmost value is not included

# Line 1 is the elements between the indices [3; 6)

# Line 2 is the elements between the indices [6; 9) 

indptr = [0, 3, 6, 9]



# Aggregate these three variables into a tuple and compose a matrix

# To display this matrix on the screen transform it into the usual "dense" matrix

csr_matrix((data, indices, indptr)).todense()
from numpy import count_nonzero



a = csr_matrix((data, indices, indptr)).todense()

sparsity_matrix = np.delete(a,4) 

sparsity = 1 - count_nonzero(sparsity_matrix) / sparsity_matrix.size

print("Разреженность матрицы", sparsity * 100, "%")
def get_auc_lr_valid(X, y, C=1.0, seed=17, ratio = 0.9):

    # Split the data into the training and validation sets

    idx = int(round(X.shape[0] * ratio))

    # Classifier training

    lr = LogisticRegression(C=C, random_state=seed, solver='liblinear').fit(X[:idx, :], y[:idx])

    # Prediction for validation set

    y_pred = lr.predict_proba(X[idx:, :])[:, 1]

    # Calculate the quality

    score = roc_auc_score(y[idx:], y_pred)

    

    return score
%%time

# Select the training set from the united dataframe (where we have the answers)

X_train = full_sites_sparse[:idx_split, :]



# Calculate metric on the validation set

print(get_auc_lr_valid(X_train, y_train))
# Function for writing predictions to a file

def write_to_submission_file(predicted_labels, out_file,

                             target='target', index_label="session_id"):

    predicted_df = pd.DataFrame(predicted_labels,

                                index = np.arange(1, predicted_labels.shape[0] + 1),

                                columns=[target])

    predicted_df.to_csv(out_file, index_label=index_label)
# Train the model on the whole training data set

# Use random_state=17 for repeatability

# Parameter C=1 by default, but here we set it explicitly

lr = LogisticRegression(C=1.0, random_state=17, solver='liblinear').fit(X_train, y_train)



# Make a prediction for test data set

X_test = full_sites_sparse[idx_split:,:]

y_test = lr.predict_proba(X_test)[:, 1]



# Write it to the file which could be submitted

write_to_submission_file(y_test, 'baseline_1.csv')
# Dataframe for new features

full_new_feat = pd.DataFrame(index=full_df.index)



# Add start_month feature

full_new_feat['start_month'] = full_df['time1'].apply(lambda ts: 

                                                      100 * ts.year + ts.month).astype('float64')
full_new_feat.groupby('start_month').size().plot(kind='bar');
# Add the new feature to the sparse matrix

tmp = full_new_feat[['start_month']].values

X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split,:]]))



# Compute the metric on the validation set

print(get_auc_lr_valid(X_train, y_train))
# Add the new standardized feature to the sparse matrix

tmp = StandardScaler().fit_transform(full_new_feat[['start_month']])

X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split,:]]))



# Compute metric on the validation set

print(get_auc_lr_valid(X_train, y_train))
#дата инфа

full_new_feat.head()
#обычная Фул инфа

full_df.head()
#новая инфа

full_new_feat = pd.concat([full_new_feat, full_df], ignore_index=False, axis=1)

full_new_feat.head()
full_new_feat['n_unique_sites'] = full_new_feat[sites].nunique(axis=1,  dropna=True)

full_new_feat.head()
tmp2 = full_new_feat[['n_unique_sites']].values

X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split,:], tmp2[:idx_split,:]]))

print(get_auc_lr_valid(X_train, y_train))
full_new_feat['start_hour'] = full_new_feat['time1'].dt.hour

full_new_feat['morning'] = full_new_feat['time1'].dt.hour <= 11 

full_new_feat.head()
tmp3 = full_new_feat[['start_hour']].values

X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split,:], tmp3[:idx_split,:]]))

print(get_auc_lr_valid(X_train, y_train))
tmp4 = full_new_feat[['morning']].values

X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split,:], tmp4[:idx_split,:]]))

print(get_auc_lr_valid(X_train, y_train))
# Compose the training set

tmp_scaled = StandardScaler().fit_transform(full_new_feat[['start_month', 

                                                           'start_hour', 

                                                           'morning']])

X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], 

                             tmp_scaled[:idx_split,:]]))



# Capture the quality with default parameters

score_C_1 = get_auc_lr_valid(X_train, y_train)

print(score_C_1)
# List of possible C-values

Cs = np.logspace(-3, 1, 10)



C_metric = [ ]

for C in Cs:

    C_metric.append(get_auc_lr_valid(X_train, y_train, C=C))

scores
plt.plot(C_metric, color='red')

plt.axhline(y=score_C_1);
Max_quality = Cs[np.argmax(C_metric, axis=0)] 

round(Max_quality, 2)
Max_quality
# Prepare the training and test data

tmp_scaled = StandardScaler().fit_transform(full_new_feat[['start_month', 'start_hour', 

                                                           'morning']])

X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], 

                             tmp_scaled[:idx_split,:]]))

X_test = csr_matrix(hstack([full_sites_sparse[idx_split:,:], 

                            tmp_scaled[idx_split:,:]]))



# Train the model on the whole training data set using optimal regularization parameter

lr = LogisticRegression(C=C, random_state=17, solver='liblinear').fit(X_train, y_train)



# Make a prediction for the test set

y_test = lr.predict_proba(X_test)[:, 1]



# Write it to the submission file

write_to_submission_file(y_test, 'baseline_2.csv')