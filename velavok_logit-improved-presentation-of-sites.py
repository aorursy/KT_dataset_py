from __future__ import division, print_function
import warnings
import os
import re
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from scipy.stats.mstats import gmean
from sklearn.model_selection import cross_val_score
%matplotlib inline
warnings.filterwarnings('ignore')
PATH_TO_DATA = '../input'
train_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_sessions.csv'),
                       index_col='session_id')
test_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_sessions.csv'),
                      index_col='session_id')
with open(os.path.join(PATH_TO_DATA, 'site_dic.pkl'), 'rb') as input_file:
    site_dict = pickle.load(input_file)
site_dict['Unknown'] = 0

#Let's sort the data by time. This is useful for cross-validation over time.
train_df = train_df.sort_values(by='time1')
def get_sites_dict_(data):
    '''
    Gets dataframe with site names (10 columns) as input.
    Returns a dictionary of sites ordered by frequency.
    '''
    m, n = data.shape #num of rows and columns
    data = pd.DataFrame(data.values.reshape(m*n, 1), columns=['site']) #transform to 1 column
    freq = data.site.value_counts().reset_index()
    key_value_df = pd.DataFrame() #contains a pair of site-frequency
    key_value_df['site'] = freq['index']
    key_value_df['count'] = freq['site']
    key_value_df.sort_values(by='count', inplace=True, ascending=False) 
    sites_dict = {} 
    sites_dict['Unknown'] = 0
    for i in np.arange(key_value_df.shape[0]):
        if key_value_df.iloc[i,0]!='Unknown':
            sites_dict[key_value_df.iloc[i,0]] = i+1
    return sites_dict

def inverse_dict(sites_dict):
    '''
    Gets a key-value dictionary. 
    Returns the dictionary by swapping the key and value.
    '''
    code_sites_dict = {}
    sites = list(sites_dict.items())
    for site in sites:
        code_sites_dict[site[1]] = site[0]
    return code_sites_dict

def make_sparse_data(data):
    '''
    Makes sparse matrix
    '''
    n = data.shape[1]
    X = data.values
    flatten_X = X.flatten()
    new_X = ([1]*flatten_X.shape[0], flatten_X, range(0, flatten_X.shape[0]+10, 10))
    X_sparse = csr_matrix(new_X)[:, 1:]
    return X_sparse

def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df=pd.DataFrame(predicted_labels,index=np.arange(1,predicted_labels.shape[0]+1),columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)
train_test_df = pd.concat([train_df, test_df])

sites = ['site%s' % i for i in range(1, 11)]
times = ['time%s' % i for i in range(1, 11)]
all_features = sites + times

train_test_df_sites = train_test_df[sites].fillna(0).astype('int')
train_test_df_times = train_test_df[times].fillna(0).astype('datetime64')
train_test_df_sites.head()
#the index by which we will separate the training sample from the test sample
idx_split = train_df.shape[0]

y = train_df['target']
train_test_sparse = make_sparse_data(train_test_df_sites)
X_train_sparse = train_test_sparse[:idx_split, :]
X_test_sparse = train_test_sparse[idx_split:,:]

print('Train DF size: {0}\nTest DF size: {1}\nTarget size: {2}'.format(
    str(X_train_sparse.shape), str(X_test_sparse.shape), str(y.shape)))
#Spoiler: this conversion gives a good boost. If you have not done it yet, be sure to use it :)
general_df = pd.concat([train_test_df_sites, train_test_df_times], axis=1) #main df

general_sites = general_df[sites].apply(lambda ts: ts.map(inverse_dict(site_dict))) #instead of numbers the names of sites
general_sites = general_sites.applymap(lambda site: re.sub("^\S*?\.*?www\S*?\.", '', site)) 

new_site_dict = get_sites_dict_(general_sites)

#on a thousand unique sites became less.
print(len(list(site_dict.keys())), len(list(new_site_dict.keys())))
general_sites.head()
general_df[sites] = general_sites.apply(lambda ts: ts.map(new_site_dict)) #new coding
general_sites_sparse = make_sparse_data(general_df[sites]) #make the first half of the sparse matrix (sites).
#1 morning
general_df['morning'] = general_df['time1'].apply(lambda ts: 1 if (ts.hour>=7) & (ts.hour<=11) else 0)
#2 evening
general_df['evening'] = general_df['time1'].apply(lambda ts: 1 if (ts.hour>=19) & (ts.hour<=23) else 0)
#3 start hour
general_df['start_hour'] = general_df['time1'].apply(lambda ts: int(ts.hour))
#4 month
general_df['month'] = general_df['time1'].apply(lambda ts: ts.month)
#5 day of week
general_df['day_of_week'] = general_df['time1'].apply(lambda ts: ts.isoweekday())

general_df.head()
other_features = list(set(general_df.columns) - set(times) - set(sites))
visual_df = general_df[other_features]
visual_df["target"] = y

#correlations
sns.heatmap(visual_df.corr())
#start_hour
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.title("Alice's activity by start hour")
plt.ylim(0, 40)
values_perc = visual_df[visual_df['target']==1]['start_hour'].value_counts(normalize=True)*100
ax = sns.barplot(values_perc.index, values_perc.values, palette="YlGnBu")
ax.set(ylabel="Percent")

plt.subplot(1, 2, 2)
plt.title("Activity of others by start hour")
plt.ylim(0, 40)
values_perc = visual_df[visual_df['target']==0]['start_hour'].value_counts(normalize=True)*100
ax = sns.barplot(values_perc.index, values_perc.values, palette="YlGnBu")
ax.set(ylabel="Percent")
plt.tight_layout()
#month
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.title("Alice's activity by start month")
plt.ylim(0, 30)
values_perc = visual_df[visual_df['target']==1]['month'].value_counts(normalize=True)*100
ax = sns.barplot(values_perc.index, values_perc.values, palette="OrRd")
ax.set(ylabel="Percent")

plt.subplot(1, 2, 2)
plt.title("Activity of others by start month")
plt.ylim(0, 30)
values_perc = visual_df[visual_df['target']==0]['month'].value_counts(normalize=True)*100
ax = sns.barplot(values_perc.index, values_perc.values, palette="OrRd")
ax.set(ylabel="Percent")
plt.tight_layout()
#day of week
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.title("Alice's activity by days of week")
plt.ylim(0, 35)
values_perc = visual_df[visual_df['target']==1]['day_of_week'].value_counts(normalize=True)*100
ax = sns.barplot(values_perc.index, values_perc.values)
ax.set(ylabel="Percent")

plt.subplot(1, 2, 2)
plt.title("Activity of others by days of week")
plt.ylim(0, 35)
values_perc = visual_df[visual_df['target']==0]['day_of_week'].value_counts(normalize=True)*100
ax = sns.barplot(values_perc.index, values_perc.values)
ax.set(ylabel="Percent")
plt.tight_layout()
general_df = pd.get_dummies(general_df, columns=['start_hour'])
general_df = pd.get_dummies(general_df, columns=['month'])
general_df = pd.get_dummies(general_df, columns=['day_of_week'])
other_features = list(set(general_df.columns) - set(times) - set(sites))
print(other_features)
general_sparse = csr_matrix(hstack([general_sites_sparse, csr_matrix(general_df[other_features])]))

X_train_notimes = general_sparse[:idx_split, :]
X_test_notimes = general_sparse[idx_split:,:]
print(X_train_notimes.shape, X_test_notimes.shape)
time_split = TimeSeriesSplit(n_splits=10)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=97)

logit = LogisticRegression(random_state=97) #add n_jobs=-1

#Use the geometric mean of the two results on cross-validation.
print(gmean([round(cross_val_score(logit, X_train_notimes, y, cv=skf, scoring='roc_auc').mean(),5),
      round(cross_val_score(logit, X_train_notimes, y, cv=time_split, scoring='roc_auc').mean(),5)]))
logit.fit(X_train_notimes, y)
logit_pred_proba = logit.predict_proba(X_test_notimes)[:,1]
write_to_submission_file(logit_pred_proba, 'submission_LOGIT.csv')