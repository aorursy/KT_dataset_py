import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb # XGBoost is short for “Extreme Gradient Boosting”

# See here for details about xgboost: http://xgboost.readthedocs.io/en/latest/model.html

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import time

import gc

sns.set_style('whitegrid')
STATIONS = ['S32', 'S33', 'S34']

train_date_part = pd.read_csv('../input/train_date.csv', nrows=10000)

# count missing value in each date column

date_cols = train_date_part.drop('Id', axis=1).count().reset_index().sort_values(by=0, ascending=False)

# create a new variable station which reads SXX in the date column name, like "L3_S37_D3949"

date_cols['station'] = date_cols['index'].apply(lambda s: s.split('_')[1])



# filter only S32, S33, and S34

date_cols = date_cols[date_cols['station'].isin(STATIONS)]

# save the date names for S32, 33,34 to a list

date_cols = date_cols.drop_duplicates('station', keep='first')['index'].tolist()

print(date_cols)
# read train_date keep only Id and the 3 date_columns

train_date1 = pd.read_csv('../input/train_date.csv', usecols=['Id'] + date_cols)

print(train_date1.columns)

train_date1.head()
# rename 3 columns

train_date1.columns = ['Id'] + STATIONS



# convert values> 0  to 1, else to 0 

for station in STATIONS:

    train_date1[station] = 1 * (train_date1[station] >= 0)

train_date1.head()
# get response from train_numeric

response = pd.read_csv('../input/train_numeric.csv', usecols=['Id', 'Response'])

print(response.shape)



# merge predictors to response

train1 = response.merge(train_date1, how='left', on='Id')

# print(train.count())

print(train1.head(3))
train1['cnt'] = 1

failure_rate = train1.groupby(STATIONS).sum()[['Response', 'cnt']]

failure_rate['failure_rate'] = failure_rate['Response'] / failure_rate['cnt']

print(failure_rate.head(20))

failure_rate = failure_rate[failure_rate['cnt'] > 1000]  # remove 

print(failure_rate.head(20))
# set data directory

DATA_DIR = "../input"



ID_COLUMN = 'Id'

TARGET_COLUMN = 'Response'



# some parameters in reading in the files

SEED = 0

CHUNKSIZE = 50000

NROWS = 250000



#Read all test and training data sets:

TRAIN_NUMERIC = "{0}/train_numeric.csv".format(DATA_DIR)

TRAIN_DATE = "{0}/train_date.csv".format(DATA_DIR)

TRAIN_CAT = "{0}/train_categorical".format(DATA_DIR)



TEST_NUMERIC = "{0}/test_numeric.csv".format(DATA_DIR)

TEST_DATE = "{0}/test_date.csv".format(DATA_DIR)

TRAIN_CAT = "{0}/test_categorical".format(DATA_DIR)
FILENAME = "etimelhoods"



train = pd.read_csv(TRAIN_NUMERIC, usecols=[ID_COLUMN, TARGET_COLUMN], nrows=NROWS)

test = pd.read_csv(TEST_NUMERIC, usecols=[ID_COLUMN], nrows=NROWS)



train["StartTime"] = -1

test["StartTime"] = -1

# **Read training and test date**

# print ("Size of training data: %int" % train.shape[0])

# print (train.head())

# print ("Size of test data: %int" %  test.shape[0])

# print (test.head())
train_date = pd.read_csv(TRAIN_DATE,  nrows=10)

# test_date = pd.read_csv(TEST_DATE,  nrows=10)

train_date.head()

# test_date.head()
train_num = pd.read_csv(TRAIN_NUMERIC,  nrows=10)

# test_num = pd.read_csv(TEST_NUMERIC,  nrows=10)

train_num.head()
train_cat = pd.read_csv(TRAIN_CAT,  nrows=10)

# test_cat = pd.read_csv(TEST_CAT,  nrows=10)

train_cat.head()
nrows = 0

# Zip: Make an iterator that aggregates elements from each of the iterables;

# Returns an iterator of tuples: zip('ABCD','xy') --> Ax By



for tr, te in zip(pd.read_csv(TRAIN_DATE, chunksize=CHUNKSIZE), pd.read_csv(TEST_DATE, chunksize=CHUNKSIZE)):

    

    # numpy.setdiff1d(ar1, ar2, assume_unique=False). Find the set difference of two arrays. 

    # Return the sorted, unique values in ar1 that are not in ar2.    

    feats = np.setdiff1d(tr.columns, [ID_COLUMN])

    # feats are the column names in _date dataset, excluding ID



    # get the min date for each ID

    stime_tr = tr[feats].min(axis=1).values

    stime_te = te[feats].min(axis=1).values

    

    # save min date for each ID, if the ID exist in data 'train'/'test'

    train.loc[train.Id.isin(tr.Id), 'StartTime'] = stime_tr

    test.loc[test.Id.isin(te.Id), 'StartTime'] = stime_te



    nrows += CHUNKSIZE

    if nrows >= NROWS:

        break
ntrain = train.shape[0] # num of rows in training set

train_test = pd.concat((train, test)).reset_index(drop=True).reset_index(drop=False)



# **Create 4 predictors based solely on ID**

# new col= kth Id - (k-1)th ID

train_test['magic1'] = train_test[ID_COLUMN].diff().fillna(9999999).astype(int)

# new col= kth Id - (k+1)th ID

train_test['magic2'] = train_test[ID_COLUMN].iloc[::-1].diff().fillna(9999999).astype(int)



# Sort by StartTime and then by ID, create another 2 columns based on ID

train_test = train_test.sort_values(by=['StartTime', 'Id'], ascending=True)

train_test['magic3'] = train_test[ID_COLUMN].diff().fillna(9999999).astype(int)

train_test['magic4'] = train_test[ID_COLUMN].iloc[::-1].diff().fillna(9999999).astype(int)



print(train_test.head())

print(train_test.tail())



# sort data back to original order

train_test = train_test.sort_values(by=['index']).drop(['index'], axis=1)

# save new train data

train = train_test.iloc[:ntrain, :]
'''# visualizing the magic features above

This is an attempt at visualizing the magic feature(outed by Faron) in how well it separates responses. Can be used to visualize any random feature's discriminating power.



**source:** https://www.kaggle.com/rithal/bosch-production-line-performance/magic-feature-visualization'''



def twoplot(df, col, xaxis=None):

    ''' scatter plot a feature split into response values as two subgraphs '''

    if col not in df.columns.values:

        print('ERROR: %s not a column' % col)

    ndf = pd.DataFrame(index = df.index)

    ndf[col] = df[col]

    ndf[xaxis] = df[xaxis] if xaxis else df.index

    ndf['Response'] = df['Response']

    

    g = sns.FacetGrid(ndf, col="Response", hue="Response")

    g.map(plt.scatter, xaxis, col, alpha=.7, s=1)

    g.add_legend();

    

    del ndf
twoplot(train, 'magic1')
twoplot(train, 'magic2')
twoplot(train, 'magic3')
twoplot(train, 'magic4')
'''The following codes are commented out for now'''



'''# features is all column in train, except ID and response:

features = np.setdiff1d(list(train.columns), [TARGET_COLUMN, ID_COLUMN])

# y: response of new training set

y = train.Response.ravel()  # numpy.ravel(): Return a flattened array

# train: 4 predictors based on columns + StartTime

train = np.array(train[features]) 



# print # rows and # cols of training predictors: 250K * 5:

print('train: {0}'.format(train.shape))



# Use failure rate in training set as prior to input into the model:

prior = np.sum(y) / (1.*len(y))



# set parameters for xgboost

xgb_params = {

    'seed': 0,

    'colsample_bytree': 0.7,

    'silent': 1,

    'subsample': 0.7,

    'learning_rate': 0.1,

    'objective': 'binary:logistic',

    'max_depth': 4,

    'num_parallel_tree': 1,

    'min_child_weight': 2,

    'eval_metric': 'auc',

    'base_score': prior

}





dtrain = xgb.DMatrix(train, label=y)

res = xgb.cv(xgb_params, dtrain, num_boost_round=10, nfold=4, seed=0, stratified=True,

             early_stopping_rounds=1, verbose_eval=1, show_stdv=True)



cv_mean = res.iloc[-1, 0]

cv_std = res.iloc[-1, 1]



print('CV-Mean: {0}+{1}'.format(cv_mean, cv_std))

'''