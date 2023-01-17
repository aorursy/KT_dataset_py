# inspired by https://www.kaggle.com/srodriguex/santander-customer-satisfaction/model-and-feature-selection-with-python

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
print(train.info())
def create_sparse_matrix():
    remove = []
    for col in train.columns:
        if train[col].std() == 0:
            remove.append(col)
    train.drop(remove, axis=1, inplace=True)
    sparse_matrix = train.replace(0, np.nan).to_sparse()
    print(sparse_matrix.info())
    return sparse_matrix

#train = create_sparse_matrix()
### no None, NaN, inf values in the train dataset

count_nulls = (train.isnull().sum()==1).sum()
print(count_nulls)
### we remove the columns with constant values

def remove_constant_columns(dataframe):
    unique_values = dataframe.apply(lambda x : len(x.unique()))
    colls_to_drop = unique_values[unique_values==1].index.values.tolist()
    print('number of train columns before dropping constants {0}'.format(len(dataframe.columns.values)))
    dataframe = dataframe.drop(colls_to_drop, axis=1)
    print('number of train columns after dropping constants {0}'.format(len(dataframe.columns.values)))
    return dataframe


train = remove_constant_columns(train)

from itertools import combinations
from numpy import array,array_equal

def identify_equal_features(dataframe):
    features_to_compare = list(combinations(dataframe.columns.tolist(),2))
    equal_features = []
    for compare in features_to_compare:
        is_equal = array_equal(dataframe[compare[0]],dataframe[compare[1]])
        if is_equal:
            equal_features.append(list(compare))
    return dataframe, equal_features


def drop_duplicates(dataframe, list_equal_features):
    ### get the columns which have duplicates and remove them. 
    list_equal_unique_features = array(list_equal_features)[:,1]
    print('number of columns to drop : {}'.format(len(list_equal_unique_features)))
    try:
        dataframe = dataframe.drop(list_equal_unique_features, axis=1)
    except ValueError as v:
        print('columns were already dropped')
    return dataframe


def identify_and_drop_equal_features(dataframe):
    dataframe, equal_features = identify_equal_features(dataframe)
    print(equal_features)
    return drop_duplicates(dataframe, equal_features)

#train = identify_and_drop_equal_features(train)
#dataframe, eaqual_features = identify_equal_features(train)
#print(eaqual_features)
train = identify_and_drop_equal_features(train)
def create_features_and_target_df(df):
    y_name = 'TARGET'
    feature_names = train.columns.tolist()
    feature_names.remove(y_name)
    X = train[feature_names]
    Y = train[y_name]
    return X, Y

X,Y = create_features_and_target_df(train)


from sklearn import cross_validation as cv
from sklearn import tree
from sklearn import metrics
from sklearn import ensemble
from sklearn import linear_model 
from sklearn import naive_bayes 

skf = cv.StratifiedKFold(Y, n_folds=3, shuffle=True)
score_metric = 'roc_auc'
scores = {}

def score_model(model):
    return cv.cross_val_score(model, X, Y, cv=skf, scoring=score_metric)


# time: 10s
scores['tree'] = score_model(tree.DecisionTreeClassifier()) 
print(scores['tree'])
# time: 7s
scores['forest'] = score_model(ensemble.RandomForestClassifier())
print(scores['forest'])

import xgboost as xgb
# time: 4min
scores['xgboost'] = score_model(xgb.XGBClassifier())
clf = xgb.XGBClassifier()
clf.fit(X,Y)
test = pd.read_csv('../input/test.csv')
test = remove_constant_columns(test)
test = identify_and_drop_equal_features(test)
X_test,Y_test = create_features_and_target_df(train)

test_pred = clf.predict_proba(X_test)
print(test_pred)
print(len(test_pred))