import os

import gc

import itertools



import numpy as np

import pandas as pd

from sklearn import preprocessing

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from pprint import pprint

from tqdm import tqdm
%%time

train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')



train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')



sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')



train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)



print(train.shape)

print(test.shape)



y_train = train['isFraud'].copy()

del train_transaction, train_identity, test_transaction, test_identity



# Drop target, fill in NaNs

X_train = train.drop(['isFraud', 'TransactionDT'], axis=1)

X_test = test.drop(['TransactionDT'], axis=1)

del train, test



X_train = X_train.fillna(-999)

X_test = X_test.fillna(-999)



# Label Encoding

for f in X_train.columns:

    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(X_train[f].values) + list(X_test[f].values))

        X_train[f] = lbl.transform(list(X_train[f].values))

        X_test[f] = lbl.transform(list(X_test[f].values))   
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df

%%time

X_train = reduce_mem_usage(X_train)

X_test = reduce_mem_usage(X_test)
class XGBGridSearch:

    """

    Source:

    https://www.kaggle.com/xhlulu/ieee-fraud-efficient-grid-search-with-xgboost

    """

    def __init__(self, param_grid, cv=3, verbose=0, 

                 shuffle=False, random_state=2019):

        self.param_grid = param_grid

        self.cv = cv

        self.random_state = random_state

        self.verbose = verbose

        self.shuffle = shuffle

        

        self.average_scores = []

        self.scores = []

    

    def fit(self, X, y):

        self._expand_params()

        self._split_data(X, y)

            

        for params in tqdm(self.param_list, disable=not self.verbose):

            avg_score, score = self._run_cv(X, y, params)

            self.average_scores.append(avg_score)

            self.scores.append(score)

        

        self._compute_best()



    def _run_cv(self, X, y, params):

        """

        Perform KFold CV on a single set of parameters

        """

        scores = []

        

        for train_idx, val_idx in self.splits:

            clf = xgb.XGBClassifier(**params)



            X_train, X_val = X.iloc[train_idx, :], X.iloc[val_idx, :]

            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            

            clf.fit(X_train, y_train)

            

            y_val_pred = clf.predict_proba(X_val)[:, 1]

            

            score = roc_auc_score(y_val, y_val_pred)

            scores.append(score)

            

            gc.collect()

        

        avg_score = sum(scores) / len(scores)

        return avg_score, scores

            

    def _split_data(self, X, y):

        kf = KFold(n_splits=self.cv, 

                   shuffle=self.shuffle, 

                   random_state=self.random_state)

        self.splits = list(kf.split(X, y))

            

    def _compute_best(self):

        """

        Compute best params and its corresponding score

        """

        idx_best = np.argmax(self.average_scores)

        self.best_score_ = self.average_scores[idx_best]

        self.best_params_ = self.param_list[idx_best]



    def _expand_params(self):

        """

        This method expands a dictionary of lists into

        a list of dictionaries (each dictionary is a single

        valid params that can be input to XGBoost)

        """

        keys, values = zip(*self.param_grid.items())

        self.param_list = [

            dict(zip(keys, v)) 

            for v in itertools.product(*values)

        ]
param_grid = {

    'n_estimators': [500],

    'missing': [-999],

    'random_state': [2019],

    'n_jobs': [1],

    'tree_method': ['gpu_hist'],

    'max_depth': [9],

    'learning_rate': [0.048, 0.05],

    'subsample': [0.85, 0.9],

    'colsample_bytree': [0.85, 0.9],

    'reg_alpha': [0, 0.1],

    'reg_lambda': [1, 0.9]

}



grid = XGBGridSearch(param_grid, cv=4, verbose=1)

grid.fit(X_train, y_train)



print("Best Score:", grid.best_score_)

print("Best Params:", grid.best_params_)
clf = xgb.XGBClassifier(**grid.best_params_)

clf.fit(X_train, y_train)



sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]

sample_submission.to_csv('simple_xgboost.csv')