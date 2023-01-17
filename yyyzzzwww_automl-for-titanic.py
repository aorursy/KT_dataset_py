# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Modeling
import lightgbm as lgb

# Evaluation of the model
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.size'] = 18
%matplotlib inline
from hyperopt import fmin
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperopt import STATUS_OK
from timeit import default_timer as timer
from hyperopt import tpe
from hyperopt import Trials
import csv
import ast
N_FOLDS = 5
MAX_EVALS = 1000
space = {
    'boosting_type': hp.choice('boosting_type', 
      [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
       {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
       {'boosting_type': 'goss', 'subsample': 1.0}]),
#     'max_depth' : hp.quniform('max_depth',1, 15, 1),
    'num_leaves': hp.quniform('num_leaves', 20, 60, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
#     'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.5, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.5, 1.0),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    'is_unbalance': hp.choice('is_unbalance', [True, False]),
}
# Create a new file and open a connection
OUT_FILE = 'bayesian_trials_1000.csv'
of_connection = open(OUT_FILE, 'w')
writer = csv.writer(of_connection)

# Write column names
headers = ['loss', 'hyperparameters', 'iteration', 'runtime', 'score']
writer.writerow(headers)
of_connection.close()
# get titanic & test csv files as a DataFrame
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

# preview the data
train_df.head()
def loadData(df, test = False):
    df.Fare[df.Fare.isnull()] = df.Fare.mean()
    df.Embarked[df.Embarked.isnull()] = 'S'
    df.Sex[df.Sex == "male"] = 1
    df.Sex[df.Sex == "female"] = 0
    df.Embarked[df.Embarked == "S"] = 0
    df.Embarked[df.Embarked == "C"] = 1
    df.Embarked[df.Embarked == "Q"] = 2

    df["Ticket_Value"] = df.Ticket.map(df.Ticket.value_counts())
    df['Embarked'] = df.Embarked.map(int)
    df['Sex'] = df.Sex.map(int)
    df.drop("Ticket", axis = 1, inplace = True)

    
    cols_list = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked"]
    
    if test:
        y = None
    else:
        y = df.Survived
    X = df[cols_list]
    
    return X, y

train_X, train_y = loadData(train_df)
test_X, test_y = loadData(test_df, test=True)
train_X.info(), train_X.head(5)
# Training set
train_set = lgb.Dataset(train_X, label = train_y)
test_set = lgb.Dataset(test_X, label = test_y)
def objective(hyperparameters):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization.
       Writes a new line to `outfile` on every iteration"""
    # Keep track of evals
    global ITERATION
    ITERATION += 1
    
    # Using early stopping to find number of trees trained
    if 'n_estimators' in hyperparameters:
        del hyperparameters['n_estimators']
    
    # Retrieve the subsample(dict.get(key, default=None))
    subsample = hyperparameters['boosting_type'].get('subsample', 1.0)
    # Extract the boosting type and subsample to top level keys
    hyperparameters['boosting_type'] = hyperparameters['boosting_type']['boosting_type']
    hyperparameters['subsample'] = subsample
    
    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'min_child_samples']:
        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])

    start = timer()
    # Perform n_folds cross validation
#     cv_results = lgb.cv(hyperparameters, train_set, num_boost_round = 10000, nfold = N_FOLDS, 
#                         early_stopping_rounds = 100, metrics = 'auc', seed = 50)
    cv_results = lgb.cv(hyperparameters, train_set, num_boost_round = 10000, nfold = N_FOLDS, 
                        early_stopping_rounds = 100, metrics = 'auc', seed = 50, verbose_eval=200)
    run_time = timer() - start
    
    # Extract the best score
    best_score = cv_results['auc-mean'][-1]
    # Loss must be minimized
    loss = 1 - best_score
    
    # Boosting rounds that returned the highest cv score
    n_estimators = len(cv_results['auc-mean'])
    # Add the number of estimators to the hyperparameters
    hyperparameters['n_estimators'] = n_estimators

    # Write to the csv file ('a' means append)
    of_connection = open(OUT_FILE, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, hyperparameters, ITERATION, run_time, best_score])
    of_connection.close()

    # Dictionary with information for evaluation
    return {'loss': loss, 'hyperparameters': hyperparameters, 'iteration': ITERATION,
            'train_time': run_time, 'status': STATUS_OK}
MAX_EVALS = 1000

# Create the algorithm
tpe_algorithm = tpe.suggest

# Record results
trials = Trials()
# Global variable
global  ITERATION

ITERATION = 0
# Run optimization
best = fmin(fn = objective, space = space, algo = tpe.suggest, trials = trials,
            max_evals = MAX_EVALS)
best
trials_dict = sorted(trials.results, key = lambda x: x['loss'])
trials_dict[:3]
type(trials_dict[0]['hyperparameters'])
def evaluate(results, name, train_features, train_labels, test_features):
    """Evaluate model on test data using hyperparameters in results
       Return dataframe of hyperparameters"""
    
    new_results = results.copy()
    # String to dictionary
    new_results['hyperparameters'] = new_results['hyperparameters'].map(ast.literal_eval)
    
    # Sort with best values on top
    new_results = new_results.sort_values('score', ascending = False).reset_index(drop = True)
    
    # Print out cross validation high score
    print('The highest cross validation score from {} was {:.5f} found on iteration {}.'.format(name, new_results.loc[0, 'score'], new_results.loc[0, 'iteration']))
    
    # Use best hyperparameters to create a model
    hyperparameters = new_results.loc[0, 'hyperparameters']
    model = lgb.LGBMClassifier(**hyperparameters)
    
    # Train and make predictions
    model.fit(train_features, train_labels)
    preds = model.predict_proba(test_features)[:, 1]
    
#     print('ROC AUC from {} on test data = {:.5f}.'.format(name, roc_auc_score(test_labels, preds)))
    
#     # Create dataframe of hyperparameters
#     hyp_df = pd.DataFrame(columns = list(new_results.loc[0, 'hyperparameters'].keys()))

#     # Iterate through each set of hyperparameters that were evaluated
#     for i, hyp in enumerate(new_results['hyperparameters']):
#         hyp_df = hyp_df.append(pd.DataFrame(hyp, index = [0]), 
#                                ignore_index = True)
        
#     # Put the iteration and score in the hyperparameter dataframe
#     hyp_df['iteration'] = new_results['iteration']
#     hyp_df['score'] = new_results['score']
    
    return preds
# Sort the trials with lowest loss (highest AUC) first
# trials_dict = sorted(trials.results, key = lambda x: x['loss'])
results = pd.read_csv(OUT_FILE)
# bayes_results = evaluate(results, name = 'Bayesian')
new_results = results.copy()
# String to dictionary
new_results['hyperparameters'] = new_results['hyperparameters'].map(ast.literal_eval)

# Sort with best values on top
new_results.sort_values('score', ascending = False).reset_index(drop = True)
del new_results
bayes_results = evaluate(results,'Bayesian', train_X, train_y, test_X)
bayes_results.shape
bayes_results
lgbmpred = (bayes_results > 0.5).astype(int)
submission = pd.DataFrame({'PassengerId':test_df.PassengerId,'Survived':lgbmpred})
submission.to_csv('LGBM.csv',index=False)
lgbmpred
