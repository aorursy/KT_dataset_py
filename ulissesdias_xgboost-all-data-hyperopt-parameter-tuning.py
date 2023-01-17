# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Plot
import seaborn as sns # Beautiful plots


## Classifier of XGBosst
from   xgboost import XGBClassifier

## Package used for fine tuning
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# The following guy used it these 3 functions as metric:
# https://www.kaggle.com/peterhurford/pets-lightgbm-baseline-with-all-the-data


# These 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)
df_train = pd.read_csv('../input/train/train.csv')
df_train.columns = ['type', 'name', 'age', 'breed1', 'breed2', 'gender', 'color1', 'color2',
       'color3', 'maturity_size', 'fur_length', 'vaccinated', 'dewormed',
       'sterilized', 'health', 'quantity', 'fee', 'state', 'rescuer_id',
       'video_amt', 'description', 'pet_id', 'photo_amt', 'adoption_speed']
df_train['photo_amt'] = pd.to_numeric(df_train['photo_amt'], downcast='integer')

## Shuffling
df_train = df_train.sample(frac=1)


chosen_columns = ['type', 'age', 'breed1', 'breed2', 'gender', 'color1', 'color2',
       'color3', 'maturity_size', 'fur_length', 'vaccinated', 'dewormed',
       'sterilized', 'health', 'quantity', 'fee', 'state',
       'video_amt', 'photo_amt']


## Splitting in features and targets
target   = df_train.adoption_speed
features = df_train[ chosen_columns  ]

## Splitting in train and valid
train_size      = int(len(features)*0.7)
features_train  = features[: train_size]
target_train    = target  [: train_size]
features_valid  = features[train_size :]
target_valid    = target  [train_size :]

features_train.head()
## It is interesting to know how XGBoost classifies each feature.
def print_feature_importance(clf) :
    sorted_idx = np.argsort(clf.feature_importances_)[::-1]
    importance = "Importance = ["
    for index in sorted_idx[:15] :
        importance += chosen_columns[index] + ","
        #print([features[index], clf.feature_importances_[index]])
    print(importance + "]")

## This function will be called several times. The parameters I'm tuning are max_depth, min_child_weight, subsample, and colsample_bytree    
def objective(space):
    clf = XGBClassifier(
        nthread          = 40,
        #n_estimators     = 10000,

        max_depth        = int(space['max_depth']),
        min_child_weight = space['min_child_weight'],
        subsample        = space['subsample'],
        colsample_bytree = space['colsample_bytree']
    )

    eval_set  = [( features_train, target_train), ( features_valid, target_valid)]
    clf.fit(features_train, target_train,
            eval_set=eval_set,
            eval_metric="merror",
            early_stopping_rounds=30,
            verbose = False
    )
    print_feature_importance(clf)
    
    prediction_train = clf.predict(features_train)
    prediction_valid = clf.predict(features_valid)
    
    kappa_train = quadratic_weighted_kappa(target_train, prediction_train)
    kappa_valid = quadratic_weighted_kappa(target_valid, prediction_valid)
    
    print("space: %s, Kappa Train: %.3f, Kappa Valid: %.3f" % (str(space), kappa_train, kappa_valid))
    print("")
    return{'loss':1-kappa_valid, 'status': STATUS_OK }
    
    
space ={
    #'max_depth'      : hp.quniform("max_depth", 5, 30, 1),
    'max_depth'       : 23,
    #'min_child_weight': hp.quniform('min_child_weight', 0, 100, 1),
    'min_child_weight': hp.quniform('min_child_weight', 20, 30, 1),
    #'subsample'       : hp.quniform('subsample', 0.1, 1, 0.1),
    'subsample'       : 1,
    #'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.1),
    'colsample_bytree': 0.6,
    }

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)
clf = XGBClassifier(
    subsample        = 0.8, 
    colsample_bytree = 0.6, 
    max_depth        = 22, 
    gamma            = 3.0, 
    min_child_weight = 52.0,    
    silent=True)

eval_set  = [( features_train, target_train), ( features_valid, target_valid)]

clf.fit(features_train, target_train,
        eval_set=eval_set,
        eval_metric="merror",
        early_stopping_rounds=30,
        verbose = False
)

## Predictions and evaluation
prediction_valid = clf.predict(features_valid)
accuracy = (target_valid == prediction_valid).mean()
kappa    = quadratic_weighted_kappa(target_valid, prediction_valid)
print(accuracy, kappa)
df_test = pd.read_csv('../input/test/test.csv')
df_test.columns = ['type', 'name', 'age', 'breed1', 'breed2', 'gender', 'color1', 'color2',
       'color3', 'maturity_size', 'fur_length', 'vaccinated', 'dewormed',
       'sterilized', 'health', 'quantity', 'fee', 'state', 'rescuer_id',
       'video_amt', 'description', 'pet_id', 'photo_amt']
features_test   = df_test[ chosen_columns ]
prediction_test = clf.predict(features_test)

submission = pd.DataFrame(
    { 
        'PetID'         : df_test.pet_id, 
        'AdoptionSpeed' : prediction_test
    }
)
## submission.set_index('PetID')
submission.to_csv('submission.csv',index=False)