import os

import sys

from pathlib import Path

from functools import partial

from typing import Optional, Union, Tuple



import numpy as np

import pandas as pd

import category_encoders as ce

from sklearn.model_selection import StratifiedKFold



import lightgbm as lgb
COMPETITION_NAME = "titanic"

ROOT = Path.cwd().parent

INPUT_ROOT = ROOT / "input"

RAW_DATA = INPUT_ROOT / COMPETITION_NAME

WORK_DIR = ROOT / "working"

# OUTPUT_ROOT = ROOT / "output"

OUTPUT_ROOT = WORK_DIR / "output"

PROC_DATA = ROOT / "processed_data"



ID = "PassengerId"

TARGET = "Survived"

RANDOM_SEED = 1086

N_FOLD = 5

NUM_THREADS = 4
for f in RAW_DATA.iterdir():

    print(f.name)
train = pd.read_csv(RAW_DATA / "train.csv")

test = pd.read_csv(RAW_DATA / "test.csv")

sample_sub = pd.read_csv(RAW_DATA / "gender_submission.csv")
def binary_accuracy_for_lgbm(

    preds: np.ndarray, data: lgb.Dataset, threshold: float=0.5,

) -> Tuple[str, float, bool]:

    """Calculate Binary Accuracy"""

    label = data.get_label()

    weight = data.get_weight()

    pred_label = (preds > threshold).astype(int)

    acc = np.average(label == pred_label, weights=weight)



    # # eval_name, eval_result, is_higher_better

    return 'my_bin_acc', acc, True
def binary_logloss_for_lgbm(preds: np.ndarray, data: lgb.Dataset) -> Tuple[str, float, bool]:

    """Calculate Binary Logloss"""

    label = data.get_label()

    weight = data.get_weight()

    p_dash = (1 - label) + (2 * label - 1) * preds

    loss_by_example = - np.log(p_dash)

    loss = np.average(loss_by_example, weights=weight)



    # # eval_name, eval_result, is_higher_better

    return 'my_lnloss', loss, False
NOT_FEATURES = [ID, TARGET]

UNUSE_COLUMNS = ["Name", "Ticket", "Cabin"]



test.insert(1, TARGET, -1)

n_train = len(train)

n_test = len(test)
all_data = pd.concat([train, test], axis=0, ignore_index=True)

all_data.head()
ord_enc = ce.OrdinalEncoder(cols=["Sex", "Embarked"])

all_data = ord_enc.fit_transform(all_data)

all_data.head()
X = all_data[

    list(filter(lambda x: x not in NOT_FEATURES + UNUSE_COLUMNS, all_data.columns))]

y = all_data[TARGET].values



X_tr_all, y_tr_all = X.iloc[:n_train].values, y[:n_train]

X_te, y_te = X.iloc[n_train:], y[n_train:]



kf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=RANDOM_SEED)

train_val_splits = list(kf.split(X_tr_all, y_tr_all))
# # In this notebook, use fold 0.

train_index, valid_index = train_val_splits[0]

X_tr, y_tr = X_tr_all[train_index], y_tr_all[train_index]

X_val, y_val = X_tr_all[valid_index], y_tr_all[valid_index]
MODEL_PARAMS_LGB = {

    "objective": "binary",  # <= set objective

    "eta": 0.01,

    "max_depth": -1,

    "seed": RANDOM_SEED,

    "num_threads": NUM_THREADS,

    "verbose": -1

}

FIT_PARAMS_LGB = {"num_boost_round": 10000, "early_stopping_rounds": 100, "verbose_eval":100,}



lgb_tr = lgb.Dataset(X_tr, y_tr)

lgb_val = lgb.Dataset(X_val, y_val)



model = lgb.train(

    MODEL_PARAMS_LGB, lgb_tr, **FIT_PARAMS_LGB,

    valid_names=['train', 'valid'], valid_sets=[lgb_tr, lgb_val],

    fobj=None,

    feval=binary_accuracy_for_lgbm  # <= set custom metric function

)
MODEL_PARAMS_LGB = {

    "objective": "binary",  # <= set objective

    "metric" : "None",  # <= set None by `string`

    "eta": 0.01,

    "max_depth": -1,

    "seed": RANDOM_SEED,

    "num_threads": NUM_THREADS,

    "verbose": -1

}   

FIT_PARAMS_LGB = {"num_boost_round": 10000, "early_stopping_rounds": 100, "verbose_eval":100}



lgb_tr = lgb.Dataset(X_tr, y_tr)

lgb_val = lgb.Dataset(X_val, y_val)



model = lgb.train(

    MODEL_PARAMS_LGB, lgb_tr, **FIT_PARAMS_LGB,

    valid_names=['train', 'valid'], valid_sets=[lgb_tr, lgb_val],

    fobj=None,

    feval=binary_accuracy_for_lgbm  # <= set custom metric function

)
MODEL_PARAMS_LGB = {

    "objective": "binary",  # <= set objective

    "first_metric_only": True,  # <= set first_metric_only

    "eta": 0.01,

    "max_depth": -1,

    "seed": RANDOM_SEED,

    "num_threads": NUM_THREADS,

    "verbose": -1

}   

FIT_PARAMS_LGB = {"num_boost_round": 10000, "early_stopping_rounds": 100, "verbose_eval":100}



lgb_tr = lgb.Dataset(X_tr, y_tr)

lgb_val = lgb.Dataset(X_val, y_val)



model = lgb.train(

    MODEL_PARAMS_LGB, lgb_tr, **FIT_PARAMS_LGB,

    valid_names=['train', 'valid'], valid_sets=[lgb_tr, lgb_val],

    fobj=None,

    feval=binary_accuracy_for_lgbm  # <= set custom metric function

)
MODEL_PARAMS_LGB = {

    "objective": "binary",  # <= set objective

    "first_metric_only": True,  # <= set first_metric_only

    "metric" : "None",  # <= set `None` by `string`

    "eta": 0.01,

    "max_depth": -1,

    "seed": RANDOM_SEED,

    "num_threads": NUM_THREADS,

    "verbose": -1

}   

FIT_PARAMS_LGB = {"num_boost_round": 10000, "early_stopping_rounds": 100, "verbose_eval":100}



lgb_tr = lgb.Dataset(X_tr, y_tr)

lgb_val = lgb.Dataset(X_val, y_val)



model = lgb.train(

    MODEL_PARAMS_LGB, lgb_tr, **FIT_PARAMS_LGB,

    valid_names=['train', 'valid'], valid_sets=[lgb_tr, lgb_val],

    fobj=None,

    feval=lambda preds, data : [  # <= set custom metric functions

        binary_accuracy_for_lgbm(preds, data),

        binary_logloss_for_lgbm(preds, data)]

)