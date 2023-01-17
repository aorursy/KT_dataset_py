%ls ../input/
%ls ../input/otto-simple-xgb/
%ls ../input/otto-simple-lgb/
%ls ../input/otto-neuralnetwork/
%ls ../input/otto-stacking-level0/
import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss

from pathlib import Path



INPUT_PATH = Path("../input/otto-group-product-classification-challenge/")

STACKING_LEVEL0_INPUT_PATH = Path("../input/otto-stacking-level0/")

XGB_PATH = Path("../input/otto-simple-xgb/")

LGB_PATH = Path("../input/otto-simple-lgb/")

NN_PATH = Path("../input/otto-neuralnetwork/")
train = pd.read_csv(INPUT_PATH / "train.csv")

test = pd.read_csv(INPUT_PATH / "test.csv")



train.shape, test.shape
drop_cols = ["id"]

target_col = "target"

target = train[target_col]

feat_cols = [col for col in train.columns if col not in drop_cols + [target_col]]



train[target_col] = train[target_col].str.replace('Class_', '')

train[target_col] = train[target_col].astype(int) - 1

target = train[target_col]



train.drop(columns=drop_cols + [target_col], inplace=True)

test.drop(columns=drop_cols, inplace=True)
keras_oof = np.load(NN_PATH / "keras_oof.npy")

lgb_oof = np.load(LGB_PATH / "lgb_oof.npy")

xgb_oof = np.load(XGB_PATH / "xgb_oof.npy")



keras_oof.shape, lgb_oof.shape, xgb_oof.shape
keras_test = pd.read_csv(NN_PATH / "submit.csv").drop(columns=["id"])

lgb_test = pd.read_csv(LGB_PATH / "submit.csv").drop(columns=["id"])

xgb_test = pd.read_csv(XGB_PATH / "submit_xgboost.csv").drop(columns=["id"])



keras_test.shape, lgb_test.shape, xgb_test.shape
train = np.concatenate([keras_oof, lgb_oof, xgb_oof], axis=1)

test = np.concatenate([keras_test, lgb_test, xgb_test], axis=1)



train.shape, test.shape
# train setting

NFOLDS = 5

RANDOM_STATE = 871972



# excluded_column = ['target', 'id']

# cols = [col for col in train.columns if col not in excluded_column]



folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, 

                        random_state=RANDOM_STATE)



# parameter calculated by LGBtuner

params = {

    'metric':'multi_logloss',

    'objective': 'multiclass',

    'num_class': 9,

    'verbosity': 1,

}
y_pred_test = np.zeros((len(test), 9))

oof = np.zeros((len(train), 9))

score = 0



for fold_n, (train_index, valid_index) in enumerate(folds.split(train, y=target)):

    print('Fold', fold_n)

    X_train, X_valid = train[train_index], train[valid_index]

    y_train, y_valid = target.loc[train_index].astype(int), target.loc[valid_index].astype(int)

    

    train_data = lgb.Dataset(X_train, label=y_train)

    valid_data = lgb.Dataset(X_valid, label=y_valid)



    lgb_model = lgb.train(params,train_data,num_boost_round=30000,

                          valid_sets=[train_data, valid_data],

                          verbose_eval=300,early_stopping_rounds=300)

    

    y_pred_valid = lgb_model.predict(X_valid,

                                     num_iteration=lgb_model.best_iteration)

    oof[valid_index] = y_pred_valid

    score += log_loss(y_valid, y_pred_valid)

    

    y_pred_test += lgb_model.predict(test, num_iteration=lgb_model.best_iteration)/NFOLDS

print('valid logloss average:', score/NFOLDS, log_loss(target, oof))
sample_submit = pd.read_csv(INPUT_PATH / "sampleSubmission.csv")

submit = pd.concat([sample_submit[['id']], pd.DataFrame(y_pred_test)], axis=1)

submit.columns = sample_submit.columns

submit.to_csv('submit.csv', index=False)