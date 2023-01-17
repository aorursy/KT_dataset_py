import gc

import numpy as np

import pandas as pd

import xgboost as xgb

# import optuna.integration.lightgbm as lgb

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss

from pathlib import Path



INPUT_PATH = Path("../input/")
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
# train setting

NFOLDS = 2

RANDOM_STATE = 871972



excluded_column = ['target', 'id']

cols = [col for col in train.columns if col not in excluded_column]



folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, 

                        random_state=RANDOM_STATE)



params = {"objective": "multi:softprob",

          "eval_metric":"mlogloss",

          "num_class": 9

}
y_pred_test = np.zeros((len(test), 9))

oof = np.zeros((len(train), 9))

score = 0



for fold_n, (train_index, valid_index) in enumerate(folds.split(train, y=target)):

    print('Fold', fold_n)

    X_train, X_valid = train.iloc[train_index], train.iloc[valid_index]

    y_train, y_valid = target.loc[train_index].astype(int), target.loc[valid_index].astype(int)

    

    train_data = xgb.DMatrix(X_train[cols], label=y_train)

    valid_data = xgb.DMatrix(X_valid[cols], label=y_valid)



    model = xgb.train(params,

                      train_data,num_boost_round=30000,

                      evals=[(valid_data, "valid")],

                      verbose_eval=4,

                      maximize=False,

                      early_stopping_rounds=300)

    

    y_pred_valid = model.predict(xgb.DMatrix(X_valid[cols]))

    oof[valid_index] = y_pred_valid

    score += log_loss(y_valid, y_pred_valid)

    

    y_pred_test += model.predict(xgb.DMatrix(test[cols]))/NFOLDS

print('valid logloss average:', score/NFOLDS, log_loss(target, oof))
sample_submit = pd.read_csv("../input/sampleSubmission.csv")

submit = pd.concat([sample_submit[['id']], pd.DataFrame(y_pred_test)], axis = 1)

submit.columns = sample_submit.columns

submit.to_csv('submit_xgboost.csv', index=False)
np.save("xgb_oof.npy", oof)