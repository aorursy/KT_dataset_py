import pandas as pd

import numpy as np

import xgboost as xgb

from sklearn.metrics import accuracy_score, roc_auc_score



if __name__ == '__main__':

    train = pd.read_csv('../input/lecture-3-data/train.csv')

    test = pd.read_csv('../input/lecture-3-data/test.csv')

    features = ['f1', 'f2', 'f3']



    params_xgb = {

        "objective": "binary:logistic",

        "eval_metric": 'logloss',

        "eta": 0.05,

        "max_depth": 2,

        "subsample": 0.8,

        "colsample_bytree": 0.8,

    }

    num_boost_round = 70

    early_stopping_rounds = 10



    dtrain = xgb.DMatrix(train[features].values, train['target'].values)

    dvalid = xgb.DMatrix(test[features].values, test['target'].values)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

    gbm = xgb.train(params_xgb, dtrain, num_boost_round, evals=watchlist,

                    early_stopping_rounds=early_stopping_rounds, verbose_eval=10)



    pred = gbm.predict(xgb.DMatrix(test[features].values), ntree_limit=gbm.best_iteration + 1)

    accuracy = accuracy_score(test['target'].values, np.round(pred))

    auc = roc_auc_score(test['target'].values, pred)

    print('Accuracy: {:.2f} %, ROC AUC: {:.2f}'.format(100*accuracy, auc))