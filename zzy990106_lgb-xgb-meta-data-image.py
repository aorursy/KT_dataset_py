import numpy as np

import pandas as pd
train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

sample = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')

train_pred = pd.read_csv('../input/trainy/1.csv')

test_pred = pd.read_csv('../input/pytorch-5-fold-efficientnet-baseline/submission.csv')
train['sex'] = train['sex'].fillna('na')

train['age_approx'] = train['age_approx'].fillna(0)

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna('na')



test['sex'] = test['sex'].fillna('na')

test['age_approx'] = test['age_approx'].fillna(0)

test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna('na')
train['y'] = train_pred['target']

test['y'] = test_pred['target']
train.head()
test.head()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()



train.sex = le.fit_transform(train.sex)

train.anatom_site_general_challenge = le.fit_transform(train.anatom_site_general_challenge)

test.sex = le.fit_transform(test.sex)

test.anatom_site_general_challenge = le.fit_transform(test.anatom_site_general_challenge)
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score



import lightgbm as lgb

import catboost as ctb



model1 = lgb.LGBMRegressor()

model2 = ctb.CatBoostRegressor(eval_metric='AUC')
feature_names = ['sex','age_approx','anatom_site_general_challenge','y']

ycol = ['target']
test['target'] = 0



kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(train[feature_names], train[ycol])):

    X_train = train.iloc[trn_idx][feature_names]

    Y_train = train.iloc[trn_idx][ycol]



    X_val = train.iloc[val_idx][feature_names]

    Y_val = train.iloc[val_idx][ycol]



    print('\nFold_{} Training ================================\n'.format(fold_id+1))



    lgb_model = model1.fit(X_train,

                          Y_train,

                          eval_names=['train', 'valid'],

                          eval_set=[(X_train, Y_train), (X_val, Y_val)],

                          verbose=100,

                          eval_metric='auc',

                          early_stopping_rounds=100)



    pred_test1 = lgb_model.predict(

        test[feature_names], num_iteration=lgb_model.best_iteration_)

    

    ctb_model = model2.fit(X_train,

                          Y_train,

                          eval_set=[(X_train, Y_train), (X_val, Y_val)],

                          verbose=100,

                          early_stopping_rounds=100)



    pred_test2 = ctb_model.predict(test[feature_names])

    test['target'] += (pred_test1+pred_test2) / kfold.n_splits
test
sample.target = test.target
sample.to_csv('submission.csv',index=False)