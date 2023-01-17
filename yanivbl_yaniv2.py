import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from xgboost import XGBClassifier

from sklearn.model_selection import KFold

from category_encoders import CountEncoder

from sklearn.pipeline import Pipeline

from sklearn.metrics import log_loss



import matplotlib.pyplot as plt

from sklearn.multioutput import MultiOutputClassifier

import pickle

import os

import warnings

warnings.filterwarnings('ignore')



train  = pd.read_csv('../input/lish-moa/train_features.csv')

test =  pd.read_csv('../input/lish-moa/test_features.csv')

targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

submission = pd.read_csv('../input/lish-moa/sample_submission.csv')



def log_loss_metric(y_true, y_pred):

    y_pred_clip = np.clip(y_pred, 1e-15, 1 - 1e-15)

    loss = - np.mean(np.mean(y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip), axis = 1))

    return loss
#Remove used ID

X_t = train.drop('sig_id',axis=1)

X_v = test.drop('sig_id',axis=1)

y_t = targets.drop('sig_id',axis=1)



#Encoding

X_t.cp_type = X_t.cp_type.apply(lambda x: 1 if x=='trt_cp' else 0)

X_v.cp_type = X_v.cp_type.apply(lambda x: 1 if x=='trt_cp' else 0)



X_t.cp_dose = X_t.cp_dose.apply(lambda x: 1 if x=='D1' else 0)

X_v.cp_dose = X_v.cp_dose.apply(lambda x: 1 if x=='D1' else 0)



X_t.cp_time = X_t.cp_time/72

X_v.cp_time = X_v.cp_time/72



from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression
#List of logistic models

log_lst=[]

for i in range(y_t.shape[1]):

    print('calc column'+str(i))

    log_lst.append(LogisticRegression().fit(X_t,y_t.iloc[:,i]))

log_preds=[]

for i in range(y_t.shape[1]):

    print ('predict column '+str(i))

    log_preds.append(log_lst[i].predict_proba(X_t)[:,1])

log_preds_mat = np.array(log_preds).T
log_preds_v=[]

for i in range(y_t.shape[1]):

    print ('predict column '+str(i))

    log_preds_v.append(log_lst[i].predict_proba(X_v)[:,1])



log_preds_v_mat = np.array(log_preds_v).T

submission.iloc[:,1:] = log_preds_v_mat

submission.set_index('sig_id').to_csv('submission.csv')