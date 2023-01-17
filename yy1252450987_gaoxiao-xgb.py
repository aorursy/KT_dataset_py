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

import sklearn.metrics as metrics
import warnings
warnings.filterwarnings("ignore")
tr_x = pd.read_csv('../input/gaoxiao-feature-v6/tr_x.v6.csv')
tr_y = pd.read_csv('../input/gaoxiao-feature-v6/tr_y.v6.csv')
va_x = pd.read_csv('../input/gaoxiao-feature-v6/va_x.v6.csv')
va_y = pd.read_csv('../input/gaoxiao-feature-v6/va_y.v6.csv')
te_x = pd.read_csv('../input/gaoxiao-feature-v6/te_x.v6.csv')
te_uid = te_x.user_id
tr_x = tr_x.drop(['user_id'], axis=1)
te_x = te_x.drop(['user_id'], axis=1)
va_x = va_x.drop(['user_id'], axis=1)
## scikit-learn XGB.classfier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
model_default = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True,
                      objective='binary:logistic', booster='gbtree', n_jobs=-1,
                      gamma=0, min_child_weight=1, max_delta_step=0, 
                      subsample=1, colsample_bytree=1, colsample_bylevel=1, 
                      reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5,
                      random_state=1)
model_default.fit(tr_x, tr_y)
# feature importance
fimp = model_default.feature_importances_
sort_feature = tr_x.columns[np.argsort(-fimp)]
from sklearn.cross_validation import PredefinedSplit
tr_val_x = pd.concat((tr_x,va_x), axis=0)
tr_val_y = pd.concat((tr_y,va_y), axis=0)

test_fold = np.zeros(tr_val_x.shape[0])   # 将所有index初始化为0,0表示第一轮的验证集
test_fold[:tr_x.shape[0]] = -1            # 将训练集对应的index设为-1，表示永远不划分到验证集中
ps = PredefinedSplit(test_fold=test_fold)
model_pred = XGBClassifier(n_estimators= 300, 
                learning_rate= 0.1, 
                max_depth= 3, min_child_weight= 1, max_delta_step=0,
                subsample= 0.6, colsample_bytree= 1, colsample_bylevel=1,
                gamma= 0.4, reg_alpha= 0, reg_lambda= 1,objective='binary:logistic', silent=True,
                random_state=1, n_jobs=-1, booster='gbtree', scale_pos_weight=1, base_score=0.5)

trd_list = [0.44, 0.5, 0.48]
feature_num_list = [34, 45, 74]
for i in range(len(feature_num_list)):
    feature_num = feature_num_list[i]
    trd = trd_list[i]
    model_pred.fit(tr_val_x[sort_feature[:feature_num]], tr_val_y)
    xgb_preds = model_pred.predict_proba(te_x[sort_feature[:feature_num]])
    results = pd.DataFrame(te_uid)                          
    results['pred'] = xgb_preds[:,1]
    actuser = results[results.pred>trd].user_id.unique()                         
    np.savetxt('xgb.v6.'+'feature'+ str(feature_num) +'.predifined_cv.trd'+str(trd)+'.txt', actuser, fmt='%d')