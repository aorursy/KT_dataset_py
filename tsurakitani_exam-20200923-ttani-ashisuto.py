import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import quantile_transform
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm.notebook import tqdm as tqdm

from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor

import sys
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

input_dir_path = '/kaggle/input/exam-for-students20200923/'

df_train = pd.read_csv(input_dir_path + 'train.csv', index_col=0, low_memory=False)
df_test = pd.read_csv(input_dir_path + 'test.csv', index_col=0, low_memory=False)

X_train = df_train.copy()
y_train = X_train['ConvertedSalary']

#y_train = np.log(y_train + 1)

X_train = X_train.drop(['ConvertedSalary'], axis=1)

X_test = df_test.copy()

#X_train.head()
y_train.head()
# 使用する数値型特徴量
col_num = ['AssessJob5', 'JobContactPriorities2', 'AssessJob6', 'AssessJob4']

# 試用するカテゴリ特徴量
col_cat = [
    'Country', 
    'SalaryType', 
    'Employment', 
    'Age',
    'LastNewJob',
    'MilitaryUS',
    'HopeFiveYears', 
    'CompanySize',
    'RaceEthnicity',
    'EducationParents',
    'AdsActions',
    'OperatingSystem',
    'AIDangerous',
    'StackOverflowJobsRecommend',
    'AdsAgreeDisagree2',
    'TimeFullyProductive'
]

# カテゴリに変換
X_train[col_cat].fillna('nothing', axis=0, inplace=True)
X_test[col_cat].fillna('nothing', axis=0, inplace=True)
ce_oe = OrdinalEncoder(cols=col_cat,handle_unknown='impute')

col_exec = []
col_exec.extend(col_num)
col_exec.extend(col_cat)

X_train = X_train[col_exec]
X_test = X_test[col_exec]

X_train = ce_oe.fit_transform(X_train)
X_test = ce_oe.fit_transform(X_test)

X_train.fillna(0, axis=0, inplace=True)
X_test.fillna(0, axis=0, inplace=True)

X_train['Country'] = X_train['Country'] * X_train['Country']
X_train['SalaryType'] = X_train['SalaryType'] * X_train['SalaryType']
X_train['Employment'] = X_train['Employment'] * X_train['Employment']
X_train['Age'] = X_train['Age'] * X_train['Age']
X_train['LastNewJob'] = X_train['LastNewJob'] * X_train['LastNewJob']

X_test['Country'] = X_test['Country'] * X_test['Country']
X_test['SalaryType'] = X_test['SalaryType'] * X_test['SalaryType']
X_test['Employment'] = X_test['Employment'] * X_test['Employment']
X_test['Age'] = X_test['Age'] * X_test['Age']
X_test['LastNewJob'] = X_test['LastNewJob'] * X_test['LastNewJob']

#df_session_ce_ordinal.head()
'''
X_train_, X_val, y_train_, y_val= train_test_split(X_train, y_train, test_size=0.05, random_state=71)

scores = []

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

#clf = LGBMClassifier(
#        max_depth=3,
#        learning_rate = 0.02,
#        colsample_bytree=0.7,
#        subsample=0.7,
#        min_split_gain=0,
#        reg_lambda=1,
#        reg_alpha=1,
#        min_child_weight=2,
#        n_estimators=9999,
#        random_state=71,
#        importance_type='gain'
#)

clf = LGBMRegressor(colsample_bytree=0.8715615575648012,
              learning_rate=0.0614629238029019,
              min_child_samples=4681.205662045211,
                n_estimators=9999,in_child_weight=1678.3084865745946, min_data_in_leaf=311,
                num_leaves=92, random_seed=42, subsample=0.9230819000143395)

clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='rmse', eval_set=[(X_val, y_val)])
y_pred = clf.predict(X_test)
'''

#CV Averaging/kFold Averaging
scores = []
lgb_y_pred_train = np.zeros(len(X_train))
lgb_y_pred_test = np.zeros(len(X_test))
skf = StratifiedKFold(n_splits=5, random_state=81, shuffle=True)

y_pred_arr = []

for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):
    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]
    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]
    
    
    #clf = LGBMClassifier(
    #    max_depth=3,
    #    learning_rate = 0.02,
    #    colsample_bytree=0.7,
    #    subsample=0.7,
    #    min_split_gain=0,
    #    reg_lambda=1,
    #    reg_alpha=1,
    #    min_child_weight=2,
    #    n_estimators=9999,
    #    random_state=71,
    #    importance_type='gain'
    #)
    clf = LGBMRegressor(
        colsample_bytree=0.8715615575648012,
        learning_rate=0.0614629238029019,
        min_child_samples=4681.205662045211,
        n_estimators=9999,
        min_child_weight=2, 
        min_data_in_leaf=311,
        num_leaves=92, 
        random_seed=42, 
        subsample=0.9230819000143395
    )
    
    y_train_ = np.log1p(y_train_)
    y_val = np.log1p(y_val)
    
    clf.fit(X_train_, y_train_,
            early_stopping_rounds=20,
            verbose=100,
            eval_metric='rmse',
            eval_set=[(X_val, y_val)]
           )
    
    #y_pred = clf.predict_proba(X_val)[:,1]
    #lgb_y_pred_train[test_ix] = y_pred
    #score = roc_auc_score(y_val, y_pred)
    #scores.append(score)
    #lgb_y_pred_test += clf.predict(X_test)[:,1]
    y_pred_arr.append(clf.predict(X_test))
    
    #print('CV Score of Fold_%d is %f' % (i, score))
#lgb_y_pred_test /= 5

y_pred = sum(y_pred_arr) / len(y_pred_arr)
print(y_pred)

# 提出用CSV作成

submission = pd.read_csv(input_dir_path + 'sample_submission.csv', index_col=0)

submission.ConvertedSalary = np.exp(y_pred) - 1
submission['ConvertedSalary'] = submission['ConvertedSalary'].round().astype(int)
submission.to_csv('mySubmission.csv')

submission.head()
