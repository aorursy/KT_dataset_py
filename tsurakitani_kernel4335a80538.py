import numpy as np

import scipy as sp

import pandas as pd

import os

import category_encoders as ce

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from pandas import DataFrame, Series

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier



import lightgbm as lgb

from lightgbm import LGBMClassifier, LGBMRegressor



#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



#df_train = pd.read_csv('../input/homework-for-students4plus/train_small.csv', index_col=0)

df_train = pd.read_csv('../input/homework-for-students4plus/train.csv', index_col=0)

df_test = pd.read_csv('../input/homework-for-students4plus/test.csv', index_col=0)



#X_train = df_train

X_train = df_train.query('issue_d >= "Jan-2015"')

y_train = X_train.loan_condition

X_train = X_train.drop(['loan_condition'], axis=1)



X_test = df_test
# ひとまず欠損情報列だけ追加してから'emp_title'を削除

TXT_train = X_train.emp_title.copy()

TXT_test = X_test.emp_title.copy()

X_train['emp_title_null'] = X_train['emp_title'].isnull().replace({True:1}).replace({False:0})

X_test['emp_title_null'] = X_test['emp_title'].isnull().replace({True:1}).replace({False:0})

X_train.drop(['emp_title'], axis=1, inplace=True)

X_test.drop(['emp_title'], axis=1, inplace=True)



# 適当に合成したり分割したりして列追加

X_train['issue_d_month'] = X_train['issue_d'].str.split('-', expand=True)[0]

X_test['issue_d_month'] = X_train['issue_d'].str.split('-', expand=True)[0]

X_train['address'] = X_train['zip_code'] + X_train['addr_state']

X_test['address'] = X_test['zip_code'] + X_test['addr_state']

X_train['grade_add'] = X_train['grade'] + X_train['sub_grade']

X_test['grade_add'] = X_test['grade'] + X_test['sub_grade']





# なんか関係なさそうなの削除

#X_train.drop(['issue_d'], axis=1, inplace=True)

#X_test.drop(['issue_d'], axis=1, inplace=True)

#X_train.drop(['addr_state'], axis=1, inplace=True)

#X_test.drop(['addr_state'], axis=1, inplace=True)

#X_train.drop(['zip_code'], axis=1, inplace=True)

#X_test.drop(['zip_code'], axis=1, inplace=True)



X_train.drop(['acc_now_delinq'], axis=1, inplace=True)

X_test.drop(['acc_now_delinq'], axis=1, inplace=True)

X_train.drop(['application_type'], axis=1, inplace=True)

X_test.drop(['application_type'], axis=1, inplace=True)

#X_train.drop(['pub_rec'], axis=1, inplace=True)

#X_test.drop(['pub_rec'], axis=1, inplace=True)

#X_train.drop(['initial_list_status'], axis=1, inplace=True)

#X_test.drop(['initial_list_status'], axis=1, inplace=True)

#X_train.drop(['collections_12_mths_ex_med'], axis=1, inplace=True)

#X_test.drop(['collections_12_mths_ex_med'], axis=1, inplace=True)

#X_train.drop(['delinq_2yrs'], axis=1, inplace=True)

#X_test.drop(['delinq_2yrs'], axis=1, inplace=True)



X_train['null_count'] = 0

X_test['null_count'] = 0

for col in X_train.columns:

    # 欠損フラグ用列の追加

    if X_train[col].isnull().any():

        X_train[col + '_null'] = X_train[col].isnull().replace({True:1}).replace({False:0})

        X_test[col + '_null'] = X_test[col].isnull().replace({True:1}).replace({False:0})

        X_train['null_count'] += X_train[col].isnull().replace({True:1}).replace({False:0})

        X_test['null_count'] += X_test[col].isnull().replace({True:1}).replace({False:0})



cats = []

for col in X_train.columns:

    # カテゴリ指定カラム

    if X_train[col].dtype == 'object':

        cats.append(col)
# 標準化列追加, いる？

scaler = StandardScaler()

num_col = ['dti', 'annual_inc', 'installment', 'tot_cur_bal', 'loan_amnt']

for col in num_col:

    X_train[col+'_SS'] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))

    X_test[col+'_SS'] = scaler.fit_transform(X_test[col].values.reshape(-1, 1))



# 欠損値の穴埋め

X_train.fillna(X_train.median(), inplace=True)

X_test.fillna(X_train.median(), inplace=True)

#X_train.fillna(0, inplace=True)

#X_test.fillna(0, inplace=True)

# Labelエンコーディングいる？

ce_ohe = ce.OrdinalEncoder(cols=cats,handle_unknown='impute')

X_train_Label = ce_ohe.fit_transform(X_train)

X_test_Label = ce_ohe.fit_transform(X_test)



for col in cats:

    X_train[col+'_Label'] = X_train_Label[col]

    X_test[col+'_Label'] = X_test_Label[col]



# カテゴリのTARGETエンコード

target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)



for col in cats:



    # X_testはX_trainでエンコーディングする

    summary = X_temp.groupby([col])[target].mean()

    X_test[col] = X_test[col].map(summary) 





    # X_trainのカテゴリ変数をoofでエンコーディングする

    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

    enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



    for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



        summary = X_train_.groupby([col])[target].mean()

        enc_train.iloc[val_ix] = X_val[col].map(summary)

        

    X_train[col]  = enc_train
scaler = StandardScaler()

num_col = ['grade', 'sub_grade', 'grade_add']

for col in num_col:

    X_train[col+'_SS'] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))

    X_test[col+'_SS'] = scaler.fit_transform(X_test[col].values.reshape(-1, 1))



X_train.fillna(X_train.mean(), axis=0, inplace=True)

X_test.fillna(X_train.mean(), axis=0, inplace=True)



X_train.head()
### こっからLightGBM



'''

clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

                    importance_type='split', learning_rate=0.05, max_depth=-1,

                    min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                    n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

                    random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

                    subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

X_train_, X_val, y_train_, y_val= train_test_split(X_train, y_train, test_size=0.1, random_state=42)

clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])

'''

'''

clf = LGBMRegressor(colsample_bytree=0.8715615575648012,

              learning_rate=0.0614629238029019,

              min_child_samples=4681.205662045211,

              n_estimators=9999,

              min_child_weight=1678.3084865745946, min_data_in_leaf=311,

              num_leaves=92, random_seed=42, subsample=0.9230819000143395)

    

X_train_, X_val, y_train_, y_val= train_test_split(X_train, y_train, test_size=0.1, random_state=42)

clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])

'''



y_pred_arr = []





for i in range(50):

    clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

                                importance_type='split', learning_rate=0.05, max_depth=-1,

                                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                                n_estimators=444, n_jobs=-1, num_leaves=15, objective=None,

                                random_state=71+i*1, reg_alpha=0.0, reg_lambda=0.0, silent=True,

                                subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

    #X_train_, X_val, y_train_, y_val= train_test_split(X_train, y_train, test_size=0.1, random_state=71+i*5)

    clf.fit(X_train, y_train)

    

    y_pred_arr.append(clf.predict_proba(X_test)[:,1])



'''

for k in range(10):

    clf = LGBMRegressor(colsample_bytree=0.8715615575648012,

              learning_rate=0.0614629238029019,

              min_child_samples=4681.205662045211,

              n_estimators=222,

              min_child_weight=1678.3084865745946, min_data_in_leaf=311,

              num_leaves=92, random_seed=71+k*10, subsample=0.9230819000143395)

    #X_train_, X_val, y_train_, y_val= train_test_split(X_train, y_train, test_size=0.1, random_state=71+k*10)

    clf.fit(X_train, y_train)

    

    y_pred_arr.append(clf.predict(X_test))

'''
'''

scores = []



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

    

    #clf = GradientBoostingClassifier()

    clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=7,min_samples_split=800, min_samples_leaf=60, subsample=0.8, random_state=10, max_features=13)

    clf.fit(X_train_, y_train_)

    y_pred = clf.predict_proba(X_val)[:,1]

    

    #lr = LogisticRegression(max_iter=300)

    #lr.fit(X_train_, y_train_)

    #y_pred = lr.predict_proba(X_val)[:,1]

    

    score = roc_auc_score(y_val, y_pred)

    scores.append(score)

    

    print('CV Score of Fold_%d is %f' % (i, score))

'''
# テキスト処理

#TXT_train.fillna('#', inplace=True)

#TXT_test.fillna('#', inplace=True)

#tfidf = TfidfVectorizer(max_features=1000, use_idf=True)

#TXT_train_enc = tfidf.fit_transform(TXT_train)

#TXT_test_enc = tfidf.transform(TXT_test)

#X_train = sp.sparse.hstack([X_train.values, TXT_train_enc])

#X_test = sp.sparse.hstack([X_test.values, TXT_test_enc])
# とりあえずパラメーターは適当

#clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=7,min_samples_split=800, min_samples_leaf=60, subsample=0.8, random_state=10, max_features=13)

#clf.fit(X_train, y_train)



#y_pred_arr.append(clf.predict_proba(X_test)[:,1])



#clf = RandomForestClassifier(max_depth=7, n_estimators=100)

#clf.fit(X_train, y_train)

#print(clf.predict_proba(X_test)[:,1])

#y_pred_arr.append(clf.predict_proba(X_test)[:,1])
y_pred = sum(y_pred_arr) / len(y_pred_arr)

print(y_pred)



submission = pd.read_csv('../input/homework-for-students4plus/sample_submission.csv', index_col=0)



submission.loan_condition = y_pred

submission.to_csv('submission.csv')