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
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm_notebook as tqdm
df_train = pd.read_csv('../input/train.csv', index_col=0)

df_test = pd.read_csv('../input/test.csv', index_col=0)

df_spi = pd.read_csv('../input/spi.csv', index_col=0)

state_latlong = pd.read_csv('../input/statelatlong.csv')

state_gdp = pd.read_csv('../input/US_GDP_by_State.csv')
df_train = df_train.reset_index().merge(state_latlong.rename(columns={"State":"addr_state"})).set_index(df_train.index.names)

df_test = df_test.reset_index().merge(state_latlong.rename(columns={"State":"addr_state"})).set_index(df_test.index.names)



df_train.head()
state_gdp = state_gdp[state_gdp.year == 2015]

state_gdp.head()
df_train = df_train.reset_index().merge(state_gdp.rename(columns={"State":"City"})).set_index(df_train.index.names)

df_test = df_test.reset_index().merge(state_gdp.rename(columns={"State":"City"})).set_index(df_test.index.names)

df_train.head()
y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition','issue_d','City','year'], axis=1)

X_test = df_test.drop(['issue_d','City','year'], axis=1)
X_train.isnull().any()
X_train['annual_inc'].fillna(X_train['annual_inc'].median(),inplace=True)

X_train['dti'].fillna(X_train['dti'].median(),inplace=True)

X_train['delinq_2yrs'].fillna(X_train['delinq_2yrs'].median(),inplace=True)

X_train['inq_last_6mths'].fillna(X_train['inq_last_6mths'].median(),inplace=True)

X_train['open_acc'].fillna(X_train['open_acc'].median(),inplace=True)

X_train['pub_rec'].fillna(X_train['pub_rec'].median(),inplace=True)

X_train['revol_util'].fillna(X_train['revol_util'].median(),inplace=True)

X_train['total_acc'].fillna(X_train['total_acc'].median(),inplace=True)

X_train['collections_12_mths_ex_med'].fillna(X_train['collections_12_mths_ex_med'].median(),inplace=True)

X_train['acc_now_delinq'].fillna(X_train['acc_now_delinq'].median(),inplace=True)

X_train['mths_since_last_delinq'].fillna(X_train['mths_since_last_delinq'].median(),inplace=True)

X_train['mths_since_last_record'].fillna(X_train['mths_since_last_record'].median(),inplace=True)

X_train['mths_since_last_major_derog'].fillna(X_train['mths_since_last_major_derog'].median(),inplace=True)

X_train['tot_coll_amt'].fillna(X_train['tot_coll_amt'].median(),inplace=True)

X_train['tot_cur_bal'].fillna(X_train['tot_cur_bal'].median(),inplace=True)

X_train.isnull().any()
X_train['emp_title'].fillna(X_train['emp_title'].mode(),inplace=True)

X_train['emp_length'].fillna(X_train['emp_length'].mode(),inplace=True)

X_train['title'].fillna(X_train['title'].mode(),inplace=True)

X_train['earliest_cr_line'].fillna(X_train['earliest_cr_line'].mode(),inplace=True)

X_train.isnull().any()
X_test['mths_since_last_delinq'].fillna(X_test['mths_since_last_delinq'].median(),inplace=True)

X_test['mths_since_last_record'].fillna(X_test['mths_since_last_record'].median(),inplace=True)

X_test['mths_since_last_major_derog'].fillna(X_test['mths_since_last_major_derog'].median(),inplace=True)

X_test['tot_coll_amt'].fillna(X_test['tot_coll_amt'].median(),inplace=True)

X_test['tot_cur_bal'].fillna(X_test['tot_cur_bal'].median(),inplace=True)

X_test['dti'].fillna(X_test['dti'].median(),inplace=True)

X_test['inq_last_6mths'].fillna(X_test['inq_last_6mths'].median(),inplace=True)

X_test['revol_util'].fillna(X_test['revol_util'].median(),inplace=True)

X_test.isnull().any()
num_cols = []

for col in X_train.columns:

    if X_train[col].dtype == 'int64':

        num_cols.append(col)

        

        print(col, X_train[col].nunique())
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train[num_cols])
X_train[num_cols] = scaler.transform(X_train[num_cols])

X_test[num_cols] = scaler.transform(X_test[num_cols])
float_cols = []

for col in X_train.columns:

    if X_train[col].dtype == 'float64':

        float_cols.append(col)

        

        print(col, X_train[col].nunique())
scaler = StandardScaler()

scaler.fit(X_train[float_cols])
X_train[float_cols] = scaler.transform(X_train[float_cols])

X_test[float_cols] = scaler.transform(X_test[float_cols])
cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, X_train[col].nunique())
dummy = pd.get_dummies(X_train[['grade','sub_grade','emp_length','home_ownership','purpose',

          'initial_list_status','application_type','addr_state']])

X_train = pd.merge(X_train, dummy, left_index=True, right_index=True)

X_train = X_train.drop(columns=['grade','sub_grade','emp_length','home_ownership','purpose',

          'initial_list_status','application_type','addr_state'])

X_train.head(5)
cats.remove('grade')

cats.remove('sub_grade')

cats.remove('emp_length')

cats.remove('home_ownership')

cats.remove('purpose')

cats.remove('initial_list_status')

cats.remove('application_type')

cats.remove('addr_state')

cats.remove('emp_title')

cats
from category_encoders import OrdinalEncoder
oe = OrdinalEncoder(cols=cats, return_df=False)
X_train[cats] = oe.fit_transform(X_train[cats])

X_test[cats] = oe.transform(X_test[cats]) 
count = X_train.emp_title.value_counts()

X_train['emp_title_Count'] = X_train['emp_title'].map(count)

X_train = X_train.drop(columns=['emp_title'])

X_train.head(5)
X_train = X_train.drop(columns=['home_ownership_OTHER','home_ownership_NONE','purpose_educational','addr_state_IA'])
X_train['emp_title_Count'].fillna(X_train['emp_title_Count'].median(),inplace=True)

X_train.isnull().any()
X_test.head(5)
dummy_te = pd.get_dummies(X_test[['grade','sub_grade','emp_length','home_ownership','purpose',

          'initial_list_status','application_type','addr_state']])

X_test = pd.merge(X_test, dummy_te, left_index=True, right_index=True)

X_test = X_test.drop(columns=['grade','sub_grade','emp_length','home_ownership','purpose',

          'initial_list_status','application_type','addr_state'])

X_test.head(5)
count = X_test.emp_title.value_counts()

X_test['emp_title_Count'] = X_test['emp_title'].map(count)

X_test = X_test.drop(columns=['emp_title'])

X_test.head(5)
X_test['emp_title_Count'].fillna(X_test['emp_title_Count'].median(),inplace=True)

X_test.isnull().any()
print (X_train.columns.values)
print (X_test.columns.values)
%%time

# CVしてスコアを見てみる

# なお、そもそもStratifiedKFoldが適切なのかは別途考える必要があります

# 次回Build Modelの内容ですが、是非各自検討してみてください

scores = []



from sklearn.model_selection import KFold



kf = KFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in enumerate(tqdm(kf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

    import lightgbm as lgb

    from lightgbm import LGBMClassifier

    

    import matplotlib.pyplot as plt

    plt.style.use('ggplot')

    %matplotlib inline

    

    clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.71,

                         importance_type='split', learning_rate=0.05, max_depth=6,

                         min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                         n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,

                         random_state=71, reg_alpha=1.0, reg_lambda=1.0, silent=True,

                         subsample=0.8, subsample_for_bin=200000, subsample_freq=0)

    

    clf.fit(X_train, y_train, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])

    

    y_pred = clf.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred)

    print(score)

    

    fig, ax = plt.subplots(figsize=(10, 15))

    lgb.plot_importance(clf, max_num_features=50, ax=ax, importance_type='gain') # 変数重要をプロット
# 全データで再学習し、testに対して予測する

clf.fit(X_train, y_train)



y_pred = clf.predict_proba(X_test)[:,1] # predict_probaで確率を出力する
# sample submissionを読み込んで、予測値を代入の後、保存する

submission = pd.read_csv('../input/sample_submission.csv', index_col=0)



submission.loan_condition = y_pred

submission.to_csv('submission.csv')