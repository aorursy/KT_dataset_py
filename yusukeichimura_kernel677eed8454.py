# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import gc

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error, log_loss

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from sklearn.preprocessing import LabelEncoder



from tqdm import tqdm_notebook as tqdm

from category_encoders import OrdinalEncoder



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



import lightgbm as lgb

from lightgbm import LGBMClassifier

import xgboost as xgb


df_train = pd.read_csv('../input/homework-for-students2/train.csv', index_col=0, parse_dates=['issue_d'])

df_test = pd.read_csv('../input/homework-for-students2/test.csv', index_col=0, parse_dates=['issue_d'])



#df_train = pd.read_csv('../input/homework-for-students2/train.csv', index_col=0, parse_dates=['issue_d'], skiprows=lambda x: x%20!=0)

#df_test = pd.read_csv('../input/homework-for-students2/test.csv', index_col=0, parse_dates=['issue_d'], skiprows=lambda x: x%20!=0)



GDP = pd.read_csv('../input/homework-for-students2/US_GDP_by_State.csv')

SL = pd.read_csv('../input/homework-for-students2/statelatlong.csv')
df_train['total_acc']
# 年月情報追加

df_train['year'] = df_train.issue_d.dt.year

df_train['month'] = df_train.issue_d.dt.month

df_test['year'] = df_test.issue_d.dt.year

df_test['month'] = df_test.issue_d.dt.month
# spiマージ

df_spi = pd.read_csv('../input/homework-for-students2/spi.csv', parse_dates=['date'])

df_spi['year'] = df_spi.date.dt.year

df_spi['month'] = df_spi.date.dt.month



df_temp = df_spi.groupby(['year', 'month'], as_index=False)['close'].mean() # 年月で GroupBy 平均

df_train = df_train.merge(df_temp, on=['year', 'month'], how='left')

df_test = df_test.merge(df_temp, on=['year', 'month'], how='left')
#SLをmarge

SL = SL.rename(columns={'State':'addr_state'})

df_train = pd.merge(df_train, SL, how='left')

df_test = pd.merge(df_test, SL, how='left')
#GDPをmarge

GDP = GDP.rename(columns={'State':'City'})

total_GDP = GDP.drop('year',axis=1)

total_GDP = total_GDP.groupby(['City']).mean()

total_GDP = total_GDP.reset_index()

#total_GDP.head()

df_train = pd.merge(df_train, total_GDP, how='left')

df_test = pd.merge(df_test,total_GDP, how='left')
#df_train['Debtfunds_month']を追加

df_train['Debtfunds_month'] = ((df_train['annual_inc']/12)*df_train['dti'])

df_test['Debtfunds_month'] = ((df_test['annual_inc']/12)*df_train['dti'])
df_train['Debtfunds_month']
#df_train['DIFacc'] = df_train['total_acc']-df_train['open_acc']

#df_train['DIVacc'] = df_train['total_acc']/df_train['open_acc']



#df_test['DIFacc'] = df_train['total_acc']-df_train['open_acc']

#df_test['DIVacc'] = df_train['total_acc']/df_train['open_acc']
#df_train['DIFacc']
#df_train['revo_acc'] = df_train['total_acc']*df_train['revol_util']

#df_test['revo_acc'] = df_test['total_acc']*df_test['revol_util']
#df_train['revol_util']
#BadFlag

#借金整理

df_train.loc[df_train['title'].astype('str').str.contains('Debt consolidation'), 'title_badflg'] = 1

df_test.loc[df_test['title'].astype('str').str.contains('Debt consolidation'), 'title_badflg'] = 1

#クレジットカード

df_train.loc[df_train['title'].astype('str').str.contains('Credit card refinancing'), 'title_badflg'] = 1

df_test.loc[df_test['title'].astype('str').str.contains('Credit card refinancing'), 'title_badflg'] = 1



#借金返済

df_train.loc[df_train['purpose'].astype('str').str.contains('debt_consolidation'), 'purpose_badflg'] = 1

df_test.loc[df_test['purpose'].astype('str').str.contains('debt_consolidation'), 'purpose_badflg'] = 1

#クレジットカード

df_train.loc[df_train['purpose'].astype('str').str.contains('credit_card'), 'purpose_badflg'] = 1

df_test.loc[df_test['purpose'].astype('str').str.contains('credit_card'), 'purpose_badflg'] = 1



df_train['emp_title'] = df_train['emp_title'].fillna('Unemployed')

df_test['emp_title'] = df_train['emp_title'].fillna('Unemployed')

df_train.loc[df_train['emp_title'].astype('str').str.contains('Unemployed'), 'emp_title_badflg'] = 1

df_test.loc[df_test['emp_title'].astype('str').str.contains('Unemployed'), 'emp_title_badflg'] = 1
#GoodFlag

#大臣

#df_train.loc[df_train['emp_title'].astype('str').str.contains('Minister'), 'emp_title_goodflg'] = '1'

#df_test.loc[df_test['emp_title'].astype('str').str.contains('Minister'), 'emp_title_goodflg'] = '1'

#コンサルタント

#df_train.loc[df_train['emp_title'].astype('str').str.contains('Consultant'), 'emp_title_goodflg'] = '1'

#df_test.loc[df_test['emp_title'].astype('str').str.contains('Consultant'), 'emp_title_goodflg'] = '1'

#金融アナリスト

#df_train.loc[df_train['emp_title'].astype('str').str.contains('Financial Analyst'), 'emp_title_goodflg'] = '1'

#df_test.loc[df_test['emp_title'].astype('str').str.contains('Financial Analyst'), 'emp_title_goodflg'] = '1'

#会計士

#df_train.loc[df_train['emp_title'].astype('str').str.contains('Accountant'), 'emp_title_goodflg'] = '1'

#df_test.loc[df_test['emp_title'].astype('str').str.contains('Accountant'), 'emp_title_goodflg'] = '1'

#ディレクター

#df_train.loc[df_train['emp_title'].astype('str').str.contains('Director'), 'emp_title_goodflg'] = '1'

#df_test.loc[df_test['emp_title'].astype('str').str.contains('Director'), 'emp_title_goodflg'] = '1'

#

#df_train.loc[df_train['emp_title'].astype('str').str.contains('Manager'), 'emp_title_goodflg'] = '1'

#df_train.loc[df_train['emp_title'].astype('str').str.contains('Manager'), 'emp_title_goodflg'] = '1'
#GoodFlag

#大臣

df_train.loc[df_train['emp_title'].astype('str').str.contains('Minister'), 'Minister'] = 1

df_test.loc[df_test['emp_title'].astype('str').str.contains('Minister'), 'Minister'] = 1

#コンサルタント

df_train.loc[df_train['emp_title'].astype('str').str.contains('Consultant'), 'Consultant'] = 1

df_test.loc[df_test['emp_title'].astype('str').str.contains('Consultant'), 'Consultant'] = 1

#金融アナリスト

df_train.loc[df_train['emp_title'].astype('str').str.contains('Financial Analyst'), 'Financial Analyst'] = 1

df_test.loc[df_test['emp_title'].astype('str').str.contains('Financial Analyst'), 'Financial Analyst'] = 1

#会計士

df_train.loc[df_train['emp_title'].astype('str').str.contains('Accountant'), 'Accountant'] = 1

df_test.loc[df_test['emp_title'].astype('str').str.contains('Accountant'), 'Accountant'] = 1

#ディレクター

df_train.loc[df_train['emp_title'].astype('str').str.contains('Director'), 'Director'] = 1

df_test.loc[df_test['emp_title'].astype('str').str.contains('Director'), 'Director'] = 1

#

df_train.loc[df_train['emp_title'].astype('str').str.contains('Manager'), 'Manager'] = 1

df_test.loc[df_test['emp_title'].astype('str').str.contains('Manager'), 'Manager'] = 1
y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition'], axis=1)

X_test = df_test
# 住所（州）をグループ識別子として分離しておく。日付とテキストも除いておく。

groups = X_train.addr_state.values



X_train.drop(['issue_d', 'emp_title', 'addr_state'], axis=1, inplace=True)

X_test.drop(['issue_d', 'emp_title', 'addr_state'], axis=1, inplace=True)

X_test
cats = []

num = []



for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

    else:

        num.append(col)
num
#cats.remove('grade')

#cats.remove('sub_grade')



#gg = ['grade', 'sub_grade']



#for col in gg:

#    summary = X_train[col].value_counts()

#    X_train[col] = X_train[col].map(summary)

#    X_test[col] = X_test[col].map(summary)
cats.remove('grade')

cats.remove('sub_grade')



gg = ['grade', 'sub_grade']



for col in gg:

    unique = pd.unique(X_train[col])

    unique.sort()

    

    items = []

    indicies = []

    for i, item in enumerate(unique):

        items.append(item)

        indicies.append(i)



    grade_vals = pd.Series(indicies, index=items)

    X_train[col] = X_train[col].map(grade_vals)

    X_test[col] = X_test[col].map(grade_vals)
oe = OrdinalEncoder(cols=cats)



X_train = oe.fit_transform(X_train)

X_test = oe.transform(X_test)
X_train['loan_amnt'] = X_train['loan_amnt'].apply(np.log1p)

X_test['loan_amnt'] = X_test['loan_amnt'].apply(np.log1p)
len(X_train) - X_train.count()
num
num.remove('mths_since_last_delinq')

num.remove('mths_since_last_record')

num.remove('mths_since_last_major_derog')

num.remove('tot_coll_amt')

num.remove('tot_cur_bal')

num.remove('title_badflg')

num.remove('purpose_badflg')

num.remove('emp_title_badflg')

num.remove('Minister')

num.remove('Consultant')

num.remove('Financial Analyst')

num.remove('Accountant')

num.remove('Director')

num.remove('Manager')



flg = ['title_badflg','purpose_badflg','emp_title_badflg','Minister','Consultant','Financial Analyst','Accountant','Director','Manager']

zero = ['mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog']

nine = ['tot_coll_amt', 'tot_cur_bal']
#X_train.fillna(-99999, inplace=True)

#X_test.fillna(-99999, inplace=True)



#X_train[cats] = X_train[cats].fillna("NaN")

#X_test[cats] = X_test[cats].fillna("NaN")



#X_train[num] = X_train[num].fillna(X_train[num].min())

#X_test[num] = X_test[num].fillna(X_test[num].min())







X_train[cats] = X_train[cats].fillna("NaN")

X_test[cats] = X_test[cats].fillna("NaN")



X_train[zero] = X_train[zero].fillna(0)

X_test[zero] = X_test[zero].fillna(0)



X_train[flg] = X_train[flg].fillna(0)

X_test[flg] = X_test[flg].fillna(0)



X_train[nine] = X_train[nine].fillna(-9999)

X_test[nine] = X_test[nine].fillna(-9999)



X_train[num] = X_train[num].fillna(X_test[num].min())

X_test[num] = X_test[num].fillna(X_test[num].min())
len(X_train) - X_train.count()
X_train[num]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train[num])



X_train[num] =scaler.transform(X_train[num])

X_test[num] =scaler.transform(X_test[num])
X_train
%%time

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm_notebook as tqdm

from sklearn.metrics import roc_auc_score

GBC = GradientBoostingClassifier(max_depth=3,max_features = 13, n_estimators=180, learning_rate=0.1)

GBC.fit(X_train, y_train)
%%time

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm_notebook as tqdm

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()

LR.fit(X_train, y_train)
LGBM = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

                                importance_type='split', learning_rate=0.05, max_depth=-1,

                                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                                n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

                                random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

                                subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
%%time



gkf = GroupKFold(n_splits=5)

y_pred_test = np.zeros(len(X_test))

scores = []



for i, (train_ix, test_ix) in enumerate(tqdm(gkf.split(X_train, y_train, groups))):

    X_train_, y_train_, groups_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix], groups[train_ix]

    X_val, y_val, groups_val = X_train.iloc[test_ix], y_train.iloc[test_ix], groups[test_ix]

    

    LGBM.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])

    y_pred = LGBM.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred)

    scores.append(score)

    

    y_pred_test += LGBM.predict_proba(X_test)[:,1] 

    

    print('CV Score of Fold_%d is %f' % (i, score))

    print('\n')

    

scores = np.array(scores)

print('Ave. CV score is %f' % scores.mean())

y_pred_test /= 5

y_pred1 = y_pred_test
XX_train = pd.DataFrame(LR.predict_proba(X_train)[:,1], columns=['LR'])

XX_train['GBC'] = GBC.predict_proba(X_train)[:,1]

XX_train['LGBM'] = LGBM.predict_proba(X_train)[:,1]



XX_test = pd.DataFrame(LR.predict_proba(X_test)[:,1], columns=['LR'])

XX_test['GBC'] = GBC.predict_proba(X_test)[:,1]

XX_test['LGBM'] = LGBM.predict_proba(X_test)[:,1]
XX_test

import xgboost as xgb

gbm = xgb.XGBClassifier(

     n_estimators= 2000,

     max_depth= 4,

     min_child_weight= 2,

     gamma=0.9,                        

     subsample=0.8,

     colsample_bytree=0.8,

     objective= 'binary:logistic',

     nthread= -1,

     scale_pos_weight=1).fit(XX_train, y_train)
y_pred = gbm.predict(XX_test)
submission = pd.read_csv('../input/homework-for-students2/sample_submission.csv', index_col=0)

submission.loan_condition = y_pred

submission.to_csv('submission.csv')