#!pip list
import gc

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error, log_loss

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer



from tqdm import tqdm_notebook as tqdm

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



import lightgbm as lgb

from lightgbm import LGBMClassifier



import re

from hyperopt import fmin, tpe, hp, rand, Trials
#df_train = pd.read_csv('../input/homework-for-students2/train.csv', index_col=0, parse_dates=['issue_d', 'earliest_cr_line'], skiprows=lambda x: x%20!=0)

#df_test = pd.read_csv('../input/homework-for-students2/test.csv', index_col=0, parse_dates=['issue_d', 'earliest_cr_line'], skiprows=lambda x: x%20!=0)

df_train = pd.read_csv('../input/homework-for-students2/train.csv', index_col=0, parse_dates=['issue_d', 'earliest_cr_line'])

df_test = pd.read_csv('../input/homework-for-students2/test.csv', index_col=0, parse_dates=['issue_d', 'earliest_cr_line'])

#df_train_all = pd.read_csv('../input/homework-for-students2/train.csv', index_col=0)

#df_test_all = pd.read_csv('../input/homework-for-students2/test.csv', index_col=0)
#df_train.describe()
#df_statelatlong = pd.read_csv('../input/homework-for-students2/statelatlong.csv')#

#df_statelatlong.columns = ['addr_state', 'state_lat', 'state_long', 'city']

#df_statelatlong.drop(['city'], axis=1, inplace=True)
#df_gdp = pd.read_csv('../input/homework-for-students2/US_GDP_by_State.csv')



#df_gdp_2013 = df_gdp[df_gdp['year'] == 2013].reset_index()

#df_gdp_2013.columns = ['index', 'city', '2013Spending', '2013GSP','2013Growth%', '2013Population', 'year']

#df_gdp_2013.drop(['index', 'year'], axis=1, inplace=True)

#df_gdp_2013['2013GSP_per_million'] = df_gdp_2013['2013GSP'] / df_gdp_2013['2013Population']



#df_gdp_2014 = df_gdp[df_gdp['year'] == 2014].reset_index()

#df_gdp_2014.columns = ['index', 'city', '2014Spending', '2014GSP','2014Growth%', '2014Population', 'year']

#df_gdp_2014.drop(['index', 'year'], axis=1, inplace=True)

#df_gdp_2014['2014GSP_per_million'] = df_gdp_2014['2014GSP'] / df_gdp_2014['2014Population']



#df_gdp_2015 = df_gdp[df_gdp['year'] == 2015].reset_index()

#df_gdp_2015.columns = ['index', 'city', '2015Spending', '2015GSP','2015Growth%', '2015Population', 'year']

#df_gdp_2015.drop(['index', 'year'], axis=1, inplace=True)

#df_gdp_2015['2015GSP_per_million'] = df_gdp_2015['2015GSP'] / df_gdp_2015['2015Population']



#df_2013_14 = pd.merge(df_gdp_2013, df_gdp_2014, on='city', how='inner')

#df_2013_15 = pd.merge(df_2013_14, df_gdp_2015, on='city', how='inner')

#df_2013_15['Spending_mean'] = (df_2013_15['2013Spending'] + df_2013_15['2014Spending'] + df_2013_15['2015Spending']) / 3

#df_2013_15['GSP_mean'] = (df_2013_15['2013GSP'] + df_2013_15['2014GSP'] + df_2013_15['2015GSP']) / 3

#df_2013_15['Growth%_mean'] = (df_2013_15['2013Growth%'] + df_2013_15['2014Growth%'] + df_2013_15['2015Growth%']) / 3

#df_2013_15['Population_mean'] = (df_2013_15['2013Population'] + df_2013_15['2014Population'] + df_2013_15['2015Population']) / 3
#df_spi = pd.read_csv('../input/homework-for-students2/spi.csv', parse_dates=['date'])

#df_spi['issue_d_yyyy'] = df_spi.date.dt.year

#df_spi['issue_d_mm'] = df_spi.date.dt.month

#df_spi_monthly = df_spi.groupby(['issue_d_yyyy', 'issue_d_mm'], as_index=False)['close'].mean()



#for j in range(12):

#    for i in range(df_spi_monthly.shape[0] - j - 1):

#        df_spi_monthly.loc[i+j+1, 'close-' + str(j+1)] = df_spi_monthly.loc[i, 'close']



#for j in range(12):

#    for i in range(df_spi_monthly.shape[0]):

#        df_spi_monthly.loc[i, 'diff-' + str(j+1)] = df_spi_monthly.loc[i, 'close'] - df_spi_monthly.loc[i, 'close-' + str(j+1)]



#for j in (3,6,12):

#    for i in range(df_spi_monthly.shape[0]-j+1):

#        tmp = 0

#        for k in range(j):

#            tmp += df_spi_monthly.loc[i+j-1-k, 'close']

#        df_spi_monthly.loc[i+j-1, 'mean-' + str(j)] = tmp / j    
#テストデータのhome_ownership列には'OTHER', 'NONE'がないため、'ANY'に変換

col = 'home_ownership'

df_train[col] = df_train[col].map(lambda x: 'ANY' if x == 'OTHER' or x == 'NONE' else x)

#df_train_all[col] = df_train_all[col].map(lambda x: 'ANY' if x == 'OTHER' or x == 'NONE' else x)
#テストデータのpurpose列には'educational'がないため、'other'に変換

col = 'purpose'

df_train[col] = df_train[col].map(lambda x: 'other' if x == 'educational' else x)

#df_train_all[col] = df_train_all[col].map(lambda x: 'other' if x == 'educational' else x)
# open_acc > total_acc のレコードの除去

df_train = df_train[(df_train['open_acc'] > df_train['total_acc']) == False] 

#df_train_all = df_train_all[(df_train_all['open_acc'] > df_train_all['total_acc']) == False]
#loan_amntの範囲で区分 low: ～$5,000, normal: $5,000～$25,000, high: $25,000～$30,000, very high: $30,000～

cat_col = 'cat_loan_amnt'

num_col = 'loan_amnt'

df_train[cat_col] = df_train[num_col].map(lambda x: 'low' if x < 5000

                                          else 'normal' if x >= 5000 and x < 25000

                                          else 'high' if x > 25000 and x < 30000

                                          else 'very high')

df_test[cat_col] = df_test[num_col].map(lambda x: 'low' if x < 5000

                                        else 'normal' if x >= 5000 and x < 25000

                                        else 'high' if x > 25000 and x < 30000

                                        else 'very high')
#annual_incの範囲で区分 very low: ～$35,000, low: $35,000～$50,000, normal: $50,000～$75,000, high: $75,000～$125,000, very high: $125,000～

cat_col = 'cat_annual_inc'

num_col = 'annual_inc'

df_train[cat_col] = df_train[num_col].map(lambda x: 'very low' if x < 35000

                                          else 'low' if x >= 35000 and x < 50000

                                          else 'normal' if x >= 50000 and x < 75000

                                          else 'high' if x >= 75000 and x < 125000

                                          else 'very high' if x >= 125000

                                          else 'missing')

df_test[cat_col] = df_test[num_col].map(lambda x: 'very low' if x < 35000

                                        else 'low' if x >= 35000 and x < 50000

                                        else 'normal' if x >= 50000 and x < 75000

                                        else 'high' if x >= 75000 and x < 125000

                                        else 'very high' if x >= 125000

                                        else 'missing')
#cat_col = 'cat_dti'

#num_col = 'dti'

#df_train[cat_col] = df_train[num_col].map(lambda x: 'very low' if x < 12

#                                          else 'low' if x >= 12 and x < 18

#                                          else 'high' if x >= 18 and x < 24

#                                          else 'very high' if x >= 24

#                                          else 'missing')

#df_test[cat_col] = df_test[num_col].map(lambda x: 'very low' if x < 12

#                                        else 'low' if x >= 12 and x < 18

#                                        else 'high' if x >= 18 and x < 24

#                                        else 'very high' if x >= 24

#                                       else 'missing')
cat_col = 'cat_installment'

num_col = 'installment'

df_train[cat_col] = df_train[num_col].map(lambda x: 'very low' if x < 260

                                          else 'low' if x >= 260 and x < 380

                                          else 'high' if x >= 380 and x < 570

                                          else 'very high')

df_test[cat_col] = df_test[num_col].map(lambda x: 'very low' if x < 260

                                        else 'low' if x >= 260 and x < 380

                                        else 'high' if x >= 380 and x < 570

                                        else 'very high')
cat_col = 'cat_last_record'

num_col = 'mths_since_last_record'

df_train[cat_col] = df_train[num_col].map(lambda x: 'no record' if x == 0

                                          else 'recorded' if x > 0

                                          else 'missing')

df_test[cat_col] = df_test[num_col].map(lambda x: 'no record' if x == 0

                                        else 'recorded' if x > 0

                                        else 'missing')
cat_col = 'cat_collections'

num_col = 'collections_12_mths_ex_med'

df_train[cat_col] = df_train[num_col].map(lambda x: 'no collection' if x == 0

                                          else 'collected' if x > 0

                                          else 'missing')

df_test[cat_col] = df_test[num_col].map(lambda x: 'no collection' if x == 0

                                        else 'collected' if x > 0

                                        else 'missing')
#平均loan_conditionで区分 good: 10%未満, normal:10%以上25%未満, bad: 25%以上

df_train['grade2'] = df_train['sub_grade'].map(lambda x: 'good' if re.match(r'A[1-5]', x) or x == 'B1'

                                           else 'normal' if re.match(r'B[2-5]', x) or re.match(r'C[1-5]', x) or x == 'D1'

                                           else 'bad')

df_test['grade2'] = df_test['sub_grade'].map(lambda x: 'good' if re.match(r'A[1-5]', x) or x == 'B1'

                                         else 'normal' if re.match(r'B[2-5]', x) or re.match(r'C[1-5]', x) or x == 'D1'

                                         else 'bad')

#df_train_all['grade2'] = df_train_all['sub_grade'].map(lambda x: 'good' if re.match(r'A[1-5]', x) or x == 'B1'

#                                           else 'normal' if re.match(r'B[2-5]', x) or re.match(r'C[1-5]', x) or x == 'D1'

#                                           else 'bad')

#df_test_all['grade2'] = df_test_all['sub_grade'].map(lambda x: 'good' if re.match(r'A[1-5]', x) or x == 'B1'

#                                         else 'normal' if re.match(r'B[2-5]', x) or re.match(r'C[1-5]', x) or x == 'D1'

#                                         else 'bad')
col1 = 'grade'

cols = ['home_ownership', 'purpose', 'initial_list_status',

        'cat_loan_amnt', 'cat_annual_inc', 'cat_installment', 'cat_last_record', 'cat_collections']

for col2 in cols:

    col = col1 + '-' + col2

    df_train[col] = df_train[col1].str.cat(df_train[col2], sep='-')

    df_test[col] = df_test[col1].str.cat(df_test[col2], sep='-')
df_train['grade-home_ownership'] = df_train['grade-home_ownership'].map(lambda x: 'ANY' if re.match('..ANY', x) else x)

df_test['grade-home_ownership'] = df_test['grade-home_ownership'].map(lambda x: 'ANY' if re.match('..ANY', x) else x)
df_train['grade-purpose'] = df_train['grade-purpose'].map(lambda x: 'G-other' if x in ('G-car', 'G-wedding', 'G-vacation', 'G-renewable_energy') else x)

df_test['grade-purpose'] = df_test['grade-purpose'].map(lambda x: 'G-other' if x in ('G-car', 'G-wedding', 'G-vacation', 'G-renewable_energy') else x)
col1 = 'home_ownership'

cols = ['purpose', 'initial_list_status', 'application_type',

        'cat_loan_amnt', 'cat_annual_inc', 'cat_installment', 'cat_last_record', 'cat_collections']

for col2 in cols:

    col = col1 + '-' + col2

    df_train[col] = df_train[col1].str.cat(df_train[col2], sep='-')

    df_test[col] = df_test[col1].str.cat(df_test[col2], sep='-')

    df_train[col] = df_train[col].map(lambda x: 'ANY' if re.match('^ANY-[A-Za-z0-9]+', x) else x)

    df_test[col] = df_test[col].map(lambda x: 'ANY' if re.match('^ANY-[A-Za-z0-9]+', x) else x)
col = 'home_ownership-cat_collections'

df_train[col] = df_train[col].map(lambda x: 'missing' if re.match('^[A-Za-z0-9]+-missing$', x) else x)

df_test[col] = df_test[col].map(lambda x: 'missing' if re.match('^[A-Za-z0-9]+-missing$', x) else x)
col1 = 'purpose'

cols = ['initial_list_status',

        'cat_loan_amnt', 'cat_annual_inc', 'cat_installment', 'cat_last_record', 'cat_collections']

for col2 in cols:

    col = col1 + '-' + col2

    df_train[col] = df_train[col1].str.cat(df_train[col2], sep='-')

    df_test[col] = df_test[col1].str.cat(df_test[col2], sep='-')
col = 'purpose-cat_loan_amnt'

df_train[col] = df_train[col].map(lambda x: 'other-high' if x in ('vacation-high', 'moving-high ', 'renewable_energy-high', 'wedding-high') 

                                  else 'other-very high' if x in ('vacation-very high', 'renewable_energy-very high')

                                  else x)

df_test[col] = df_test[col].map(lambda x: 'other-high' if x in ('vacation-high', 'moving-high ', 'renewable_energy-high', 'wedding-high') 

                                else 'other-very high' if x in ('vacation-very high', 'renewable_energy-very high')

                                else x)
col = 'purpose-cat_collections'

df_train[col] = df_train[col].map(lambda x: 'other-collected' if x in ('renewable_energy-collected') 

                                  else x)

df_test[col] = df_test[col].map(lambda x: 'other-collected' if x in ('renewable_energy-collected') 

                                else x)
col1 = 'initial_list_status'

cols = ['application_type', 'cat_loan_amnt', 'cat_annual_inc', 'cat_installment', 'cat_last_record', 'cat_collections']

for col2 in cols:

    col = col1 + '-' + col2

    df_train[col] = df_train[col1].str.cat(df_train[col2], sep='-')

    df_test[col] = df_test[col1].str.cat(df_test[col2], sep='-')
col = 'initial_list_status-cat_last_record'

df_train[col] = df_train[col].map(lambda x: 'no record' if re.match('[fw]-no record', x) 

                                  else x)

df_test[col] = df_test[col].map(lambda x: 'no record' if re.match('[fw]-no record', x) 

                                else x)
col1 = 'application_type'

cols = ['cat_annual_inc', 'cat_installment']

for col2 in cols:

    col = col1 + '-' + col2

    df_train[col] = df_train[col1].str.cat(df_train[col2], sep='-')

    df_test[col] = df_test[col1].str.cat(df_test[col2], sep='-')
col = 'application_type-cat_annual_inc'

df_train[col] = df_train[col].map(lambda x: 'Joint App-high' if x == 'Joint App-very high' 

                                  else x)

df_test[col] = df_test[col].map(lambda x: 'Joint App-high' if x == 'Joint App-very high'  

                                else x)
col1 = 'cat_loan_amnt'

cols = ['cat_annual_inc', 'cat_installment', 'cat_last_record', 'cat_collections']

for col2 in cols:

    col = col1 + '-' + col2

    df_train[col] = df_train[col1].str.cat(df_train[col2], sep='-')

    df_test[col] = df_test[col1].str.cat(df_test[col2], sep='-')
col = 'cat_loan_amnt-cat_annual_inc'

df_train[col] = df_train[col].map(lambda x: 'very high-low' if x == 'high-low' 

                                  else 'normal-very low' if x in ('high-very low', 'very high-very low')

                                  else x)

df_test[col] = df_test[col].map(lambda x: 'very high-low' if x == 'high-low' 

                                else 'normal-very low' if x in ('high-very low', 'very high-very low')

                                else x)
col = 'cat_loan_amnt-cat_installment'

df_train[col] = df_train[col].map(lambda x: 'very high-low' if x == 'high-low' 

                                  else 'normal-very low' if x in ('high-very low', 'very high-very low')

                                  else x)

df_test[col] = df_test[col].map(lambda x: 'very high-low' if x == 'high-low' 

                                else 'normal-very low' if x in ('high-very low', 'very high-very low')

                                else x)
col1 = 'cat_annual_inc'

cols = ['cat_installment', 'cat_last_record', 'cat_collections']

for col2 in cols:

    col = col1 + '-' + col2

    df_train[col] = df_train[col1].str.cat(df_train[col2], sep='-')

    df_test[col] = df_test[col1].str.cat(df_test[col2], sep='-')
#col1 = 'cat_dti'

#cols = ['cat_installment', 'cat_last_record', 'cat_collections']

#for col2 in cols:

#    col = col1 + '-' + col2

#    df_train[col] = df_train[col1].str.cat(df_train[col2], sep='-')

#    df_test[col] = df_test[col1].str.cat(df_test[col2], sep='-')
col1 = 'cat_installment'

cols = ['cat_last_record', 'cat_collections']

for col2 in cols:

    col = col1 + '-' + col2

    df_train[col] = df_train[col1].str.cat(df_train[col2], sep='-')

    df_test[col] = df_test[col1].str.cat(df_test[col2], sep='-')
col1 = 'cat_last_record'

col2 = 'cat_collections'

col = col1 + '-' + col2

df_train[col] = df_train[col1].str.cat(df_train[col2], sep='-')

df_test[col] = df_test[col1].str.cat(df_test[col2], sep='-')
# 欠損の総数

col = 'num_nulls'

df_train[col] = df_train.isnull().sum(axis=1)

df_test[col] = df_test.isnull().sum(axis=1)
# 欠損フラグ(テストデータに欠損があるもののみ)

cols = ['emp_title', 'emp_length', 'title', 'dti', 'inq_last_6mths', 'mths_since_last_delinq',

       'mths_since_last_record', 'revol_util', 'mths_since_last_major_derog']

for col in cols:

    flag_name = col + '_isnull'

    df_train[flag_name] = df_train[col].map(lambda x: 1 if pd.isnull(x) else 0) 

    df_test[flag_name] = df_test[col].map(lambda x: 1 if pd.isnull(x) else 0)
# 欠損パターン(テストデータに欠損があるもののみ)

col1 = 'missing_pattern'

cols = ['emp_length_isnull', 'title_isnull', 'dti_isnull', 'inq_last_6mths_isnull', 'mths_since_last_delinq_isnull',

       'mths_since_last_record_isnull', 'revol_util_isnull', 'mths_since_last_major_derog_isnull']



df_train[col1] = df_train['emp_title_isnull'].astype(str)

df_test[col1] = df_test['emp_title_isnull'].astype(str)

for col2 in cols:

    df_train[col1] = df_train[col1].str.cat(df_train[col2].astype(str))

    df_test[col1] = df_test[col1].str.cat(df_test[col2].astype(str))

#df_test['missing_pattern'].value_counts()
y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition'], axis=1)



X_test = df_test
X_all = pd.concat([X_train, X_test])
X_all_desc = X_all.describe()
#X_all_desc
#内容の確認(count: nullでない行数、mean: 平均値、std: 標準偏差、min：最小値、25%：25%タイル値、50%: 中央値、75%: タイル値、max: 最大値)

#X_all_desc.loc['count', 'loan_amnt']
X_all_mode = X_all.mode()
#内容の確認 最頻値

#X_all_mode.loc[0,'loan_amnt']
col = 'issue_d'

X_train['issue_d_yyyyq'] = X_train[col].dt.year * 10 + X_train[col].dt.quarter 

X_train['issue_d_yyyymm'] = X_train[col].dt.year * 100 + X_train[col].dt.month 

X_train['issue_d_yyyy'] = X_train[col].dt.year

X_train['issue_d_q'] = X_train[col].dt.quarter

X_train['issue_d_mm'] = X_train[col].dt.month

X_test['issue_d_yyyyq'] = X_test[col].dt.year * 10 + X_test[col].dt.quarter 

X_test['issue_d_yyyymm'] = X_test[col].dt.year * 100 + X_test[col].dt.month 

X_test['issue_d_yyyy'] = X_test[col].dt.year

X_test['issue_d_q'] = X_test[col].dt.quarter

X_test['issue_d_mm'] = X_test[col].dt.month
col = 'earliest_cr_line'

X_train['earliest_cr_line_yyyyq'] = X_train[col].map(lambda x: -1 if pd.isnull(x) else x.year * 10 + x.quarter) 

X_train['earliest_cr_line_yyyymm'] = X_train[col].map(lambda x: -1 if pd.isnull(x) else x.year * 100 + x.month) 

X_train['earliest_cr_line_yyyy'] = X_train[col].map(lambda x: -1 if pd.isnull(x) else x.year)

X_train['earliest_cr_line_q'] = X_train[col].map(lambda x: -1 if pd.isnull(x) else x.quarter)

X_train['earliest_cr_line_mm'] = X_train[col].map(lambda x: -1 if pd.isnull(x) else x.month)

X_test['earliest_cr_line_yyyyq'] = X_test[col].dt.year * 10 + X_test[col].dt.quarter 

X_test['earliest_cr_line_yyyymm'] = X_test[col].dt.year * 100 + X_test[col].dt.month 

X_test['earliest_cr_line_yyyy'] = X_test[col].dt.year

X_test['earliest_cr_line_q'] = X_test[col].dt.quarter

X_test['earliest_cr_line_mm'] = X_test[col].dt.month
# ローン発行日までの口座開設期間(日)

col = 'from_first_cr_line'

X_train[col] = (X_train['issue_d'] - X_train['earliest_cr_line']).dt.days

X_test[col] = (X_test['issue_d'] - X_test['earliest_cr_line']).dt.days

X_train[col].fillna(-1, inplace=True)

X_train[col].fillna(-1, inplace=True)
#クリッピング

X_train['earliest_cr_line_yyyy'].clip(1974, 2019, axis=0, inplace=True)

X_test['earliest_cr_line_yyyy'].clip(1974, 2019, axis=0, inplace=True)



X_train['earliest_cr_line_yyyymm'].clip(197412, 201901, axis=0, inplace=True)

X_test['earliest_cr_line_yyyymm'].clip(197412, 201901, axis=0, inplace=True)



X_train['earliest_cr_line_yyyyq'].clip(19744, 20191, axis=0, inplace=True)

X_test['earliest_cr_line_yyyyq'].clip(19744, 20191, axis=0, inplace=True)
#X_train = pd.merge(X_train, df_spi_monthly, on=['issue_d_yyyy', 'issue_d_mm'], how='inner')

#X_test = pd.merge(X_test, df_spi_monthly, on=['issue_d_yyyy', 'issue_d_mm'], how='inner')
#X_train = pd.merge(X_train, df_statelatlong, on='addr_state', how='inner')

#X_test = pd.merge(X_test, df_statelatlong, on='addr_state', how='inner')
#X_train = pd.merge(X_train, df_2013_15, on='city', how='inner')

#X_test = pd.merge(X_test, df_2013_15, on='city', how='inner')
#X_train.drop(['city'], axis=1, inplace=True)

#X_test.drop(['city'], axis=1, inplace=True)
#X_train.drop(['state_lat', 'state_long'], axis=1, inplace=True)

#X_test.drop(['state_lat', 'state_long'], axis=1, inplace=True)
#df_gdp_2013.columns = ['index', 'city', '2013Spending', '2013GSP','2013Growth%', '2013Population', 'year']

#X_train.drop(['2013Spending', '2013GSP','2013Growth%', '2013Population', '2013GSP_per_million'], axis=1, inplace=True)

#X_test.drop(['2013Spending', '2013GSP','2013Growth%', '2013Population', '2013GSP_per_million'], axis=1, inplace=True)
TXT_emp_title_train = X_train.emp_title.copy()

TXT_emp_title_test = X_test.emp_title.copy()



TXT_title_train = X_train.title.copy()

TXT_title_test = X_test.title.copy()



X_train.drop(['emp_title', 'title'], axis=1, inplace=True)

X_test.drop(['emp_title', 'title'], axis=1, inplace=True)
cols = ['grade', 'sub_grade', 'addr_state', 'zip_code', 'cat_loan_amnt', 'cat_annual_inc', 'cat_installment', 'cat_last_record', 'cat_collections', 'grade2',

        'grade-home_ownership', 'grade-purpose', 'grade-initial_list_status', 'grade-cat_loan_amnt', 'grade-cat_annual_inc', 

        'grade-cat_installment', 'grade-cat_last_record', 'grade-cat_collections',

        'home_ownership-purpose', 'home_ownership-initial_list_status', 'home_ownership-application_type', 'home_ownership-cat_loan_amnt',

        'home_ownership-cat_annual_inc', 'home_ownership-cat_installment', 'home_ownership-cat_last_record', 'home_ownership-cat_collections',

        'purpose-initial_list_status', 'purpose-cat_loan_amnt', 'purpose-cat_annual_inc', 'purpose-cat_installment', 'purpose-cat_last_record', 'purpose-cat_collections',

        'initial_list_status-application_type', 'initial_list_status-cat_loan_amnt', 'initial_list_status-cat_annual_inc', 'initial_list_status-cat_installment',

        'initial_list_status-cat_last_record', 'initial_list_status-cat_collections',

        'application_type-cat_annual_inc', 'application_type-cat_installment',

        'cat_loan_amnt-cat_annual_inc', 'cat_loan_amnt-cat_installment', 'cat_loan_amnt-cat_last_record', 'cat_loan_amnt-cat_collections',

        'cat_annual_inc-cat_installment', 'cat_annual_inc-cat_last_record', 'cat_annual_inc-cat_collections',

        'cat_installment-cat_last_record', 'cat_installment-cat_collections', 'cat_last_record-cat_collections']

target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)



for col in cols:

    # X_testはX_trainでエンコーディングする

    summary = X_temp.groupby([col])[target].mean()

    enc_test = X_test[col].map(summary) 



    # X_trainのカテゴリ変数をoofでエンコーディングする

    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





    enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



    for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



        summary = X_train_.groupby([col])[target].mean()

        enc_train.iloc[val_ix] = X_val[col].map(summary)



    enc_train[enc_train.isnull()] = 0.17447602091409

    enc_test[enc_test.isnull()] = 0.17447602091409

    

    X_train[col] = enc_train

    X_test[col] = enc_test
col = 'emp_length'

encoder = OrdinalEncoder(mapping=[{'col':col,'mapping':{'< 1 year':0, '1 year':1, '2 years':2, '3 years':3, '4 years':4, '5 years':5, 

                                                        '6 years':6, '7 years':7, '8 years':8, '9 years':9, '10+ years':10}}], return_df=True)

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)
cols = ['home_ownership', 'purpose', 

#        'loan_amnt_range', 'annual_inc_range'

       ] 

#ohe = OneHotEncoder(cols=cols, handle_unknown='indicator', use_cat_names=True)

ohe = OneHotEncoder(cols=cols, use_cat_names=True)



enc_train = ohe.fit_transform(X_train[cols])

enc_test = ohe.transform(X_test[cols])
X_train = pd.concat([X_train, enc_train], axis=1)

X_test = pd.concat([X_test, enc_test], axis=1)
#cols = ['missing_pattern'] 

#ohe = OneHotEncoder(cols=cols, handle_unknown='indicator', use_cat_names=True)

#ohe = OneHotEncoder(cols=cols, use_cat_names=True)



#enc_train = ohe.fit_transform(X_train[cols])

#enc_test = ohe.transform(X_test[cols])
#X_train = pd.concat([X_train, enc_train], axis=1)

#X_test = pd.concat([X_test, enc_test], axis=1)
cols = ['home_ownership', 'purpose', 'initial_list_status', 'application_type', 'missing_pattern']

for col in cols:

    summary = X_all[col].value_counts()

    X_train[col] = X_train[col].map(summary)

    X_test[col] = X_test[col].map(summary)
#X_all['home_ownership'].value_counts()
#X_test['home_ownership'].value_counts()
# dtypeがobjectのカラム名とユニーク数を確認してみましょう。

#cats = []

#for col in X_train.columns:

#    if X_train[col].dtype == 'object':

#        cats.append(col)

        

#        print(col, X_train[col].nunique())
#X_train.drop(['cat_dti'], axis=1, inplace=True)

#X_test.drop(['cat_dti'], axis=1, inplace=True)
cols = ['loan_amnt', 'installment', 'revol_bal', 'tot_coll_amt', 'tot_cur_bal']

col1 = 'annual_inc'



for col in cols:

    new_col = col + '_to_' + col1 + '_ratio'



    X_train[new_col] = X_train[col] / X_train[col1]

    X_test[new_col] = X_test[col] / X_test[col1]

    

    #分母、分子のいずれかがNull、または0/0のときは-1

    X_train[new_col].fillna(-1, inplace=True)

    X_test[new_col].fillna(-1, inplace=True)

    

    #分母が0、分子が0以外の数値の場合は999999

    X_train[new_col].replace([np.inf, -np.inf], np.nan, inplace=True)

    X_test[new_col].replace([np.inf, -np.inf], np.nan, inplace=True)

    X_train[new_col].fillna(999999, inplace=True)

    X_test[new_col].fillna(999999, inplace=True)
cols = ['open_acc', 'acc_now_delinq']

col1 = 'total_acc'



for col in cols:

    new_col = col + '_to_' + col1 + '_ratio'



    X_train[new_col] = X_train[col] / X_train[col1]

    X_test[new_col] = X_test[col] / X_test[col1]

    

    #分母、分子のいずれかがNull、または0/0のときは-1

    X_train[new_col].fillna(-1, inplace=True)

    X_test[new_col].fillna(-1, inplace=True)

    

    #分母が0、分子が0以外の数値の場合は9999

    X_train[new_col].replace([np.inf, -np.inf], np.nan, inplace=True)

    X_test[new_col].replace([np.inf, -np.inf], np.nan, inplace=True)

    X_train[new_col].fillna(9999, inplace=True)

    X_test[new_col].fillna(9999, inplace=True)
#for col in X_train.columns:

#    print (col, X_train[col].isnull().sum())
X_train.drop(['issue_d', 'earliest_cr_line'], axis=1, inplace=True)

X_test.drop(['issue_d', 'earliest_cr_line'], axis=1, inplace=True)

    

X_train['mths_since_last_record'].fillna(X_all_mode.loc[0, 'mths_since_last_record'], inplace=True)

X_test['mths_since_last_record'].fillna(X_all_mode.loc[0, 'mths_since_last_record'], inplace=True)



X_train.fillna(X_all_desc.loc['50%',], inplace=True)

X_test.fillna(X_all_desc.loc['50%',], inplace=True)



#X_train.fillna(-9999, inplace=True)

#X_test.fillna(-9999, inplace=True)
#X_train.drop(['title_isnull', 'dti_isnull', 'inq_last_6mths_isnull', 'revol_util_isnull'], axis=1, inplace=True)

#X_test.drop(['title_isnull', 'dti_isnull', 'inq_last_6mths_isnull', 'revol_util_isnull'], axis=1, inplace=True)
#密行列の状態を保存

X_train_base = X_train

X_test_base = X_test
TXT_emp_title_train.fillna('#', inplace=True)

TXT_emp_title_test.fillna('#', inplace=True)



TXT_title_train.fillna('#', inplace=True)

TXT_title_test.fillna('#', inplace=True)
TXT_emp_title_train_base = TXT_emp_title_train

TXT_emp_title_test_base = TXT_emp_title_test

TXT_title_train_base = TXT_title_train

TXT_title_test_base = TXT_title_test
tfidf_emp_title = TfidfVectorizer(max_features=200, use_idf=True)

tfidf_title = TfidfVectorizer(max_features=150, use_idf=True)
TXT_emp_title_train = tfidf_emp_title.fit_transform(TXT_emp_title_train_base)

TXT_emp_title_test = tfidf_emp_title.transform(TXT_emp_title_test_base)



TXT_title_train = tfidf_title.fit_transform(TXT_title_train_base)

TXT_title_test = tfidf_title.transform(TXT_title_test_base)
X_train_1 = sp.sparse.hstack([X_train_base.values, TXT_emp_title_train])

X_test_1 = sp.sparse.hstack([X_test_base.values, TXT_emp_title_test])



X_train_2 = sp.sparse.hstack([X_train_1, TXT_title_train])

X_test_2 = sp.sparse.hstack([X_test_1, TXT_title_test])



X_train = X_train_2.tocsr()

X_test = X_test_2.tocsr()
#issue_dで完全に区別できるのでAUC1.0に。issue_d外してやる時間なかった。

#X = sp.sparse.vstack([X_train, X_test], format='csr')

#y = np.concatenate([np.zeros(X_train.shape[0]), np.ones(X_test.shape[0])])



#scores = []



#y_pred_test_av = np.zeros(X.shape[0])

#cv_iteration = 0



#skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



#for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X, y))):

#    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

#    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

#    X_, y_ = X[train_ix], y[train_ix]

#    X_val, y_val = X[test_ix], y[test_ix]

    

#    clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

#                     importance_type='split', learning_rate=0.05, max_depth=-1,

#                     min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

#                     n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

#                     random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

#                     subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

    

#    clf.fit(X_, y_, early_stopping_rounds=20, eval_metric='binary_logloss', eval_set=[(X_val, y_val)])

#    y_pred = clf.predict_proba(X_val)[:,1]

#    score = roc_auc_score(y_val, y_pred)

#    scores.append(score)

    

#    y_pred_test_av += clf.predict_proba(X)[:,1]

#    cv_iteration += clf.best_iteration_

#    print('CV Score of Fold_%d is %f' % (i, score))
#print(np.mean(scores))

#print(scores)

#y_pred_test_av /= 5

#X_train_pred = y_pred_test_av[:X_train.shape[0]]

#X_train_alike = X_train[np.argsort(X_train_pred[:, 0])][:X_train.shape[0] // 2]

#y_train_alike = y_train[np.argsort(X_train_pred[:, 0])][:X_train.shape[0] // 2]

#def objective(space):

#    scores = []



#    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



#    for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

#        X_train_, y_train_ = X_train[train_ix], y_train.values[train_ix]

#        X_val, y_val = X_train[test_ix], y_train.values[test_ix]



#        clf = LGBMClassifier(n_estimators=9999, **space) 



#        clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])

#        y_pred = clf.predict_proba(X_val)[:,1]

#        score = roc_auc_score(y_val, y_pred)

#        scores.append(score)

        

#    scores = np.array(scores)

#    print(scores.mean())

    

#    return -scores.mean()
#space ={

#        'max_depth': hp.choice('max_depth', np.arange(10, 30, dtype=int)),

#        'subsample': hp.uniform ('subsample', 0.8, 1),

#        'learning_rate' : hp.quniform('learning_rate', 0.025, 0.5, 0.025),

#        'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05)

#    }
#trials = Trials()



#best = fmin(fn=objective,

#              space=space, 

#              algo=tpe.suggest,

#              max_evals=30, 

#              trials=trials, 

#              rstate=np.random.RandomState(71) 

#             )
# CVしてスコアを見てみる。層化抽出で良いかは別途よく考えてみてください。

#scores = []

#y_pred_test_gbc = np.zeros(X_test.shape[0])



#skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



##skf.split(X_train_2, y_train)

#for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

#    X_train_, y_train_ = X_train[train_ix], y_train.values[train_ix]

#    X_val, y_val = X_train[test_ix], y_train.values[test_ix]

        

#    clf = GradientBoostingClassifier(learning_rate=0.15, n_estimators=70, min_samples_split=500, max_depth=6, min_samples_leaf=70, max_features=25, subsample=0.7)

    

#    clf.fit(X_train_, y_train_)

#    y_pred = clf.predict_proba(X_val)[:,1]

#    score = roc_auc_score(y_val, y_pred)

#    scores.append(score)



#    y_pred_test_gbc += clf.predict_proba(X_test)[:,1]



#    print('CV Score of Fold_%d is %f' % (i, score))
#print(np.mean(scores))

#print(scores)

#y_pred_test_gbc /= 5
# 全データで再学習し、testに対して予測する

#clf.fit(X_train, y_train)

#y_pred_gbc = clf.predict_proba(X_test)[:,1]
#LGBMClassifier(**best)
%%time

# CVしてスコアを見てみる

# なお、そもそもStratifiedKFoldが適切なのかは別途考える必要があります

scores = []

y_pred_test = np.zeros(X_test.shape[0])

cv_iteration = 0



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

#    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

#    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    X_train_, y_train_ = X_train[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train[test_ix], y_train.values[test_ix]

    

    clf = LGBMClassifier(boosting_type='gbdt', class_weight=None,

               colsample_bytree=0.6000000000000001, importance_type='split',

               learning_rate=0.025, max_depth=-1, min_child_samples=20,

               min_child_weight=0.001, min_split_gain=0.0, n_estimators=9999,

               n_jobs=-1, num_leaves=31, objective=None, random_state=71,

               reg_alpha=0.0, reg_lambda=0.0, silent=True,

               subsample=0.8638066621345636, subsample_for_bin=200000,

               subsample_freq=0)

    

    clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])

    y_pred = clf.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred)

    scores.append(score)

    

    y_pred_test += clf.predict_proba(X_test)[:,1]

    cv_iteration += clf.best_iteration_

    print('CV Score of Fold_%d is %f' % (i, score))
print(np.mean(scores))

print(scores)

y_pred_test /= 5

cv_iteration = (cv_iteration // 5) + 1
cv_iteration
#fi = clf.booster_.feature_importance(importance_type='gain')

#for i in range(100):

#    print(i,  fi[i])



#for i in range(len(fi)):

#    if fi[i] == 0:

#        print(i)



#idx = np.argsort(fi)[::-1]

#print(idx[fi == 0])

#top_cols, top_importances = idx[:100], fi[idx[:100]]

#print(top_cols, top_importances)
fig, ax = plt.subplots(figsize=(5, 30))

lgb.plot_importance(clf, max_num_features=100, ax=ax, importance_type='gain')
#for i in range(X_train_base.shape[1]):

#    print(i, X_train_base.columns[i])
# 全データで再学習し、testに対して予測する

clf = LGBMClassifier(boosting_type='gbdt', class_weight=None,

               colsample_bytree=0.6000000000000001, importance_type='split',

               learning_rate=0.025, max_depth=-1, min_child_samples=20,

               min_child_weight=0.001, min_split_gain=0.0, n_estimators=int(cv_iteration),

               n_jobs=-1, num_leaves=31, objective=None, random_state=71,

               reg_alpha=0.0, reg_lambda=0.0, silent=True,

               subsample=0.8638066621345636, subsample_for_bin=200000,

               subsample_freq=0)

clf.fit(X_train, y_train, eval_metric='auc')

y_pred = clf.predict_proba(X_test)[:,1]
#fi = clf.booster_.feature_importance(importance_type='gain')



#for i in range(len(fi)):

#    if fi[i] == 0:

#        print(i)
fig, ax = plt.subplots(figsize=(5, 30))

lgb.plot_importance(clf, max_num_features=100, ax=ax, importance_type='gain')
# こちらもスムーズな進行のために20分の１に間引いていますが、本番では"skiprows=lambda x: x%20!=0"を削除して用いてください。

#submission = pd.read_csv('../input/homework-for-students2/sample_submission.csv', index_col=0, skiprows=lambda x: x%20!=0)

#submission = pd.read_csv('../input/homework-for-students2/sample_submission.csv', index_col=0)



#submission.loan_condition = (y_pred + y_pred_test + y_pred_gbc + y_pred_test_gbc) / 4

#submission.to_csv('submission_1126_ensemble_all_1.csv')
len(y_pred)
submission = pd.read_csv('../input/homework-for-students2/sample_submission.csv', index_col=0)



submission.loan_condition = (y_pred + y_pred_test) / 2

submission.to_csv('submission_1126_ensemble_lgbm_1.csv')