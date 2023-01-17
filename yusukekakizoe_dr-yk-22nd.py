import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from category_encoders import OrdinalEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm

from lightgbm import LGBMClassifier



from sklearn.ensemble import GradientBoostingClassifier

import itertools



from numba import jit

from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

import warnings



warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)



import scipy as sp

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline
df_train = pd.read_csv('../input/homework-for-students2/train.csv', index_col=0, parse_dates=['issue_d','earliest_cr_line'])

df_test = pd.read_csv('../input/homework-for-students2/test.csv', index_col=0, parse_dates=['issue_d','earliest_cr_line'])



X_train = df_train

X_test = df_test



## 2014年以降を見る

X_train = X_train[X_train.issue_d.dt.year>=2014]



y_train = X_train.loan_condition.values

X_train = X_train.drop(['loan_condition'], axis=1 )



len_X_train = len(X_train)

len_X_test = len(X_test)



## ティズニー用:正規化した数値特徴量に関して処理する。

# Disney_train = X_train

# Disney_test = X_test
# cats = []

# for col in Disney_train.columns:

#     if Disney_train[col].dtype == 'object':

#         cats.append(col)

        

#         print(col, Disney_train[col].nunique())
# Disney_train.drop(cats, axis=1, inplace=True)

# Disney_test.drop(cats, axis=1, inplace=True)



# Disney_train.drop(['mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog'], axis=1, inplace=True)

# Disney_test.drop(['mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog'], axis=1, inplace=True)



# Disney_japan = pd.concat([Disney_train, Disney_test],  ignore_index=True)
# log_money = ('loan_amnt', 'annual_inc', 'tot_cur_amt' ,'tot_cur_bal')



# for i in log_money:

#     Disney_japan[i] = np.log1p(Disney_japan[i])
# Disney_col = Disney_train.columns

# Disney_col
# scaler = StandardScaler()



# for i in Disney_col:

#     Disney_japan[Disney_col] = scaler.fit_transform(Disney_japan[Disney_col])

    

# Disney_japan.head()
# for i in Disney_japan.columns:

#     print(i, Disney_japan[i].isnull().sum())
# drops = ['delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_util', 'collections_12_mths_ex_med', 'acc_now_delinq']
# Disney_japan.drop(drops, axis=1, inplace=True)

# Disney_japan.head()
# Disney_japan.fillna(0, inplace=True)
# tsne = TSNE(n_jobs=5)

# Disney_Tsney = tsne.fit_transform(Disney_japan)

# ティズニーしようと思ったが、結局計算時間が長すぎて諦める。
## 欠損値の情報をのこしておく

nan_train = X_train.isnull().sum(axis=1)

nan_test = X_test.isnull().sum(axis=1)



X_train['nan_num']=nan_train

X_test['nan_num']=nan_test



## 欠損であることがプラスなのかマイナスなのかよくわからないもの

nan_que_train = X_train.loc[:, ['mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog']].isnull().sum(axis=1)

nan_que_test = X_test.loc[:, ['mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog']].isnull().sum(axis=1)



X_train['nan_que'] = nan_que_train

X_test['nan_que'] = nan_que_test
## テキスト変換予定のものを避難させる

TXT_train1 = X_train.emp_title.copy()

TXT_test1 = X_test.emp_title.copy()



X_train.drop(['emp_title'], axis=1, inplace=True)

X_test.drop(['emp_title'], axis=1, inplace=True)



TXT_train2= X_train.title.copy()

TXT_test2 = X_test.title.copy()



X_train.drop(['title'], axis=1, inplace=True)

X_test.drop(['title'], axis=1, inplace=True)
## お金に関わっていそうなものの比をとる

## 計算失敗したものは見なかったことに...

moneys = ['loan_amnt', 'installment', 'annual_inc', 'revol_bal', 'tot_cur_bal']

iter_moneys = list(itertools.combinations(moneys,2))



for i, j in iter_moneys: 

    X_train['ratio_' + i + '_' + j] = X_train[i]/X_train[j]

    X_test['ratio_' + i + '_' + j] = X_test[i]/X_test[j]

    X_train['ratio_' + i + '_' + j].fillna(-9999, inplace=True)

    X_test['ratio_' + i + '_' + j].fillna(-9999, inplace=True)

    

## debt to income dti, to money    

X_train['dti_2_money'] = (X_train['dti']/100)*12*X_train['annual_inc']

X_test['dti_2_money'] = (X_test['dti']/100)*12*X_test['annual_inc']

    

X_train['annual_inc'].fillna(-9999, inplace=True)

X_test['annual_inc'].fillna(-9999, inplace=True)
X_train.head()
X_test.head()
cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, X_train[col].nunique())
# X_train.drop(['initial_list_status'], axis=1, inplace=True)

# X_test.drop(['initial_list_status'], axis=1, inplace=True)



# X_train.drop(['collections_12_mths_ex_med'], axis=1, inplace=True)

# X_test.drop(['collections_12_mths_ex_med'], axis=1, inplace=True)



# X_train.drop(['mths_since_last_delinq'], axis=1, inplace=True)

# X_test.drop(['mths_since_last_delinq'], axis=1, inplace=True)



# X_train.drop(['application_type'], axis=1, inplace=True)

# X_test.drop(['application_type'], axis=1, inplace=True)



# X_train.drop(['delinq_2yrs'], axis=1, inplace=True)

# X_test.drop(['delinq_2yrs'], axis=1, inplace=True)
##代入

X_train['year_d']=df_train.issue_d.dt.year

X_train['month_d']=df_train.issue_d.dt.month

X_test['year_d']=df_test.issue_d.dt.year

X_test['month_d']=df_test.issue_d.dt.month



X_train['year_cr']=df_train.earliest_cr_line.dt.year

X_train['month_cr']=df_train.earliest_cr_line.dt.month

X_test['year_cr']=df_test.earliest_cr_line.dt.year

X_test['month_cr']=df_test.earliest_cr_line.dt.year
df_spi = pd.read_csv('../input/homework-for-students2/spi.csv', parse_dates=['date'])

df_spi['year_d']=df_spi.date.dt.year

df_spi['month_d']=df_spi.date.dt.month
df_temp =df_spi.groupby(['year_d', 'month_d'], as_index=False)['close'].mean()
X_train = X_train.merge(df_temp, on=['year_d', 'month_d'], how='left')

X_test = X_test.merge(df_temp, on=['year_d', 'month_d'], how='left')
X_train.drop(['issue_d'], axis=1, inplace=True)

X_test.drop(['issue_d'], axis=1, inplace=True)

X_train.drop(['earliest_cr_line'], axis=1, inplace=True)

X_test.drop(['earliest_cr_line'], axis=1, inplace=True)
gdp_data = pd.read_csv('../input/homework-for-students2/US_GDP_by_State.csv', index_col=0)

state_data = pd.read_csv('../input/homework-for-students2/statelatlong.csv', index_col=0)
gdp_ind = gdp_data.index

state_ind = state_data.index
gdp_data['City']=gdp_ind

state_data['addr_state']=state_ind



gdp_data['per_gdp'] = (gdp_data['Gross State Product']-gdp_data['State & Local Spending'])/gdp_data['Population (million)']
gdp_data.drop(['Gross State Product'], axis=1, inplace=True)

gdp_data.drop(['Real State Growth %'], axis=1, inplace=True)

gdp_data.drop(['Population (million)'], axis=1, inplace=True)

gdp_data.drop(['State & Local Spending'], axis=1, inplace=True)



mean_state_gdp = gdp_data.groupby(['City'], as_index=False)['per_gdp'].mean()

gdp_state_data = state_data.merge(mean_state_gdp, on=['City'], how='left')

gdp_state_data.drop(['City'], axis=1, inplace=True)
gdp_state_data.head()
X_train=X_train.merge(gdp_state_data, on=['addr_state'], how='left')

X_test=X_test.merge(gdp_state_data, on=['addr_state'], how='left')
X_train.drop(['zip_code'], axis=1, inplace=True)

X_test.drop(['zip_code'], axis=1, inplace=True)
cnt_enc = ['sub_grade', 'home_ownership', 'purpose', 'addr_state' ]



len_X_train = len(X_train)

X_japan = pd.concat([X_train, X_test], ignore_index=True)



for i in cnt_enc:

    X_japan['cnt_' + i] = X_japan[i].map(X_japan[i].value_counts())

    X_train['cnt_' + i] = X_japan['cnt_' + i][:len_X_train]

    X_test['cnt_' + i]  = np.array(X_japan['cnt_' + i][len_X_train:])
X_test.head()
X_train.head()
# @jit

# def counting (XX):

#     amnt_class_ = []

#     for ind in range( len(XX) ):

#         if XX['loan_amnt'][ind]<=5000:

#             amnt_class_.append(5000)

#         elif XX['loan_amnt'][ind]>5000 and XX['loan_amnt'][ind]<= 10000:

#             amnt_class_.append(10000)

#         elif XX['loan_amnt'][ind]>10000 and XX['loan_amnt'][ind]<= 15000:

#             amnt_class_.append(15000)

#         elif XX['loan_amnt'][ind]>15000 and XX['loan_amnt'][ind]<= 20000:

#             amnt_class_.append(20000)

#         elif XX['loan_amnt'][ind]>20000 and XX['loan_amnt'][ind]<= 25000:

#             amnt_class_.append(25000)

#         elif XX['loan_amnt'][ind]>25000 and XX['loan_amnt'][ind]<= 30000:

#             amnt_class_.append(30000)

#         elif XX['loan_amnt'][ind]>30000 and XX['loan_amnt'][ind]<= 35000:

#             amnt_class_.append(35000)

#         elif XX['loan_amnt'][ind] >35000 and XX['loan_amnt'][ind]<= 40000:

#             amnt_class_.append(40000)

#         else: 

#             amnt_class_.append(-9999)

    

#     return amnt_class_
# amnt_class = counting(X_japan)

# X_japan['amnt_class']=np.array(amnt_class)
# X_japan['cnt_amnt_class'] = X_japan['amnt_class'].map( X_japan['amnt_class'].value_counts() )



# X_train['cnt_amnt_class'] = X_japan['cnt_amnt_class'][:len_X_train]

# X_test['cnt_amnt_class'] =  np.array( X_japan['cnt_amnt_class'][len_X_train:] )
X_japan.head()
X_train.head()
X_test.head()
emp_length_mapping = {'6 years':6, '4 years':4, '< 1 year':0, '10+ years':10, '3 years':3, '2 years':2,

 '8 years':8, '1 year':1, '9 years':9, '7 years':7, '5 years':5}



X_train['emp_length']=X_train['emp_length'].map(emp_length_mapping)

X_test['emp_length']=X_test['emp_length'].map(emp_length_mapping)

X_train['emp_length'].fillna(-9999, inplace=True)

X_test['emp_length'].fillna(-9999, inplace=True)
grade_mapping ={'A':7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1}

X_train['grade']=X_train['grade'].map(grade_mapping)

X_test['grade']=X_test['grade'].map(grade_mapping)
sub_grade_mapping ={'A1':35, 'A2':34, 'A3':33, 'A4':32, 'A5':31, 'B1':30, 'B2':29, 'B3':28,

                'B4':27, 'B5':26, 'C1':25, 'C2':24, 'C3':23, 'C4':22, 'C5':21, 'D1':20,

                'D2':19, 'D3':18, 'D4':17, 'D5':16, 'E1':15, 'E2':14, 'E3':13, 'E4':12,

                'E5':11, 'F1':10, 'F2':9, 'F3':8, 'F4':7, 'F5':6, 'G1':5, 'G2':4, 'G3':3,

                'G4':2, 'G5':1 }



X_train['sub_grade']=X_train['sub_grade'].map(sub_grade_mapping)

X_test['sub_grade']=X_test['sub_grade'].map(sub_grade_mapping)
home_mapping ={'OWN':5, 'MORTGAGE':4, 'RENT':3, 'OTHER':2, 'NONE':1, 'ANY':-9999 }



X_train['home_ownership']=X_train['home_ownership'].map(home_mapping)

X_test['home_ownership']=X_test['home_ownership'].map(home_mapping)
## 効いてそうな量を序列とかけまくる

multi = ['grade', 'sub_grade', 'home_ownership', 'emp_length']

passive = ['loan_amnt', 'annual_inc', 'installment', 'revol_bal', 'tot_cur_bal', 'year_cr']



for i in multi:

    for j in passive:

        X_train[ i + '_mul_' + j] = X_train[i]*X_train[j]

        X_test[ i + '_mul_' + j] = X_test[i]*X_test[j]
X_train.head()
X_test.head()
## 欠損埋め

X_train['month_cr'].fillna(-9999, inplace=True)

X_test['month_cr'].fillna(-9999, inplace=True)

X_train['year_cr'].fillna(-9999, inplace=True)

X_test['year_cr'].fillna(-9999, inplace=True)



## 整数値に変換

X_train['month_cr']=X_train['month_cr'].astype(int)

X_test['month_cr']=X_test['month_cr'].astype(int)

X_train['year_cr']=X_train['year_cr'].astype(int)

X_test['year_cr']=X_test['year_cr'].astype(int)
X_train['sub_mul_length'] = X_train['sub_grade']*X_train['emp_length']

X_test['sub_mul_length'] = X_test['sub_grade']*X_test['emp_length']
X_train.head()
X_test.head()
## ダメ押しのカテゴリカルエンコード（エラー回避）

cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, X_train[col].nunique())
encoder = OrdinalEncoder(cols=cats)

X_train[cats]=encoder.fit_transform(X_train[cats])

X_test[cats]=encoder.transform(X_test[cats])
TXT_train1.fillna('#', inplace=True)

TXT_test1.fillna('#', inplace=True)
# counts_vec = CountVectorizer(max_features=15000)

# Txitter1 = pd.concat([TXT_train1, TXT_test1])

# counts_vec.fit(Txitter1)

tfidf = TfidfVectorizer(max_features=10000)

Txitter1 = pd.concat([TXT_train1, TXT_test1])

tfidf.fit(Txitter1)
# TXT_train1_trans = counts_vec.transform(TXT_train1)

# TXT_test1_trans = counts_vec.transform(TXT_test1)



TXT_train1_trans = tfidf.transform(TXT_train1)

TXT_test1_trans = tfidf.transform(TXT_test1)
X_train = sp.sparse.hstack([X_train, TXT_train1_trans])

X_test = sp.sparse.hstack([X_test, TXT_test1_trans])
TXT_train2.fillna('#', inplace=True)

TXT_test2.fillna('#', inplace=True)
# counts_vec = CountVectorizer(max_features=10000)

# Txitter2 = pd.concat([TXT_train2, TXT_test2])

# counts_vec.fit(Txitter2)



tfidf = TfidfVectorizer(max_features=10000)

Txitter2 = pd.concat([TXT_train2, TXT_test2])

tfidf.fit(Txitter2)
TXT_train2_trans = tfidf.transform(TXT_train2)

TXT_test2_trans = tfidf.transform(TXT_test2)



# TXT_train2_trans = counts_vec.transform(TXT_train2)

# TXT_test2_trans = counts_vec.transform(TXT_test2)
X_train = sp.sparse.hstack([X_train, TXT_train2_trans])

X_test = sp.sparse.hstack([X_test, TXT_test2_trans])
## 疎行列のままLight GBMに投げる

X_train = X_train.tocsr()

X_test  = X_test.tocsr()
# X_train
# X_test
# アンサンブル的なもの



# scores = []

skf = StratifiedKFold(n_splits=10, random_state=71, shuffle=True)

y_pred_test = np.zeros(len_X_test)



for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train[train_ix], y_train[train_ix]

    X_val, y_val = X_train[test_ix], y_train[test_ix]

    

    clf = LGBMClassifier(

        learning_rate = 0.05,

        num_leaves=31,

        colsample_bytree=0.9,

        subsample=0.9,

        n_estimators=9999,

        random_state=71,

        importance_type='gain')



    clf.fit(X_train_, y_train_, 

            early_stopping_rounds=100,

            eval_metric='auc',

            eval_set=[(X_val, y_val)])

#     y_pred = clf.predict_proba(X_val)[:,1]

    y_pred_test += clf.predict_proba(X_test)[:,1]



#     score = roc_auc_score(y_val, y_pred)

#     scores.append(score)

#     print('CV Score of Fold_%d is %f' % (i, score))

    

    y_pred_test += clf.predict_proba(X_test)[:,1]

    y_pred_test /= 10
submission = pd.read_csv('../input/homework-for-students2/sample_submission.csv', index_col=0)



# y_pred = clf.predict_proba(X_test)[:,1]

submission.loan_condition = y_pred_test

submission.to_csv('submission.csv')
# pd.set_option('display.max_columns', 100)

# pd.set_option('display.max_rows', 100)



# importance = pd.DataFrame(clf.feature_importances_, index=X_train.columns, columns=['importance'])

# importance = importance.sort_values('importance', ascending=False)

# display(importance)