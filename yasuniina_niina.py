# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm



from sklearn.ensemble import GradientBoostingClassifier

import itertools

from sklearn.decomposition import PCA

import lightgbm as lgb

import eli5

import math

from sklearn.cluster import MiniBatchKMeans

from lightgbm import LGBMClassifier
#ハンズオンのスムーズな進行のために全体の20分の1だけ読み込むことにします。

#実際に課題でモデリングする際には"skiprows=lambda x: x%20!=0"を削除してください。

df_train = pd.read_csv('/kaggle/input/homework-for-students4plus/train.csv', index_col=0) #, skiprows=lambda x: x%20!=0)

#df_train = pd.read_csv('../input/homework-for-students2/train.csv', index_col=0)

df_test = pd.read_csv('/kaggle/input/homework-for-students4plus/test.csv', index_col=0)

df_spi = pd.read_csv('/kaggle/input/homework-for-students4plus/spi.csv', index_col=0)



df_train['issue_d'] = pd.to_datetime(df_train['issue_d'])

df_test['issue_d'] = pd.to_datetime(df_test['issue_d'])

df_spi.index = pd.to_datetime(df_spi.index)
df_spi_fet = pd.DataFrame()

df_spi_fet = df_spi.resample('M').mean()

df_spi_fet = df_spi_fet.rename(columns={'close':'mean'})



#df_spi_fet['max'] = df_spi.resample('M').max()

#df_spi_fet['min'] = df_spi.resample('M').min()

#df_spi_fet['median'] = df_spi.resample('M').median()



#df_spi_fet['mean_shift1'] = df_spi_fet['mean'].shift(1)

#df_spi_fet['mean_diff1'] = df_spi_fet['mean'].diff(1)



#for i in range(12):

#    i+=1

#    df_spi_fet['mean_shift'+str(i)] = df_spi_fet['mean'].shift(i)

#    df_spi_fet['mean_diff' +str(i)] = df_spi_fet['mean'].diff(i)

#    df_spi_fet['mean_shift-'+str(i)] = df_spi_fet['mean'].shift(-i)

#    df_spi_fet['mean_diff-' +str(i)] = df_spi_fet['mean'].diff(-i)



df_spi_fet = df_spi_fet.reset_index()
df_train = pd.merge(df_train.assign(grouper=df_train['issue_d'].dt.to_period('M')),

                    df_spi_fet.assign(grouper=df_spi_fet['date'].dt.to_period('M')),

                    how='left', on='grouper').drop(['grouper', 'date'], axis=1)

df_test = pd.merge(df_test.assign(grouper=df_test['issue_d'].dt.to_period('M')),

                   df_spi_fet.assign(grouper=df_spi_fet['date'].dt.to_period('M')),

                   how='left', on='grouper').drop(['grouper', 'date'], axis=1)
# まずは2013年から取り込む

df_train = df_train[df_train.issue_d.dt.year >= 2013]



df_train = df_train.drop([#'issue_d', 

                          'initial_list_status'], axis=1)

df_test = df_test.drop([#'issue_d', 

                        'initial_list_status'], axis=1)
y_train = df_train.loan_condition

#X_train = df_train.drop(['loan_condition'], axis=1)

X_train = df_train



X_test = df_test
# dtypeがobject（数値でないもの）のカラム名とユニーク数を確認してみましょう。

cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, X_train[col].nunique(), X_train[col].isnull().sum())
# dtypeが数値のカラム名とユニーク数を確認してみましょう。

nums = []

for col in X_train.columns:

    if X_train[col].dtype != 'object':

        nums.append(col)

        

        print(col, X_train[col].nunique(), X_train[col].isnull().sum())
X_test_pr = X_test

X_train_pr = X_train
col = 'loan_amnt'

X_test[col] = X_test[col].apply(lambda x: 35000 if x>35000 else x)

X_train_pr[col] = np.log1p(X_train[col])

X_test_pr[col] = np.log1p(X_test[col])
col = 'installment'

X_train_pr[col] = np.log1p(X_train[col])

X_test_pr[col] = np.log1p(X_test[col])
col = 'annual_inc'

X_train_pr[col] = np.log1p(X_train[col])

X_test_pr[col] = np.log1p(X_test[col])
col = 'dti'

X_train_pr[col] = X_train[col].apply(lambda x: 0 if x<0 else x)

X_test_pr[col] = X_test[col].apply(lambda x: 0 if x<0 else x)

X_train_pr[col] = np.log1p(X_train[col])

X_test_pr[col] = np.log1p(X_test[col])
col = 'delinq_2yrs'

X_train_pr[col] = np.log1p(X_train[col])

X_test_pr[col] = np.log1p(X_test[col])
col = 'mths_since_last_delinq'

X_train_pr[col] = np.log1p(X_train[col])

X_test_pr[col] = np.log1p(X_test[col])
col = 'open_acc'

X_train_pr[col] = np.log1p(X_train[col])

X_test_pr[col] = np.log1p(X_test[col])
col = 'pub_rec'

X_train_pr[col] = np.log1p(X_train[col])

X_test_pr[col] = np.log1p(X_test[col])
col = 'revol_bal'

X_train_pr[col] = np.log1p(X_train[col])

X_test_pr[col] = np.log1p(X_test[col])
col = 'revol_util'

X_train_pr[col] = np.log1p(X_train[col])

X_test_pr[col] = np.log1p(X_test[col])
col = 'total_acc'

X_train_pr[col] = np.log1p(X_train[col])

X_test_pr[col] = np.log1p(X_test[col])
col = 'collections_12_mths_ex_med'

X_train_pr[col] = np.log1p(X_train[col])

X_test_pr[col] = np.log1p(X_test[col])
col = 'mths_since_last_major_derog'

X_train_pr[col] = np.log1p(X_train[col])

X_test_pr[col] = np.log1p(X_test[col])
col = 'acc_now_delinq'

X_train_pr[col] = np.log1p(X_train[col])

X_test_pr[col] = np.log1p(X_test[col])
col = 'tot_coll_amt'

X_train_pr[col] = np.log1p(X_train[col])

X_test_pr[col] = np.log1p(X_test[col])
col = 'tot_cur_bal'

X_train_pr[col] = np.log1p(X_train[col])

X_test_pr[col] = np.log1p(X_test[col])
# emp_title

emp_title_train = X_train_pr.emp_title.copy()

emp_title_test = X_test_pr.emp_title.copy()



emp_title_train.fillna('#', inplace=True)

emp_title_test.fillna('#', inplace=True)



emp_tfidf = TfidfVectorizer(max_features=1000, use_idf=True)

TXT_train = emp_tfidf.fit_transform(emp_title_train).tocsr()

TXT_test = emp_tfidf.transform(emp_title_test).tocsr()



from sklearn.linear_model import LogisticRegression



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

stack_train = np.zeros(len(X_train))

stack_test = np.zeros(len(X_test))



for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

    X_train_tfidf, y_train_tfidf = TXT_train[train_ix], y_train.values[train_ix]

    X_val_tfidf, y_val_tfidf = TXT_train[test_ix], y_train.values[test_ix]



    

    clf = LGBMClassifier(

        max_depth=3,

        learning_rate = 0.02,

        colsample_bytree=0.7,

        subsample=0.7,

        min_split_gain=0,

        reg_lambda=1,

        reg_alpha=1,

        min_child_weight=2,

        n_estimators=9999,

        random_state=71,

        importance_type='gain')

    clf.fit(X_train_tfidf, y_train_tfidf,

            early_stopping_rounds=500,

            verbose=100,

            eval_metric='auc',

            eval_set=[(X_val_tfidf, y_val_tfidf)])

    

    y_pred = clf.predict_proba(X_val_tfidf)[:,1]

    stack_train[test_ix] = y_pred

    score = roc_auc_score(y_val_tfidf, y_pred)

    print('CV Score of Fold_%d is %f' % (i, score))

    

    stack_test += clf.predict_proba(TXT_test)[:,1]



stack_test /= 5



X_train_pr['tfidf_emp_title'] = stack_train

X_test_pr['tfidf_emp_title'] = stack_test



# 他の特徴量は2015年のみ使う

X_train_pr = X_train_pr[X_train_pr.issue_d.dt.year >= 2015]

#X_train_pr = X_train_pr[X_train_pr.issue_d.dt.month >= 4]

X_train_pr = X_train_pr.drop(['issue_d'], axis=1)

X_test_pr  = X_test_pr.drop(['issue_d'], axis=1)



y_train = X_train_pr.loan_condition

X_train_pr = X_train_pr.drop(['loan_condition'], axis=1)
titles = ['Debt consolidation','Credit card refinancing','Home improvement','Other',

          'Major purchase','Medical expenses','Car financing','Business','Moving and relocation',

          'Vacation','Home buying','Green loan']



X_train_pr['title'] = X_train_pr['title'].apply(lambda x: x if x in titles else '#####')

X_test_pr['title'] = X_test_pr['title'].apply(lambda x: x if x in titles else '#####')



X_train_pr['title'] = X_train_pr['title'].fillna('#####')

X_test_pr['title'] = X_test_pr['title'].fillna('#####')
X_train_pr = X_train_pr.replace({'grade':{'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}})

X_test_pr = X_test_pr.replace({'grade':{'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}})

X_train_pr['grade'] = X_train_pr['grade'].astype(int)

X_test_pr['grade'] = X_test_pr['grade'].astype(int)
X_train_pr = X_train_pr.replace({'sub_grade':{'A1':1,'A2':2,'A3':3,'A4':4,'A5':5,

                                    'B1':6,'B2':7,'B3':8,'B4':9,'B5':10,

                                    'C1':11,'C2':12,'C3':13,'C4':14,'C5':15,

                                    'D1':16,'D2':17,'D3':18,'D4':19,'D5':20,

                                    'E1':21,'E2':22,'E3':23,'E4':24,'E5':25,

                                    'F1':26,'F2':27,'F3':28,'F4':29,'F5':30,

                                   'G1':31,'G2':32,'G3':33,'G4':34,'G5':35}})

X_test_pr = X_test_pr.replace({'sub_grade':{'A1':1,'A2':2,'A3':3,'A4':4,'A5':5,

                                    'B1':6,'B2':7,'B3':8,'B4':9,'B5':10,

                                    'C1':11,'C2':12,'C3':13,'C4':14,'C5':15,

                                    'D1':16,'D2':17,'D3':18,'D4':19,'D5':20,

                                    'E1':21,'E2':22,'E3':23,'E4':24,'E5':25,

                                    'F1':26,'F2':27,'F3':28,'F4':29,'F5':30,

                                   'G1':31,'G2':32,'G3':33,'G4':34,'G5':35}})

X_train_pr['sub_grade'] = X_train_pr['sub_grade'].astype(int)

X_test_pr['sub_grade'] = X_test_pr['sub_grade'].astype(int)
X_train_pr = X_train_pr.replace({'emp_length':{'< 1 year':0.5,'1 year':1,'2 years':2,'3 years':3,

                                        '4 years':4,'5 years':5,'6 years':6,'7 years':7,

                                        '8 years':8,'9 years':9,'10+ years':10}})  

X_test_pr = X_test_pr.replace({'emp_length':{'< 1 year':0.5,'1 year':1,'2 years':2,'3 years':3,

                                        '4 years':4,'5 years':5,'6 years':6,'7 years':7,

                                        '8 years':8,'9 years':9,'10+ years':10}})
X_train_pr['earliest_cr_line'] = pd.to_datetime(X_train_pr['earliest_cr_line'])

X_test_pr['earliest_cr_line']  = pd.to_datetime(X_test_pr['earliest_cr_line'])



X_train_pr['earliest_cr_line'] = X_train_pr['earliest_cr_line'].dt.year

X_test_pr['earliest_cr_line']  = X_test_pr['earliest_cr_line'].dt.year
# dtypeがobject（数値でないもの）のカラム名とユニーク数を確認してみましょう。

cats = []

for col in X_train_pr.columns:

    if X_train_pr[col].dtype == 'object':

        cats.append(col)

        print(col, X_train_pr[col].nunique(), X_train_pr[col].isnull().sum())

        print(col, X_test_pr[col].nunique(), X_test_pr[col].isnull().sum())
# dtypeが数値のカラム名とユニーク数を確認してみましょう。

nums = []

for col in X_train_pr.columns:

    if X_train_pr[col].dtype != 'object':

        nums.append(col)

        print(col, X_train_pr[col].nunique(), X_train_pr[col].isnull().sum())

        print(col, X_test_pr[col].nunique(), X_test_pr[col].isnull().sum())
# カテゴリ変数のOrdinalエンコーディング

encoder = OrdinalEncoder(cols=cats)

X_train_numcat = pd.concat([X_train_pr[nums], encoder.fit_transform(X_train_pr[cats])], axis=1)

X_test_numcat  = pd.concat([X_test_pr[nums],  encoder.transform(X_test_pr[cats])], axis=1)
#Countエンコーディング

for col in cats:

    summary = X_train_numcat[col].value_counts()

    X_train_numcat[col+'_cnt'] = X_train_numcat[col].map(summary)

    X_test_numcat[col+'_cnt']  = X_test_numcat[col].map(summary)
# Targetエンコーディング



target = 'loan_condition'

X_temp = pd.concat([X_train_numcat, y_train], axis=1)

cats += ['sub_grade', 'grade'] #, 'earliest_cr_line']

#cats += ['earliest_cr_line', 'emp_length', 'sub_grade', 'grade', 'emp_title', 'title']

for col in cats:

    summary = X_temp.groupby([col])[target].mean()

    X_test_numcat[col+'_tgt'] = X_test_numcat[col].map(summary)

    skf = StratifiedKFold(n_splits=7, random_state=42, shuffle=True)

    enc_train = Series(np.zeros(len(X_train_numcat)), index=X_train_numcat.index)

    for i, (train_ix, val_ix) in enumerate((skf.split(X_train_numcat, y_train))):

        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]

        summary = X_train_.groupby([col])[target].mean()

        enc_train.iloc[val_ix] = X_val[col].map(summary)

    X_train_numcat[col+'_tgt'] = enc_train
X_train_numcat = X_train_numcat.replace([np.inf, -np.inf, np.nan], -99999)

X_test_numcat = X_test_numcat.replace([np.inf, -np.inf, np.nan], -99999)
nums2comb = ['loan_amnt', 'installment', 'annual_inc', 'dti', 'revol_bal', 'revol_util', 'tot_cur_bal',

            'mths_since_last_delinq', 'open_acc', 'total_acc', 'mths_since_last_major_derog', 'tot_coll_amt']

#            'acc_now_delinq', 'collections_12_mths_ex_med', 'pub_rec', 'mths_since_last_record', 'inq_last_6mths', 'delinq_2yrs']



X_train_numcatfet = X_train_numcat

X_test_numcatfet  = X_test_numcat



X_train_numcatfet['tot_cur_bal+annual_inc*2'] = X_train_numcatfet['tot_cur_bal'] + X_train_numcatfet['annual_inc']*2

X_test_numcatfet['tot_cur_bal+annual_inc*2'] = X_test_numcatfet['tot_cur_bal'] + X_test_numcatfet['annual_inc']*2



#特徴の追加

X_train_numcatfet['annual_inc_emp_length']=round(X_train_numcatfet['annual_inc']*X_train_numcatfet['emp_length'],5)

X_test_numcatfet['annual_inc_emp_length']=round(X_test_numcatfet['annual_inc']*X_test_numcatfet['emp_length'],5)





#X_train_numcatfet['annual_inc_grade']=round(X_train_numcatfet['annual_inc'] / X_train_numcatfet['sub_grade'],5)

#X_test_numcatfet['annual_inc_grade'] =round(X_test_numcatfet['annual_inc'] / X_test_numcatfet['sub_grade'],5)



X_train_numcatfet['tot_cur_bal_grade']=round(X_train_numcatfet['tot_cur_bal'] * X_train_numcatfet['sub_grade'],5)

X_test_numcatfet['tot_cur_bal_grade'] =round(X_test_numcatfet['tot_cur_bal'] * X_test_numcatfet['sub_grade'],5)



X_train_numcatfet['dti_grade']=round(X_train_numcatfet['dti'] * X_train_numcatfet['sub_grade'],5)

X_test_numcatfet['dti_grade'] =round(X_test_numcatfet['dti'] * X_test_numcatfet['sub_grade'],5)



X_train_numcatfet['loan_amnt_grade']=round(X_train_numcatfet['loan_amnt'] * X_train_numcatfet['sub_grade'],5)

X_test_numcatfet['loan_amnt_grade']=round(X_test_numcatfet['loan_amnt'] * X_test_numcatfet['sub_grade'],5)



X_train_numcatfet['open_acc_grade']=round(X_train_numcatfet['open_acc'] * X_train_numcatfet['sub_grade'],5)

X_test_numcatfet['open_acc_grade']=round(X_test_numcatfet['open_acc'] * X_test_numcatfet['sub_grade'],5)



X_train_numcatfet['home_ownership_grade']=round(X_train_numcatfet['home_ownership'] * X_train_numcatfet['sub_grade'],5)

X_test_numcatfet['home_ownership_grade']=round(X_test_numcatfet['home_ownership'] * X_test_numcatfet['sub_grade'],5)





X_train_numcatfet['installment_grade']=round(X_train_numcatfet['installment'] * X_train_numcatfet['sub_grade'],5)

X_test_numcatfet['installment_grade']=round(X_test_numcatfet['installment'] * X_test_numcatfet['sub_grade'],5)



X_train_numcatfet['revol_bal_grade']=round(X_train_numcatfet['revol_bal'] * X_train_numcatfet['sub_grade'],5)

X_test_numcatfet['revol_bal_grade']=round(X_test_numcatfet['revol_bal'] * X_test_numcatfet['sub_grade'],5)





for elm in list(itertools.combinations(nums2comb, 2)):

    a = elm[0]

    b = elm[1]

    X_train_numcatfet[a+'/'+b] = round(X_train_numcatfet[a] / X_train_numcatfet[b], 5)

    X_test_numcatfet[a+'/'+b]  = round(X_test_numcatfet[a] / X_test_numcatfet[b], 5)

#    X_train_numcatfet[a+'*'+b] = round(X_train_numcatfet[b] * X_train_numcatfet[a], 5)

#    X_test_numcatfet[a+'*'+b]  = round(X_test_numcatfet[b] * X_test_numcatfet[a], 5)

#    X_train_numcatfet[a+'-'+b] = round(X_train_numcatfet[a] - X_train_numcatfet[b], 5)

#    X_test_numcatfet[a+'-'+b]  = round(X_test_numcatfet[a] - X_test_numcatfet[b], 5)
"""

def make_radian_row(pca_result):

    rad = []

    for r in pca_result:

        rad.append(math.atan(r[0]/r[1]))

    return rad



X_train_numcatfet = X_train_numcatfet.replace([np.inf, -np.inf, np.nan], -99999)

X_test_numcatfet  = X_test_numcatfet.replace([np.inf, -np.inf, np.nan], -99999)



# 主成分分析

pca = PCA(n_components=2)

pca.fit(X_train_numcatfet[['open_acc*grade', 'dti*grade', 'loan_amnt*grade', 'home_ownership*grade', 'tfidf_emp_title']])



# 角度データの追加

X_train_numcatfetrad = X_train_numcatfet

X_test_numcatfetrad  = X_test_numcatfet



X_train_numcatfetrad["rad"] = make_radian_row(pca.transform(X_train_numcatfetrad[['open_acc*grade', 'dti*grade', 'loan_amnt*grade', 'home_ownership*grade', 'tfidf_emp_title']]))

X_test_numcatfetrad["rad"]  = make_radian_row(pca.transform(X_test_numcatfetrad[['open_acc*grade', 'dti*grade', 'loan_amnt*grade', 'home_ownership*grade', 'tfidf_emp_title']]))

"""
#X_train_numcatfetradclst = X_train_numcatfetrad

#X_test_numcatfetradclst  = X_test_numcatfetrad
drop1=[

'collections_12_mths_ex_med',

'open_acc/tot_coll_amt',

'delinq_2yrs',

'application_type_tgt',

'revol_util/tot_coll_amt',

'mths_since_last_delinq/tot_coll_amt',

'installment/mths_since_last_major_derog',

'annual_inc/mths_since_last_major_derog',

'loan_amnt/tot_coll_amt',

'mths_since_last_major_derog/tot_coll_amt',

'home_ownership_cnt',

'annual_inc/tot_coll_amt',

'installment/tot_coll_amt',

'application_type',

'acc_now_delinq',

'pub_rec',

'title',

'grade',

'application_type_cnt']



drop2=[

'revol_util/mths_since_last_delinq',

'home_ownership',

'total_acc/tot_coll_amt',

'open_acc',

'mths_since_last_delinq/mths_since_last_major_derog',

'loan_amnt/mths_since_last_major_derog',

'purpose_cnt',

'revol_util/mths_since_last_major_derog',

'tot_cur_bal/tot_coll_amt',

'dti/mths_since_last_major_derog',

'loan_amnt/mths_since_last_delinq',

'annual_inc/mths_since_last_delinq',

'total_acc/mths_since_last_major_derog',

'dti/tot_coll_amt',

'revol_bal/mths_since_last_major_derog',

'revol_bal/mths_since_last_delinq',

'grade_tgt',

'revol_bal/tot_coll_amt',



'loan_amnt/total_acc',

'revol_bal/tot_cur_bal',

'installment/total_acc',

'installment/revol_util',

'installment/open_acc',

'loan_amnt/revol_util',

'title_cnt',

'revol_bal/total_acc',

'dti/open_acc',

'revol_util/total_acc',

'total_acc',

'installment/dti',

'tot_coll_amt',

'loan_amnt/open_acc',

'mths_since_last_delinq/total_acc',

'loan_amnt/dti',

'revol_util/open_acc',

'purpose_tgt'

]





X_train_numcatfet = X_train_numcatfet.drop(drop1, axis=1)

X_test_numcatfet = X_test_numcatfet.drop(drop1, axis=1)



X_train_numcatfet = X_train_numcatfet.drop(drop2, axis=1)

X_test_numcatfet = X_test_numcatfet.drop(drop2, axis=1)
from sklearn.neural_network import MLPClassifier



scores = []

y_pred_test = np.zeros(len(X_test_numcatfet))



random_states = [71, 42, 99, 55]

n_splits = 5

for r in random_states:

    skf = StratifiedKFold(n_splits=n_splits, random_state=r, shuffle=True)

    for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train_numcatfet, y_train))):

        X_train_, y_train_ = X_train_numcatfet.values[train_ix], y_train.values[train_ix]

        X_val, y_val = X_train_numcatfet.values[test_ix], y_train.values[test_ix]

        clf = LGBMClassifier(

            max_depth=3,

            learning_rate = 0.02,

            colsample_bytree=0.7,

            subsample=0.7,

            min_split_gain=0,

            reg_lambda=1,

            reg_alpha=1,

            min_child_weight=2,

            n_estimators=9999,

            random_state=r,

            importance_type='gain')

        clf.fit(X_train_, y_train_,

                early_stopping_rounds=500,

                verbose=100,

                eval_metric='auc',

                eval_set=[(X_val, y_val)])

        y_pred = clf.predict_proba(X_val)[:,1]

        score = roc_auc_score(y_val, y_pred)

        scores.append(score)    

        print('CV Score of Fold_%d is %f' % (i, score))

        y_pred_test += clf.predict_proba(X_test_numcatfet)[:,1]



    

# 平均スコアを算出

print(np.array(scores).mean())

y_pred_test /= n_splits * len(random_states)

submission = pd.read_csv('/kaggle/input/homework-for-students4plus/sample_submission.csv', index_col=0)

submission.loan_condition = y_pred_test

submission.to_csv('submission.csv')
# 全データで再学習し、testに対して予測した際のfeature importanceを表示

X_train_numcatfet = X_train_numcatfet.replace([np.inf, -np.inf, np.nan], -99999)

X_test_numcatfet = X_test_numcatfet.replace([np.inf, -np.inf, np.nan], -99999)

clf = LGBMClassifier(

    max_depth=3,

    learning_rate = 0.02,

    colsample_bytree=0.7,

    subsample=0.7,

    min_split_gain=0,

    reg_lambda=1,

    reg_alpha=1,

    min_child_weight=2,

    n_estimators=9999,

    random_state=71,

    importance_type='gain')

    

clf.fit(X_train_, y_train_,

        early_stopping_rounds=500,

        verbose=100,

        eval_metric='auc',

        eval_set=[(X_val, y_val)])

#y_pred = clf.predict_proba(X_test_numcatfet)[:,1]

# sample submissionを読み込んで、予測値を代入の後、保存する

#submission = pd.read_csv('./data/sample_submission.csv', index_col=0)

#submission.loan_condition = y_pred

#submission.to_csv('submission.csv')

eli5.show_weights(clf, feature_names = X_test_numcatfet.columns.tolist(),top=300)