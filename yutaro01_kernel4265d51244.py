# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

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
keizai=pd.read_csv('../input/homework-for-students3/spi.csv')

df_train = pd.read_csv('../input/homework-for-students3/train.csv', index_col=0)



df_train['issue_d']=pd.to_datetime(df_train['issue_d'])

df_train['earliest_cr_line']=pd.to_datetime(df_train['earliest_cr_line'])

df_train=df_train.query('issue_d=="2014" | issue_d=="2015"')



df_test = pd.read_csv('../input/homework-for-students3/test.csv', index_col=0)

df_test['issue_d']=pd.to_datetime(df_test['issue_d'])

df_test['earliest_cr_line']=pd.to_datetime(df_test['earliest_cr_line'])

# issue_dの月の株価を示す変数を入れる



df_train['date']=df_train['issue_d'].map(lambda x:str(x)[:7])

df_test['date']=df_test['issue_d'].map(lambda x:str(x)[:7])



keizai['date']=pd.to_datetime(keizai['date'])

keizai['diff_rate']=keizai['close'].diff()/keizai['close'].shift()



keizai=keizai.set_index('date')

keizai_m=keizai.resample(rule='M').mean()

keizai_m=keizai_m['2007':'2016'].reset_index()

keizai_m['date']=keizai_m['date'].map(lambda x: str(x)[:7])



df_train=pd.merge(df_train.reset_index(),keizai_m[['date','diff_rate']],on='date',how='left')

df_test=pd.merge(df_test.reset_index(),keizai_m[['date','diff_rate']],on='date',how='left')



df_train.set_index(['ID'],inplace=True)

df_test.set_index(['ID'],inplace=True)



df_train=df_train.drop(['date'],axis=1)

df_test=df_test.drop(['date'],axis=1)
df_test.title.value_counts()
df_train.title.value_counts()


# df_test.query('title=="Debt consolidation"').ix[:,['purpose']]='debt_consolidation'

# df_test.query('title=="Credit card refinancing"').ix[:,['purpose']]='credit_card'

# df_test.query('title=="Home improvement"').ix[:,['purpose']]='home_improvement'

# df_test.query('title=="Major purchase"').ix[:,['purpose']]='major_purchase'

# df_test.query('title=="Business"').ix[:,['purpose']]='small_business'

# df_test.query('title=="Medical expenses"').ix[:,['purpose']]='medical'



# # df_test.purpose.value_counts()
df_test.query('purpose=="other" & title!="Other"')['title'].value_counts()
df_test.query('title=="Green loan"')[['title','purpose']].head(100)
# # df_train['edu']=df_train['title'].map(lambda x: 1 if 'educat' in x else 0)

# df_train.title=df_train.title.fillna('none')

# df_train.title=df_train.title.map(lambda x : x.lower())



# df_train['educat']=df_train.title.map(lambda x: 1 if ('educat' in x) | ('college' in x) | ('school' in x) else 0)
df_train.query('educat==1').ix[:,['purpose']]='educational'
df_train.query('purpose=="educational"')[['title','purpose']]
df_train.title.value_counts()
pd.set_option('display.max_columns', 1000)

pd.set_option('display.max_rows', 700)

#emp_title->clusterに変換

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans



df_train['istrain']=1

df_test['istrain']=0

df_train.emp_title=df_train.emp_title.fillna('None')

df_test.emp_title=df_test.emp_title.fillna('None')

df_train.title=df_train.title.fillna('None')

df_test.title=df_test.title.fillna('None')



df_train.emp_title=df_train.emp_title.map(lambda x: x.lower())

df_test.emp_title=df_test.emp_title.map(lambda x: x.lower())

df_train.title=df_train.title.map(lambda x: x.lower())

df_test.title=df_test.title.map(lambda x: x.lower())





temp=pd.concat([df_train[['title','emp_title','istrain']],df_test[['title','emp_title','istrain']]])



vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b',max_features=1000)

vecs_emp = vectorizer.fit_transform(temp['emp_title'])

vecs_tit = vectorizer.fit_transform(temp['title'])



clusters_emp= KMeans(n_clusters=20, random_state=0).fit_predict(vecs_emp)

clusters_tit= KMeans(n_clusters=13, random_state=0).fit_predict(vecs_tit)



temp=pd.DataFrame(pd.concat([df_train[['emp_title','istrain']],df_test[['emp_title','istrain']]]))

temp['cluster_emp']=clusters_emp

temp['cluster_tit']=clusters_tit

# temp=pd.read_csv('../output/kaggle/working/clusters.csv')

import gc

del vecs, clusters

gc.collect()
emp_tr=temp.query('istrain==1')['cluster_emp']

emp_ts=temp.query('istrain==0')['cluster_emp']

tit_tr=temp.query('istrain==1')['cluster_tit']

tit_ts=temp.query('istrain==0')['cluster_tit']



df_train=pd.concat([df_train,emp_tr,tit_tr],axis=1)

df_test=pd.concat([df_test,emp_ts,tit_ts],axis=1)



df_train.drop(['istrain'],axis=1,inplace=True)

df_test.drop(['istrain'],axis=1,inplace=True)
#cluster -> 破産割合でReplace

# wariai_tr=(df_train.query('loan_condition==1').groupby('cluster')['loan_condition'].count()/df_train.groupby('cluster')['loan_condition'].count()).sort_values(ascending=False)

# wariai_ts=(df_test.query('loan_condition==1').groupby('cluster')['loan_condition'].count()/df_test.groupby('cluster')['loan_condition'].count()).sort_values(ascending=False)

# df_train['cluster']=df_train['cluster'].map(wariai_tr)

# df_test['cluster']=df_test['cluster'].map(wariai_tr)
y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition'], axis=1)



X_test = df_test
# # #purposeのカテゴリをloan_condition=1となっている人数順に数値化

# def purpose_flg(x):

#     if x=='debt_consolidation'  : return 14

#     if x=='credit_card'  : return 13

#     if x=='home_improvement'  : return 12

#     if x=='other'  : return 11

#     if x=='major_purchase'  : return 10

#     if x=='small_business'  : return 9

#     if x=='medical'  : return 8

#     if x=='car'  : return 7

#     if x=='moving'  : return 6

#     if x=='vacation'  : return 5

#     if x=='house'  : return 4

#     if x=='wedding'  : return 3

#     if x=='renewable_energy'  : return 2

#     if x=='educational'  : return 1



# X_train['purpose']=X_train['purpose'].map(lambda x:purpose_flg(x))

# X_test['purpose']=X_test['purpose'].map(lambda x:purpose_flg(x))
def flg_cov(x):

    if 'A1' in x : return 35

    if 'A2' in x : return 34

    if 'A3' in x : return 33

    if 'A4' in x : return 32

    if 'A5' in x : return 31

    if 'B1' in x : return 30

    if 'B2' in x : return 29

    if 'B3' in x : return 28

    if 'B4' in x : return 27

    if 'B5' in x : return 26

    if 'C1' in x : return 25

    if 'C2' in x : return 24

    if 'C3' in x : return 23

    if 'C4' in x : return 22

    if 'C5' in x : return 21

    if 'D1' in x : return 20

    if 'D2' in x : return 19

    if 'D3' in x : return 18

    if 'D4' in x : return 17

    if 'D5' in x : return 16

    if 'E1' in x : return 15

    if 'E2' in x : return 14

    if 'E3' in x : return 13

    if 'E4' in x : return 12

    if 'E5' in x : return 11

    if 'F1' in x : return 10

    if 'F2' in x : return 9

    if 'F3' in x : return 8

    if 'F4' in x : return 7

    if 'F5' in x : return 6

    if 'G1' in x : return 5

    if 'G2' in x : return 4

    if 'G3' in x : return 3

    if 'G4' in x : return 2

    if 'G5' in x : return 1





    

X_train['grade']=X_train['grade']+X_train['sub_grade']

X_train['grade']=X_train['grade'].map(lambda x: flg_cov(x))



X_test['grade']=X_test['grade']+X_test['sub_grade']

X_test['grade']=X_test['grade'].map(lambda x: flg_cov(x))



X_train=X_train.drop(['sub_grade'],axis=1)

X_test=X_test.drop(['sub_grade'],axis=1)
def state_GDP(x):

    if 'CA' in x : return 51

    if 'TX' in x : return 50

    if 'NY' in x : return 49

    if 'FL' in x : return 48

    if 'IL' in x : return 47

    if 'PA' in x : return 46

    if 'OH' in x : return 45

    if 'NJ' in x : return 44

    if 'GA' in x : return 43

    if 'NC' in x : return 42

    if 'MA' in x : return 41

    if 'VA' in x : return 40

    if 'MI' in x : return 39

    if 'WA' in x : return 38

    if 'MD' in x : return 37

    if 'IN' in x : return 36

    if 'MN' in x : return 35

    if 'TN' in x : return 34

    if 'CO' in x : return 33

    if 'WI' in x : return 32

    if 'MO' in x : return 31

    if 'AZ' in x : return 30

    if 'CT' in x : return 29

    if 'LA' in x : return 28

    if 'OR' in x : return 27

    if 'SC' in x : return 26

    if 'AL' in x : return 25

    if 'KY' in x : return 24

    if 'OK' in x : return 23

    if 'IA' in x : return 22

    if 'KS' in x : return 21

    if 'UT' in x : return 20

    if 'NV' in x : return 19

    if 'DC' in x : return 18

    if 'AR' in x : return 17

    if 'NE' in x : return 16

    if 'MS' in x : return 15

    if 'NM' in x : return 14

    if 'HI' in x : return 13

    if 'NH' in x : return 12

    if 'WV' in x : return 11

    if 'DE' in x : return 10

    if 'ID' in x : return 9

    if 'ME' in x : return 8

    if 'ND' in x : return 7

    if 'RI' in x : return 6

    if 'AK' in x : return 5

    if 'SD' in x : return 4

    if 'MT' in x : return 3

    if 'WY' in x : return 2

    if 'VT' in x : return 1

#     if 'CA' in x : return 2491.6

#     if 'TX' in x : return 1611.2

#     if 'NY' in x : return 1445.6

#     if 'FL' in x : return 883.9

#     if 'IL' in x : return 772.2

#     if 'PA' in x : return 708.4

#     if 'OH' in x : return 607.3

#     if 'NJ' in x : return 564.4

#     if 'GA' in x : return 501.5

#     if 'NC' in x : return 499.7

#     if 'MA' in x : return 488.1

#     if 'VA' in x : return 481.7

#     if 'MI' in x : return 470.6

#     if 'WA' in x : return 446.4

#     if 'MD' in x : return 366.2

#     if 'IN' in x : return 333.4

#     if 'MN' in x : return 326.8

#     if 'TN' in x : return 316.7

#     if 'CO' in x : return 313.3

#     if 'WI' in x : return 301.6

#     if 'MO' in x : return 292.7

#     if 'AZ' in x : return 291.4

#     if 'CT' in x : return 256.3

#     if 'LA' in x : return 238.1

#     if 'OR' in x : return 216.5

#     if 'SC' in x : return 201.8

#     if 'AL' in x : return 200

#     if 'KY' in x : return 191.9

#     if 'OK' in x : return 188

#     if 'IA' in x : return 175.8

#     if 'KS' in x : return 151.8

#     if 'UT' in x : return 149.1

#     if 'NV' in x : return 141.1

#     if 'DC' in x : return 121.1

#     if 'AR' in x : return 118.7

#     if 'NE' in x : return 113.5

#     if 'MS' in x : return 105.9

#     if 'NM' in x : return 93.2

#     if 'HI' in x : return 80.6

#     if 'NH' in x : return 74.3

#     if 'WV' in x : return 73.4

#     if 'DE' in x : return 68.9

#     if 'ID' in x : return 65.5

#     if 'ME' in x : return 57.3

#     if 'ND' in x : return 55.9

#     if 'RI' in x : return 55.7

#     if 'AK' in x : return 53.4

#     if 'SD' in x : return 47.1

#     if 'MT' in x : return 45.8

#     if 'WY' in x : return 39.4

#     if 'VT' in x : return 30.3



df_train['addr_state']=df_train['addr_state'].map(lambda x:state_GDP(x))

df_test['addr_state']=df_test['addr_state'].map(lambda x:state_GDP(x))
X_train['tot_cur_bal_flg'] =X_train['tot_cur_bal'].map(lambda x: 1 if x!=x else 0)

X_test['tot_cur_bal_flg']  =X_test['tot_cur_bal'].map(lambda x: 1 if x!=x else 0)

X_train['tot_coll_bal_flg']=X_train['tot_coll_amt'].map(lambda x: 1 if x!=x else 0)

X_test['tot_coll_bal_flg'] =X_test['tot_coll_amt'].map(lambda x: 1 if x!=x else 0)

X_train['kakochien']       =X_train['delinq_2yrs'].map(lambda x: 1 if x!=0 else 0)

X_test['kakochien']        =X_test['delinq_2yrs'].map(lambda x: 1 if x!=0 else 0)

X_train['emp_flg']         =X_train['emp_length'].map(lambda x: 1 if x!=x else 0)

X_test['emp_flg']          =X_test['emp_length'].map(lambda x: 1 if x!=x else 0)

X_train['emp_title_flg']   =X_train['emp_title'].map(lambda x: 1 if x=='none' else 0)

X_test['emp_title_flg']    =X_test['emp_title'].map(lambda x: 1 if x=='none' else 0)

X_train['dti_flg']         =X_train['dti'].map(lambda x: 1 if x!=x else 0)

X_test['dti_flg']          =X_test['dti'].map(lambda x: 1 if x!=x else 0)

X_train['inc_loan']        =X_train['loan_amnt']/(X_train['annual_inc']+1)

X_test['inc_loan']         =X_test['loan_amnt']/(X_test['annual_inc']+1)

X_train['loanrate_over25'] =X_train['inc_loan'].map(lambda x: 1 if x>0.25 else 0)

X_test['loanrate_over25']  =X_test['inc_loan'].map(lambda x: 1 if x>0.25 else 0)

X_train['inc_installment'] =X_train['installment']/(X_train['annual_inc']+1)/12

X_test['inc_installment']  =X_test['installment']/(X_test['annual_inc']+1)/12

X_train['revol_inc_rate']  =X_train['revol_bal']/(X_train['annual_inc']+1)

X_test['revol_inc_rate']   =X_test['revol_bal']/(X_test['annual_inc']+1)

X_train['line_spend']      =(X_train['issue_d']-X_train['earliest_cr_line']).astype('timedelta64[D]')

X_test['line_spend']       =(X_test['issue_d']-X_test['earliest_cr_line']).astype('timedelta64[D]')





X_train['annual_inc'] = X_train['annual_inc'].apply(np.log1p)

X_test['annual_inc'] = X_test['annual_inc'].apply(np.log1p)

# X_train['loan_amnt'] = X_train['loan_amnt'].apply(np.log1p)

# X_test['loan_amnt'] = X_test['loan_amnt'].apply(np.log1p)



X_train=X_train.drop(['issue_d','earliest_cr_line'],axis=1)

X_test=X_test.drop(['issue_d','earliest_cr_line'],axis=1)



X_train=X_train.drop(['emp_title'],axis=1)

X_test=X_test.drop(['emp_title'],axis=1)
#emp_lengthカテゴリ→数値

def emp_length_cov(x):

    if '10+ years' in x : return 10

    if '2 years' in x : return 2

    if '< 1 year' in x : return 0.5

    if '3 years' in x : return 3

    if '1 year' in x : return 1

    if '5 years' in x : return 5

    if '4 years' in x : return 4

    if '7 years' in x : return 7

    if '8 years' in x : return 8

    if '6 years' in x : return 6

    if '9 years' in x : return 9

    if '_None_' in x : return 0



X_train['emp_length']=X_train['emp_length'].fillna('_None_')

X_test['emp_length']=X_test['emp_length'].fillna('_None_')



X_train['emp_length']=X_train['emp_length'].map(lambda x:emp_length_cov(x))

X_test['emp_length']=X_test['emp_length'].map(lambda x:emp_length_cov(x))
cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

#         print(col, X_train[col].nunique())

cats.remove('title')

cats

oe = OrdinalEncoder(cols=cats, return_df=False)



X_train[cats] = oe.fit_transform(X_train[cats])

X_test[cats] = oe.transform(X_test[cats])
X_train.drop(['title','zip_code'], axis=1, inplace=True)

X_test.drop(['title','zip_code'], axis=1, inplace=True)



X_train.fillna(X_train.median(), inplace=True)

X_test.fillna(X_train.median(), inplace=True)
# for i in X_train.columns:

#     print('----------------')

#     print(i,':')

#     print('df_train:',df_train[i].value_counts().count())

#     print('df_test :',df_test[i].value_counts().count())

#     print()

#     print('train:',X_train[i].value_counts().count())

#     print('test :',X_test[i].value_counts().count())

#     print()
from sklearn.model_selection import GridSearchCV

import lightgbm as lgb

from lightgbm import LGBMClassifier



clf = LGBMClassifier(boosting_type='gbdt',

                     class_weight=None,

                     colsample_bytree=0.8,

                     importance_type='split',

                     learning_rate=0.05,

                     min_child_samples=20,

                     min_child_weight=0.01,

                     min_split_gain=0.0,

                     n_jobs=-1,

                     n_estimators=9999,          #Number of boosted trees to fit.

                     max_depth=-1,               #Maximum tree depth for base learners, <=0 means no limit.

                     num_leaves=15,              #Maximum tree leaves for base learners.

                     objective=None,

                     random_state=71,

                     reg_alpha=0.0,              #L1 regularization term on weights.

                     reg_lambda=0.0,               #L2 regularization term on weights.

                     silent=True,   

                     subsample=1.0,              #Subsample ratio of the training instance

                     subsample_for_bin=20000,

                     subsample_freq=0,

                    )
# CVしてスコアを見てみる。層化抽出で良いかは別途よく考えてみてください。

from sklearn.model_selection import GridSearchCV

import lightgbm as lgb

from lightgbm import LGBMClassifier



scores = []

best_iter=[]

y_pred=pd.DataFrame()



# skf = StratifiedKFold(n_splits=5, random_state=58, shuffle=True)





# clf = LGBMClassifier(boosting_type='gbdt',class_weight=None,colsample_bytree=0.8,

#                     importance_type='split',learning_rate=0.05,max_depth=-1,

#                     min_child_samples=20,min_child_weight=0.01,min_split_gain=0.0,

#                     n_estimators=9999,n_jobs=-1,num_leaves=15,objective=None,

#                     random_state=71,reg_alpha=0.0,reg_lambda=0,silent=True,

#                     subsample=0.8,subsample_for_bin=200000,subsample_freq=0,verbosity=1)



for t in range(10):

    print('----',t,'----')

    skf = StratifiedKFold(n_splits=10, random_state=t+71, shuffle=True)

    for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

        X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

        X_val, y_val = X_train.values[test_ix], y_train.values[test_ix] 

        clf.fit(X_train_, y_train_,early_stopping_rounds=20,eval_metric='auc',eval_set=[(X_val,y_val)])

        y_pred_val = clf.predict_proba(X_val)[:,1]

        y_pred=pd.concat([y_pred,pd.DataFrame(clf.predict_proba(X_test,num_iteration =clf.best_iteration_)[:,1])],axis=1)

        scores.append(roc_auc_score(y_val, y_pred_val))

        best_iter.append(clf.best_iteration_)

#     print('CV Score of Fold_%d is %f' % (i, score))



print('best scores:',scores)

print('best ave:',sum(scores)/len(scores))

print('best iter',best_iter)

s=pd.Series(clf.feature_importances_,index=X_train.columns)

s.sort_values(ascending=False).plot.barh(color='C0')

s.sort_values(ascending=False)
print('best scores:',scores)

print('best ave:',sum(scores)/len(scores))

print('best iter',best_iter)

s=pd.Series(clf.feature_importances_,index=X_train.columns)

s.sort_values(ascending=False).plot.barh(color='C0')

s.sort_values(ascending=False)
y_pred['mean']=y_pred.mean(axis=1)

# y_pred['std']=y_pred.std(axis=1)

# y_pred.sort_values(by='std',ascending=False)
y_pred['mean']
submission = pd.read_csv('../input/homework-for-students3/sample_submission.csv')



submission.loan_condition = y_pred['mean']

submission.set_index('ID',inplace=True)

submission.to_csv('submission.csv')
# submission
import seaborn as sns

# sns.heatmap(X_train.corr()[['grade','cluster']],linewidths=.5)

plt.figure(figsize=(20, 20)) 

sns.heatmap(X_train.corr()<-0.5,linewidths=.5,square=True)
df_train[['title','purpose']]
# temp=pd.concat([X_train.mean(),X_test.mean()],axis=1)

# temp['diff']=(temp[0]-temp[1])/temp[0]

# temp
# for i in X_train.columns:

#     print(i)

#     print('train:',X_train[i].std())

#     print('test :',X_test[i].std())

#     print()
# for i in X_train.columns:

#     print(i)

# #     plt.hist([X_train[i],X_test[i]],bins=20,alpha=0.5)

#     plt.show()



# #     plt.hist([df_test[i],df_train[i]],bins=20,alpha=0.5)

# #     plt.show()
