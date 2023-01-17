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
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder
from sklearn.metrics import mean_squared_log_error
from tqdm import tqdm_notebook as tqdm

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

import lightgbm as lgb
#関数
#項目一括削除
def del_val(train,test,array):
    train.drop(array,axis=1, inplace=True)
    test.drop(array,axis=1, inplace=True)
    return

#ターゲット分割
def y_x(train,target):
    y_train = train[target]
    X_train = df_train.drop(target, axis=1)
    return y_train,X_train

#文字数値指定変換
def char_num(train,test,char_var,num_var,c_list):
    train[num_var]=X_train[char_var].replace(c_list).astype(float)
    test[num_var]=X_train[char_var].replace(c_list).astype(float)
    return train,test

#文字変換
def char_rep(train,test,char_var,c_list):
    train[char_var]=X_train[char_var].replace(c_list)
    test[char_var]=X_train[char_var].replace(c_list)
    return train,test

#yyyymm文字から、date,yyyymm数値,yyyy,mmを作成
def yyyymm_date(train,test,yyyymm_var):
    train[yyyymm_var+'_date']=pd.to_datetime(X_train[yyyymm_var]+"-01",format='%b-%Y-%d')
    test[yyyymm_var+'_date']=pd.to_datetime(X_test[yyyymm_var]+"-01",format='%b-%Y-%d')
    train[yyyymm_var+'_ym']=X_train[yyyymm_var+'_date'].dt.year*100+X_train[yyyymm_var+'_date'].dt.month
    test[yyyymm_var+'_ym']=X_test[yyyymm_var+'_date'].dt.year*100+X_test[yyyymm_var+'_date'].dt.month
    train[yyyymm_var+'_year']=X_train[yyyymm_var+'_date'].dt.year
    test[yyyymm_var+'_year']=X_test[yyyymm_var+'_date'].dt.year
    train[yyyymm_var+'_month']=X_train[yyyymm_var+'_date'].dt.month
    test[yyyymm_var+'_month']=X_test[yyyymm_var+'_date'].dt.month
    train.drop([yyyymm_var+'_date',yyyymm_var],axis=1, inplace=True)
    test.drop([yyyymm_var+'_date',yyyymm_var],axis=1, inplace=True)
    return train,test

#文字列長取得,欠損は0
def char_len(train,test,char_var):
    X_train[char_var+'_len']=X_train[char_var].str.len()
    X_train[char_var+'_len'].fillna(0, inplace=True)
    X_test[char_var+'_len']=X_test[char_var].str.len()
    X_test[char_var+'_len'].fillna(0, inplace=True)
    return train,test

#merge
def data_merge(train,test,merge,key1,key2,how):
    train = pd.merge(train,merge, left_on =key1 , right_on=key2 , how = how )
    test = pd.merge(test, merge, left_on =key1 , right_on=key2 , how = how )
    return train,test

#欠損値 個別対応 0など
def var_missing(train,test,var_list,value):
    for var in var_list:
        train=train.fillna({var:value})
        test=test.fillna({var:value})
#全体を中央値
def all_missing(train,test):
    train.fillna(train.median(), inplace=True)
    test.fillna(train.median(), inplace=True)
    
#ZIP3
def zip3(train,test,zipcode):
    train[zipcode+'_3']=train[zipcode].str[:3].astype(int)
    test[zipcode+'_3']=test[zipcode].str[:3].astype(int)
    return train,test

#文字列長取得,欠損は0
def label_encoder(train,test,var_list):
    for var in var_list:
        le = LabelEncoder()
        le = le.fit(X_train[var])
        X_train[var] = le.transform(X_train[var])
        X_test[var] = le.transform(X_test[var])
    return train,test

#実際に課題でモデリングする際には"skiprows=lambda x: x%20!=0"を削除してください。
df_train = pd.read_csv('../input/exam-for-students20200923/train.csv', index_col=0,encoding="utf-8")
df_test = pd.read_csv('../input/exam-for-students20200923/test.csv', index_col=0,encoding="utf-8")
df_country = pd.read_csv('../input/exam-for-students20200923/country_info.csv', encoding="utf-8")
def comma_dot(data,var_list):
 for var in var_list:
  data[var]=data[var].str.replace(',', '.').astype('float16')
 return data

var_list=["Pop. Density (per sq. mi.)","Coastline (coast/area ratio)","Net migration",
      "Infant mortality (per 1000 births)",
      "Literacy (%)","Phones (per 1000)",
      "Arable (%)","Crops (%)","Other (%)","Climate","Birthrate",
      "Deathrate","Agriculture","Industry","Service"]
df_country=comma_dot(df_country,var_list)


df_country=df_country[["Country","Phones (per 1000)",
                       "GDP ($ per capita)","Crops (%)","Area (sq. mi.)"]]
#マージ
#key1=["Country"]
#key2=["Country"]
#df_train,df_test =data_merge(df_train,df_test,df_country,key1,key2,"left")
#ターゲット分割
y='ConvertedSalary'
y_train,X_train =y_x(df_train,y)
X_test = df_test
#var=["Country"]
#del_val(X_train,X_test,var)
#数値へのデコード
#朝いつ起きるか 一旦、orderで実施。早いほどよいのか？


#年齢
list={
'18 - 24 years old':21,
'25 - 34 years old':30,
'35 - 44 years old':40,
'45 - 54 years old':50,
'55 - 64 years old':60,
'65 years or older':65,
'Under 18 years old':18
}
X_train,X_test = char_num(X_train,X_test,'Age','Age_num',list)
#企業規模
list={
'1,000 to 4,999 employees':3000,
'10 to 19 employees':15,
'10,000 or more employees':10000,
'100 to 499 employees':300,
'20 to 99 employees':60,
'5,000 to 9,999 employees':7500,
'500 to 999 employees':750,
'Fewer than 10 employees':10}
X_train,X_test = char_num(X_train,X_test,'CompanySize','CompanySize_num',list)

#PC使用時間
list={
'1 - 4 hours':3,
'5 - 8 hours':7,
'9 - 12 hours':11,
'Less than 1 hour':1,
'Over 12 hours':12}
X_train,X_test = char_num(X_train,X_test,'HoursComputer','HoursComputer_num',list)

#外出時間
list={
'1 - 2 hours':1.5,
'3 - 4 hours':3.5,
'30 - 59 minutes':0.75,
'Less than 30 minutes':0.5,
'Over 4 hours':4
}
X_train,X_test = char_num(X_train,X_test,'HoursOutside','HoursOutside_num',list)

#食事抜き
list={
'1 - 2 times per week':1.5,
'3 - 4 times per week':3.5,
'Daily or almost every day':7,
'Never':0
}
X_train,X_test = char_num(X_train,X_test,'SkipMeals','SkipMeals_num',list)
    

#StackOverflowRecommend?
list={
'1':1,
'2':2,
'3':3,
'4':4,
'5':5,
'6':6,
'7':7,
'8':8,
'9':9,
'0 (Not Likely)':0,
'10 (Very Likely)':10
}
X_train,X_test = char_num(X_train,X_test,'StackOverflowRecommend','StackOverflowRecommend_num',list)
#StackOverflowJobsRecommend?
list={
'1':1,
'2':2,
'3':3,
'4':4,
'5':5,
'6':6,
'7':7,
'8':8,
'9':9,
'0 (Not Likely)':0,
'10 (Very Likely)':10
}
X_train,X_test = char_num(X_train,X_test,'StackOverflowJobsRecommend','StackOverflowJobsRecommend_num',list)

#経験年数？
list={  
'1 - 2 times per week':1.5,
'3 - 4 times per week':3.5,
'Daily or almost every day':7,
"I don't typically exercise":0}
X_train,X_test = char_num(X_train,X_test,'Exercise','Exercise_num',list)

#在籍年数？
list={  
'0-2 years':1,
'12-14 years':13,
'15-17 years':14,
'18-20 years':19,
'21-23 years':22,
'24-26 years':25,
'27-29 years':28,
'30 or more years':30,
'3-5 years':4,
'6-8 years':7,
'9-11 years':10}
X_train,X_test = char_num(X_train,X_test,'YearsCoding','YearsCoding_num',list)

#前職在籍年数？
list={  
'0-2 years':1,
'12-14 years':13,
'15-17 years':14,
'18-20 years':19,
'21-23 years':22,
'24-26 years':25,
'27-29 years':28,
'30 or more years':30,
'3-5 years':4,
'6-8 years':7,
'9-11 years':10,'n/a':0}
X_train,X_test = char_num(X_train,X_test,'YearsCodingProf','YearsCodingProf_num',list)
#数値化済みは除く
array=['Age','CompanySize','HoursComputer','HoursOutside','SkipMeals','StackOverflowRecommend','StackOverflowJobsRecommend','Exercise','YearsCoding','YearsCodingProf']
del_val(X_train,X_test,array)
X_train,X_test=char_len(X_train,X_test,"Gender")

array=['Gender']
del_val(X_train,X_test,array)
#全体を中央値
X_train['Gender_len'].fillna(3, inplace=True)
X_test['Gender_len'].fillna(3, inplace=True)

# dtypeがobject（数値でないもの）のカラム名とユニーク数を確認してみましょう。
cats = []
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        cats.append(col)
        print(col, X_train[col].nunique())
var=['FrameworkWorkedWith','CommunicationTools','DevType','CurrencySymbol']
for v in var:
 cats.remove(v)

del_val(X_train,X_test,var)


#順序関係ない場合のencoding
oe = OrdinalEncoder(cols=cats, return_df=False)

X_train[cats] = oe.fit_transform(X_train[cats])
X_test[cats] = oe.transform(X_test[cats])
#Countエンコーディング
#for col in cats:
# summary = X_train[col].value_counts()
# X_train[col+'_cnt'] = X_train[col].map(summary)
# X_test[col+'_cnt']  = X_test[col].map(summary)
#全体を中央値
def all_missing_0(train,test):
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)
#最後,欠損値を0に
all_missing(X_train,X_test)
#all_missing_0(X_train,X_test)

var=["JobSearchStatus",
"StackOverflowJobsRecommend_num",
"SurveyEasy",
"AIFuture",
"AgreeDisagree3",
"EthicsResponsible",
"SexualOrientation",
"AdsPriorities2",
"HypotheticalTools2",
"Dependents",
"AdBlocker",
"TimeAfterBootcamp",
"HypotheticalTools4",
"AdsAgreeDisagree3",
"StackOverflowConsiderMember",
"SkipMeals_num",
"StackOverflowJobs",
"TimeFullyProductive",
"AdsAgreeDisagree2",
"EthicsChoice",
"SurveyTooLong",
"StackOverflowHasAccount",
"Hobby"]
del_val(X_train,X_test,var)
#訓練データとテストデータに分割する
#from sklearn.model_selection import train_test_split
#X_train_1, X_valid, y_train_1, y_valid = train_test_split(X_train, y_train,random_state=71)

params={'lambda_l1': 0.001625781371297295,
 'lambda_l2': 8.662399290811956,
 'num_leaves': 91,
 'feature_fraction': 0.9799999999999999,
 'bagging_fraction': 1.0,
 'bagging_freq': 0,
 'min_child_samples': 10,
 'objective': 'mean_squared_error',
 'metric': 'rmse'}

y_train = np.log(y_train + 1)
n_splits=5
random_states=[42,71,1010]

scores = []
y_pred_test = np.zeros(len(X_test)) # テストデータに対する予測格納用array
for random_state in random_states:
 skf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
 for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):
  X_train_, y_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix]
  X_val, y_val = X_train.iloc[test_ix], y_train.iloc[test_ix ]
  clf = lgb.LGBMRegressor(**params)
  clf = lgb.LGBMRegressor()
  clf.fit(X_train_, y_train_, early_stopping_rounds=200, eval_metric='rmse', eval_set=[(X_val, y_val)])
  oof = np.exp(clf.predict(X_val))-1
  oof = np.where(oof < 0, 0, oof)
  rmsle = np.sqrt(mean_squared_log_error(np.exp(y_val), oof))
  scores.append(rmsle)
  print(f"RMSLE : {rmsle}")
  y_pred_test += np.exp(clf.predict(X_test))-1 # テストデータに対する予測値を足していく
print('Ave. RMSLE score is %f' % np.mean(scores))
y_pred_test /= n_splits * len(random_states) # 最後にfold数で割る
# 訓練データとテストデータに分割する
from sklearn.model_selection import train_test_split
X_train_1, X_valid, y_train_1, y_valid = train_test_split(X_train, y_train,random_state=71)

trains=lgb.Dataset(X_train_1, y_train_1)
valids=lgb.Dataset(X_valid, y_valid)

import optuna.integration.lightgbm as lgb_p

params = {
    'objective': 'mean_squared_error',
    'metric': 'rmse'
}
best_params = {}
history = []
model = lgb_p.train(params, trains, valid_sets=valids,
                    verbose_eval=False,
                    num_boost_round=100,
                    early_stopping_rounds=50,
#                    best_params=best_params,
#                    categorical_feature=categorical_feature,
                    tuning_history=history)
import eli5
eli5.show_weights(clf, feature_names = X_test.columns.tolist(),top=300)
# sample submissionを読み込んで、予測値を代入の後、保存する
submission = pd.read_csv('../input/exam-for-students20200923/sample_submission.csv', index_col=0)

submission.ConvertedSalary = y_pred_test
submission.to_csv('submission.csv')