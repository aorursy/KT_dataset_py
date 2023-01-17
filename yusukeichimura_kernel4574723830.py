import pandas as pd

import numpy as np

import datetime as dt

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error, log_loss

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import TfidfVectorizer





from sklearn.ensemble import GradientBoostingClassifier

import lightgbm as lgb

from lightgbm import LGBMClassifier
df_train = pd.read_csv('../input/exam-for-students20200129/train.csv')

df_test = pd.read_csv('../input/exam-for-students20200129/test.csv')

country_info = pd.read_csv('../input/exam-for-students20200129/country_info.csv')

#df = pd.read_csv('./train.csv',parse_dates=['Date'],skiprows=lambda x: x%5!=0)



#レコードが多い場合は、行数を絞って読み取る

#df = pd.read_csv('./train.csv',parse_dates=['Date'],skiprows=lambda x: x%20!=0)

#ディレクトリ指定については以下参照

#df_train = pd.read_csv('../input/homework-for-students2/train.csv', index_col=0, parse_dates=['issue_d'], skiprows=lambda x: x%20!=0)
#効果イマイチなので一旦やめる

#df_train = pd.merge(df_train, country_info, how='left')

#df_test = pd.merge(df_test, country_info, how='left')
imp=[

    'Respondent',

    'Hobby',

    'OpenSource',

    'Country',

    'Student',

    'Employment',

    'FormalEducation',

    'UndergradMajor',

    'CompanySize',

    'DevType',

    'YearsCoding',

    'YearsCodingProf',

    'JobSatisfaction',

    'CareerSatisfaction',

    'HopeFiveYears',

    'JobSearchStatus',

    'LastNewJob',

    'AssessJob1',

    'AssessJob2',

    'AssessJob3',

    'AssessJob4',

    'AssessJob5',

    'AssessJob6',

    'AssessJob7',

    'AssessJob8',

    'AssessJob9',

    'AssessJob10',

    'AssessBenefits1',

    'AssessBenefits2',

    'AssessBenefits3',

    'AssessBenefits4',

    'AssessBenefits5',

    'AssessBenefits6',

    'AssessBenefits7',

    'AssessBenefits8',

    'AssessBenefits9',

    'AssessBenefits10',

    'AssessBenefits11',

    'JobContactPriorities1',

    'JobContactPriorities2',

    'JobContactPriorities3',

    'JobContactPriorities4',

    'JobContactPriorities5',

    'JobEmailPriorities1',

    'JobEmailPriorities2',

    'JobEmailPriorities3',

    'JobEmailPriorities4',

    'JobEmailPriorities5',

    'JobEmailPriorities6',

    'JobEmailPriorities7',

    'ConvertedSalary'

]
#上位の特徴量だけに絞ってみたが効果薄

#df_train = df_train[imp]

#df_train.shape
#上位の特徴量だけに絞ってみたが効果薄

#imp.remove('ConvertedSalary')

#df_test = df_test[imp]
df_test.shape
df_train.loc[df_train['Currency'].astype('str').str.contains('U.S.'), 'US2'] = 1

df_test.loc[df_test['Currency'].astype('str').str.contains('U.S.'), 'US2'] = 1

df_train['US2'] = df_train['US2'].fillna(0)

df_test['US2'] = df_test['US2'].fillna(0)

df_train['US2']
df_train.loc[df_train['Country'].astype('str').str.contains('United States'), 'US'] = 1

df_test.loc[df_test['Country'].astype('str').str.contains('United States'), 'US'] = 1

df_train['US'] = df_train['US'].fillna(0)

df_test['US'] = df_test['US'].fillna(0)

df_train['US']
df_train.loc[df_train['Employment'].astype('str').str.contains('full-time'), 'full-time'] = 1

df_test.loc[df_test['Employment'].astype('str').str.contains('full-time'), 'full-time'] = 1

df_train['full-time'] = df_train['full-time'].fillna(0)

df_test['full-time'] = df_test['full-time'].fillna(0)

df_train['full-time']
X_train = df_train.drop(['ConvertedSalary'], axis=1).copy()

y_train = df_train.ConvertedSalary.copy()



X_test = df_test.copy()
X_train.head()
X_test.head()
#Object型のカラムを"cats",それ以外を"num"に格納する

cats = []

num = []



for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col) 

    else:

        num.append(col)
cats
num
#欠損値をNaNで埋める

X_train[cats] = X_train[cats].fillna("NaN")

X_test[cats] = X_test[cats].fillna("NaN")



#欠損値を0で埋める

#X_train[num] = X_train[num].fillna(0)

#X_test[num] = X_train[num].fillna(0)



#欠損値を前の値で埋める

#X_train[num] = X_train[num].fillna(method='ffill')

#X_test[num] = X_test[num].fillna(method='ffill')



#欠損値を最小値で埋める

#X_train[num] = X_train[num].fillna(df[num].min())

#X_test[num] = X_test[num].fillna(df[num].min())



#欠損値を平均値で埋める

X_train[num] = X_train[num].fillna(X_train[num].mean())

X_test[num] = X_test[num].fillna(X_train[num].mean())
#訓練データの欠損数⇒全部0になっていればOK

len(X_train) - X_train.count()
#テストデータの欠損数⇒全部0になっていればOK

len(X_test) - X_test.count()
num
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train[num])



X_train[num] =scaler.transform(X_train[num])

X_test[num] =scaler.transform(X_test[num])
X_test.head()
X_train.head()
cats
oe = OrdinalEncoder(cols=cats)

X_train = oe.fit_transform(X_train)

X_test = oe.transform(X_test)
X_train.head()
X_test.head()
"True" if X_train.shape[1] == X_test.shape[1] else "False"
"True" if X_train.shape[0] == y_train.shape[0] else "False"
len(X_train) - X_train.count()
len(X_test) - X_test.count()
#学習前にy_trainを変換する

y_train = np.log(y_train + 1)
X_train['US']
y_train.shape
import numpy as np

import lightgbm as lgb

from sklearn.datasets import load_boston

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold, train_test_split

import gc



df_cv_avg = pd.DataFrame()



skf = KFold(n_splits=5, random_state=71, shuffle=True)



lgb_y_pred_train = np.zeros(len(X_train))

#lgb_y_pred_test = np.zeros(len(X_test))



for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

    # LightGBM parameters

    params = {

            'task' : 'train',

            'boosting_type' : 'gbdt',

            'objective' : 'regression',

            'metric' : {'l2'},

            'num_leaves' : 31,

            'learning_rate' : 0.1,

            'feature_fraction' : 0.9,

            'bagging_fraction' : 0.8,

            'bagging_freq': 5,

            'verbose' : 0

    }

    

    

    lgb_train = lgb.Dataset(X_train_, y_train_)

    lgb_eval = lgb.Dataset(X_train_, y_train_, reference=lgb_train)



    gbm = lgb.train(params,

            lgb_train,

            num_boost_round=100,

            valid_sets=lgb_eval,

            early_stopping_rounds=10)

    

    #y_pred = gbm.predict(X_val)

    y_pred = gbm.predict(X_test)

    #score = mean_squared_error(y_val, y_pred)**0.5

    #scores.append(score)

    series = pd.Series(y_pred, name='rslt_' + str(i))

    df_cv_avg = pd.concat([df_cv_avg, series], axis=1)

    print('CV Score of Fold_%d is completed.' % (i))

 

    del X_train_, y_train_, X_val, y_val

    gc.collect()

    

df_cv_avg['rslt_avg'] = df_cv_avg.mean(axis=1)

display(df_cv_avg)
importance = pd.DataFrame(gbm.feature_importance(), index=X_train.columns, columns=['importance']).sort_values('importance', ascending=True)

importance.head(50)
#学習が終わった後でy_trainを補正
df_cv_avg['rslt_avg']=np.exp(df_cv_avg['rslt_avg'])
df_cv_avg['rslt_avg']=df_cv_avg['rslt_avg'].round()
submission = pd.read_csv('../input/exam-for-students20200129/sample_submission.csv')



df_cv_avg.set_index(submission.index, inplace=True)

submission.ConvertedSalary = df_cv_avg['rslt_avg']

submission.to_csv('submission.csv',index=False)

submission