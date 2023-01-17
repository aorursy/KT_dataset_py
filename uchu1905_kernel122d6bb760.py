# 準備

import pandas as pd

from pandas import Series, DataFrame

import numpy as np

import scipy as sp



import datetime as dt



import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import seaborn as sns

%matplotlib inline

plt.style.use('ggplot')



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 20)
# 特徴量エンジニアリングとモデリングで使うパッケージのインポート

from sklearn.preprocessing import StandardScaler



from sklearn.metrics import roc_auc_score

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder, CatBoostEncoder

from tqdm import tqdm_notebook as tqdm



import lightgbm as lgb

from lightgbm import LGBMClassifier





# トレーニングデータ読み込み

df_train = pd.read_csv('../input/exam-for-students20200129/train.csv', index_col=0)

#df_train = pd.read_csv('../input/train.csv/train.csv', index_col=0, skiprows=lambda x: x%20!=0)

#df_train.head()
# テストデータ読み込み

df_test = pd.read_csv('../input/exam-for-students20200129/test.csv', index_col=0)

#df_test = pd.read_csv('../input/test.csv/test.csv', index_col=0, skiprows=lambda x: x%20!=0)

#df_test.head()
# ターゲットのカラムを指定

tgt_col = 'ConvertedSalary'
# 欠損値の個数を特徴量として使用

df_train2 = df_train.copy()

df_test2 = df_test.copy()



df_train2['missing_amnt'] = df_train2.isnull().sum(axis=1)

df_test2['missing_amnt'] = df_test2.isnull().sum(axis=1)
# 評価指標がRMSLEなのでターゲットを対数変換

df_train2[tgt_col] = np.log1p(df_train2[tgt_col])
# トレーニングデータを目的変数(y)と説明変数(X)に分割

y_train = df_train2[tgt_col]

X_train = df_train2.drop([tgt_col], axis=1)



X_test = df_test2
num_cols = []

for col in X_train.columns:

    if X_train[col].dtype == 'int64' or X_train[col].dtype == 'float64':

        num_cols.append(col)



scaler = StandardScaler()

scaler.fit(X_train[num_cols])



X_train[num_cols] = scaler.transform(X_train[num_cols])

X_test[num_cols] = scaler.transform(X_test[num_cols])
# 変数Studentをエンコーディングする関数定義

def student_enc(x):

    if x == 'No':

        return 3

    elif x == 'Yes, part-time':

        return 2

    elif x == 'Yes, full-time':

        return 1
# 変数Studentをエンコーディング

X_train['Student'] = X_train['Student'].apply(student_enc)

X_test['Student'] = X_test['Student'].apply(student_enc)
# 変数CompanySizeをエンコーディングする関数定義

def company_size_enc(x):

    if x == 'Fewer than 10 employees':

        return 1

    elif x == '10 to 19 employees':

        return 2

    elif x == '20 to 99 employees':

        return 3

    elif x == '100 to 499 employees':

        return 4

    elif x == '500 to 999 employees':

        return 5

    elif x == '1,000 to 4,999 employees':

        return 6

    elif x == '5,000 to 9,999 employees':

        return 7

    elif x == '10,000 or more employees':

        return 8
# 変数CompanySizeをエンコーディング

X_train['CompanySize'] = X_train['CompanySize'].apply(company_size_enc)

X_test['CompanySize'] = X_test['CompanySize'].apply(company_size_enc)
# 変数YearsCodingをエンコーディングする関数定義

def years_coding_enc(x):

    if x == '0-2 years':

        return 1

    elif x == '3-5 years':

        return 2

    elif x == '6-8 years':

        return 3

    elif x == '9-11 years':

        return 4

    elif x == '12-14 years':

        return 5

    elif x == '15-17 years':

        return 6

    elif x == '18-20 years':

        return 7

    elif x == '21-23 years':

        return 8

    elif x == '24-26 years':

        return 9

    elif x == '27-29 years':

        return 10

    elif x == '30 or more years':

        return 11
# 変数YearsCodingをエンコーディング

X_train['YearsCoding'] = X_train['YearsCoding'].apply(years_coding_enc)

X_test['YearsCoding'] = X_test['YearsCoding'].apply(years_coding_enc)
# 変数YearsCodingProfをエンコーディングする関数定義

def years_coding_prof_enc(x):

    if x == '0-2 years':

        return 1

    elif x == '3-5 years':

        return 2

    elif x == '6-8 years':

        return 3

    elif x == '9-11 years':

        return 4

    elif x == '12-14 years':

        return 5

    elif x == '15-17 years':

        return 6

    elif x == '18-20 years':

        return 7

    elif x == '21-23 years':

        return 8

    elif x == '24-26 years':

        return 9

    elif x == '27-29 years':

        return 10

    elif x == '30 or more years':

        return 11
# 変数YearsCodingProfProfをエンコーディング

X_train['YearsCodingProf'] = X_train['YearsCodingProf'].apply(years_coding_prof_enc)

X_test['YearsCodingProf'] = X_test['YearsCodingProf'].apply(years_coding_prof_enc)
# 変数JobSatisfactionをエンコーディングする関数定義

def job_satisfaction_enc(x):

    if x == 'Extremely dissatisfied':

        return 1

    elif x == 'Moderately dissatisfied':

        return 2

    elif x == 'Slightly dissatisfied':

        return 3

    elif x == 'Neither satisfied nor dissatisfied':

        return 4

    elif x == 'Slightly satisfied':

        return 5

    elif x == 'Moderately satisfied':

        return 6

    elif x == 'Extremely satisfied':

        return 7
# 変数JobSatisfactionをエンコーディング

X_train['JobSatisfaction'] = X_train['JobSatisfaction'].apply(job_satisfaction_enc)

X_test['JobSatisfaction'] = X_test['JobSatisfaction'].apply(job_satisfaction_enc)
# 変数CareerSatisfactionをエンコーディングする関数定義

def career_satisfaction_enc(x):

    if x == 'Extremely dissatisfied':

        return 1

    elif x == 'Moderately dissatisfied':

        return 2

    elif x == 'Slightly dissatisfied':

        return 3

    elif x == 'Neither satisfied nor dissatisfied':

        return 4

    elif x == 'Slightly satisfied':

        return 5

    elif x == 'Moderately satisfied':

        return 6

    elif x == 'Extremely satisfied':

        return 7
# 変数CareerSatisfactionをエンコーディング

X_train['CareerSatisfaction'] = X_train['CareerSatisfaction'].apply(career_satisfaction_enc)

X_test['CareerSatisfaction'] = X_test['CareerSatisfaction'].apply(career_satisfaction_enc)
# 変数Ageをエンコーディングする関数定義

def age_enc(x):

    if x == 'Under 18 years old':

        return 1

    elif x == '18 - 24 years old':

        return 2

    elif x == '25 - 34 years old':

        return 3

    elif x == '35 - 44 years old':

        return 4

    elif x == '45 - 54 years old':

        return 5

    elif x == '55 - 64 years old':

        return 6

    elif x == '65 years or older':

        return 7
# 変数Ageをエンコーディング

X_train['Age'] = X_train['Age'].apply(age_enc)

X_test['Age'] = X_test['Age'].apply(age_enc)
# 変数EducationParentsをエンコーディングする関数定義

def edu_enc(x):

    if x == 'They never completed any formal education':

        return 1

    elif x == 'Primary/elementary school':

        return 2

    elif x == 'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)':

        return 3

    elif x == 'Some college/university study without earning a degree':

        return 4

    elif x == 'Bachelor’s degree (BA, BS, B.Eng., etc.)':

        return 6

    elif x == 'Master’s degree (MA, MS, M.Eng., MBA, etc.)':

        return 7

    elif x == 'Associate degree':

        return 5

    elif x == 'Other doctoral degree (Ph.D, Ed.D., etc.)':

        return 8

    elif x == 'Professional degree (JD, MD, etc.)':

        return 9
# カテゴリ変数のカラム名とユニーク数を取得

cat_cols = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cat_cols.append(col)

        

        #print(col, X_train[col].nunique())
#CountEncoding

for i in cat_cols:

    summary = X_train[i].value_counts() / len(X_train)

    X_train[i] = X_train[i].map(summary)

    X_test[i] = X_test[i].map(summary)
X_train['Country_LastNewJob'] = X_train['Country'] * X_train['LastNewJob']

X_train['Country_YearsCodingProf'] = X_train['Country'] * X_train['YearsCodingProf']

X_train['LastNewJob_YearsCodingProf'] = X_train['LastNewJob'] * X_train['YearsCodingProf']
X_test['Country_LastNewJob'] = X_test['Country'] * X_test['LastNewJob']

X_test['Country_YearsCodingProf'] = X_test['Country'] * X_test['YearsCodingProf']

X_test['LastNewJob_YearsCodingProf'] = X_test['LastNewJob'] * X_test['YearsCodingProf']
X_train.fillna(X_train.median(), inplace=True)

X_test.fillna(X_train.median(), inplace=True)
# 層化抽出5回のCVを行い、LightGBMでモデルに学習させる

scores = []



# skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

# StratifiedKFoldは分類問題でしか使えないのでKFoldを使う

kf = KFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in enumerate(tqdm(kf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

        

    model = lgb.LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

                                importance_type='split', learning_rate=0.05, max_depth=-1,

                                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                                n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

                                random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

                                subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

    

    model.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='rmse', eval_set=[(X_val, y_val)])

    y_pred = model.predict(X_val)

    mse = mean_squared_error(y_val, y_pred)

    rmse = np.sqrt(mse)

    scores.append(rmse)



    print('CV Score of Fold_%d is %f' % (i, rmse))
# モデルのRMSEを確認

print(np.mean(scores))

print(scores)
# 全データで再学習し、testに対して予測する

model.fit(X_train, y_train)



y_pred = model.predict(X_test)
# sample submissionを読み込んで、予測値を代入の後、保存する



submission = pd.read_csv('../input/exam-for-students20200129/sample_submission.csv', index_col=0)



submission[tgt_col] = y_pred

# ここで予測値に指数変換をかけて元に戻す

submission[tgt_col] = np.exp(submission[tgt_col]) - 1



submission.to_csv('submission.csv')
