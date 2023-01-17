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
from tqdm import tqdm_notebook as tqdm



import numpy as np 

import pandas as pd 

from pandas import DataFrame, Series

import scipy as sp



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline

import seaborn as sns



# Encoder

from sklearn.preprocessing import LabelEncoder

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder



# Numeric Encoder

from sklearn.preprocessing import StandardScaler, MinMaxScaler, quantile_transform



# Decomposer

from sklearn.decomposition import PCA



# Text Encoder

from nltk.stem import PorterStemmer

from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer



# Modeling

import lightgbm as lgb

from lightgbm import LGBMClassifier

from lightgbm import LGBMRegressor



import xgboost as xgb

from xgboost import XGBClassifier



# Validation Scheme

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit



# Scores

from sklearn.model_selection import cross_val_score

# Classification: AUC, f1, Logloss

from sklearn.metrics import roc_auc_score, f1_score, log_loss

# Regression: MAR, RMSE

from sklearn.metrics import mean_absolute_error, mean_squared_error
fname = '../input/exam-for-students20200923/train.csv'

df_train = pd.read_csv(fname, index_col=0)

print(df_train.shape)



fname = '../input/exam-for-students20200923/test.csv'

df_test = pd.read_csv(fname, index_col=0)

print(df_test.shape)



fname = '../input/exam-for-students20200923/country_info.csv'

country = pd.read_csv(fname) #, index_col=0

print(country.shape)
# Europeぽい国の人だけで学習する→学習データが減りすぎたのかよくわからんけどスコア下がってしまったので今度は下位50国だけ削除する

# Sub 5

# Sub 4 

"""

countries = [

    'United Kingdom'

    , 'United States'

    , 'Netherlands'

    , 'Sweden'

    , 'Australia'

    , 'India'

    , 'Greece'

    , 'Poland'

    , 'Russian Federation'

    , 'Ireland'

    , 'China'

    , 'Colombia'

    , 'Japan'

    , 'Romania'

    , 'Portugal'

    , 'Italy'

    , 'New Zealand'

    , 'Turkey'

    , 'Denmark'

    , 'Canada'

    , 'Spain'

]

"""



"""

countries = ['Latvia', 'Nepal', 'Taiwan', 'Thailand', 'United Arab Emirates',

       'South Korea', 'Kenya', 'Morocco',

       'Venezuela, Bolivarian Republic of...', 'Kazakhstan',

       'Bosnia and Herzegovina', 'Dominican Republic', 'Saudi Arabia',

       'Other Country (Not Listed Above)', 'Peru', 'Malta', 'Armenia',

       'Lebanon', 'Costa Rica', 'Albania', 'Uruguay', 'Tunisia', 'Luxembourg',

       'Jordan', 'Ecuador', 'Algeria', 'Republic of Moldova', 'Georgia',

       'Cyprus', 'Ghana', 'The former Yugoslav Republic of Macedonia',

       'El Salvador', 'Azerbaijan', 'Uganda', 'Iceland', 'Cuba', 'Ethiopia',

       'Paraguay', 'Panama', 'Zimbabwe', 'Republic of Korea', 'Uzbekistan',

       'Guatemala', 'Syrian Arab Republic', 'Bolivia', 'Mauritius', 'Cambodia',

       'Sudan', 'United Republic of Tanzania', 'Kuwait', 'Nicaragua',

       'Cameroon', 'Trinidad and Tobago', 'Kyrgyzstan', 'Madagascar', 'Oman',

       'Iraq', 'Afghanistan', 'Yemen', 'Montenegro', 'Mozambique', 'Myanmar',

       'Mongolia', 'Honduras', 'Somalia', 'Fiji', 'Maldives', 'Andorra',

       'Bahrain', 'Benin', 'Jamaica', 'Qatar', 'Rwanda', 'Liechtenstein',

       'Senegal', 'Swaziland', 'Libyan Arab Jamahiriya', 'Barbados', 'Bhutan',

       'Democratic Republic of the Congo', 'Botswana', 'Bahamas', 'Togo',

       'Tajikistan', 'Namibia', 'Dominica', 'Turkmenistan', 'Lesotho',

       'Guyana', 'Malawi', 'Monaco', 'Zambia', 'Suriname', 'Marshall Islands',

       'Congo, Republic of the...', 'Saint Lucia', 'Gambia', 'Sierra Leone',

       "Côte d'Ivoire", 'Eritrea']

"""
countries = ['Nicaragua', 'Cameroon', 'Trinidad and Tobago', 'Kyrgyzstan',

       'Madagascar', 'Oman', 'Iraq', 'Afghanistan', 'Yemen', 'Montenegro',

       'Mozambique', 'Myanmar', 'Mongolia', 'Honduras', 'Somalia', 'Fiji',

       'Maldives', 'Andorra', 'Bahrain', 'Benin', 'Jamaica', 'Qatar', 'Rwanda',

       'Liechtenstein', 'Senegal', 'Swaziland', 'Libyan Arab Jamahiriya',

       'Barbados', 'Bhutan', 'Democratic Republic of the Congo', 'Botswana',

       'Bahamas', 'Togo', 'Tajikistan', 'Namibia', 'Dominica', 'Turkmenistan',

       'Lesotho', 'Guyana', 'Malawi', 'Monaco', 'Zambia', 'Suriname',

       'Marshall Islands', 'Congo, Republic of the...', 'Saint Lucia',

       'Gambia', 'Sierra Leone', "Côte d'Ivoire", 'Eritrea']
# Europeぽい国の人だけで学習する→学習データが減りすぎたのかよくわからんけどスコア下がってしまったので今度は下位100国だけ削除する

print(df_train.shape)

df_train = df_train[~df_train.Country.isin(countries)]

print(df_train.shape)
"""

top_countries = ['United States', 'India', 'United Kingdom', 'Canada',

       'Russian Federation', 'Australia', 'Brazil', 'Netherlands', 'Spain',

       'Poland']



#df_train.loc[~df_train.Country.isin(top_countries), 'Country'] = 'others'



df_train.Country.unique()

"""
# Train and Test

# targetを指定

target = 'ConvertedSalary'



X_train = df_train.drop(columns=target)

print('X_train', X_train.shape)



y_train = df_train[target]

print('y_train', y_train.shape)



X_test = df_test

print('X_test', X_test.shape)
X = pd.concat([X_train, X_test])

print('X', X.shape)
country.info()
# Country Info:　全てのカラム使う→処理めんどいからいったんRegion, GDP, Poplulaion,  Areaだけ

columns = ['Country', 'Region']

#columns = ['Country', 'Region', 'Population', 'Area (sq. mi.)', 'GDP ($ per capita)']

country_ = country[columns]

print(X.shape)

X = pd.merge(X, country_, on='Country', how='left')

print(X.shape)

X.head()
# cats and nums

cats = []

for col in df_train.columns:

    if df_train[col].dtype == 'object':

        cats.append(col)

        print(col, df_train[col].nunique())



nums =[]

for col in df_train.columns:

    col_type = df_train[col].dtypes

    if (col != target) & ((str(col_type)[:3] == 'int') | (str(col_type)[:5] == 'float')):

        print(col)

        nums.append(col)

        

print('数値特徴量 {}, カテゴリ特徴量 {}'.format(len(nums), len(cats)))
years_dict = {

    '0-2 years': 2

    , '3-5 years' : 5

    , '6-8 years' : 8

    , '9-11 years': 11

    , '12-14 years' : 14

    , '15-17 years' : 17

    , '18-20 years' : 20

    , '21-23 years' : 23

    , '24-26 years' : 26

    , '27-29 years' : 29

    , '30 or more years' : 30

}



X['YearsCoding'] = X['YearsCoding'].replace(years_dict)

X['YearsCodingProf'] = X['YearsCodingProf'].replace(years_dict)
X['YearsCoding'].fillna(-1, inplace=True)

X['YearsCodingProf'].fillna(-1, inplace=True)
X['YearsCoding'] = X['YearsCoding'].astype(int)

X['YearsCodingProf'] = X['YearsCodingProf'].astype(int)
years_dict = {

    'Under 18 years old': 17

    , '18 - 24 years old' : 24

    , '25 - 34 years old' : 34

    , '35 - 44 years old' : 44

    , '45 - 54 years old' : 54

    , '55 - 64 years old' : 64

    , '65 years or older' : 65

}



X['Age'] = X['Age'].replace(years_dict)
X['Age'].fillna(-1, inplace=True)
X['Age'] = X['Age'].astype(int)
size_dict = {

    'Fewer than 10 employees' : 9

    , '10 to 19 employees' : 10

    , '20 to 99 employees': 20

    , '100 to 499 employees' : 100

    , '500 to 999 employees' : 500

    , '1,000 to 4,999 employees' : 1000

    , '5,000 to 9,999 employees' : 5000

    , '10,000 or more employees' : 10000

}



X['CompanySize'] = X['CompanySize'].replace(size_dict)
X['CompanySize'].fillna(-1, inplace=True)
X['CompanySize'] = X['CompanySize'].astype(int)
hours_dict = {

    'Less than 1 hour' : 0

    , '1 - 4 hours' : 1

    ,  '5 - 8 hours' : 5

    , '9 - 12 hours' : 9

    , 'Over 12 hours' :12

}

       

X['HoursComputer'] = X['HoursComputer'].replace(hours_dict)
hours_dict = {

    'Less than 30 minutes' : 0

    , '30 - 59 minutes' : 1

    , '1 - 2 hours' : 2

    , '3 - 4 hours' : 3

    , 'Over 4 hours' : 4

}



X['HoursOutside'] = X['HoursOutside'].replace(hours_dict)
X['HoursComputer'].fillna(-1, inplace=True)

X['HoursOutside'].fillna(-1, inplace=True)
X['HoursComputer'] = X['HoursComputer'].astype(int)

X['HoursOutside'] = X['HoursOutside'].astype(int)
col = 'DevType'



X['Full-stack'] = 0

X['Back-end'] = 0

X['Front-end'] = 0

X['Mobile'] = 0

X['executive'] = 0

X['DataScientist'] = 0

X['Student'] = 0



X[col].fillna('#', inplace=True)



X.loc[X[col].str.contains('Full-stack'), 'Full-stack'] = 1

X.loc[X[col].str.contains('Back-end'), 'Back-end'] = 1

X.loc[X[col].str.contains('Front-end'), 'Front-end'] = 1

X.loc[X[col].str.contains('Mobile'), 'Mobile'] = 1

X.loc[X[col].str.contains('executive'), 'executive'] = 1

X.loc[X[col].str.contains('Data scientist'), 'DataScientist'] = 1

X.loc[X[col].str.contains('Student'), 'Student'] = 1
X.DevType.value_counts()
stu_dict = {

    'No' : 0

    , 'Yes, full-time' : 1

    , 'Yes, part-time': 2

}



#X['Student'] = X['Student'].replace(stu_dict)
X['Student'].fillna(-1, inplace=True)
X['Student'] = X['Student'].astype(int)
# null flag

X['Emp_null'] = 0

X.loc[X['Employment'].isnull(), 'Emp_null'] = 1
# 無職フラグ

X['Employment'].fillna('#', inplace=True)

X['NotEmployed'] = 0

X.loc[X['Employment'].str.contains('Not employed,'), 'NotEmployed'] = 1
# Part-Time flag

X['Part-time'] = 0

X.loc[X['Employment'].str.contains('part-time'), 'Part-time'] = 1



# Fleelance

X['Fleelance'] = 0

X.loc[X['Employment'].str.contains('free'), 'Fleelance'] = 1
#White or of European descentW                                                                                                                                                           0.913455

#Middle Eastern                                                                                                                                                                          0.021991

#South Asian                                                                                                                                                                             0.012296

#Hispanic or Latino/Latina



#Majority = ['White or of European descent', 'Middle Eastern', 'South Asian', 'Hispanic or Latino/Latina']

#X['Minority'] = 0

#X.loc[~X['RaceEthnicity'].isin(Majority), 'Minority'] = 1
job_dict = {

    "I've never had a job" : 0

    , 'Less than a year ago' : 1

    , 'Between 1 and 2 years ago' : 2

    , 'Between 2 and 4 years ago' : 4

    , 'More than 4 years ago' : 5

}



X['LastNewJob'] = X['LastNewJob'].replace(job_dict)

X['LastNewJob'].fillna(-1, inplace=True)
X['LastNewJob'] = X['LastNewJob'].astype(int)
# trainとtestに戻す

X_train = X.iloc[:len(X_train),:]

X_test = X.iloc[len(X_train):,:]

print('TRAIN{}, TEST{}'.format(X_train.shape, X_test.shape))
cats_oe = [

    'Hobby',

 'OpenSource',

 #'Country',

 #'Student',

 #'Employment',

 'FormalEducation',

 'UndergradMajor',

 #'CompanySize',

 #'DevType',

 #'YearsCoding',

 #'YearsCodingProf',

 'JobSatisfaction',

 'CareerSatisfaction',

 'HopeFiveYears',

 'JobSearchStatus',

 #'LastNewJob',

 'UpdateCV',

 #'Currency',

 'SalaryType',

 #'CurrencySymbol',

 'CommunicationTools',

 'TimeFullyProductive',

 'TimeAfterBootcamp',

 'AgreeDisagree1',

 'AgreeDisagree2',

 'AgreeDisagree3',

 'FrameworkWorkedWith',

 'OperatingSystem',

 'NumberMonitors',

 'CheckInCode',

 'AdBlocker',

 'AdBlockerDisable',

 'AdsAgreeDisagree1',

 'AdsAgreeDisagree2',

 'AdsAgreeDisagree3',

 'AdsActions',

 'AIDangerous',

 'AIInteresting',

 'AIResponsible',

 'AIFuture',

 'EthicsChoice',

 'EthicsReport',

 'EthicsResponsible',

 'EthicalImplications',

 'StackOverflowRecommend',

 'StackOverflowVisit',

 'StackOverflowHasAccount',

 'StackOverflowParticipate',

 'StackOverflowJobs',

 'StackOverflowDevStory',

 'StackOverflowJobsRecommend',

 'StackOverflowConsiderMember',

 'HypotheticalTools1',

 'HypotheticalTools2',

 'HypotheticalTools3',

 'HypotheticalTools4',

 'HypotheticalTools5',

 'WakeTime',

 #'HoursComputer',

 #'HoursOutside',

 'SkipMeals',

 'ErgonomicDevices',

 'Exercise',

 'Gender',

 'SexualOrientation',

 'EducationParents',

 'RaceEthnicity',

 #'Age',

 'Dependents',

 #'MilitaryUS',

 'SurveyTooLong',

 'SurveyEasy',

 'Region' #CountryInfoと結合したから

]
# Ordinal Encoding: 最低限のEncoding

oe = OrdinalEncoder(cols=cats_oe)

X_train = oe.fit_transform(X_train)

X_test = oe.transform(X_test)
cats_oe = [

 'FrameworkWorkedWith',

 'OperatingSystem'

]
# Count Encoding

for col in cats_oe:

    summary = X_train[col].value_counts()

    col_name = col + '_ce'

    X_train[col_name]  = X_train[col].map(summary)

    X_test[col_name] = X_test[col].map(summary)
leakage = ['Country', 'Currency', 'CurrencySymbol', 'MilitaryUS']

drop_cols = ['DevType', 'Employment']





# Countryをグループ識別子として分離。

group = X_train['Country'].values 

print('group max{}'.format(group.max()))





# 元の特徴量数

print('Original Cols{}'.format(X_train.shape[1]))



X_train_ = X_train.drop(columns=leakage)

X_test_ = X_test.drop(columns=leakage)



X_train_ = X_train_.drop(columns=drop_cols)

X_test_ = X_test_.drop(columns=drop_cols)

print('Cols for Use{}'.format(X_train_.shape[1]))
X_train_.info()
scores = []

lgb_models = []

df_importance = pd.DataFrame()



# 対数平均二乗誤差（RMSLE）

metric = 'rmse'

y_train_log = np.log1p(y_train) # ターゲットを対数変換



# GroupKFold

gkf = GroupKFold(n_splits=5)



# Seed Averaging

seeds = [51,61,71,81,91]

for i, seed in enumerate(seeds):

    for train_index, test_index in gkf.split(X_train_, y_train_log, group):

        # Training

        X_train__, y_train_ = X_train_.iloc[train_index], y_train_log.iloc[train_index]

        X_val_, y_val_ = X_train_.iloc[test_index], y_train_log.iloc[test_index]

        

        # Model

        model = LGBMRegressor(random_state=seed)

        model.fit(X_train__, y_train_)

        lgb_models.append(model)

        

        # Validation Score

        y_pred_ = model.predict(X_val_)

        scores.append(mean_squared_error(y_val_, y_pred_))

        #squared:boolean value, optional (default = True)

        #If True returns MSE value, if False returns RMSE value.



        # Feature Importance

        if i ==0:

            df_importance = pd.DataFrame(model.booster_.feature_importance(), index=X_train__.columns, columns=['importance'])

            

        else:

            df_importance += pd.DataFrame(model.booster_.feature_importance(), index=X_train__.columns, columns=['importance'])



print('Max RMSLE:{}, Min RMSLE{}, Ave Score{}'.format(np.round(max(scores), 5), np.round(min(scores), 5), np.round(np.mean(scores), 5)))        
#df_importance_ = df_importance.sort_values(by='importance', ascending=False).head(20)

#df_importance_.index



df_importance = df_importance.sort_values(by='importance', ascending=False)

x  = df_importance.index

y = df_importance.importance

fig = plt.figure(figsize=(10, 25))

ax = plt.subplot()

ax = sns.barplot(y=x, x=y,data=df_importance)
lgb_models
# CV Averaging

y_pred_test = np.zeros(len(X_test))



for model in lgb_models:

    pred = model.predict(X_test_,num_iteration=model.booster_.best_iteration)

    pred = np.expm1(pred) #対数から変換

    pred[pred<0] = 0 

    y_pred_test += pred # テストデータに対する予測値を足していく



y_pred_test /= len(lgb_models) # 最後にfold数で割る

y_pred_test_ = np.round(y_pred_test, 0) #整数に丸める

y_pred_test_
y_pred_test_
fname = '../input/exam-for-students20200923/sample_submission.csv'

submission = pd.read_csv(fname, index_col=0)



submission[target] = y_pred_test_

submission.to_csv('submission.csv')