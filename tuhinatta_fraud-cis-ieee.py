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
import gc, datetime, random





# Classifier Libraries

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier



# Other Libraries

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss

from imblearn.metrics import classification_report_imbalanced

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report

from collections import Counter

from sklearn.model_selection import KFold, StratifiedKFold



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import lightgbm as lgb

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.metrics import roc_auc_score



import altair as alt

from altair.vega import v5

from IPython.display import HTML

from sklearn import preprocessing



import gc, datetime, random



from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
train = pd.read_csv('/kaggle/input/test-train/train.csv')

test = pd.read_csv('/kaggle/input/test-train/test.csv')
def setDevice(df):

    df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()

    df['device_name'] = df['DeviceInfo'].str.split('/', expand=True)[0]

    df.loc[df['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'

    df.loc[df['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'

    df.loc[df['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'

    df.loc[df['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'

    df.loc[df['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'

    df.loc[df['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'

    df.loc[df['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'

    df.loc[df['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'

    df.loc[df['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'

    df.loc[df['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'

    df.loc[df['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'

    df.loc[df['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'

    df.loc[df['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'

    df.loc[df['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'

    df.loc[df['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'

    df.loc[df['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'

    df.loc[df['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'



    df.loc[df.device_name.isin(df.device_name.value_counts()[df.device_name.value_counts() < 200].index), 'device_name'] = "Others"

    df['had_id'] = 1

    gc.collect()

    

    return df



train=setDevice(train)

test=setDevice(test)
train["lastest_browser"] = np.zeros(train.shape[0])

test["lastest_browser"] = np.zeros(test.shape[0])



train.loc[train['id_31'].notnull(), 'lastest_browser'] = 1

test.loc[test['id_31'].notnull(), 'lastest_browser'] = 1
i_cols = ['card1','card2','card3','card5',

          'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',

          'D1','D2','D3','D4','D5','D6','D7','D8',

          'addr1','addr2',

          'dist1','dist2',

          'P_emaildomain', 'R_emaildomain',

          'DeviceInfo','device_name',

          'id_30','id_33',

          'uid','uid2','uid3',

         ]



for col in i_cols:

    temp_df = pd.concat([train[[col]], test[[col]]])

    fq_encode = temp_df[col].value_counts(dropna=False).to_dict()   

    train[col+'_fq_enc'] = train[col].map(fq_encode)

    test[col+'_fq_enc']  = test[col].map(fq_encode)





for col in ['DT_M','DT_W','DT_D']:

    temp_df = pd.concat([train[[col]], test[[col]]])

    fq_encode = temp_df[col].value_counts().to_dict()

            

    train[col+'_total'] = train[col].map(fq_encode)

    test[col+'_total']  = test[col].map(fq_encode)

    

periods = ['DT_M','DT_W','DT_D']

i_cols = ['uid']

for period in periods:

    for col in i_cols:

        new_column = col + '_' + period

            

        temp_df = pd.concat([train[[col,period]], test[[col,period]]])

        temp_df[new_column] = temp_df[col].astype(str) + '_' + (temp_df[period]).astype(str)

        fq_encode = temp_df[new_column].value_counts().to_dict()

            

        train[new_column] = (train[col].astype(str) + '_' + train[period].astype(str)).map(fq_encode)

        test[new_column]  = (test[col].astype(str) + '_' + test[period].astype(str)).map(fq_encode)

        

        train[new_column] /= train[period+'_total']

        test[new_column]  /= test[period+'_total']
def get_too_many_null_attr(data):

    many_null_cols = [col for col in data.columns if data[col].isnull().sum() / data.shape[0] > 0.9]

    return many_null_cols
def get_too_many_repeated_val(data):

    big_top_value_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

    return big_top_value_cols
def get_useless_columns(data):

    too_many_null = get_too_many_null_attr(data)

    print("More than 90% null: " + str(len(too_many_null)))

    too_many_repeated = get_too_many_repeated_val(data)

    print("More than 90% repeated value: " + str(len(too_many_repeated)))

    cols_to_drop = list(set(too_many_null + too_many_repeated))

    cols_to_drop.remove('isFraud')

    return cols_to_drop
cols_to_drop = get_useless_columns(train)
train = train.drop(cols_to_drop, axis=1)
class ModifiedLabelEncoder(LabelEncoder):

    def fit_transform(self, y, *args, **kwargs):

        return super().fit_transform(y).reshape(-1, 1)



    def transform(self, y, *args, **kwargs):

        return super().transform(y).reshape(-1, 1)
class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attr):

        self.attributes = attr

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attributes].values
noisy_cols = [

    'TransactionID','TransactionDT',                      

    'uid','uid2','uid3',                                 

    'DT','DT_M','DT_W','DT_D',       

    'DT_hour','DT_day_week','DT_day',

    'DT_D_total','DT_W_total','DT_M_total',

    'id_30','id_31','id_33',

    'D1', 'D2', 'D9',

]



noisy_cat_cols = list(train[noisy_cols].select_dtypes(include=['object']).columns) 

noisy_num_cold = list(train[noisy_cols].select_dtypes(exclude=['object']).columns)
cat_attr = list(train.select_dtypes(include=['object']).columns)

num_attr = list(train.select_dtypes(exclude=['object']).columns)

num_attr.remove('isFraud')



for col in noisy_cat_cols:

    if col in cat_attr:

        print("Deleting " + col)

        cat_attr.remove(col)

for col in noisy_num_cold:

    if col in num_attr:

        print("Deleting " + col)

        num_attr.remove(col)
num_pipeline = Pipeline([

        ('selector', DataFrameSelector(num_attr)),

        ('imputer', SimpleImputer(strategy="median")),

        ('scaler', StandardScaler()),

    ]) 



cat_pipeline = Pipeline([

        ('selector', DataFrameSelector(cat_attr)),

        ('imputer', SimpleImputer(strategy="most_frequent")),

    ])





full_pipeline = FeatureUnion(transformer_list=[

    ('num_pipeline', num_pipeline),

    ('cat_pipeline', cat_pipeline),

])
def encodeCategorical(df_train, df_test):

    for f in df_train.drop('isFraud', axis=1).columns:

        if df_train[f].dtype=='object' or df_test[f].dtype=='object': 

            lbl = preprocessing.LabelEncoder()

            lbl.fit(list(df_train[f].values) + list(df_test[f].values))

            df_train[f] = lbl.transform(list(df_train[f].values))

            df_test[f] = lbl.transform(list(df_test[f].values))

    return df_train, df_test
y_train = train['isFraud']

train, test = encodeCategorical(train, test)
X_train = pd.DataFrame(full_pipeline.fit_transform(train))

gc.collect()
del train
test = test.drop(cols_to_drop, axis=1)

test = pd.DataFrame(full_pipeline.transform(test))
c=0

for i in range(len(y_train)) :

    if y_train[i]==1 :

        c+=1

        

print(c)

print(len(y_train))
print(len(X_train))
#XG Boost on df

from xgboost import XGBClassifier

gg = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, gamma=0,

              learning_rate=0.1, max_delta_step=0, max_depth=2,

              min_child_weight=1, missing=None, n_estimators=70, n_jobs=-1,

              nthread=None, objective='binary:logistic', random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

              silent=None, subsample=1, verbosity=1)

gg.fit(X_train, y_train)

y_xgb = gg.predict_proba(test)[:, 1]
#Light GBM on df

lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(test, reference=lgb_train)



params = {'objective': 'binary','feature_fraction': 1,'bagging_fraction': 1,'verbose': -1}



gbm = lgb.train(params,lgb_train,num_boost_round=20)

y_gbm = gbm.predict(test)
outcome = pd.DataFrame(y_xgb)

outcome.to_csv('y_xgb.csv', index=False)
outcome = pd.DataFrame(y_gbm)

outcome.to_csv('y_gbm.csv', index=False)