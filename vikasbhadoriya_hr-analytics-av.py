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

import pandas as pd

import seaborn as sns

import regex as re

import math 

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder,StandardScaler 

from sklearn.impute import KNNImputer

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score,accuracy_score,auc,confusion_matrix,log_loss,roc_curve

from sklearn.metrics import roc_auc_score as roc

import warnings

warnings.filterwarnings('ignore')
df_train_temp = pd.read_csv("../input/train.csv")

df_test_temp = pd.read_csv("../input/test.csv")
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
c =np.nan

str(c)
df_train.head()
df_train.isnull().sum()
df_test.isnull().sum()
df_train.nunique()
df_train.target.value_counts()
train_size = len(df_train)

train_size
y = df_train['target'].values

ids = df_test.enrollee_id

df_train_temp = df_train.drop('target',axis = 1)
df = pd.concat([df_train_temp,df_test_temp],axis = 0)

df.drop('enrollee_id',axis = 1,inplace = True)
df.tail()
df.last_new_job.value_counts()
df.experience.value_counts()
exp_list = []

for exp in df.experience:

    if(exp == '>20'):

        exp_list.append(21.0)

    elif(exp == '<1'):

        exp_list.append(0.0)

    else:

        exp_list.append(float(exp))

        

df.experience = exp_list

        
df.info()
df.company_size.value_counts()
df.company_size.value_counts()
df.company_size.value_counts()
def company_size_pre(x):

    x = str(x)

    if(x == 'nan'):

        return 0

    elif(re.search('[-]',x)!= None):

        

        return int((int(x.split('-')[0])+int(x.split('-')[1]))/2)

    elif(re.search('[/]',x)!= None):

        return int((int(x.split('/')[0])+int(x.split('/')[1]))/2)

    return re.sub('[><+]','',x)
df.company_size = df.company_size.apply(lambda x : company_size_pre(x))
df.company_size = df.company_size.astype('float')

def zero_to_nan_handler(x):

    if(x == 0.0):

        return np.nan

    else:

        return x

df.company_size = df.company_size.apply(lambda x : zero_to_nan_handler(x))
df.company_size.isnull().sum()
def last_new_job(x):

    x = str(x)

    if(re.search('>4',x)!= None):

        return re.sub('>4','4',x)

    elif(x == 'never'):

        return '0'

    else:

        return x

df.last_new_job = df.last_new_job.apply(lambda x : last_new_job(x))
df.last_new_job.value_counts()
def last_new_job_pre(x):

    x = str(x)

    if(x == 'nan'):

        return 0

    else:

        return x

    

df.last_new_job = df.last_new_job.apply(lambda x : last_new_job_pre(x))
df.last_new_job = df.last_new_job.astype('float')
df.last_new_job = df.last_new_job.apply(lambda x: zero_to_nan_handler(x))
df.last_new_job.isnull().sum()
df.info()
cat_col = ['city', 'gender', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'company_type']

con_col = ['city_development_index', 'experience', 'training_hours', 'last_new_job', 'company_size']

cols = list(df.columns.values)
df_cat = pd.get_dummies(df[cat_col])
pd.set_option("display.max_columns",None)
df_cat.head()
df_final = pd.concat([df[con_col],df_cat],axis=1)
df_final.shape
imputer = KNNImputer(n_neighbors=3)

new_df = imputer.fit_transform(df_final)
new_df
df1 = pd.DataFrame()

j = 0 

for i in list(df_final.columns.values):

  df1[i] = new_df[:,j]

  j += 1

df1.shape
df1.isnull().sum()
df_train1 = df1[:train_size].copy()

df_test1 = df1[train_size:].copy()
df_train1.shape,df_test1.shape
df_train1.head()
X_train, X_test, y_train, y_test = train_test_split(df_train1,y,test_size=0.2, random_state=0, stratify=y)

X_train.shape, X_test.shape
from catboost import CatBoostClassifier, Pool

cat = CatBoostClassifier(iterations=160, custom_metric=['AUC'], class_weights=[1,10], logging_level='Silent')

cat.fit(X_train, y_train, eval_set=(X_test,y_test), use_best_model=True)

print(roc_auc_score(y_test,cat.predict_proba(X_test)[:,1]))

#print(confusion_matrix(y_test,cat.predict(X_test)))
from xgboost import XGBClassifier

xgb = XGBClassifier(n_esttimators=100, max_depth=1, scale_pos_weight=10)

xgb.fit(X_train, y_train)

print(roc_auc_score(y_test,xgb.predict_proba(X_test)[:,1]))

#print(confusion_matrix(y_test,cat.predict(X_test)))
from lightgbm import LGBMClassifier

lgb = LGBMClassifier(n_estimators=20, class_weight={0:1,1:10})

lgb.fit(X_train, y_train)

print(roc_auc_score(y_test,lgb.predict_proba(X_test)[:,1]))
lgb.feature_importances_
from sklearn.linear_model import LogisticRegression as LR

lr = LR(max_iter=300, class_weight={0:1,1:10})

lr.fit(X_train, y_train)

print(roc_auc_score(y_test,lr.predict_proba(X_test)[:,1]))


from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

cnt = 0

for train_index, test_index in kfold.split(df_train1, y):

  X_train, X_test = df_train1.loc[train_index], df_train1.loc[test_index]

  y_train, y_test = y[train_index], y[test_index] ;cnt+=1



  print('Fold '+str(cnt)+' : ')



  lr = LR(max_iter=500)

  lr.fit(X_train, y_train)

  print(' LR : ',end='')

  print(roc(y_test,lr.predict_proba(X_test)[:,1]))



  lgb = LGBMClassifier(n_estimators=80)

  lgb.fit(X_train, y_train, eval_metric='AUC',eval_set=(X_train,y_train), early_stopping_rounds=50, verbose=False)

  print('LGM : ',end='')

  print(roc(y_test,lgb.predict_proba(X_test)[:,1]))



  cat = CatBoostClassifier(iterations=1500, logging_level='Silent')

  cat.fit(X_train, y_train, eval_set=(X_test,y_test), use_best_model=True)

  print('Cat : ',end='')

  print(roc(y_test,cat.predict_proba(X_test)[:,1]))



  xgb = XGBClassifier(n_estimators=80)

  xgb.fit(X_train, y_train, eval_metric='auc',eval_set=[(X_train,y_train)], early_stopping_rounds=50, verbose=False)

  print('XGB : ',end='')

  print(roc(y_test,xgb.predict_proba(X_test)[:,1]))



  t_fpr, t_tpr, _ = roc_curve(y_test, [0 for _ in y_test])

  plt.plot(t_fpr, t_tpr, label='Base')



  xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb.predict_proba(X_test)[:,1])

  plt.plot(xgb_fpr, xgb_tpr, linestyle='--', label='XGB')



  lgb_fpr, lgb_tpr, _ = roc_curve(y_test, lgb.predict_proba(X_test)[:,1])

  plt.plot(lgb_fpr, lgb_tpr, linestyle='--', label='LGB')



  cat_fpr, cat_tpr, _ = roc_curve(y_test, cat.predict_proba(X_test)[:,1])

  plt.plot(cat_fpr, cat_tpr, linestyle='--', label='CAT')

  

  lr_fpr, lr_tpr, _ = roc_curve(y_test, lr.predict_proba(X_test)[:,1])

  plt.plot(lr_fpr, lr_tpr, linestyle='--', label=' LR')



  plt.xlabel('False Positive Rate')

  plt.ylabel('True Positive Rate')

  plt.legend()

  plt.show()
pd.set_option("display.max_rows",None)
#final_df.to_csv(r"hr_submission.csv",index = False)