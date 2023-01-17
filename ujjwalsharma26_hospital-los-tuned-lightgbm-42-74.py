# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/av-healthcare-analytics-ii/healthcare/train_data.csv')

test_df = pd.read_csv('/kaggle/input/av-healthcare-analytics-ii/healthcare/test_data.csv')



train_df.head()
train_df['grouped'] = train_df['Hospital_code'].astype(str) + train_df['Hospital_type_code'].astype(str) + train_df['City_Code_Hospital'].astype(str)+ train_df['Hospital_region_code'].astype(str) + train_df['Ward_Facility_Code'].astype(str)



test_df['grouped'] = test_df['Hospital_code'].astype(str) + test_df['Hospital_type_code'].astype(str) + test_df['City_Code_Hospital'].astype(str)+ test_df['Hospital_region_code'].astype(str) + test_df['Ward_Facility_Code'].astype(str)

test_df.head()
train_df.isnull().sum()
x1=train_df['Bed Grade'].fillna(train_df['Bed Grade'].mode()[0])

x2=train_df['City_Code_Patient'].fillna(train_df['City_Code_Patient'].mode()[0])

train_df['Bed Grade'] = x1

train_df['City_Code_Patient'] =x2

train_df.isnull().sum()



x1=test_df['Bed Grade'].fillna(test_df['Bed Grade'].mode()[0])

x2=test_df['City_Code_Patient'].fillna(test_df['City_Code_Patient'].mode()[0])

test_df['Bed Grade'] = x1

test_df['City_Code_Patient'] =x2

test_df.isnull().sum()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le1 = LabelEncoder()

train_df['Hospital_type_code'] = le.fit_transform(train_df['Hospital_type_code'])

train_df['City_Code_Hospital'] = le.fit_transform(train_df['City_Code_Hospital'])

train_df['Hospital_region_code'] = le.fit_transform(train_df['Hospital_region_code'])

train_df['Department'] = le.fit_transform(train_df['Department'])

train_df['Ward_Type'] = le.fit_transform(train_df['Ward_Type'])

train_df['Ward_Facility_Code'] = le.fit_transform(train_df['Ward_Facility_Code'])

train_df['Type of Admission'] = le.fit_transform(train_df['Type of Admission'])

train_df['Severity of Illness'] = le.fit_transform(train_df['Severity of Illness'])

train_df['Age'] = le.fit_transform(train_df['Age'])

# train_df['grouped'] = le.fit_transform(train_df['grouped'])

train_df['Stay'] = le1.fit_transform(train_df['Stay'])

train_df=train_df.drop('case_id',axis=1)

# train_df=train_df.drop('Admission_Deposit',axis=1)

# train_df=train_df.drop('Hospital_code',axis=1)

train_df=train_df.drop('patientid',axis=1)

# train_df=train_df.drop('Visitors with Patient',axis=1)

# train_df=train_df.drop('Hospital_code',axis=1)

# train_df=train_df.drop('Hospital_type_code',axis=1)

# train_df=train_df.drop('City_Code_Hospital',axis=1)

# train_df=train_df.drop('Hospital_region_code',axis=1)

# train_df=train_df.drop('Ward_Facility_Code',axis=1)



test_df['Hospital_type_code'] = le.fit_transform(test_df['Hospital_type_code'])

test_df['City_Code_Hospital'] = le.fit_transform(test_df['City_Code_Hospital'])

test_df['Hospital_region_code'] = le.fit_transform(test_df['Hospital_region_code'])

test_df['Department'] = le.fit_transform(test_df['Department'])

test_df['Ward_Type'] = le.fit_transform(test_df['Ward_Type'])

test_df['Ward_Facility_Code'] = le.fit_transform(test_df['Ward_Facility_Code'])

test_df['Type of Admission'] = le.fit_transform(test_df['Type of Admission'])

test_df['Severity of Illness'] = le.fit_transform(test_df['Severity of Illness'])

test_df['Age'] = le.fit_transform(test_df['Age'])

# test_df['grouped'] = le.fit_transform(test_df['grouped'])

test_ids=test_df['case_id']

test_df=test_df.drop('case_id',axis=1)

# test_df=test_df.drop('Admission_Deposit',axis=1)

# test_df=test_df.drop('Hospital_code',axis=1)

test_df=test_df.drop('patientid',axis=1)

# test_df=test_df.drop('City_Code_Patient',axis=1)

# test_df=test_df.drop('Visitors with Patient',axis=1)

# test_df=test_df.drop('Hospital_code',axis=1)

# test_df=test_df.drop('Hospital_type_code',axis=1)

# test_df=test_df.drop('City_Code_Hospital',axis=1)

# test_df=test_df.drop('Hospital_region_code',axis=1)

# test_df=test_df.drop('Ward_Facility_Code',axis=1)





train_df.head()
test_df.head()
sns.countplot('Department',data=train_df)
sns.countplot('Ward_Type',data=train_df)
sns.countplot('Type of Admission',data=train_df)
sns.countplot('Severity of Illness',data=train_df)
sns.countplot('Age',data=train_df)
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



X_train, X_val, y_train, y_val = train_test_split(train_df.drop('Stay',axis=1),train_df['Stay'] , test_size=0.2, random_state=0,shuffle=True)
from sklearn.ensemble import RandomForestClassifier



clf_rf = RandomForestClassifier(max_depth=10, random_state=6)

clf_rf.fit(X_train,y_train)

preds=clf_rf.predict(X_val)



print('Accuracy: ', accuracy_score(y_val, preds)*100, '%')
from sklearn.naive_bayes import GaussianNB



clf_gnb = GaussianNB()

clf_gnb.fit(X_train,y_train)

preds=clf_gnb.predict(X_val)



print('Accuracy: ', accuracy_score(y_val, preds)*100, '%')
from xgboost import XGBClassifier

clf_xgb = XGBClassifier()

clf_xgb.fit(X_train,y_train)

preds=clf_xgb.predict(X_val)



print('Accuracy: ', accuracy_score(y_val, preds)*100, '%')
from lightgbm import LGBMClassifier



clf_lgb = make_pipeline(StandardScaler(), LGBMClassifier(random_state=444,n_estimators=825,learning_rate=0.07,colsample_bytree=0.7,

                        min_data_in_leaf=65,reg_alpha=1.6,reg_lambda=1.1))

clf_lgb.fit(X_train,y_train)

preds=clf_lgb.predict(X_val)



print('Accuracy: ', accuracy_score(y_val, preds)*100, '%')
from catboost import Pool, CatBoostClassifier



clf_ctb = CatBoostClassifier(iterations=100,

                           learning_rate=0.08,

                           depth=7,

                           loss_function='MultiClass',

                           eval_metric='Accuracy')

clf_ctb.fit(X_train,y_train)

preds=clf_ctb.predict(X_val)



print('Accuracy: ', accuracy_score(y_val, preds)*100, '%')
df_sub = pd.DataFrame()

df_sub["case_id"] = test_ids

df_sub["Stay"] = le1.inverse_transform(clf_lgb.predict(test_df))

df_sub.head()



df_sub.to_csv("Submission.csv",index=False)
df_sub.head()