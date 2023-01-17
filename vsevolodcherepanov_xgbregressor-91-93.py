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
df=pd.read_csv('/kaggle/input/flights-data/flights.csv')

df.isnull().sum() # observe all nans 

df.nunique() #unique values per col

df.drop('Unnamed: 0',axis=1, inplace=True)

cols=df.columns

print("Time period:", df.year.min()," to ",df.year.max(),"\n", 

      df.head())
from sklearn.preprocessing import LabelEncoder

cat_cols = [col for col in df.columns if df[col].dtype == "object"]

label_encoder = LabelEncoder()

for col in set(cat_cols):

    df[col]=df[col].astype('str')

    df[col] = label_encoder.fit_transform(df[col])

from sklearn.impute import SimpleImputer

imputer = SimpleImputer()

imputed = pd.DataFrame(imputer.fit_transform(df))

imputed.columns=cols

print(imputed.head())
from sklearn.model_selection import train_test_split

train, test = train_test_split(imputed, test_size=0.2) 
y_train=train['arr_delay']

X_train=train.drop(['arr_delay'], axis=1)

y_test=test['arr_delay']

X_test=test.drop(['arr_delay'], axis=1)
from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score

my_model = XGBRegressor(n_estimators=300,learning_rate=0.05, n_jobs=4)

my_model.fit(X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(X_test, y_test)],

             verbose=False)

cv_scores=cross_val_score(my_model, X_train, y_train,cv=5)

pr_arr_delay=my_model.predict(X_test)

print('mean cross-validation score:', cv_scores.mean())
y_train_2=train['dep_delay']

X_train_2=train.drop(['dep_delay'], axis=1)

y_test_2=test['dep_delay']

X_test_2=test.drop(['dep_delay'], axis=1)
my_model.fit(X_train_2, y_train_2, 

             early_stopping_rounds=5, 

             eval_set=[(X_test_2, y_test_2)],

             verbose=False)

cv_scores_2 = cross_val_score(my_model, X_train_2, y_train_2,cv=5)

pr_dep_delay=my_model.predict(X_test_2)

print('mean cross-validation score:', cv_scores_2.mean())
arr_delay=pd.DataFrame({"Prediction":pr_arr_delay, "Test data":y_test})

arr_delay.plot.kde(bw_method=3,figsize=(10,5))
dep_delay=pd.DataFrame({"Prediction":pr_dep_delay, "Test data":y_test_2})

dep_delay.plot.kde(bw_method=3,figsize=(10,5))
a=[]

b=[]

for i in range(len(imputed)):

    if imputed.dep_delay[i]<0:

        imputed.dep_delay[i]=imputed.dep_delay[i]%1

        a.append(1)

    else: 

        a.append(0)

for e in range(len(imputed)):

    if imputed.arr_delay[e]<0:

        imputed.arr_delay[e]=imputed.arr_delay[e]%1

        b.append(1)

    else: 

        b.append(0)

imputed['negative value dep?']=a

imputed['negative value arr?']=b
train_new, test_new = train_test_split(imputed, test_size=0.2) 
y_train_new=train_new['arr_delay']

X_train_new=train_new.drop(['arr_delay'], axis=1)

y_test_new=test_new['arr_delay']

X_test_new=test_new.drop(['arr_delay'], axis=1)
my_model.fit(X_train_new, y_train_new, 

             early_stopping_rounds=5, 

             eval_set=[(X_test_new, y_test_new)],

             verbose=False)

cv_scores_2 = cross_val_score(my_model, X_train_new, y_train_new,cv=5)

pr_arr_delay_new=my_model.predict(X_test_new)

print('mean cross-validation score:', cv_scores_2.mean())
arr_delay=pd.DataFrame({"Prediction":pr_arr_delay_new, "Test data":y_test_new})

arr_delay.plot.kde(bw_method=3,figsize=(10,5))