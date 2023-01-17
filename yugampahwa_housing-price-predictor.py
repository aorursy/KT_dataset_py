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
#reading the training data and setting index to Id

data=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',index_col='Id')

cname=[col for col in data.columns if data[col].isnull().sum() >400]

data1=data.drop(cname,axis=1)

data1.info()
test_data=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv',index_col='Id')

test_data.drop(['Neighborhood','Exterior2nd','Exterior1st'],axis=1,inplace=True)

test=test_data.drop(cname,axis=1)

test.info()
y=data1.SalePrice

x=data1.drop(['SalePrice'],axis=1)

categorical_cols=[cname for cname in x.columns if x[cname].nunique()<10 and x[cname].dtype=='object']

categorical_col1=[cname for cname in categorical_cols if set(x[cname])==set(test[cname])]

numerical_cols=[cname for cname in x.columns if x[cname].dtype in ['int64','float64']]

my_cols=categorical_col1+numerical_cols

x1=data1[my_cols].copy()

x1
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x1,y,train_size=0.8,test_size=0.2,random_state=0)

x_train.head()
from sklearn.impute import SimpleImputer

impute=SimpleImputer(strategy='mean')

impute1=SimpleImputer(strategy='constant')

x_train_imputed_num=pd.DataFrame(impute.fit_transform(x_train[numerical_cols]))

x_test_imputed_num=pd.DataFrame(impute.transform(x_test[numerical_cols]))

x_train_imputed_num.columns=x_train[numerical_cols].columns

x_test_imputed_num.columns=x_test[numerical_cols].columns

x_train_imputed_num.index=x_train.index

x_test_imputed_num.index=x_test.index

x_train_imputed_col=pd.DataFrame(impute1.fit_transform(x_train[categorical_col1]))

x_test_imputed_col=pd.DataFrame(impute1.transform(x_test[categorical_col1]))

x_train_imputed_col.columns=x_train[categorical_col1].columns

x_test_imputed_col.columns=x_test[categorical_col1].columns

x_train_imputed_col.index=x_train.index

x_test_imputed_col.index=x_test.index

x_train_org=x_train_imputed_num.merge(x_train_imputed_col,left_index=True,right_index=True)

x_test_org=x_test_imputed_num.merge(x_test_imputed_col,left_index=True,right_index=True)

x_train_org.head()
from sklearn.preprocessing import OneHotEncoder

encoder=OneHotEncoder(handle_unknown='ignore',sparse=False)

oh_train_cols=pd.DataFrame(encoder.fit_transform(x_train_org[categorical_col1]))

oh_test_cols=pd.DataFrame(encoder.transform(x_test_org[categorical_col1]))

oh_train_cols.index=x_train_org.index

oh_test_cols.index=x_test_org.index

x_train_org1=x_train_org.drop(categorical_col1,axis=1)

x_test_org2=x_test_org.drop(categorical_col1,axis=1)

oh_x_train=pd.concat([x_train_org1,oh_train_cols],axis=1)

oh_x_test=pd.concat([x_test_org2,oh_test_cols],axis=1)

oh_x_train.head()
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

model=XGBRegressor(n_estimators=500,learning_rate=0.05)

model.fit(oh_x_train,y_train,early_stopping_rounds=5,eval_set=[(oh_x_test,y_test)],verbose=False)

pred=model.predict(oh_x_test)

mae=mean_absolute_error(y_test,pred)

mae
import matplotlib.pyplot as plt

plt.plot(range(1,31),y_test[:30],marker='o',linestyle='--',label='Actual',color='b')

plt.plot(range(1,31),pred[:30],marker='o',linestyle='--',label='Predicted',color='y')

plt.xlabel('House number')

plt.ylabel('House prices')

plt.legend()

plt.show()
numerical_col1=[col for col in test.columns if test[col].dtype in ['int64','float64']]

my_col1=numerical_col1+categorical_col1

test1=test[my_col1].copy()

test_num=pd.DataFrame(impute.fit_transform(test1[numerical_col1]))

test_num.columns=test1[numerical_col1].columns

test_num.index=test1.index

test_cat=pd.DataFrame(impute1.fit_transform(test1[categorical_col1]))

test_cat.columns=test1[categorical_col1].columns

test_cat.index=test1.index

test_cat_num=pd.DataFrame(encoder.fit_transform(test_cat))

test_cat_num.index=test1.index

test_data1=pd.concat([test_num,test_cat_num],axis=1)

test_data1.head()
pred2=model.predict(test_data1)

output=pd.DataFrame({'Id':test_data.index,'SalePrice':pred2})

output.to_csv('submission.csv',index=False)