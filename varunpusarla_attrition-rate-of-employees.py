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
df_train= pd.read_csv('/kaggle/input/Train.csv')
df_test=pd.read_csv('/kaggle/input/Test.csv')
df_train.columns
df_train.head()
df_train.info()
df_test.head()
df_test.info()
#Searching for missing values and then analysing
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
features_with_na= [features for features in df_train.columns if df_train[features].isnull().sum()>1]
features_with_na_test= [features for features in df_test.columns if df_test[features].isnull().sum()>1]

for feature in features_with_na:
    print( feature, np.round(df_train[feature].isnull().mean(),4),'% missing values')
features_with_num= [feature for feature in df_train.columns if df_train[feature].dtypes!='O']
print(features_with_num)
for feature in features_with_num:
    df_train.groupby(feature)['Attrition_rate'].median().plot()
    plt.xlabel(feature)
    plt.ylabel("Attrition_rate")
  
    plt.show()

discrete_feature=[feature for feature in features_with_num and df_train if len(df_train[feature].unique())<10 or df_train[feature].dtypes=='O']
discrete_feature.remove('Hometown')
discrete_feature.remove('Employee_ID')
print(discrete_feature)
for feature in discrete_feature:
    df_train.groupby(feature)['Attrition_rate'].mean().plot.bar()
    plt.xlabel(feature)
    plt.yscale('log')
    plt.ylabel('Attrition_rate')
    
    plt.show()
#Using log transformation

continous_feature=[feature for feature in features_with_num if feature not in discrete_feature]
print(continous_feature)
for feature in continous_feature:
    df_train[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.show()
df_train.info()
mean_value=df_train[features_with_na].mean()
df_train.fillna(mean_value, inplace=True)

mean_value_test=df_test[features_with_na_test].mean()
df_test.fillna(mean_value_test, inplace=True)
df_train.info()
object_features=[feature for feature in df_train if df_train[feature].dtypes=='O']
object_features.remove('Employee_ID')
print(object_features)

object_features_test=[feature for feature in df_test if df_test[feature].dtypes=='O']
object_features_test.remove('Employee_ID')
#Encoding categ features
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()

for f in object_features:
   df_train[f]=encoder.fit_transform(df_train[f])


for f in object_features_test:
    df_test[f]=encoder.fit_transform(df_test[f])

df_train.info()
df_test.info()
#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scale_feat= [f for f in df_train.columns if f not in ['Employee_ID','Attrition_rate']]
df_train[scale_feat]=scaler.fit_transform(df_train[scale_feat])

scale_feat_test=[f for f in df_test.columns if f not in ['Employee_ID','Attrition_rate']]
df_test[scale_feat_test]=scaler.fit_transform(df_test[scale_feat_test])
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

feat=['Gender', 'Age', 'Education_Level', 'Relationship_Status', 'Hometown', 'Unit', 'Decision_skill_possess','Time_of_service', 'Time_since_promotion', 'growth_rate', 'Travel_Rate','Post_Level', 'Pay_Scale', 'Compensation_and_Benefits','Work_Life_balance', 'VAR1', 'VAR2', 'VAR3', 'VAR4', 'VAR5', 'VAR6','VAR7']
X=df_train[feat]
y=df_train.Attrition_rate
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.33, random_state=42) 

from sklearn.model_selection import RandomizedSearchCV
import xgboost

regressor=xgboost.XGBRegressor()

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}
random_search=RandomizedSearchCV(regressor,param_distributions=params,n_iter=5,scoring='neg_root_mean_squared_error',n_jobs=-1,cv=5,verbose=3)
random_search.fit(X_train, y_train)
random_search.best_estimator_
random_search.best_params_
new_model= xgboost.XGBRegressor(min_child_weight= 3,
 max_depth= 3,
 learning_rate=0.2,
 gamma= 0.4,
 colsample_bytree= 0.7)
new_model.fit(X_train,y_train)
y_pred=new_model.predict(X_test)
rmse=mean_squared_error(y_test, y_pred)**(1/2)
print(rmse)
new_model.fit(X,y)
predictions=new_model.predict(df_test[feat])
output= pd.DataFrame({'Employee_ID': df_test.Employee_ID, 'Attrition_rate': predictions})
output.to_csv('MySubmission.csv', index=False)
print('Success')
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping

n_cols= X.shape[1]

n_model=Sequential()
n_model.add(Dense(200,activation='relu',input_shape=(n_cols,)))
n_model.add(Dense(190,activation='relu'))
n_model.add(Dense(180, activation='relu'))
n_model.add(Dense(1))

mycb=[EarlyStopping(patience=10)]


n_model.compile(optimizer='adam', loss='mean_squared_error')
n_model.fit(X.values,y.values,validation_split=0.3,callbacks=mycb, epochs=100)
y_pred_n=n_model.predict(X_test)
rmse_n=mean_squared_error(y_test, y_pred_n)**(1/2)
print(rmse_n)
