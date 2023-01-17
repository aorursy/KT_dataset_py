import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
file=pd.read_csv('../input/insurance-company-dataset/train.csv')

data=pd.DataFrame(file)

file=pd.read_csv('../input/insurance-company-dataset/test.csv')

test=pd.DataFrame(file)

data.head()
data.rename(columns={"Count_3-6_months_late":"3-6_late","Count_6-12_months_late":"6-12_late","Count_more_than_12_months_late":"more_than_12"},inplace=True)

test.rename(columns={"Count_3-6_months_late":"3-6_late","Count_6-12_months_late":"6-12_late","Count_more_than_12_months_late":"more_than_12"},inplace=True)

data.head()
data.info()
data.describe()
data.isnull().sum()
from sklearn.impute import SimpleImputer



imputer=SimpleImputer(strategy='mean')

data.iloc[:,[4,5,6,7]]=imputer.fit_transform(data.iloc[:,[4,5,6,7]])

test.iloc[:,[4,5,6,7]]=imputer.fit_transform(test.iloc[:,[4,5,6,7]])
data.isnull().sum()
categorical_cols=[col for col in data.columns if data[col].dtype==object]

categorical_cols
for col in categorical_cols:

    print("Unique values in {} are {}".format(col,data[col].nunique()))
from sklearn.preprocessing import OneHotEncoder



OH=OneHotEncoder(handle_unknown='ignore',sparse=False)

OH_data_train=pd.DataFrame(OH.fit_transform(data[categorical_cols]))

OH_data_test=pd.DataFrame(OH.fit_transform(test[categorical_cols]))

OH_data_train.index=data.index

OH_data_test.index=test.index

data=pd.concat([data,OH_data_train],axis=1)

test=pd.concat([test,OH_data_test],axis=1)

data.head()
data.drop(labels=categorical_cols,axis=1,inplace=True)

test.drop(labels=categorical_cols,axis=1,inplace=True)

data.head()
sns.scatterplot(x=data['Income'],y=data['id'])
data['Check_Outliers']=pd.cut(data['Income'],5)

data[['Income','Check_Outliers']].groupby('Check_Outliers',as_index=False).count()
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

scaler = scaler.fit(data[['Income']])

scaled_income_train = scaler.transform(data[['Income']])

scaled_income_test = scaler.transform(test[['Income']])

scaled_income_train
data.drop(labels='Check_Outliers',axis=1,inplace=True)

data['Scaled Income']=scaled_income_train

test['Scaled Income']=scaled_income_test
data['Check_Outliers']=pd.cut(data['Scaled Income'],5)

data[['Scaled Income','Check_Outliers']].groupby('Check_Outliers',as_index=False).count()
lower_bound=(0.1)

upper_bound=0.95



limit_to_bounds=data['Income'].quantile([lower_bound, upper_bound])

limit_to_bounds
considerable_data=(data['Income']<limit_to_bounds.loc[upper_bound])

considerable_data.value_counts()
final_data=data[considerable_data].copy()

final_data.head()
sns.scatterplot(x=final_data['Income'],y=data['id'])
data.drop(labels='Check_Outliers',axis=1,inplace=True)
final_data['Check_Outliers']=pd.cut(final_data['Income'],5)

final_data[['Check_Outliers','Income']].groupby('Check_Outliers',as_index=False).count()
encode_together=[test,final_data]



for dataset in encode_together:

    dataset.loc[dataset['Income']<=23603.99,'Income']=0

    dataset.loc[(dataset['Income']>23603.99) & (dataset['Income']<=109232.0),'Income']=1

    dataset.loc[(dataset['Income']>109232.0) & (dataset['Income']<=194434.0),'Income']=2

    dataset.loc[(dataset['Income']>194434.0) & (dataset['Income']<=279636.0),'Income']=3

    dataset.loc[(dataset['Income']>279636.0) & (dataset['Income']<=364838.0),'Income']=4

    dataset.loc[(dataset['Income']>364838.0) & (dataset['Income']<=450040.0),'Income']=5

    dataset.loc[dataset['Income']>450040.0,'Income']=6

    

final_data.head()
non_considerable_data=~considerable_data

final_data.loc[non_considerable_data,'Income']=5

final_data.drop(labels=['Scaled Income','Check_Outliers'],axis=1,inplace=True)

final_data.head(10)
final_data['age_in_days']=final_data['age_in_days']/365

final_data['age_in_days']=final_data['age_in_days'].astype('int64')

test['age_in_days']=test['age_in_days']/365

test['age_in_days']=test['age_in_days'].astype('int64')

final_data.head()
age_above_100=[age for age in data['age_in_days'].values if age>99]

age_above_100
final_data['age_in_days'].describe()
sns.scatterplot(x=final_data['age_in_days'],y=final_data['id'])
final_data.rename(columns={'age_in_days':'age'},inplace=True)

test.rename(columns={'age_in_days':'age'},inplace=True)

final_data.columns
final_data['Check_Outliers']=pd.cut(final_data['age'],5)

final_data[['Check_Outliers','age']].groupby('Check_Outliers',as_index=False).count()
final_data.drop(labels='Check_Outliers',axis=1,inplace=True)



final_data['age']=final_data['age'].astype('int64')    

test['age']=test['age'].astype('int64')



scaler = scaler.fit(final_data[['age']])

scaled_age_train = scaler.transform(final_data[['age']])

scaled_age_test = scaler.transform(test[['age']])

scaled_age_train
final_data['Scaled Age']=scaled_age_train

test['Scaled Age']=scaled_age_test

final_data['Check_Outliers']=pd.cut(final_data['Scaled Age'],5)

final_data[['Check_Outliers','Scaled Age']].groupby('Check_Outliers',as_index=False).count()
lower_bound=0.1

upper_bound=0.95



limit_to_bounds=final_data['Scaled Age'].quantile([lower_bound, upper_bound])

limit_to_bounds
considerable_data=(final_data['Scaled Age']<limit_to_bounds.loc[upper_bound])

considerable_data.value_counts()
final_data2=final_data[considerable_data].copy()

final_data2.head()
final_data2['Scaled Age']=final_data['Scaled Age']

final_data2.drop(labels=['Check_Outliers'],axis=1,inplace=True)

test.drop(labels='id',axis=1,inplace=True)
final_data2.columns
sns.scatterplot(x=final_data2['Scaled Age'],y=final_data2['id'])
test.drop(labels=['Scaled Income','Scaled Age'],axis=1,inplace=True)

final_data2.drop(labels='Scaled Age',axis=1,inplace=True)

print(test.columns)

final_data2.columns
from sklearn.model_selection import train_test_split



y=final_data2.target

final_data2.drop(labels=['target','id','premium'],axis=1,inplace=True)

X=final_data2



X_train,X_valid,y_train,y_valid=train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)
from imblearn.over_sampling import SMOTE

print('Number of positive and negative reviews:\n',y_train.value_counts())

sm = SMOTE(random_state=0)

X_train_res,y_train_res = sm.fit_sample(X_train,y_train)

print('Shape after oversampling\n',X_train_res.shape) 

print('Equal 1s and 0s \n', np.bincount(y_train_res))
model1=XGBClassifier(colsample_bytree= 0.7,

 learning_rate= 0.01,

 max_depth= 6,

 min_child_weight= 11,

 missing= -999,

 n_estimators= 5000,

 nthread= 4,

 objective= 'binary:logistic',

 subsample= 0.8)

model1.fit(X_train,y_train)



preds=model1.predict(X_valid)



from sklearn.metrics import roc_curve



fpr1, tpr1, thresh1 = roc_curve(y_valid, preds, pos_label=1)

fpr2, tpr2, thresh2 = roc_curve(y_valid, preds, pos_label=1)

 

random_probs = [0 for i in range(len(y_valid))]

p_fpr, p_tpr, _ = roc_curve(y_valid, random_probs, pos_label=1)
from sklearn.metrics import roc_auc_score,mean_absolute_error



MAE=mean_absolute_error(y_valid,preds)

auc_score = roc_auc_score(y_valid, preds)

print(auc_score)

MAE
predictions=model1.predict(test)

predictions
flag1=0



for val in predictions:

    if val==1:

        flag1+=1

        

flag0=len(predictions)-flag1



print(flag0)

flag1