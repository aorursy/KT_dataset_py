import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
k_train= pd.read_csv("/kaggle/input/liberty-mutual-group-property-inspection-prediction/train.csv.zip")

k_test = pd.read_csv("/kaggle/input/liberty-mutual-group-property-inspection-prediction/test.csv.zip")
x=pd.concat([k_train,k_test],axis=0)
# hepls in finding out total number of unique variables in each columns of dataset



for col_name in x.columns :

  if x[col_name].dtypes == 'object':

    unique_cat = len(x[col_name].unique())

    print("feature'{col_name}' has {unique_cat} unique categories.".format(

        col_name=col_name, unique_cat=unique_cat))
#copy pasted form cat_features 

dummy_list=['T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11', 'T1_V12', 'T1_V15', 'T1_V16', 'T1_V17', 'T2_V3', 'T2_V5', 'T2_V11', 'T2_V12', 'T2_V13']
#function to create dummy variables in dataset



def dummy_df(df, dummy_list):

  for x in dummy_list:

    dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)

    df=df.drop(x, 1)

    df=pd.concat([df,dummies],axis =1)

  return df 



x= dummy_df(x,dummy_list)
#devideing data into test and train 

df_Train=x.iloc[:50999,:]

df_Test=x.iloc[50999:,:]
df_Test.drop(['Hazard'],axis=1,inplace=True)
x_train=df_Train.drop(['Hazard'],axis=1)

y_train=df_Train['Hazard']
from sklearn.ensemble import RandomForestRegressor

m=RandomForestRegressor(bootstrap = True,

 max_depth= 60,

 max_features= 'sqrt',

 min_samples_leaf= 2,

 min_samples_split= 10,

 n_estimators= 230,n_jobs=-1)
Random_forest= m.fit(x_train,y_train)
y_pred=m.predict(df_Test)

y_pred
pred=pd.DataFrame(y_pred)

sub_df=pd.read_csv('/kaggle/input/liberty-mutual-group-property-inspection-prediction/sample_submission.csv.zip')

datasets=pd.concat([sub_df['Id'],pred],axis=1)

datasets.columns=['Id','Hazard']

datasets.to_csv('sample_submission.csv',index=False)