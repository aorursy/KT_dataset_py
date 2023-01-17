# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd 
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold,StratifiedKFold,RepeatedStratifiedKFold
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')
# import eli5
# from eli5.sklearn import PermutationImportance
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
train_df      = pd.read_csv('../input/av-healthcare2/train.csv')
test_df       = pd.read_csv('../input/av-healthcare2/test.csv')
submission_df = pd.read_csv('../input/av-healthcare2/sample_submission.csv')
#Creating Addition Features
combine_set=pd.concat([train_df,test_df],axis=0)
combine_set['City_Code_Patient'].fillna(-99,inplace=True)
combine_set['Bed Grade'].fillna(5,inplace=True)
combine_set['Unique_Hospital_per_patient']=combine_set.groupby(['patientid'])['Hospital_code'].transform('nunique')
combine_set['Unique_patient_per_hospital']=combine_set.groupby(['Hospital_code'])['patientid'].transform('nunique')
combine_set['Unique_patient_per_Department']=combine_set.groupby(['Department'])['patientid'].transform('nunique')
combine_set['Total_deposit_paid_by_patient_in_each_hospital']=combine_set.groupby(['Hospital_code','patientid'])['Admission_Deposit'].transform('sum')
combine_set['Min_Severity_of_Illness'] = combine_set.groupby('patientid')['Severity of Illness'].transform('min')
# In[4]:

le = LabelEncoder()
#Encoding categorical variables by frequency encoding and label encoding
for col in combine_set.select_dtypes(include='object').columns:
    if col not in ['Age','Stay']:
        fe=combine_set.groupby([col]).size()/len(combine_set)
        combine_set[col]=combine_set[col].apply(lambda x: fe[x])   
        # combine_set[col]  = pd.get_dummies(combine_set[col].astype(str))         
    elif col!='Stay':
        combine_set[col]=le.fit_transform(combine_set[col].astype(str))
    else:
        pass
# In[5]:

#Splitting train and test

X=combine_set[combine_set['Stay'].isnull()==False].drop(['case_id','Stay'],axis=1)
y=le.fit_transform(combine_set[combine_set['Stay'].isnull()==False]['Stay'])
y=pd.DataFrame(y,columns=['Stay'])
X_main_test=combine_set[combine_set['Stay'].isnull()==True].drop(['case_id','Stay'],axis=1)
# In[6]:

# kf=KFold(n_splits=10,shuffle=True,random_state=294)
kf=KFold(n_splits=5,shuffle=True)

preds_1   = {}
y_pred_1  = []
acc_score = []

for i,(train_idx,val_idx) in enumerate(kf.split(X)):    
    
    X_train, y_train = X.iloc[train_idx,:], y.iloc[train_idx]
    
    X_val, y_val = X.iloc[val_idx, :], y.iloc[val_idx]
   
    print('\nFold: {}\n'.format(i+1))

    lg=LGBMClassifier(device="gpu", 
                      gpu_platform_id= 0,
                      max_bin=63,#Theoretically best speeds using LGBM
                      gpu_device_id= 0,
                      boosting_type='gbdt',
                      learning_rate=0.04,
                      # max_depth=15,
                      # num_leaves = 150,
                      objective='multi_class',
                      num_class=11,                      
                      n_estimators=50000,
                      metric='multi_error',
                      colsample_bytree=0.8,
                      min_child_samples=228,
                      reg_alpha=1,
                      reg_lambda=1,
                      # random_state=294,
                      n_jobs=-1,
                     
                      ) 
    
    # lg.fit(X_train,y_train)
    lg.fit(X_train, y_train
#                         ,categorical_feature = categorical_features
                        ,eval_metric='multi_error'
                        ,eval_set=[(X_train, y_train),(X_val, y_val)]
                        ,early_stopping_rounds=100
                        ,verbose=50
                       )
    
    print(accuracy_score(y_val,lg.predict(X_val))*100)
    
    acc = accuracy_score(y_val,lg.predict(X_val))*100
    acc_score.append(acc)
    print("Score : ",acc)    
    y_pred_1.append(lg.predict_proba(X_main_test))
    
    # preds_1[i+1]=lg.predict_proba(X_main_test)
    # y_pred_1.append(lg.predict_proba(X_main_test))

y_pred_final_1          = np.mean(np.array(y_pred_1),axis=0)
    
print('mean accuracy score: {}'.format((sum(acc_score)/10)))

preds_1=np.argmax(y_pred_final_1,axis=1)

print(preds_1.shape)
submission_df['Stay']=le.inverse_transform(preds_1.astype(int))
# submission_df[0] = y_pred_final_1[:,0]
# submission_df[1] =y_pred_final_1[:,1]

# Download Submission File :
display("submission_df",submission_df)
sub_file_name_1 = "BEST_11_CV=42.96_LB=WAIT_LGBM-1.csv"

submission_df.to_csv(sub_file_name_1,index=False)
submission_df.head(5)
from catboost                         import CatBoostClassifier
import timeit

catboost = CatBoostClassifier(eval_metric='Accuracy', max_depth=4, task_type="GPU", devices="0:1", n_estimators=1000, verbose=500)
catboost.fit( X, y, verbose=10 )   
y_pred_3  = []
y_pred_3.append(catboost.predict_proba(X_main_test))
y_pred_final_3          = np.mean(np.array(y_pred_3),axis=0)
# gpu_time = timeit.timeit('train_on_gpu()', setup="from __main__ import train_on_gpu", number=1)
# print('Time to fit and predict model on GPU: {} sec'.format(int(gpu_time)))

preds_3=np.argmax(y_pred_final_3,axis=1)

submission_df['Stay']=le.inverse_transform(preds_3.astype(int))
# submission_df[0] = y_pred_final_1[:,0]
# submission_df[1] =y_pred_final_1[:,1]

# Download Submission File :
display("submission_df",submission_df)
sub_file_name_3 = "BEST_11_CV=42.96_LB=WAIT_LGBM-1.csv"

submission_df.to_csv(sub_file_name_3,index=False)
submission_df.head(5)
# Ensemble of LGBM + CatBoost :

preds = (y_pred_final_1*0.2 + y_pred_final_3*0.8) /2
preds=np.argmax(preds,axis=1)
print(preds)

# In[9]:
# Download Submission File :
submission_df['Stay']=le.inverse_transform(preds.astype(int))
display("submission_df",submission_df)
sub_file_name = "ENSEMBLE_1_CV=42.22_42.17_LB=WAIT_LGB-1_0.2_LBG-2_0.8.csv"

submission_df.to_csv(sub_file_name,index=False)
submission_df.head(5)