# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = df.drop('customerID',axis=1)
df.head()
df.describe()
df.dtypes
df.TotalCharges = df.TotalCharges.apply(lambda x: x.strip())
df.TotalCharges[df.TotalCharges == '' ] = np.NAN
df.TotalCharges = df.TotalCharges.astype('float')

for col in df.columns:
    if df[col].dtypes == 'object':
        print(df.groupby(col)['Churn'].count())
df = df.dropna()
# Shufle
df = df.sample(frac=1)
df_x = df.drop('Churn',axis = 1) 
df['Churn'] = df['Churn'].astype('category')
df_y = df['Churn'].cat.codes
for col in df_x.columns:
    if df_x[col].dtypes == 'object':
        df_x = pd.concat([df_x,pd.get_dummies(df_x[col],prefix=col)],axis=1)
        df_x = df_x.drop(col,axis=1)
train_x,test_x,train_y,test_y = train_test_split(df_x,df_y,train_size = 0.2,random_state=1)
in_fold = []
out_fold = []
# Use 5 fold cross validation
kf = StratifiedKFold(5)
for train_index, val_index in kf.split(train_x,train_y):
    # split into train ad val index
    in_fold_x,in_fold_y = train_x.iloc[train_index],train_y.iloc[train_index]
    out_fold_x,out_fold_y = train_x.iloc[val_index], train_y.iloc[val_index]
    
    rf = RandomForestClassifier(n_estimators= 10,max_depth = 3)
    rf.fit(in_fold_x,in_fold_y)
    
    # caculate prediction_probability from this spilt
    in_fold_pred = rf.predict_proba(in_fold_x)
    out_fold_pred = rf.predict_proba(out_fold_x)
    
    # caculate log_loss
    in_fold_log_loss = log_loss(in_fold_y,in_fold_pred)
    out_fold_log_loss = log_loss(out_fold_y,out_fold_pred)
    
    # append to array
    in_fold.append(in_fold_log_loss)
    out_fold.append(out_fold_log_loss)
    #print("in fold log_loss {} out_fold log_loss {}".format(in_fold_log_loss,out_fold_log_loss))
    
# print out loss the infold loss should be lower than out fold loss if not check for overfitting
print("in fold average logloss {} with std {}".format(np.mean(in_fold),np.std(in_fold)))
print("out fold average logloss {} with std {}".format(np.mean(out_fold),np.std(out_fold)))    
rf = RandomForestClassifier(n_estimators= 10,max_depth = 3)
rf.fit(train_x,train_y)
train_prob = rf.predict_proba(train_x)
test_prob = rf.predict_proba(test_x)
train_log_loss = log_loss(train_y,train_prob)
test_log_loss = log_loss(test_y,test_prob)
print("train log_loss {} test log_loss {}".format(train_log_loss,test_log_loss))
    
pd.DataFrame(rf.feature_importances_,columns=['importance'],index=train_x.columns).sort_values(by=['importance'],ascending = False)
# Use something cooler https://github.com/slundberg/shap?
import shap

# load JS visualization code to notebook
shap.initjs()

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(train_x)

# visualize the first prediction's explanation
shap.force_plot(explainer.expected_value[1], shap_values[1], train_x)
shap.summary_plot(shap_values[1], train_x)


