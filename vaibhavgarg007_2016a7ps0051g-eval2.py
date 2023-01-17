import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import math
import xgboost as xgb
import lightgbm as lgb
import random
from sklearn.metrics import accuracy_score
%matplotlib inline
df_test = pd.read_csv('test.csv')
df_train = pd.read_csv('train.csv')
y_train = df_train['class']
df_train = df_train.drop(columns = ['class','id'])
df_test_id = df_test['id']
df_test = df_test.drop(columns=['id'])
df_train_3 = pd.DataFrame()
for i in df_train.columns.values:
    for j in df_train.columns.values:
        if i > j:
            df_train_3['{}_{}'.format(i,j)] = df_train[i]*df_train[j]       
df_test_3 = pd.DataFrame()
for i in df_test.columns.values:
    for j in df_test.columns.values:
        if i > j:
            df_test_3['{}_{}'.format(i,j)] = df_test[i]*df_test[j]            
from sklearn.preprocessing import PowerTransformer
scaler = PowerTransformer()
df_train = pd.DataFrame(scaler.fit_transform(df_train.values), index=df_train.index, columns=df_train.columns)
df_test = pd.DataFrame(scaler.transform(df_test.values), index=df_test.index, columns=df_test.columns)
from sklearn.model_selection import train_test_split
x_train_val,x_test_val,y_train_val,y_test_val = train_test_split(df_train_3,y_train,test_size=0.3,random_state=27)
from sklearn.ensemble import RandomForestClassifier
rf_classifier =  RandomForestClassifier(n_estimators=200,max_depth=10,random_state=27)
rf_classifier.fit(x_train_val, y_train_val)
x=0
n_estimators_rf = 100
max_depth_rf = 1
random_state_rf = 30
min_samples_leaf_rf = 1
min_samples_split_rf = 1
for i in range(100,10000,100):
    rf_classifier = RandomForestClassifier(n_estimators=i).fit(x_train_val,y_train_val)
    if accuracy_score(y_test_val,rf_classifier.predict(x_test_val)) > x:
        x = accuracy_score(y_test_val,rf_classifier.predict(x_test_val))
        n_estimators_rf = i
for i in range(1,100,1):
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators_rf,max_depth=i).fit(x_train_val,y_train_val)
    if accuracy_score(y_test_val,rf_classifier.predict(x_test_val)) > x:
        x = accuracy_score(y_test_val,rf_classifier.predict(x_test_val))
        max_depth_rf = i
for i in range(0,10000,1):
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators_rf,max_depth=max_depth_rf,random_state=i).fit(x_train_val,y_train_val)
    if accuracy_score(y_test_val,rf_classifier.predict(x_test_val)) > x:
        x = accuracy_score(y_test_val,rf_classifier.predict(x_test_val))
        random_state_rf = i
        
rf_classifier = RandomForestClassifier(n_estimators=n_estimators_rf,max_depth=max_depth_rf,random_state=random_state_rf).fit(df_train_3,y_train)
df_test['class'] = rf_classifier.predict(df_test_3)
from xgboost import XGBClassifier

xgb_clf = XGBClassifier().fit(x_train_val,y_train_val)
y_pred_xgb = xgb_clf.predict(x_test_val)
xgb_acc = accuracy_score(y_test_val,y_pred_xgb)
n_estimators_xgb = 0
max_depth_xgb = 0
min_child_weight_xgb = 1
gamma_xgb = 0.0
subsample_xgb = 0.5
colsample_bytree_xgb = 0.5
x=0

for i in range(100,10000,100):
    xgb_clf = XGBClassifier(n_estimators=i).fit(x_train_val,y_train_val)
    if accuracy_score(y_test_val,xgb_clf.predict(x_test_val)) > x:
        x = accuracy_score(y_test_val,xgb_clf.predict(x_test_val))
        n_estimators_xgb = i

for i in range(0,5,1):
    for j in range(0,5,1):
        xgb_clf = XGBClassifier(n_estimators=n_estimators_xgb,max_depth=i,min_child_weight=j).fit(x_train_val,y_train_val)
        if accuracy_score(y_test_val,xgb_clf.predict(x_test_val)) > x:
            x = accuracy_score(y_test_val,xgb_clf.predict(x_test_val))
            max_depth_xgb = i
            min_child_weight_xgb = j
            
for i in range(5,10,1):
    xgb_clf = XGBClassifier(n_estimators=n_estimators_xgb,max_depth=max_depth_xgb,min_child_weight=min_child_weight_xgb,gamma=i/10).fit(x_train_val,y_train_val)
    if accuracy_score(y_test_val,xgb_clf.predict(x_test_val)) > x:
        x = accuracy_score(y_test_val,xgb_clf.predict(x_test_val))
        gamma_xgb = i/10
            
xgb_clf = XGBClassifier(n_estimators=n_estimators_xgb,max_depth=max_depth_xgb,min_child_weight=min_child_weight_xgb,gamma=gamma_xgb).fit(df_train_3,y_train)
df_test['class1'] = xgb_clf.predict(df_test_3)
chnge = random.sample(range(1, len(df_test['class1'])), 3)
df_test['class'].iloc[chnge] = df_test['class1'].iloc[chnge]
from sklearn.ensemble import ExtraTreesClassifier
xtra_classifier =  ExtraTreesClassifier(n_estimators=200,max_depth=10,random_state=27)
xtra_classifier.fit(x_train_val, y_train_val)
x=0
n_estimators_xtra = 100
max_depth_xtra = 1
random_state_xtra = 30
min_samples_leaf_xtra = 1
min_samples_split_xtra = 1
for i in range(100,10000,100):
    xtra_classifier = ExtraTreesClassifier(n_estimators=i).fit(x_train_val,y_train_val)
    if accuracy_score(y_test_val,xtra_classifier.predict(x_test_val)) > x:
        x = accuracy_score(y_test_val,xtra_classifier.predict(x_test_val))
        n_estimators_xtra = i
for i in range(1,200,1):
    xtra_classifier = ExtraTreesClassifier(n_estimators=n_estimators_xtra,max_depth=i).fit(x_train_val,y_train_val)
    if accuracy_score(y_test_val,xtra_classifier.predict(x_test_val)) > x:
        x = accuracy_score(y_test_val,xtra_classifier.predict(x_test_val))
        max_depth_xtra = i
for i in range(0,10000,1):
    xtra_classifier = ExtraTreesClassifier(n_estimators=n_estimators_xtra,max_depth=max_depth_xtra,random_state=i).fit(x_train_val,y_train_val)
    if accuracy_score(y_test_val,xtra_classifier.predict(x_test_val)) > x:
        x = accuracy_score(y_test_val,xtra_classifier.predict(x_test_val))
        random_state_xtra = i
        
xtra_classifier = ExtraTreesClassifier(n_estimators=n_estimators_xtra,max_depth=max_depth_xtra,random_state=random_state_xtra).fit(df_train_3,y_train)
df_test['class2'] = xtra_classifier.predict(df_test_3)
chnge = random.sample(range(1, len(df_test['class2'])), 3)
df_test['class'].iloc[chnge] = df_test['class2'].iloc[chnge]
from lightgbm import LGBMClassifier
classifier = LGBMClassifier().fit(df_train_3,y_train)
x=0
n_estimators_lgb = 100
max_depth_lgb = 1
random_state_lgb = 1
for i in range(0,10000,1):
    lgb_classifier = LGBMClassifier(random_state_lgb=i).fit(x_train_val,y_train_val)
    if accuracy_score(y_test_val,lgb_classifier.predict(x_test_val)) > x:
        x = accuracy_score(y_test_val,lgb_classifier.predict(x_test_val))
        random_state_lgb = i
for i in range(100,10000,100):
    lgb_classifier = LGBMClassifier(random_state=random_state_lgb,n_estimators=i).fit(x_train_val,y_train_val)
    if accuracy_score(y_test_val,lgb_classifier.predict(x_test_val)) > x:
        x = accuracy_score(y_test_val,xtra_classifier.predict(x_test_val))
        n_estimators_lgb = i
for i in range(1,200,1):
    lgb_classifier = LGBMClassifier(random_state=random_state_lgb,n_estimators=n_estimators_lgb,max_depth=i).fit(x_train_val,y_train_val)
    if accuracy_score(y_test_val,lgb_classifier.predict(x_test_val)) > x:
        x = accuracy_score(y_test_val,lgb_classifier.predict(x_test_val))
        max_depth_lgb = i

lgb_classifier = LGBMClassifier(random_state=random_state_lgb,n_estimators=n_estimators_lgb,max_depth=max_depth_lgb).fit(x_train_val,y_train_val)       
df_test['class3'] = lgb_classifier.predict(df_test_3)
chnge = random.sample(range(1, len(df_test['class3'])), 3)
df_test['class'].iloc[chnge] = df_test['class3'].iloc[chnge]
out = pd.concat([df_test_id,df_test['class']],axis=1)
out.columns = ['id','class']
out.to_csv('submission.csv', index=False)
from sklearn.ensemble import RandomForestClassifier
xgb_clf = RandomForestClassifier(random_state=1234,min_samples_leaf=1,min_samples_split=2).fit(df_train_3,y_train)
df_test['class'] = xgb_clf.predict(df_test_3)
out = pd.concat([df_test_id,df_test['class']],axis=1)
out.columns = ['id','class']
out.to_csv('submission.csv', index=False)