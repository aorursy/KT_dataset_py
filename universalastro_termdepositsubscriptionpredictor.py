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
import plotly.graph_objects as go

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from category_encoders import HashingEncoder
from category_encoders import WOEEncoder
from category_encoders import BinaryEncoder

from sklearn import metrics
import itertools
import gc
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
!pip install pandas_profiling
!pip install category_encoders
train = pd.read_csv("/kaggle/input/term-deposit-prediction-data-set/train.csv")
test = pd.read_csv("/kaggle/input/term-deposit-prediction-data-set/test.csv")

train_back = train.copy()
test_back = test.copy()

numerical_features = [feature for feature in train.columns if train[feature].dtypes != 'O']
print("Numerical_features: ",numerical_features)
categorical_features = [feature for feature in train.columns if train[feature].dtypes == 'O']
print("Categorical_features: ",categorical_features)

train.info()
from pandas_profiling import ProfileReport
profile = ProfileReport(train)
profile.to_widgets()
# Hack - There are some zero or 01 values. So ading 1 or 2, before doing log transformation.
train['duration'] = train['duration'] + 1
train['previous'] = train['previous'] + 2
train['pdays'] = train['pdays'] + 2
train['balance'] = train['balance'] + 1
cols = ['age','pdays','previous','campaign','duration']

for feature in cols:
    print("\nMin/Max values of {} are {}, {}".format(feature, train[feature].min(), train[feature].max()))    
    if 0 in train[feature].unique():
        pass
    else:
        try:
            train[feature] = np.log(train[feature])           
        except:
            print("some error in train: ", feature)
    print("After log, transformation - Min/Max values of {} are {} - {}".format(feature, train[feature].min(), train[feature].max()))

print(" = " * 60)
test['duration'] = test['duration'] + 1
test['previous'] = test['previous'] + 2
test['pdays'] = test['pdays'] + 2

for feature in cols:
    print("\nMin/Max values of {} are {}, {}".format(feature, test[feature].min(), test[feature].max()))    
    if 0 in test[feature].unique():
        pass
    else:
        try:
            test[feature] = np.log(test[feature])
        except:
            print("some error in test: ", feature)
    print("After log transformation, Min/Max values of {} are {} - {}".format(feature, test[feature].min(), test[feature].max()))
x_data = ['Age', 'Pdays', 'Previous', 'Duration', 'Campaign']

N = 50

y0 = train['age']
y1 = train['pdays']
y2 = train['previous']
y3 = train['duration']
y4 = train['campaign']

y_data = [y0, y1, y2, y3, y4]

colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)']

fig = go.Figure()

for xd, yd, cls in zip(x_data, y_data, colors):
        fig.add_trace(go.Box(
            y=yd,
            name=xd,
            boxpoints='outliers', notched=True,
            jitter=0.5,
            whiskerwidth=0.4,
            fillcolor=cls,
            marker_size=2,
            line_width=1)
        )

fig.update_layout(
    title='Box plots of numerical columns',
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        dtick=5,
        gridcolor='rgb(255, 255, 255)',
        gridwidth=1,
        zerolinecolor='rgb(255, 255, 255)',
        zerolinewidth=2,
    ),
    margin=dict(l=40, r=30, b=80, t=100,),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
    showlegend=False
)
fig.show()
# Discover the number of categories within each categorical feature:
len(train.job.unique()),  len(train.poutcome.unique()),len(train.month.unique()),len(train.contact.unique()), len(train.marital.unique()), len(train.loan.unique()), len(train.education.unique()), len(train.housing.unique()),len(train.default.unique())
train_back
cat_features = [feature for feature in train_back.columns if ( train_back[feature].dtypes == 'O') ]
cat_features.append('day')
print("Removed columns - ", cat_features.pop(-2))

# We create a helper function to get the scores for each encoding method:
def get_score(model, X, y, X_val, y_val,X_test):
    model.fit(X, y)
    y_pred = model.predict_proba(X_val)[:,1]
    score = roc_auc_score(y_val, y_pred)
    y_pred = model.predict(X_test)
    return score,y_pred

target_feature =  'subscribed'

SEED = 123
logit = LogisticRegression(random_state=SEED)
rf = RandomForestClassifier(random_state=SEED)
lb_train = train_back.copy()
lb_test = test_back.copy()

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
for feature in cat_features:
  lb_train[feature]= label_encoder.fit_transform(lb_train[feature]) 
  lb_test[feature]= label_encoder.fit_transform(lb_test[feature]) 

lb_train[target_feature] = lb_train[target_feature].map({"yes":1, "no":0})
lb_y = lb_train[target_feature]
lb_train.drop([target_feature],axis= 1, inplace=True)

# feature scaling
scaler = StandardScaler()
lb_train = scaler.fit_transform(lb_train)
lb_test = scaler.transform(lb_test)

# Split dataset into train and validation subsets:
X_train, X_val, y_train, y_val = train_test_split(lb_train, lb_y, test_size=0.2, random_state = SEED)

baseline_logit_with_standard,y_pred_logit = get_score(logit, X_train, y_train, X_val, y_val, lb_test)
print('Logistic Regression score without feature engineering:', baseline_logit_with_standard)

baseline_rf_with_standard,y_pred_rf = get_score(rf, X_train, y_train, X_val, y_val, lb_test)
print('Random Forest score without feature engineering:', baseline_rf_with_standard)

del lb_train;
gc.collect() 
del lb_test;
gc.collect() 
lb_train = train_back.copy()
lb_test = test_back.copy()

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
for feature in cat_features:
  lb_train[feature]= label_encoder.fit_transform(lb_train[feature]) 
  lb_test[feature]= label_encoder.fit_transform(lb_test[feature]) 

lb_train[target_feature] = lb_train[target_feature].map({"yes":1, "no":0})
lb_y = lb_train[target_feature]
lb_train.drop([target_feature],axis= 1, inplace=True)

# feature scaling
scaler = MinMaxScaler()
lb_train = scaler.fit_transform(lb_train)
lb_test = scaler.transform(lb_test)

# Split dataset into train and validation subsets:
X_train, X_val, y_train, y_val = train_test_split(lb_train, lb_y, test_size=0.2, random_state = SEED)

baseline_logit_with_minmax, y_pred_logit_minmax = get_score(logit, X_train, y_train, X_val, y_val, lb_test)
print('Logistic Regression score without feature engineering:', baseline_logit_with_minmax)

baseline_rf_with_minmax,y_pred_rf_minmax = get_score(rf, X_train, y_train, X_val, y_val, lb_test)
print('Random Forest score without feature engineering:', baseline_rf_with_minmax)

del lb_train;
gc.collect() 
del lb_test;
gc.collect() 
ohe_train = train_back.copy()
ohe_test = test_back.copy()

one_hot_enc = OneHotEncoder(sparse=False)

ohe_train[target_feature] = ohe_train[target_feature].map({"yes":1, "no":0})
ohe_y = ohe_train[target_feature]
ohe_train.drop([target_feature],axis= 1, inplace=True)

print("Before Target Encoder - Shape of Train/Test: ", ohe_train.shape, ohe_test.shape)
ohe_train = (one_hot_enc.fit_transform(ohe_train[cat_features]))
ohe_test = (one_hot_enc.transform(ohe_test[cat_features]))
print("After One-Hot Encoder - Shape of Train/Test: ", ohe_train.shape, ohe_test.shape)

# feature scaling
scaler = StandardScaler()
ohe_train = scaler.fit_transform(ohe_train)
ohe_test = scaler.transform(ohe_test)

# Split dataset into train and validation subsets:
ohe_X_train, ohe_X_val, ohe_y_train, ohe_y_val = train_test_split(ohe_train, ohe_y, test_size=0.2, random_state = SEED)

ohe_logit_score, y_pred_logit_ohe = get_score(logit, ohe_X_train, ohe_y_train, ohe_X_val, ohe_y_val, ohe_test)
print('Logistic Regression score without feature engineering:', ohe_logit_score)

ohe_rf_score, y_pred_rf_ohe = get_score(rf, ohe_X_train, ohe_y_train, ohe_X_val, ohe_y_val, ohe_test)
print('Random Forest score without feature engineering:', ohe_rf_score)

del ohe_train;
gc.collect() 
del ohe_test;
gc.collect() 
hash_train = train_back.copy()
hash_test = test_back.copy()

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
for feature in cat_features:
  hash_train[feature]= label_encoder.fit_transform(hash_train[feature]) 
  hash_test[feature]= label_encoder.fit_transform(hash_test[feature]) 

hash_train[target_feature] = hash_train[target_feature].map({"yes":1, "no":0})
hash_y = hash_train[target_feature]
hash_train.drop([target_feature],axis= 1, inplace=True)

from category_encoders import TargetEncoder
columns = ['job', 'marital', 'education', 'default', \
       'housing', 'loan', 'contact', 'day', 'month', 'poutcome']

targ_enc = TargetEncoder(cols = columns, smoothing=8, min_samples_leaf=5).fit(hash_train, hash_y)

print("Before Target Encoder - Shape of Train/Test: ", hash_train.shape, hash_test.shape)
hash_train_te = targ_enc.transform(hash_train.reset_index(drop=True))
hash_test_te = targ_enc.transform(hash_test.reset_index(drop=True))
print("After Target Encoder - Shape of Train/Test: ", hash_train_te.shape, hash_test_te.shape)

# Split dataset into train and validation subsets:
X_train, X_val, y_train, y_val = train_test_split(hash_train_te, hash_y, test_size=0.2, random_state = SEED)

te_logit_score, y_pred_logit_te = get_score(logit, X_train, y_train, X_val, y_val, hash_test)
print('Logistic Regression score with target encoding:', te_logit_score)

te_rf_score, y_pred_rf_te = get_score(rf, X_train, y_train, X_val, y_val, hash_test)
print('Random Forest score with target encoding:', te_rf_score)

del hash_train;
gc.collect() 
del hash_test;
gc.collect() 
he_train = train_back.copy()
he_test = test_back.copy()

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
for feature in cat_features:
  he_train[feature]= label_encoder.fit_transform(he_train[feature]) 
  he_test[feature]= label_encoder.fit_transform(he_test[feature]) 

he_train[target_feature] = he_train[target_feature].map({"yes":1, "no":0})
he_y = he_train[target_feature]
he_train.drop([target_feature],axis= 1, inplace=True)

columns = ['job', 'marital', 'education', 'default', \
       'housing', 'loan', 'contact', 'day', 'month', 'poutcome']

targ_enc = HashingEncoder(cols = columns,  n_components=1000).fit(he_train, he_y)
print("Before Hashing Encoder - Shape of Train/Test: ", he_train.shape, he_test.shape)
he_train = targ_enc.transform(he_train.reset_index(drop=True))
he_test = targ_enc.transform(he_test.reset_index(drop=True))
print("After Hashing Encoder - Shape of Train/Test: ", he_train.shape, he_test.shape)

# Split dataset into train and validation subsets:
X_train_te, X_val_te, y_train_te, y_val_te = train_test_split(he_train, he_y, test_size=0.2, random_state = SEED)

he_logit_score, y_pred_logit_he = get_score(logit, X_train_te, y_train_te, X_val_te, y_val_te, he_test)
print('Logistic Regression score with Hashing encoding:', he_logit_score)

he_rf_score, y_pred_rf_he = get_score(rf, X_train_te, y_train_te, X_val_te, y_val_te, he_test)
print('Random Forest score with Hashing encoding:', he_rf_score)

del he_train;
gc.collect() 
del he_test;
gc.collect() 
woe_train = train_back.copy()
woe_test = test_back.copy()

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
for feature in cat_features:
  woe_train[feature]= label_encoder.fit_transform(woe_train[feature]) 
  woe_test[feature]= label_encoder.fit_transform(woe_test[feature]) 

woe_train[target_feature] = woe_train[target_feature].map({"yes":1, "no":0})
woe_y = woe_train[target_feature]
woe_train.drop([target_feature],axis= 1, inplace=True)

columns = ['job', 'marital', 'education', 'default', \
       'housing', 'loan', 'contact', 'day', 'month', 'poutcome']
woe_enc = WOEEncoder(cols=columns, random_state=17).fit(woe_train, woe_y)

print("Before WOE Encoder - Shape of Train/Test: ", woe_train.shape, woe_test.shape)
woe_train_wo = woe_enc.transform(woe_train.reset_index(drop=True))
woe_test_wo = woe_enc.transform(woe_test.reset_index(drop=True))
print("After WOE Encoder - Shape of Train/Test: ", woe_train_wo.shape, woe_test_wo.shape)

# Split dataset into train and validation subsets:
X_train_woe, X_val_woe, y_train_woe, y_val_woe = train_test_split(woe_train_wo, woe_y, test_size=0.2, random_state = SEED)

woe_logit_score, y_pred_logit_woe = get_score(logit, X_train_woe, y_train_woe, X_val_woe, y_val_woe, woe_test)
print('Logistic Regression score with Weight Of Evidence encoding:', woe_logit_score)

woe_rf_score, y_pred_rf_woe = get_score(rf, X_train_woe, y_train_woe, X_val_woe, y_val_woe, woe_test)
print('Random Forest score with Weight Of Evidence encoding:', woe_rf_score)

del woe_train;
gc.collect() 
del woe_test;
gc.collect() 
be_train = train_back.copy()
be_test = test_back.copy()

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
for feature in cat_features:
  be_train[feature]= label_encoder.fit_transform(be_train[feature]) 
  be_test[feature]= label_encoder.fit_transform(be_test[feature]) 

be_train[target_feature] = be_train[target_feature].map({"yes":1, "no":0})
be_y = be_train[target_feature]
be_train.drop([target_feature],axis= 1, inplace=True)

columns = ['job', 'marital', 'education', 'default', \
       'housing', 'loan', 'contact', 'day', 'month', 'poutcome']
be_enc = BinaryEncoder(cols=columns).fit(be_train, be_y)

print("Before Binary Encoder - Shape of Train/Test: ", be_train.shape, be_test.shape)
be_train = be_enc.transform(be_train.reset_index(drop=True))
be_test = be_enc.transform(be_test.reset_index(drop=True))
print("After Binary Encoder - Shape of Train/Test: ",be_train.shape, be_test.shape)

# Split dataset into train and validation subsets:
X_train_be, X_val_be, y_train_be, y_val_be = train_test_split(be_train, be_y, test_size=0.2, random_state = SEED)

be_logit_score, y_pred_logit_be = get_score(logit, X_train_be, y_train_be, X_val_be, y_val_be, be_test)
print('Logistic Regression score with Binary encoding:', be_logit_score)

be_rf_score, y_pred_rf_be = get_score(rf, X_train_be, y_train_be, X_val_be, y_val_be, be_test)
print('Random Forest score with Binary encoding:', be_rf_score)

del be_train;
gc.collect() 
del be_test;
gc.collect() 
from prettytable import PrettyTable

myTable = PrettyTable(["SNo.", "Encoder", "Logistic", "Random Forest", "No. of Cols added"]) 
  
# Add rows 
myTable.add_row(["1", "One Hot", round(ohe_logit_score,4), round(ohe_rf_score,4), 58 ]) 
myTable.add_row(["2", "Hashing", round(he_logit_score,4), round(he_rf_score,4), 0]) 
myTable.add_row(["3", "Target", round(te_logit_score,4), round(te_rf_score,4), 0 ]) 
myTable.add_row(["4", "Weight of Evaluation", round(woe_logit_score,4), round(woe_rf_score,4), 0]) 
myTable.add_row(["5", "Binary", round(be_logit_score,4), round(be_rf_score,4) , 24 ]) 
myTable.add_row(["6", "Label Encoder + Standard Scaler", round(baseline_logit_with_standard,4), round(baseline_rf_with_standard,4), 0 ]) 
myTable.add_row(["7", "Label Encoder + MinMax Scaler", round(baseline_logit_with_minmax,4), round(baseline_rf_with_minmax, 4), 0 ]) 

print(myTable)
models = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(), AdaBoostClassifier(), GradientBoostingClassifier(), KNeighborsClassifier(), SVC(), XGBClassifier()]
model_names = ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'KNeighborsClassifier', 'SVC', 'XGBClassifier']
accuracy_train = []
accuracy_val = []
for model in models:
    mod = model
    mod.fit(X_train, y_train)
    y_pred_train = mod.predict(X_train)
    y_pred_val = mod.predict(X_val)
    accuracy_train.append(accuracy_score(y_train, y_pred_train))
    accuracy_val.append(accuracy_score(y_val, y_pred_val))
data = {'Modelling Algorithm' : model_names, 'Train Accuracy' : accuracy_train, 'Validation Accuracy' : accuracy_val}
data = pd.DataFrame(data)
data['Difference'] = ((np.abs(data['Train Accuracy'] - data['Validation Accuracy'])) * 100)/(data['Train Accuracy'])
data.sort_values(by = 'Difference')
xgb = XGBClassifier()

parameters = {   'eta': [0.1], 'colsample_bytree':[0.7],
               'min_child_weight': [5], 'max_depth' :[7], 'max_features':[5],'subsample': [0.7],
               'reg_alpha':[1], 'n_estimators': [100] ,'seed':[11] }

xgb_clf = GridSearchCV(xgb, parameters, cv = 5, n_jobs = -1, verbose=1)
xgb_train = train_back.copy()
xgb_test = test_back.copy()

xgb_train['duration'] = xgb_train['duration'] + 1
xgb_train['previous'] = xgb_train['previous'] + 2
xgb_train['pdays'] = xgb_train['pdays'] + 2

cols = ['age','pdays','previous','campaign','duration']

for feature in cols:
    print("\nMin/Max values of {} are {}, {}".format(feature, xgb_train[feature].min(), xgb_train[feature].max()))    
    if 0 in xgb_train[feature].unique():
        pass
    else:
        try:
            xgb_train[feature] = np.log(xgb_train[feature])           
        except:
            print("some error in train: ", feature)
    print("After log, transformation - Min/Max values of {} are {} - {}".format(feature, xgb_train[feature].min(), xgb_train[feature].max()))

print(" = " * 60)

xgb_test['duration'] = xgb_test['duration'] + 1
xgb_test['previous'] = xgb_test['previous'] + 2
xgb_test['pdays'] = xgb_test['pdays'] + 2

for feature in cols:
    print("\nMin/Max values of {} are {}, {}".format(feature, xgb_test[feature].min(), xgb_test[feature].max()))    
    if 0 in xgb_test[feature].unique():
        pass
    else:
        try:
            xgb_test[feature] = np.log(xgb_test[feature])
        except:
            print("some error in test: ", feature)
    print("After log transformation, Min/Max values of {} are {} - {}".format(feature, xgb_test[feature].min(), xgb_test[feature].max()))

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
for feature in cat_features:
  xgb_train[feature]= label_encoder.fit_transform(xgb_train[feature]) 
  xgb_test[feature]= label_encoder.fit_transform(xgb_test[feature]) 

xgb_train[target_feature] = xgb_train[target_feature].map({"yes":1, "no":0})
xgb_y = xgb_train[target_feature]
xgb_train.drop([target_feature],axis= 1, inplace=True)

xgb_train.drop(['ID'],axis= 1, inplace=True)
xgb_test.drop(['ID'],axis= 1, inplace=True)

# feature scaling
scaler = StandardScaler()
xgb_train = scaler.fit_transform(xgb_train)
xgb_test = scaler.transform(xgb_test)

# Split dataset into train and validation subsets:
X_train, X_val, y_train, y_val = train_test_split(xgb_train, xgb_y, test_size=0.25, random_state = SEED)

xgb_clf.fit(X_train, y_train)
predictions = xgb_clf.predict_proba(X_val)[:,1]
score = roc_auc_score(y_val,predictions)
print("XGB score: ",score)

y_pred = xgb_clf.predict(X_val)

del xgb_train;
gc.collect() 
del xgb_test;
gc.collect() 
y_pred_test = pd.DataFrame(y_pred, columns = ['Prediction'])
print(y_pred_test.head())

y_pred_test.to_csv('Prediction.csv')
#Generate Confusion Matrix

conf_matrix = confusion_matrix(y_pred,y_val)
print(conf_matrix)

print(" = "*60)
print(classification_report(y_pred,y_val))