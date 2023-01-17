# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Set your own project id here
PROJECT_ID = 'your-google-cloud-project'
from google.cloud import bigquery
bigquery_client = bigquery.Client(project=PROJECT_ID)
from google.cloud import storage
storage_client = storage.Client(project=PROJECT_ID)
from google.cloud import automl_v1beta1 as automl
automl_client = automl.AutoMlClient()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lightgbm as lgb
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

# Importing the dataset
df = pd.read_csv("../input/creditcard.csv")
df.head()
df.describe()
print(round(df['Class'].value_counts()[0]/len(df) * 100,2))
print(round(df['Class'].value_counts()[1]/len(df) * 100,2))

# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)
from sklearn.preprocessing import StandardScaler, RobustScaler

# RobustScaler is less prone to outliers.

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))
df.drop(['Time','Amount'], axis=1, inplace=True)

scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)

df2=df.copy()
# The dataset being divided into 2 parts
X = df.drop('Class', axis=1)
y = df.iloc[:, 30].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import roc_auc_score
print('Logistic Regression: ', roc_auc_score(y_test, y_pred))
lr_sc=roc_auc_score(y_test, y_pred)
df2=df.copy()
# For loop for most insignificant column
x=list(range(2,30))
flag=0
for i in x:
    df2.drop(df2.columns[i], axis = 1, inplace = True)
    X = df2.drop('Class', axis=1)
    y = df2.iloc[:, 29].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)
    y_pred_loop = classifier.predict(X_test)
    df2=df.copy()
    if (roc_auc_score(y_test, y_pred_loop))>lr_sc :
        lr_sc=roc_auc_score(y_test, y_pred_loop)
        flag=i
print(flag)
#Dropping flag=14 column
df2.drop(df2.columns[14], axis = 1, inplace = True)

X = df2.drop('Class', axis=1)
y = df2.iloc[:, 29].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred_drop = classifier.predict(X_test)
print(roc_auc_score(y_test, y_pred_drop))
#Drop another column
df3=df2.copy()
# For loop to find most insignificant column
x=list(range(2,29))
flag=0
for i in x:
    df3.drop(df3.columns[i], axis = 1, inplace = True)
    X = df3.drop('Class', axis=1)
    y = df3.iloc[:, 28].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)
    y_pred_loop = classifier.predict(X_test)
    df3=df2.copy()
    if (roc_auc_score(y_test, y_pred_loop))>lr_sc :
        lr_sc=roc_auc_score(y_test, y_pred_loop)
        flag=i

print(flag)
#Dropping flag=15 column
df3.drop(df3.columns[15], axis = 1, inplace = True)

X = df3.drop('Class', axis=1)
y = df3.iloc[:, 28].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred_drop2 = classifier.predict(X_test)
print(roc_auc_score(y_test, y_pred_drop2))
#Drop another variable
df4=df3.copy()
# For loop for most insignificant var
x=list(range(2,28))
flag=0
for i in x:
    df4.drop(df4.columns[i], axis = 1, inplace = True)
    X = df4.drop('Class', axis=1)
    y = df4.iloc[:, 27].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)
    y_pred_loop = classifier.predict(X_test)
    df4=df3.copy()
    if (roc_auc_score(y_test, y_pred_loop))>lr_sc :
        lr_sc=roc_auc_score(y_test, y_pred_loop)
        flag=i

print(flag)
# No more need for reducing columns
X_test
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train,y_train)
y_pred_RF=regressor.predict(X_test)
y_pred_RF=y_pred_RF>0.5
print(roc_auc_score(y_test, y_pred_RF))
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_RF = confusion_matrix(y_test, y_pred_RF)
print(cm_RF)
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier=Sequential()
classifier.add(Dense(output_dim=15, init='uniform', activation='relu', input_dim=27))
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train, y_train, validation_split=0.2, batch_size=25, epochs=50, shuffle=True, verbose=2)
y_pred_ANN=classifier.predict(X_test)
y_pred_ANN=y_pred_ANN>0.5
print(roc_auc_score(y_test, y_pred_ANN))
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_ANN = confusion_matrix(y_test, y_pred_ANN)
print(cm_ANN)
#XGB
gg = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=2,
              min_child_weight=1, missing=None, n_estimators=70, n_jobs=-1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)
gg.fit(X_train, y_train)
y_pred_dropxg = gg.predict_proba(X_test)[:, 1]
len(y_pred_dropxg)
y_pred_dropxgb=np.zeros(71202, dtype = int) 
for i in range(len(y_pred_dropxg)):
        if y_pred_dropxg[i] > 0.75:
            y_pred_dropxgb[i]=1
print(roc_auc_score(y_test, y_pred_dropxgb))

accuracy_score(y_test, y_pred_dropxgb)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_xgb = confusion_matrix(y_test, y_pred_dropxgb)
cm_xgb
df2 = sklearn.utils.shuffle(df2)
fraud_df = df2.loc[df2['Class'] == 1][:369] 
non_fraud_df = df2.loc[df2['Class'] == 0][:369] 
undersampled_df_train = pd.concat([fraud_df, non_fraud_df])
undersampled_df_train = sklearn.utils.shuffle(undersampled_df_train)
X_t=undersampled_df_train.drop('Class', axis=1)
y_t = undersampled_df_train.iloc[:, 28].values
fraud_df_tt = df2.loc[df2['Class'] == 1][369:492] 
non_fraud_df_tt = df2.loc[df2['Class'] == 0][369:] #ratio in test case same 
undersampled_df_test = pd.concat([fraud_df_tt, non_fraud_df_tt])
undersampled_df_test = sklearn.utils.shuffle(undersampled_df_test)
X_tt=undersampled_df_test.drop('Class', axis=1)
y_tt = undersampled_df_test.iloc[:, 28].values
#XGB
gg = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=2,
              min_child_weight=1, missing=None, n_estimators=70, n_jobs=-1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)
gg.fit(X_t, y_t)
y_pred_undr = gg.predict_proba(X_tt)[:, 1]

y_pred_dropxgb=np.zeros(284069, dtype = int) 
for i in range(len(y_pred_undr)):
        if y_pred_undr[i] > 0.95:
            y_pred_dropxgb[i]=1

print(roc_auc_score(y_tt, y_pred_dropxgb))
accuracy_score(y_tt, y_pred_dropxgb)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_xgb = confusion_matrix(y_tt, y_pred_dropxgb)
for i in range(769) :
    non_fraud_df = df2.loc[df2['Class'] == 0][369*i:738*i] 
    undersampled_df_train = pd.concat([fraud_df, non_fraud_df])
    undersampled_df_train = sklearn.utils.shuffle(undersampled_df_train)
    X_t=undersampled_df_train.drop('Class', axis=1)
    y_t=undersampled_df_train.iloc[:, 28].values
        
    #XGB
    gg = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=2,
              min_child_weight=1, missing=None, n_estimators=70, n_jobs=-1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)
    gg.fit(X_t, y_t)
    y_pred_undr = gg.predict_proba(X_tt)[:, 1]
    
    for j in range(len(y_pred_undr)) :
        if (y_pred_undr[j])>0.95 :
            y_pred_dropxgb[j]=1
        
        if (y_pred_undr[j])<0.75 :
            y_pred_dropxgb[j]=0
print(roc_auc_score(y_tt, y_pred_dropxgb))
print(accuracy_score(y_tt, y_pred_dropxgb))
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_xgb = confusion_matrix(y_tt, y_pred_dropxgb)
print(cm_xgb)
outcome = pd.DataFrame(y_pred_dropxgb)
outcome.to_csv('y_undersample.csv', index=False)