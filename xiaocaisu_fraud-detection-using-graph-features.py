# Necessary imports



## Data loading, processing and for more

import pandas as pd

import numpy as np

from imblearn.over_sampling import SMOTE



## Visualization

import seaborn as sns

import matplotlib.pyplot as plt

# set seaborn style because it prettier

sns.set()



## Metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import roc_curve, auc



## Models

import xgboost as xgb

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.preprocessing import StandardScaler
# Find out the paths of the datasets

import os

for dirname, _, filenames in os.walk('../input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# read the data and show first 5 rows

data = pd.read_csv("../input/banksim1/bs140513_032310.csv")

data.head(5)
# dropping zipcodeori and zipMerchant since they have only one unique value

data_reduced = data.drop(['zipcodeOri','zipMerchant'],axis=1)
# turning object columns type to categorical for easing the transformation process

col_categorical = data_reduced.select_dtypes(include= ['object']).columns

for col in col_categorical:

    data_reduced[col] = data_reduced[col].astype('category')

# categorical values ==> numeric values

data_reduced[col_categorical] = data_reduced[col_categorical].apply(lambda x: x.cat.codes)



# add transaction_id

data_reduced['transaction_id'] = np.arange(len(data_reduced))+1



data_reduced.head(5)
X = data_reduced.drop(['fraud'],axis=1)

y = data['fraud']

print(X.head(),"\n")

print(y.head())
# Data is highly imbalanced. Use resampling technique.

sm = SMOTE(random_state=42)

X_res, y_res = sm.fit_resample(X, y)

y_res = pd.DataFrame(y_res)

print(y_res[0].value_counts())
# %% Random Forest Classifier

# only training set



rf_clf = RandomForestClassifier(n_estimators=100,max_depth=8,random_state=42,

                                verbose=1,class_weight="balanced")



rf_clf.fit(X_res,y_res)

y_pred = rf_clf.predict(X_res)



print("Classification Report for Random Forest Classifier: \n", classification_report(y_res, y_pred))

print("Confusion Matrix of Random Forest Classifier: \n", confusion_matrix(y_res,y_pred))
# See importance of the features

X_res = pd.DataFrame(StandardScaler().fit_transform(X_res), columns=X.columns)

feature_importances = pd.DataFrame(rf_clf.feature_importances_,

                                   index = X_res.columns,

                                   columns=['importance']).sort_values('importance',                                                                 

                                   ascending=False)

print(feature_importances)
# read the data and show first 5 rows

graph_data = pd.read_csv("../input/trans-9-30/Transaction.csv")

graph_data.head(5)
# drop columns already in data_reduce df

graph_data = graph_data.drop(['fraud','category', 'amount', 'c_age_group', 'c_gender', 'time_step', 

                              'Unnamed: 20'],axis=1)



# join with original data

graph_data = graph_data.set_index('transaction_id').join(data_reduced.set_index('transaction_id'))



graph_data = graph_data.sort_values(by=['transaction_id'])

# drop not significant features

graph_data = graph_data.drop(['c_tot_diff_merchants', 'is_top_merchants_amt', 'is_top_categories_amt', 'merchant', 'gender',

                              'is_top_merchants_cnt', 'step', 'customer', 'age', 'c_avg_num_of_transactions_per_day',

                              'c_num_of_fraud', 'fraud_cnt_3hop', 'c_avg_num_of_diff_merchants_per_day'],axis=1)

graph_data.head(5)
graph_X = graph_data.drop(['fraud'],axis=1)

graph_y = graph_data['fraud']

print(graph_X.head(),"\n")

print(graph_y.head())
sm = SMOTE(random_state=42)

graph_X_res, graph_y_res = sm.fit_resample(graph_X, graph_y)

graph_y_res = pd.DataFrame(graph_y_res)

print(graph_y_res[0].value_counts())
# %% Random Forest Classifier



graph_rf_clf = RandomForestClassifier(n_estimators=100,max_depth=8,random_state=42,

                                verbose=1,class_weight="balanced")



graph_rf_clf.fit(graph_X_res,graph_y_res)

graph_y_pred = graph_rf_clf.predict(graph_X_res)

print("Classification Report for Random Forest Classifier: \n", classification_report(y_res, y_pred))

print("Confusion Matrix of Random Forest Classifier: \n", confusion_matrix(y_res,y_pred))
# See importance of the features

graph_X_res = pd.DataFrame(StandardScaler().fit_transform(graph_X_res), columns=graph_X.columns)

feature_importances = pd.DataFrame(graph_rf_clf.feature_importances_,

                                   index = graph_X_res.columns,

                                   columns=['importance']).sort_values('importance',                                                                 

                                   ascending=False)

print(feature_importances)