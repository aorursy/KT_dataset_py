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
import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

sns.set_style('whitegrid')

import tensorflow as tf

from sklearn.model_selection import train_test_split, cross_val_predict

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.decomposition import PCA

from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import time

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline



SMALL_SIZE = 10

MEDIUM_SIZE = 12



plt.rc('font', size=SMALL_SIZE)

plt.rc('axes', titlesize=MEDIUM_SIZE)

plt.rc('axes', labelsize=MEDIUM_SIZE)

plt.rcParams['figure.dpi']=150



#sdss_df = pd.read_csv('Skyserver_SQL2_27_2018 6_51_39 PM.csv', skiprows=1)

sdss_df = pd.read_csv('/kaggle/input/sloan-digital-sky-survey/Skyserver_SQL2_27_2018 6_51_39 PM.csv', skiprows=0)



sdss_df.head()



sdss_df.info()



sdss_df.describe()



sdss_df['class'].value_counts()



sdss_df.columns.values



sdss_df.drop(['objid', 'run', 'rerun', 'camcol', 'field', 'specobjid'], axis=1, inplace=True)

sdss_df.head(1)



fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(16, 4))

ax = sns.distplot(sdss_df[sdss_df['class']=='STAR'].redshift, bins = 30, ax = axes[0], kde = False)

ax.set_title('Star')

ax = sns.distplot(sdss_df[sdss_df['class']=='GALAXY'].redshift, bins = 30, ax = axes[1], kde = False)

ax.set_title('Galaxy')

ax = sns.distplot(sdss_df[sdss_df['class']=='QSO'].redshift, bins = 30, ax = axes[2], kde = False)

ax = ax.set_title('QSO')



fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(16, 4))

ax = sns.lvplot(x=sdss_df['class'], y=sdss_df['dec'], palette='coolwarm')

ax.set_title('dec')



fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(16, 4))

fig.set_dpi(100)

ax = sns.heatmap(sdss_df[sdss_df['class']=='STAR'][['u', 'g', 'r', 'i', 'z']].corr(), ax = axes[0], cmap='coolwarm')

ax.set_title('Star')

ax = sns.heatmap(sdss_df[sdss_df['class']=='GALAXY'][['u', 'g', 'r', 'i', 'z']].corr(), ax = axes[1], cmap='coolwarm')

ax.set_title('Galaxy')

ax = sns.heatmap(sdss_df[sdss_df['class']=='QSO'][['u', 'g', 'r', 'i', 'z']].corr(), ax = axes[2], cmap='coolwarm')

ax = ax.set_title('QSO')



sns.lmplot(x='ra', y='dec', data=sdss_df, hue='class', fit_reg=False, palette='coolwarm', size=6, aspect=2)

plt.title('Equatorial coordinates')



sdss_df_fe = sdss_df



# encode class labels to integers

le = LabelEncoder()

y_encoded = le.fit_transform(sdss_df_fe['class'])

sdss_df_fe['class'] = y_encoded



# Principal Component Analysis

pca = PCA(n_components=3)

ugriz = pca.fit_transform(sdss_df_fe[['u', 'g', 'r', 'i', 'z']])



# update dataframe 

sdss_df_fe = pd.concat((sdss_df_fe, pd.DataFrame(ugriz)), axis=1)

sdss_df_fe.rename({0: 'PCA_1', 1: 'PCA_2', 2: 'PCA_3'}, axis=1, inplace = True)

sdss_df_fe.drop(['u', 'g', 'r', 'i', 'z'], axis=1, inplace=True)

sdss_df_fe.head()



scaler = MinMaxScaler()

sdss = scaler.fit_transform(sdss_df_fe.drop('class', axis=1))



X_train, X_test, y_train, y_test = train_test_split(sdss, sdss_df_fe['class'], test_size=0.33)



knn = KNeighborsClassifier()

training_start = time.perf_counter()

knn.fit(X_train, y_train)

training_end = time.perf_counter()

prediction_start = time.perf_counter()

preds = knn.predict(X_test)

prediction_end = time.perf_counter()

acc_knn = (preds == y_test).sum().astype(float) / len(preds)*100

knn_train_time = training_end-training_start

knn_prediction_time = prediction_end-prediction_start

print("Scikit-Learn's K Nearest Neighbors Classifier's prediction accuracy is: %3.2f" % (acc_knn))

print("Time consumed for training: %4.3f seconds" % (knn_train_time))

print("Time consumed for prediction: %6.5f seconds" % (knn_prediction_time))



from sklearn.preprocessing import MaxAbsScaler

scaler_gnb = MaxAbsScaler()

sdss = scaler_gnb.fit_transform(sdss_df_fe.drop('class', axis=1))

X_train_gnb, X_test_gnb, y_train_gnb, y_test_gnb = train_test_split(sdss, sdss_df_fe['class'], test_size=0.33)



gnb = GaussianNB()

training_start = time.perf_counter()

gnb.fit(X_train_gnb, y_train_gnb)

training_end = time.perf_counter()

prediction_start = time.perf_counter()

preds = gnb.predict(X_test_gnb)

prediction_end = time.perf_counter()

acc_gnb = (preds == y_test_gnb).sum().astype(float) / len(preds)*100

gnb_train_time = training_end-training_start

gnb_prediction_time = prediction_end-prediction_start

print("Scikit-Learn's Gaussian Naive Bayes Classifier's prediction accuracy is: %3.2f" % (acc_gnb))

print("Time consumed for training: %4.3f seconds" % (gnb_train_time))

print("Time consumed for prediction: %6.5f seconds" % (gnb_prediction_time))



xgb = XGBClassifier(n_estimators=100)

training_start = time.perf_counter()

xgb.fit(X_train, y_train)

training_end = time.perf_counter()

prediction_start = time.perf_counter()

preds = xgb.predict(X_test)

prediction_end = time.perf_counter()

acc_xgb = (preds == y_test).sum().astype(float) / len(preds)*100

xgb_train_time = training_end-training_start

xgb_prediction_time = prediction_end-prediction_start

print("XGBoost's prediction accuracy is: %3.2f" % (acc_xgb))

print("Time consumed for training: %4.3f" % (xgb_train_time))

print("Time consumed for prediction: %6.5f seconds" % (xgb_prediction_time))



rfc = RandomForestClassifier(n_estimators=10)

training_start = time.perf_counter()

rfc.fit(X_train, y_train)

training_end = time.perf_counter()

prediction_start = time.perf_counter()

preds = rfc.predict(X_test)

prediction_end = time.perf_counter()

acc_rfc = (preds == y_test).sum().astype(float) / len(preds)*100

rfc_train_time = training_end-training_start

rfc_prediction_time = prediction_end-prediction_start

print("Scikit-Learn's Random Forest Classifier's prediction accuracy is: %3.2f" % (acc_rfc))

print("Time consumed for training: %4.3f seconds" % (rfc_train_time))

print("Time consumed for prediction: %6.5f seconds" % (rfc_prediction_time))



svc = SVC()

training_start = time.perf_counter()

svc.fit(X_train, y_train)

training_end = time.perf_counter()

prediction_start = time.perf_counter()

preds = svc.predict(X_test)

prediction_end = time.perf_counter()

acc_svc = (preds == y_test).sum().astype(float) / len(preds)*100

svc_train_time = training_end-training_start

svc_prediction_time = prediction_end-prediction_start

print("Scikit-Learn's Support Vector Machine Classifier's prediction accuracy is: %3.2f" % (acc_svc))

print("Time consumed for training: %4.3f seconds" % (svc_train_time))

print("Time consumed for prediction: %6.5f seconds" % (svc_prediction_time))



results = pd.DataFrame({

    'Model': ['KNN', 'Naive Bayes', 

              'XGBoost', 'Random Forest', 'SVC'],

    'Score': [acc_knn, acc_gnb, acc_xgb, acc_rfc, acc_svc],

    'Runtime Training': [knn_train_time, gnb_train_time, xgb_train_time, rfc_train_time, 

                         svc_train_time],

    'Runtime Prediction': [knn_prediction_time, gnb_prediction_time, xgb_prediction_time, rfc_prediction_time,

                          svc_prediction_time]})

result_df = results.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Model')

result_df



from sklearn.model_selection import cross_val_score

rfc_cv = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(rfc_cv, X_train, y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())



xgb_cv = XGBClassifier(n_estimators=100)

scores = cross_val_score(xgb_cv, X_train, y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())



importances = pd.DataFrame({

    'Feature': sdss_df_fe.drop('class', axis=1).columns,

    'Importance': xgb.feature_importances_

})

importances = importances.sort_values(by='Importance', ascending=False)

importances = importances.set_index('Feature')

importances



importances.plot.bar()



scaler = MinMaxScaler()

sdss = pd.DataFrame(scaler.fit_transform(sdss_df_fe.drop(['mjd', 'class'], axis=1)), columns=sdss_df_fe.drop(['mjd', 'class'], axis=1).columns)

sdss['class'] = sdss_df_fe['class']



sdss.head()



sdss.to_csv('sdss_data.csv')



X_train, X_test, y_train, y_test = train_test_split(sdss.drop('class', axis=1), sdss['class'],

                                                   test_size=0.33)



xgboost = XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators=100, gamma=0, 

                        min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005)



xgboost.fit(X_train, y_train)

preds = xgboost.predict(X_test)



accuracy = (preds == y_test).sum().astype(float) / len(preds)*100



print("XGBoost's prediction accuracy WITH optimal hyperparameters is: %3.2f" % (accuracy))



xgb_cv = XGBClassifier(n_estimators=100)

scores = cross_val_score(xgb_cv, X_train, y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())



unique, counts = np.unique(sdss['class'], return_counts=True)

dict(zip(unique, counts))



predictions = cross_val_predict(xgb, sdss.drop('class', axis=1), sdss['class'], cv=3)

confusion_matrix(sdss['class'], predictions)



print("Precision:", precision_score(sdss['class'], predictions, average='micro'))

print("Recall:",recall_score(sdss['class'], predictions, average='micro'))



print("F1-Score:", f1_score(sdss['class'], predictions, average='micro'))