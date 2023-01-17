import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import time



import sklearn

from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB



from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import MinMaxScaler

column_names = ["Id", "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Income"]
test_data = pd.read_csv("../input/test_data.csv", names = column_names, na_values='?').drop(0, axis = 0).reset_index(drop = True)

train_data = pd.read_csv("../input/train_data.csv", names = column_names, na_values='?').drop(0, axis = 0).reset_index(drop = True)
train_data = train_data.drop("Id", axis = 1)

train_data = train_data.drop("Education", axis = 1)

test_data = test_data.drop("Id", axis = 1)

test_data = test_data.drop("Education", axis = 1)

test_data = test_data.drop("Income", axis = 1)
test_data.head()
train_data.columns
train_data.shape
train_data.head()
train_data.info()
train_data['Age'] = train_data['Age'].astype('int64')

train_data['fnlwgt'] = train_data['fnlwgt'].astype('int64')

train_data['Education-Num'] = train_data['Education-Num'].astype('int64')

train_data['Capital Gain'] = train_data['Capital Gain'].astype('int64')

train_data['Capital Loss'] = train_data['Capital Loss'].astype('int64')

train_data['Hours per week'] = train_data['Hours per week'].astype('int64')

train_data.info()
train_data.describe()
train_data.describe(include=['object'])
train_data = train_data.fillna('missing')  # Preencher dados faltantes com a string 'missing'

test_data = test_data.fillna('missing')
def count_null_values(data):  # Returns a DataFrame with count of null values

    

    

    counts_null = []

    for column in data.columns:

        counts_null.append(data[column].isnull().sum())

    counts_null = np.asarray(counts_null)



    counts_null = pd.DataFrame({'feature': data.columns, 'count.': counts_null,

                                'freq. [%]': 100*counts_null/data.shape[0]}).set_index('feature', drop = True)

    counts_null = counts_null.sort_values(by = 'count.', ascending = False)

    

    return counts_null
count_null_values(train_data)
train_data_analysis = train_data.apply(preprocessing.LabelEncoder().fit_transform)
corr_mat = train_data_analysis.corr()

sns.set()

plt.figure(figsize=(15,12))

sns.heatmap(corr_mat, annot=True)
train_data_analysis.corr().Income.sort_values()
abs(train_data_analysis.corr().Income).sort_values(ascending=False)
x_train = train_data[["Capital Gain", "Education-Num", "Relationship", "Age", "Hours per week", "Sex", "Marital Status", 

                      "Capital Loss", "Race", "Occupation"]].apply(preprocessing.LabelEncoder().fit_transform)

y_train = train_data.Income



x_test = test_data[["Capital Gain", "Education-Num", "Relationship", "Age", "Hours per week", "Sex", "Marital Status", 

                      "Capital Loss", "Race", "Occupation"]].apply(preprocessing.LabelEncoder().fit_transform)
scaler = MinMaxScaler()  # Scaler para normalizar os dados contidos nos atributos



x_train = scaler.fit_transform(x_train)

x_test = scaler.fit_transform(x_test)
knn = KNeighborsClassifier(n_neighbors = 23, p = 1)

start = time.time()

scores = cross_val_score(knn, x_train, y_train, cv = 10)

print('K-Nearest Neighbors CV accuracy: {0:1.4f} +-{1:2.5f}\n'.format(scores.mean(), scores.std()))

print ('Time elapsed: {0:1.2f}\n'.format(time.time()-start))
start = time.time()

knn.fit(x_train, y_train)

y_predict_knn = knn.predict(x_test)

print ('Time elapsed: {0:1.2f}\n'.format(time.time()-start))
log_clf = LogisticRegression(solver = 'lbfgs', C = 1.0, penalty = 'l2', warm_start =  True)

start = time.time()

log_scores = cross_val_score(log_clf, x_train, y_train, cv = 10)

print('Logistic Regression CV accuracy: {0:1.4f} +-{1:2.5f}\n'.format(log_scores.mean(), log_scores.std()))

print ('Time elapsed: {0:1.2f}\n'.format(time.time()-start))
start = time.time()

log_clf.fit(x_train, y_train)

y_predict_log = log_clf.predict(x_test)

print ('Time elapsed: {0:1.2f}\n'.format(time.time()-start))
rf_clf = RandomForestClassifier(n_estimators = 700, max_depth = 12)

start = time.time()

rf_scores = cross_val_score(rf_clf, x_train, y_train, cv = 10)

print('Random Forest CV accuracy: {0:1.4f} +-{1:2.5f}\n'.format(rf_scores.mean(), rf_scores.std()))

print ('Time elapsed: {0:1.2f}\n'.format(time.time()-start))
start = time.time()

rf_clf.fit(x_train, y_train)

y_predict_rf = rf_clf.predict(x_test)

print ('Time elapsed: {0:1.2f}\n'.format(time.time()-start))
gnb_clf = GaussianNB()

start = time.time()

gnb_scores = cross_val_score(gnb_clf, x_train, y_train, cv = 10)

print('Gaussian Naive Bayes CV accuracy: {0:1.4f} +-{1:2.5f}\n'.format(gnb_scores.mean(), gnb_scores.std()))

print ('Time elapsed: {0:1.2f}\n'.format(time.time()-start))
start = time.time()

gnb_clf.fit(x_train, y_train)

y_predict_gnb = gnb_clf.predict(x_test)

print ('Time elapsed: {0:1.2f}\n'.format(time.time()-start))
df_pred_knn = pd.DataFrame({'Income':y_predict_knn})

df_pred_log = pd.DataFrame({'Income':y_predict_log})

df_pred_rf = pd.DataFrame({'Income':y_predict_rf})

df_pred_gnb = pd.DataFrame({'Income':y_predict_gnb})
df_pred_knn.to_csv("knn_prediction.csv", index = True, index_label = 'Id')

df_pred_log.to_csv("log_prediction.csv", index = True, index_label = 'Id')

df_pred_rf.to_csv("rf_prediction.csv", index = True, index_label = 'Id')

df_pred_gnb.to_csv("gnb_prediction.csv", index = True, index_label = 'Id')