# The challenge here is to predict whether a user will click on some ad presented to him

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/taboola-dataset/data.csv', encoding= 'unicode_escape')
df.shape
df.head()
df.info()
df.columns
for col in df.columns:
    print(col, "- number of unique values:",df[col].nunique())
len(df[df['is_click'] == 1])
# correlation metrics

import seaborn as sns

fig, ax = plt.subplots(figsize=(8,8))

corr = df.corr()

sns.heatmap(corr, cmap='cubehelix', ax = ax)
df.groupby('quality_level')['is_click'].size()
#define data1

data1 = df[['is_click', 'user_recs', 'user_clicks', 'empiric_recs', 'empiric_clicks']]
data1_10000 = data1.iloc[0:10000]
data1_10000.head(5)
# knn model data1

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

model = KNeighborsClassifier(n_neighbors=27)
features = data1_10000.drop(columns=['is_click'])
label = data1_10000['is_click']

scaler = StandardScaler()
features_scaled = scaler.fit(features)
features_scaled = scaler.transform(features)

x_train, x_test, y_train, y_test = train_test_split(features_scaled, label, test_size=0.3)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(metrics.accuracy_score(y_test, y_pred))
# k optimization knn data1

k_range = range(1,50, 2)
scores_list = []
for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))
print(scores_list)
print('max of score_list is:', max(scores_list))
plt.plot(k_range, scores_list)
plt.xlabel('Value of k for KNN')
plt.ylabel('Testing Accuracy')
# applying random forest model for data1

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)

rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)

print(metrics.accuracy_score(y_test, y_pred))
# k optimization rf data1

k_range = range(100,1000, 100)
scores_list = []
for k in k_range:
    rf = RandomForestClassifier(n_estimators =k, random_state = 42)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))
print(scores_list)
print('max of score_list is:', max(scores_list))
# logistic regression model for data1

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state=0)

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

print(metrics.accuracy_score(y_test, y_pred))
# apply cnn (convolutional neural network) model for data1

from keras.utils import to_categorical

# convert class vectors to binary class matrices
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape)
print(y_test.shape)
y_train
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

num_classes = 2
epochs = 10

n_steps = 4

x_train = x_train.reshape(7000, n_steps, 1)
x_test = x_test.reshape(3000, n_steps, 1)

print(x_train.shape)
print(y_train.shape)

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(n_steps, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# The results of cnn in data1 are not better than the previous models

# define data2

data2 = df[['is_click', 'quality_level', 'ad_type', 'content_category', 'user_recs', 'user_clicks', 'empiric_recs', 'empiric_clicks', 'target_item_alchemy_taxonomies']]
data2_10000 = data2.iloc[0:10000]
data2_10000.head(5)
# one-hot encoding

data2_10000_OHE = pd.get_dummies(data2_10000)
data2_10000_OHE.head()
data2_10000_OHE.shape
# pca

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data2_10000_OHE.drop(columns=['is_click']))
# knn model for data2

model = KNeighborsClassifier(n_neighbors=27)
features = principalComponents
label = data2_10000_OHE['is_click']

scaler = StandardScaler()
features_scaled = scaler.fit(features)
features_scaled = scaler.transform(features)

x_train, x_test, y_train, y_test = train_test_split(features_scaled, label, test_size=0.3)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(metrics.accuracy_score(y_test, y_pred))
# k - optimization

k_range = range(1,100, 2)
scores_list = []
for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))
print(scores_list)
print('max of score_list is:', max(scores_list))
plt.plot(k_range, scores_list)
plt.xlabel('Value of k for KNN')
plt.ylabel('Testing Accuracy')
# results of data2 are same to those of data1
# adding features and feature engineering

df['user_ratio'] = df.apply(lambda row: row.user_clicks / (row.user_recs+1), axis=1)
# adding this to models do not improve the results
# adding another features?