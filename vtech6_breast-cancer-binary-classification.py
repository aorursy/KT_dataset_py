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
from sklearn.svm import SVC

from sklearn import datasets

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

import tensorflow as tf

from scipy.stats import uniform

import keras
data = datasets.load_breast_cancer()

X = data.data

y = data.target



numerical_transformer = StandardScaler()

X = numerical_transformer.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
SVClassifier = SVC(kernel='rbf', gamma=0.5)



all_data = pd.DataFrame(columns=data.feature_names, data=X)



score = cross_val_score(SVClassifier, all_data, y, cv=5)

score = sum([i for i in score])/len(score)

score
keras_clf = keras.models.Sequential()



keras_clf.add(keras.layers.Dense(1024, activation='relu'))

keras_clf.add(keras.layers.Dense(512, activation='relu'))

keras_clf.add(keras.layers.Dropout(0.2))

keras_clf.add(keras.layers.Dense(128, activation='relu'))

keras_clf.add(keras.layers.Dense(1, activation='sigmoid'))



keras_clf.compile(optimizer='adam',

               loss='binary_crossentropy',

               metrics=['accuracy'])

keras_clf.fit(X_train, y_train, epochs=10)
keras_preds = keras_clf.predict_classes(X_test)



accuracy_score(keras_preds, y_test)
import seaborn

import matplotlib.pyplot as plt

correlation_matrix = all_data.corr()

plt.figure(figsize=(12, 10))

seaborn.heatmap(correlation_matrix, linewidths=.5)

important_features = ['mean concave points', 'mean radius', 'mean perimeter', 'mean area', 'mean compactness', 'mean concavity', 'mean fractal dimension',

                     'worst radius', 'worst perimeter', 'worst area', 'worst compactness', 'worst concavity', 'worst concave points']

len(important_features)

new_features_df = all_data[important_features]
new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(new_features_df, y, test_size=0.2)
score = cross_val_score(SVClassifier, new_features_df, y, cv=5)

score = sum([i for i in score])/len(score)

score
keras_clf_2 = keras.models.Sequential()



keras_clf_2.add(keras.layers.Dense(1024, activation='relu'))

keras_clf_2.add(keras.layers.Dense(512, activation='relu'))

keras_clf_2.add(keras.layers.Dropout(0.2))

keras_clf_2.add(keras.layers.Dense(128, activation='tanh'))

keras_clf_2.add(keras.layers.Dense(1, activation='sigmoid'))



keras_clf_2.compile(optimizer='adam',

               loss='binary_crossentropy',

               metrics=['accuracy'])

keras_clf_2.fit(new_x_train.values, new_y_train, epochs=10)
preds_2 = keras_clf_2.predict_classes(new_x_test)

accuracy_score(preds_2, new_y_test)
distributions = dict(C=[0.1, 0.5, 1, 3, 10, 15,20, 25, 50, 100],

                     kernel=['poly', 'rbf', 'linear'],

                    gamma=['auto', 'scale'])

random_clf = GridSearchCV(SVClassifier, param_grid=distributions)



search = random_clf.fit(new_x_train, new_y_train)

best_params=search.best_params_
optimized_clf = SVC(C=best_params['C'], kernel=best_params['kernel'], gamma=best_params['gamma'])



optimized_scores=cross_val_score(optimized_clf, new_x_train, new_y_train, cv=10)

print(sum(optimized_scores)/len(optimized_scores))