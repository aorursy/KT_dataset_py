# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split # splitting our data into training and testing data

import seaborn as sns # for creating a correlation heatmap

import matplotlib.pyplot as plt # for displaying our heatmap for analysis

from xgboost import XGBClassifier # eventually, we will use an XGBClassifier for our model

from sklearn.metrics import accuracy_score # to score our model



# Input data files are available in the "../input/" directory.



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Read the dataset

X_full = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv', index_col='id')



# Assign y to the diagnosis column

y = X_full.diagnosis



# Assigning our index_col to be the column 'id' shifted our data over, leaving a column with all NaN entries.

# We drop that here

X = X_full.drop(columns=['Unnamed: 32'])



# Show all values whenever we call head.

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

# If we run .dtypes on our data frame, we notice that all columns, aside from the diagnosis being a string, our integers.



# We replace a malignant diagnosis with 1, and benign with 0

X['diagnosis'].replace('M', 1, inplace=True)

X['diagnosis'].replace('B', 0, inplace=True)

y.replace('M', 1, inplace=True)

y.replace('B', 0, inplace=True)
# Here, we use the seaborn correlation heatmap to visualize the correlatons of features in our dataset on one another.

# Using the filter method, we will drop features which have an absolute value of less than 0.5 on the feature 'diagnosis'



# Setting up and displaying our heatmap correlation

plt.figure(figsize=(20,20))

cor = X.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds, fmt='.2f')

plt.show()
# Keep features which have a med-high correlation on the diagnosis

features = ['radius_mean', 'perimeter_mean', 'area_mean', 'compactness_mean', 'concavity_mean', 

            'concave points_mean', 'radius_se', 'perimeter_se', 'area_se', 'radius_worst', 'perimeter_worst',

           'area_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst']

X = X[features]



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)
# We will use an XGBoostClassifier, and score the model using SKLearn Accuracy Score



model = XGBClassifier()

model.fit(X_train, y_train)

preds = model.predict(X_valid)

accuracy_score(y_valid, preds)