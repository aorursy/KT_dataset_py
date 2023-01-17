# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
os.chdir(r'/kaggle/input/glass')

train = pd.read_csv('glass.csv')

train.head(5)
train.describe()
train.head()
train.dtypes
# Clean up data

# Use the .isnull() method to locate missing data

missing_values = train.isnull()

missing_values.head(5)
# Visualize the data

# data -> argument refers to the data to creat heatmap

# yticklabels -> argument avoids plotting the column names

# cbar -> argument identifies if a colorbar is required or not

# cmap -> argument identifies the color of the heatmap

sns.heatmap(data = missing_values, yticklabels=False, cbar=False, cmap='viridis')
import seaborn as sns

sns.countplot(x="Type", data=train)
# Split data into 'X' features and 'y' target label sets

from sklearn.model_selection import train_test_split

X = train[['Al','Na','K']]

y = train['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

output_model = model.fit(X_train, y_train)

output_model
#Save the model in pickle

#Save to file in the current working directory

pkl_filename = "/kaggle/working/pickle_model.pkl"

with open(pkl_filename, 'wb') as file:

    pickle.dump(model, file)



# Load from file

with open(pkl_filename, 'rb') as file:

    pickle_model = pickle.load(file)



# Calculate the accuracy score and predict target values

score = pickle_model.score(X_test, y_test)

print("Test score: {0:.2f} %".format(100 * score))

Ypredict = pickle_model.predict(X_test)
y_predictions = model.predict(X_test)
from sklearn.metrics import classification_report

cls_report = classification_report(y_test,y_predictions)

print (cls_report)
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test,y_predictions)

print (conf_matrix)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_predictions)