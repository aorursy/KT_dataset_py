# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb # XGBoost!



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import os

os.system("ls ../input")



# get otto & test csv files as a DataFrame

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



# preview the data

train_df.head()
# map target to numerics



# map each class to numerical value from 0 to 8(i.e. 9 classes)

range_of_classes = range(1, 10)

map_values_dic   = {}



for n in range_of_classes:

    map_values_dic['Class_{}'.format(n)] = n - 1



train_df['target'] = train_df['target'].map(map_values_dic)



# Plot

import seaborn as sns

sns.countplot(x='target', data=train_df)
del train_df["id"]
train_data = train_df.values
X = train_data[:, :93]

y = train_data[:, 93]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
len(y_test)
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
model = XGBClassifier(learning_rate = 0.1)

model.fit(X_train, y_train)

# make predictions for test data

y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]

# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))