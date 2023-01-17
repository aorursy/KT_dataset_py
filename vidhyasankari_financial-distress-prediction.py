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



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#load data set

df=pd.read_csv("/kaggle/input/financial-distress/Financial Distress.csv",index_col="Company")


df.head()
df.describe()
df.columns
from sklearn.model_selection import train_test_split



# Obtain target and predictors

y = df["Financial Distress"]

features = ['Time', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7',

       'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17',

       'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27',

       'x28', 'x29', 'x30', 'x31', 'x32', 'x33', 'x34', 'x35', 'x36', 'x37',

       'x38', 'x39', 'x40', 'x41', 'x42', 'x43', 'x44', 'x45', 'x46', 'x47',

       'x48', 'x49', 'x50', 'x51', 'x52', 'x53', 'x54', 'x55', 'x56', 'x57',

       'x58', 'x59', 'x60', 'x61', 'x62', 'x63', 'x64', 'x65', 'x66', 'x67',

       'x68', 'x69', 'x70', 'x71', 'x72', 'x73', 'x74', 'x75', 'x76', 'x77',

       'x78', 'x79', 'x80', 'x81', 'x82', 'x83']

X = df[features].copy()



# Break off validation set from training data

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)
from sklearn.ensemble import RandomForestRegressor



# Define the models

model_1 = RandomForestRegressor(n_estimators=50, random_state=0)

model_2 = RandomForestRegressor(n_estimators=100, random_state=0)

model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)

model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)

model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)



models = [model_1, model_2, model_3, model_4, model_5]
from sklearn.metrics import mean_absolute_error



# Function for comparing different models

def score_model(model, X_tr=X_train, X_t=X_test, y_tr=y_train, y_t=y_test):

    model.fit(X_tr, y_tr)

    preds = model.predict(X_t)

    return mean_absolute_error(y_t, preds)



for i in range(0, len(models)):

    mae = score_model(models[i])

    print("Model %d MAE: %d" % (i+1, mae))


model_1.fit(X, y)



# Generate test predictions

preds = model_1.predict(X_test)

print(mean_absolute_error(y_test, preds))
