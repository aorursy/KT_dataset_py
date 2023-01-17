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
# Ignore warnings

import warnings

warnings.filterwarnings('ignore')
# Read and preview data

df = pd.read_csv('/kaggle/input/boston-housing-dataset/HousingData.csv')

df.head()
df.info()
df.isnull().sum()
df = df.fillna(df.mean())
df.isnull().sum()
# Declare feature vector and target variable

X = df[['LSTAT','RM','NOX','PTRATIO','DIS','AGE']]

y = df['MEDV']
# Split the data into train and test data:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
# Build the model with Random Forest Regressor :

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)**(0.5)

mse
import lime

import lime.lime_tabular
# LIME has one explainer for all the models

explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns.values.tolist(),

                                                  class_names=['MEDV'], verbose=True, mode='regression')
# Choose the 5th instance and use it to predict the results

j = 5

exp = explainer.explain_instance(X_test.values[j], model.predict, num_features=6)

# Show the predictions

exp.show_in_notebook(show_table=True)
exp.as_list()
# Choose the 10th instance and use it to predict the results

j = 10

exp = explainer.explain_instance(X_test.values[j], model.predict, num_features=6)
# Show the predictions

exp.show_in_notebook(show_table=True)
exp.as_list()
# Choose the 15th instance and use it to predict the results

j = 15

exp = explainer.explain_instance(X_test.values[j], model.predict, num_features=6)
# Show the predictions

exp.show_in_notebook(show_table=True)
exp.as_list()
# Choose the 20th instance and use it to predict the results

j = 20

exp = explainer.explain_instance(X_test.values[j], model.predict, num_features=6)
# Show the predictions

exp.show_in_notebook(show_table=True)
exp.as_list()
# Choose the 25th instance and use it to predict the results

j = 25

exp = explainer.explain_instance(X_test.values[j], model.predict, num_features=6)
# Show the predictions

exp.show_in_notebook(show_table=True)
exp.as_list()