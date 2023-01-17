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
import seaborn as sns

import matplotlib.pyplot as plt





df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print ("Num Cols: ", len(df.columns))

print ("Num Rows: ",len(df.index))
df.info(verbose=True, null_counts=True)
df.describe()
ax = sns.heatmap(df.isnull(),yticklabels=False,cbar=False)

ax.set(xlabel='columns', ylabel='rows (white if null)')

plt.show()





from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



# Define column categories

categorical_cols = [col for col in df.columns if df[col].dtype == 'object']

numeric_cols = df.drop(columns=categorical_cols+["SalePrice"]).columns



print ("Categorical columns: ", categorical_cols)

print ("Numeric columns: ", numeric_cols)
# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='mean')
# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])
from sklearn.model_selection import train_test_split



# Grab target as y, remove target from X

train_test = df.copy()

y = train_test.SalePrice

X = train_test.drop(columns=['SalePrice'])



# Split into train, test

train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, random_state = 17)
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score



# Time to tune params!

def display_validation(pipeline):

    # Preprocessing of training data, fit model 

    pipeline.fit(train_X,train_y)

    # Preprocessing of validation data, get predictions

    preds = pipeline.predict(val_X)



    # Evaluate the model

    score = mean_absolute_error(val_y, preds)

    print('MAE:', score)
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



# Define column categories

categorical_cols = [col for col in df.columns if df[col].dtype == 'object']

numeric_cols = df.drop(columns=categorical_cols+["SalePrice"]).columns



print ("Categorical columns: ", categorical_cols)

print ("Numeric columns: ", numeric_cols)





# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='mean')





# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('numeric', numerical_transformer, numeric_cols),

        ('categorical', categorical_transformer, categorical_cols)

    ])
from sklearn.ensemble import RandomForestRegressor

import random



for n in [50,100, 500]:

    model = RandomForestRegressor(n_estimators=n, random_state = 17)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])

    print("n_estimators: ", n)

    display_validation(pipeline)

    
final_model = RandomForestRegressor(n_estimators=100, random_state = 17)

final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', final_model)

                             ])



# Preprocessing of validation data, get predictions

final_pipeline.fit(train_X,train_y)

test_data_labels = final_pipeline.predict(test)


