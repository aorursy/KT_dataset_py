# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex4 import *

print("Setup Complete")
import pandas as pd

from sklearn.model_selection import train_test_split



# Read the data

X_full = pd.read_csv('../input/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X_full.SalePrice

X_full.drop(['SalePrice'], axis=1, inplace=True)



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 

                                                                train_size=0.8, test_size=0.2,

                                                                random_state=0)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_train_full.columns if 

                    X_train_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if 

                X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
X_train.head()
X_train[categorical_cols].head()
X_train[numerical_cols].head()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_absolute_error



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



# Define model

#model = LogisticRegression(random_state=0,max_iter=500,solver='saga')

model = RandomForestRegressor(n_estimators=100, random_state=0)



# Bundle preprocessing and modeling code in a pipeline

clf = Pipeline(steps=[('preprocessor',numerical_transformer ),

                      ('model', model)

                     ])



# Preprocessing of training data, fit model 

clf.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = clf.predict(X_valid)



print('MAE:', mean_absolute_error(y_valid, preds))
from xgboost import XGBRegressor

# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='median')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])

from lightgbm import LGBMRegressor

# Define model

#model = RandomForestRegressor(n_estimators=150, random_state=0)

xgb = XGBRegressor(learning_rate=0.01, n_estimators=3460,

                       max_depth=3, min_child_weight=0,

                       gamma=0, subsample=0.7, reg_alpha = 0.0001,

                       colsample_bytree=0.7,

                       objective='reg:squarederror', nthread=-1,

                       scale_pos_weight=1, seed=27)

lgbm = LGBMRegressor(objective='regression',

                         num_leaves=4,

                         reg_alpha = 0.0001,

                         learning_rate=0.01,

                         n_estimators=5000,

                         max_bin=200,

                         bagging_fraction=0.75,

                         bagging_freq=5,

                         bagging_seed=7,

                         feature_fraction=0.2,

                         feature_fraction_seed=7,

                         verbose=-1,

                                       )



# Check your answer

step_1.a.check()
# Lines below will give you a hint or solution code

#step_1.a.hint()

#step_1.a.solution()
# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', xgb)

                             ])



# Preprocessing of training data, fit model 

my_pipeline.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = my_pipeline.predict(X_valid)



# Evaluate the model

score = mean_absolute_error(y_valid, preds)

print('MAE:', score)



# Check your answer

step_1.b.check()
# Line below will give you a hint

step_1.b.hint()
# Preprocessing of test data, fit model

preds_test = my_pipeline.predict(X_test)

# Your code here



# Check your answer

step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)