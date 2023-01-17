%load_ext autoreload

%autoreload 2
import numpy as np

import pandas as pd



import time

from sklearn.linear_model import (LinearRegression, Lasso,

                                  Ridge, ElasticNet)

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error



# Refer to https://www.kaggle.com/product-feedback/91185 to import functions and

# classes from Kaggle kernels

from preprocess import DataPreprocessModule
data_preprocess_module = DataPreprocessModule(

    train_path='../input/hdb-resale-price-prediction/train.csv',

    test_path='../input/hdb-resale-price-prediction/test.csv')

X_train, X_val, X_test, y_train, y_val = data_preprocess_module.get_preprocessed_data()

print('Shape of X_train:', X_train.shape)

print('Shape of X_val:', X_val.shape)

print('Shape of X_test:', X_test.shape)

print('Shape of y_train:', y_train.shape)

print('Shape of y_val:', y_val.shape)
# Define RMSE

metric = lambda y1_real, y2_real: np.sqrt(mean_squared_error(y1_real, y2_real))

# Claculate exp(y) - 1 for all elements in y

y_trfm = lambda y: np.expm1(y)

# Define function to get score given model and data

def get_score(model, X, y):

    # Predict

    preds = model.predict(X)

    # Transform

    preds = y_trfm(preds)

    y = y_trfm(y)

    return metric(preds, y)
# Build pipeline

pipeline_reg = data_preprocess_module.build_pipeline(LinearRegression())



# Train model

pipeline_reg.fit(X_train, y_train)

get_score(pipeline_reg, X_val, y_val)
# Build pipeline

pipeline_lasso = data_preprocess_module.build_pipeline(Lasso())



# Hyperparameter tuning

params = {

    'model__alpha': [10, 1, 0.1, 0.01, 0.001]

}

lasso = GridSearchCV(pipeline_lasso, params, cv=5,

                     scoring='neg_mean_squared_error', n_jobs=-1)

time_start = time.time()

lasso.fit(X_train, y_train)

print('Time taken for hyperparameter tuning: {:.2f} min'.

      format((time.time() - time_start) / 60))

get_score(lasso, X_val, y_val)
# Build pipeline

pipeline_ridge = data_preprocess_module.build_pipeline(Ridge())



# Hyperparameter tuning

params = {

    'model__alpha': [10, 1, 0.1, 0.01, 0.001]

}

ridge = GridSearchCV(pipeline_ridge, params, cv=5,

                     scoring='neg_mean_squared_error', n_jobs=-1)

time_start = time.time()

ridge.fit(X_train, y_train)

print('Time taken for hyperparameter tuning: {:.2f} min'.

      format((time.time() - time_start) / 60))

get_score(ridge, X_val, y_val)
# Build pipeline

pipeline_elast = data_preprocess_module.build_pipeline(ElasticNet())



# Hyperparameter tuning

params = {

    'model__alpha': [10, 1, 0.1, 0.01, 0.001],

    'model__l1_ratio': [0.25, 0.5, 0.75]

}

elast = GridSearchCV(pipeline_elast, params, cv=5,

                     scoring='neg_mean_squared_error', n_jobs=-1)

time_start = time.time()

elast.fit(X_train, y_train)

print('Time taken for hyperparameter tuning: {:.2f} min'.

      format((time.time() - time_start) / 60))

get_score(elast, X_val, y_val)
# Selected parameter for elastic net

elast.best_params_
# Choosing elstic net as it has the lowest validation score

preds_test = elast.predict(X_test)

preds_test = y_trfm(preds_test)



output = pd.DataFrame({'id': X_test.index,

                       'resale_price': preds_test})

output.to_csv('submission.csv', index=False)