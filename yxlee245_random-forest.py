%load_ext autoreload

%autoreload 2
import numpy as np

import pandas as pd



import time

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

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

model = RandomForestRegressor(max_features='sqrt', n_jobs=-1)

pipeline_rfr = data_preprocess_module.build_pipeline(model)

# Hyperparameter tuning

params = {

    'model__n_estimators': [1000, 1500, 2000],

    'model__max_depth': [10, 15, 20]

}

rfr = GridSearchCV(pipeline_rfr, params, cv=5, scoring='neg_mean_squared_error')

time_start = time.time()

rfr.fit(X_train, y_train)

print('Time taken for hyperparameter tuning: {:.2f} min'.

      format((time.time() - time_start) / 60))

get_score(rfr, X_val, y_val)
# Show selected params

rfr.best_params_
preds_test = rfr.predict(X_test)

preds_test = y_trfm(preds_test)



output = pd.DataFrame({'id': X_test.index,

                       'resale_price': preds_test})

output.to_csv('submission.csv', index=False)