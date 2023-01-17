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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn

from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor
# We have non-numeric as well as empty data in our dataset. Let's clean them up to avoid model fit errors.

def preprocess_data(df):

    """

    Performs transformation on df to match ideal_model

    """

    df["saleyear"] = df.saledate.dt.year

    df["salemonth"] = df.saledate.dt.month

    df["saleday"] = df.saledate.dt.day

    df["saledayofweek"] = df.saledate.dt.dayofweek

    df["saledayofyear"] = df.saledate.dt.dayofyear

    

    df.drop("saledate", axis=1, inplace=True)

    

    # Fill numeric rows with median

    for label, content in df.items():

        #if pd.api.types.is_string_dtype(content):

            #df_tmp[label] = content.astype("category").cat.as_ordered()

        if pd.api.types.is_numeric_dtype(content):

            if pd.isnull(content).sum():

                # Add binary column which signifies if data is missing

                df[label+"_is_missing"] = pd.isnull(content)

                # Fill missing values with median

                df[label] = content.fillna(content.median())



        # Fill missing categorical data and convert them to numbers

        if not pd.api.types.is_numeric_dtype(content):

            # Add binary column to indicate missing value

            df[label+"_is_missing"] = pd.isnull(content)

            # Convent to number and add 1

            df[label] = pd.Categorical(content).codes + 1

        

    return df
df_tmp = pd.read_csv("/kaggle/input/bluebook-for-bulldozers/TrainAndValid.csv",

                low_memory=False,

                parse_dates=["saledate"])



df_tmp = preprocess_data(df_tmp)



df_val = df_tmp[df_tmp.saleyear == 2012]

df_train = df_tmp[df_tmp.saleyear != 2012]



# Split X & y

X_train, y_train = df_train.drop("SalePrice", axis=1), df_train.SalePrice

X_valid, y_valid = df_val.drop("SalePrice", axis=1), df_val.SalePrice



X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
# Create evaluation function (RMSLE)

from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score



def rmsle(y_test, y_preds):

    """

    Calculate root mean squared log error between predictions and actuals

    """

    return np.sqrt(mean_squared_log_error(y_test, y_preds))



# Create function to evaluate model on different levels

def show_scores(model):

    train_preds = model.predict(X_train)

    valid_preds = model.predict(X_valid)

    scores = {"Training MAE": mean_absolute_error(y_train, train_preds),

             "Valid MAE": mean_absolute_error(y_valid, valid_preds),

             "Training RMSLE": rmsle(y_train, train_preds),

             "Valid RMSLE": rmsle(y_valid, valid_preds),

             "Training R^2": r2_score(y_train, train_preds),

              "Valid R^2": r2_score(y_valid, valid_preds)}

    return scores
# %%time

# from sklearn.model_selection import RandomizedSearchCV

# from sklearn.ensemble import RandomForestRegressor



# # Different RandomForestRegressor hyperparameters

# rf_grid = {"n_estimators": np.arange(10, 100, 1),

#           "max_depth": [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100],

#           "min_samples_split": np.arange(2, 20, 2),

#           "min_samples_leaf": np.arange(1, 20, 2),

#           "max_features": [0.5, 1, "sqrt", "auto"],

#           "max_samples": [10000],

#           "bootstrap": [True, False]}



# # Instantiate RandomizedSearchCV model

# rs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=3, random_state=654),

#                               param_distributions=rf_grid,

#                               n_iter=10,

#                               cv=5,

#                               verbose=True)



# # Fit model

# rs_model.fit(X_train, y_train)



# # Get best params from RandomSearchCV

# rs_model.best_params_
%%time

best_model = RandomForestRegressor(n_estimators=96,

                                    min_samples_leaf=7,

                                    min_samples_split=10,

                                    max_features=0.5,

                                    n_jobs=2,

                                    max_samples=10000,

                                    max_depth=40,

                                    bootstrap=False)

best_model.fit(X_train, y_train)



show_scores(best_model)
# Make predictions on test data

# Import test data

df_test = pd.read_csv("/kaggle/input/bluebook-for-bulldozers/Test.csv", low_memory=False,

                      parse_dates=["saledate"])



df_test = preprocess_data(df_test)



# We can find how columsn differ using sets

# set(X_train.columns) - set(df_test.columns)



# Add df_test to include auctioneerID_is_missing

df_test["auctioneerID_is_missing"] = False



# Predict using ideal model

test_preds = best_model.predict(df_test)



# Format predictions as per Kaggle requirements

df_preds = pd.DataFrame()

df_preds["SalesID"] = df_test["SalesID"]

df_preds["SalesPrice"] = test_preds

df_preds



# Export prediction data

#df_preds.to_csv("/kaggle/input/bluebook-for-bulldozers/test_predictions.csv")