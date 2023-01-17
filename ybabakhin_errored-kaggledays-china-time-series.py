import warnings

warnings.simplefilter(action="ignore")



import datetime

import os

import random

import numpy as np

import pandas as pd

import seaborn as sns

import lightgbm as lgb

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

plt.style.use("ggplot")



# Seed Everything

seed = 123

random.seed(seed)

os.environ['PYTHONHASHSEED'] = str(seed)

np.random.seed(seed)
def read_data(train_path, test_path):

    train = pd.read_csv(train_path)

    test = pd.read_csv(test_path)

    

    # Set date to the index

    train['date'] = pd.to_datetime(train['date'])

    train.set_index(["date"], inplace=True)



    test['date'] = pd.to_datetime(test['date'])

    test.set_index(["date"], inplace=True)

    

    return train, test
def get_date_features(df):

    

    # Add date-specific features

    df['dayofweek'] = df.index.dayofweek

    df['is_weekend'] = df.index.dayofweek // 5

    df['day'] = df.index.day

    df['month'] = df.index.month

    df['year'] = df.index.year

    df['dayofyear'] = df.index.dayofyear

    df['weekofyear'] = df.index.weekofyear



    return df





def generate_features(train, test):



    # Get the time delta between last test and last train observations

    model_delta = max(test.index) - max(train.index)



    # Initialize variables

    lags = [model_delta.days]

    window = 7

    lag_features = []



    # Concatenate train and test together

    data = pd.concat([train, test], sort=False)



    for lag in lags:

        feat = data.groupby(["store", "item"])[["sales"]].rolling(window=window).mean().shift(lag)

        feat.columns = ['sales_mean_lag_{}'.format(lag)]

        lag_features.append(feat)



        feat = data.groupby(["store", "item"])[["sales"]].rolling(window=window).std().shift(lag)

        feat.columns = ['sales_std_lag_{}'.format(lag)]

        lag_features.append(feat)



    # Concatenate all features together

    lag_features = pd.concat(lag_features, axis=1)



    # Add date features to the train and test

    train = get_date_features(train)

    test = get_date_features(test)



    # Join lag features to the train and test by (store, item, date)

    train.set_index(["store", "item"], append=True, inplace=True)

    train = train.reorder_levels(["store", "item", "date"])

    test.set_index(["store", "item"], append=True, inplace=True)

    test = test.reorder_levels(["store", "item", "date"])



    train = train.join(lag_features)

    test = test.join(lag_features)



    # Resetting index back

    train.reset_index(level=[0, 1], inplace=True)

    test.reset_index(level=[0, 1], inplace=True)

    

    return train, test, model_delta, window
# Read data

train, test = read_data(train_path="../input/demand-forecasting-kernels-only/train.csv",

                        test_path="../input/demand-forecasting-kernels-only/test.csv")
train.head()
test.head()
# Look at the data

plt.figure(figsize=(12, 8))

plt.plot(train[(train.store==1) & (train.item==1)]["sales"].diff())

plt.title("Train data for store 1; item 1")

plt.show()
# There are multiple approaches for multi-step time series predictions: https://arxiv.org/pdf/1108.3259.pdf



# Generate some features

train, test, model_delta, window = generate_features(train, test)
train.head()
# Train start date

start = min(train.index) + model_delta + datetime.timedelta(days=window)



# Validation start date

split_date = max(train.index) - model_delta



# Keep only non-NA values in the train data

train = train[train.index >= start]



# Train-Validation split

val_train = train[train.index < split_date]

validation = train[train.index > split_date]
def smape(y_true, y_pred):

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    diff = 100 * np.abs(y_true - y_pred) / denominator

    diff[denominator == 0] = 0.0

    

    return np.nanmean(diff)





def smape_lightgbm(y_pred, y_true):

    return "smape", smape(y_true, y_pred), False
# Select all the features except for the target variable

features = [x for x in train.columns if x != "sales"]



params = {

    "learning_rate": 0.1,

    "objective": "regression",

    "metric": "None",

    "n_estimators": 1000,

    "colsample_bytree": 0.9,

    "num_leaves": 32,

    "subsample": 0.8,

    "subsample_freq": 1,

    "lambda_l2": 1,

}



# Train LightGBM model

reg = lgb.LGBMRegressor(**params)

reg = reg.fit(

    val_train[features],

    val_train["sales"],

    eval_set=[(validation[features], validation["sales"])],

    verbose=100,

    early_stopping_rounds=200,

    eval_metric=smape_lightgbm,

)
predictions = reg.predict(validation[features])

validation['pred'] = predictions

print("Validation SMAPE: {:.3f}".format(smape(validation["sales"], validation["pred"])))
plt.figure(figsize=(12, 8))



to_plot = validation[(validation.store==1) & (validation.item==1)]



plt.plot(to_plot["sales"], alpha=0.8)

plt.plot(to_plot["pred"], linestyle="--")

plt.title("True values vs Predicted values for store 1; item 1")

plt.legend(["TRUE VALUES", "PREDICTED VALUES"])

plt.xticks(None)

plt.show()
# Re-train the model on the whole train data

params['n_estimators'] = reg.best_iteration_



# Train LightGBM model

reg = lgb.LGBMRegressor(**params)

reg = reg.fit(

    train[features],

    train["sales"],

)
feature_importances = pd.DataFrame(

    {"feature": val_train[features].columns,

     "importance": reg.feature_importances_}

)



plt.figure(figsize=(8, 8))

sns.barplot(

    data=feature_importances.sort_values("importance", ascending=False).head(10),

    x="importance",

    y="feature",

)

plt.yticks(fontsize=14)

plt.title("Top 10 features")

plt.show()
test["sales"] = reg.predict(test[features])
plt.figure(figsize=(12, 8))

plt.plot(train[(train.store==1) & (train.item==1)][["sales"]])

plt.plot(test[(test.store==1) & (test.item==1)][["sales"]])

plt.title("Train data and Test predictions for store 1; item 1")

plt.legend(["TRAIN", "TEST"])

plt.show()
test[["id", "sales"]].to_csv("submission.csv", index=False)