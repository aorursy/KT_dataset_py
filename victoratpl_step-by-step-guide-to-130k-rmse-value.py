# Disable warnings

import warnings

warnings.filterwarnings('ignore')



# Install required packages

!pip install wheel matplotlib pandas scikit-learn xgboost seaborn hyperopt
# Import libraries and set settings of seaborn (lib for nicer plotting)

from math import sqrt



from hyperopt import hp, tpe, STATUS_OK, Trials

from hyperopt.fmin import fmin

import matplotlib

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns;

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import MinMaxScaler

from tabulate import tabulate

import xgboost as xgb



sns.set(rc={'figure.figsize': (11.7, 8.27)})
# Specify the filename

file_name = "../input/kc_house_data.csv"



# Read dataset to pandas dataframe (also parse date to datetime object and set column #0 as index for dataframe)

df = pd.read_csv(file_name, parse_dates=[1], date_parser=lambda x: pd.to_datetime(x, format='%Y%m%dT000000'), index_col=0)
# Brief overview of the dataset

df.head()
# How many examples and features do we have in the dataset?

df.shape
# What are the types of the features?

df.dtypes
# Check if we need to handle with some NA values in dataframe

df.apply(lambda x: sum(x.isnull()) / len(df))
# Show basic statistics about our dataframe's columns (columns are ours features)

df.describe()
# Plot histograms of the features

df.hist(bins=50, figsize=(20, 15));
# Plot the histogram of pricing dates grouped by years and months

df['date'].groupby([df["date"].dt.year, df["date"].dt.month]).count().plot(kind="bar");
# Replace date feature with new ones

df['pricing_yr'] = df['date'].dt.year

df['pricing_month'] = df['date'].dt.month

df['pricing_yrmonth'] = df['date'].dt.year.map(str) + df['date'].dt.month.map('{:02d}'.format)

df['pricing_yrmonth'] = df['pricing_yrmonth'].astype(int)

df = df.drop(['date'], axis=1)
# Create a dataframe for showing purposes with feature name and feature type

feature_type = ["discrete", "discrete", "continuous", "continuous", "discrete",

                "dichotomous", "ordinal", "ordinal", "ordinal", "continuous",

                "continuous", "discrete", "discrete", "discrete", "continuous", "continuous",

                "continuous", "continuous", "discrete", "ordinal", "discrete*"]

feature_array = df.drop(['price'], axis=1).columns.values

pd.DataFrame({"feature": feature_array, "type": feature_type})
# Plot the correlation of every two variables

sns.pairplot(df);
# Remove the outlier

df = df[df["bedrooms"] < 20]
sns.jointplot(x="bedrooms", y="price", data=df, kind="reg");
# Count the pearson correlation of features

corr_matrix = df.corr()

corr_matrix["price"].sort_values(ascending=False)
# Source: https://seaborn.pydata.org/examples/many_pairwise_correlations.html



# Compute the correlation matrix

# Plot figsize

fig, ax = plt.subplots(figsize=(20, 20))

# Generate Color Map

colormap = sns.diverging_palette(220, 10, as_cmap=True)

# Generate Heat Map, allow annotations and place floats in map

sns.heatmap(corr_matrix, cmap=colormap, annot=True, fmt=".2f")

# Apply xticks

plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns);

# Apply yticks

plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)

# Show plot

plt.show()
# Remove pricing and zipcode features

df = df.drop(['pricing_yr', 'pricing_yrmonth', 'pricing_month', 'zipcode'], axis=1)
# Prepare data for model

Y = df['price'].values

X = df.drop(['price'], axis=1).values



# Split dataset on: train (for cross-validation) and test (hold-out) sets 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
# Simple naive model with Cross-Validation on train set

root_mean_squared_errors = []



# When we compare models it's good to use CV, because it's going to be robust for randomness of train/val split.

kf = KFold(n_splits=10)

for train_indices, val_indices in kf.split(X_train):

    predictions = [Y_train[train_indices].mean()] * Y_train[val_indices].shape[0]

    actuals = Y_train[val_indices]

    root_mean_squared_errors.append(sqrt(mean_squared_error(actuals, predictions)))



rmse = np.mean(root_mean_squared_errors)

print("average of CV rmse score: {0:.0f}".format(rmse))
# Train Linear Regression with Cross-Validation on train set

root_mean_squared_errors = []



kf = KFold(n_splits=10)

for train_indices, val_indices in kf.split(X_train):

    linreg_model = LinearRegression(n_jobs=-1).fit(X_train[train_indices], Y_train[train_indices])

    predictions = linreg_model.predict(X_train[val_indices])

    actuals = Y_train[val_indices]

    root_mean_squared_errors.append(sqrt(mean_squared_error(actuals, predictions)))



rmse = np.mean(root_mean_squared_errors)

print("average of CV rmse score: {0:.0f}".format(rmse))
# Train RandomForest model with Cross-Validation on train set

root_mean_squared_errors = []



kf = KFold(n_splits=10)

for train_indices, val_indices in kf.split(X_train):

    rf_model = RandomForestRegressor(n_jobs=-1).fit(X_train[train_indices], Y_train[train_indices])

    predictions = rf_model.predict(X_train[val_indices])

    actuals = Y_train[val_indices]

    root_mean_squared_errors.append(sqrt(mean_squared_error(actuals, predictions)))



rmse = np.mean(root_mean_squared_errors)

print("average of CV rmse score: {0:.0f}".format(rmse))
# Train RandomForest model with Cross-Validation on train set

root_mean_squared_errors = []



kf = KFold(n_splits=10)

for train_indices, val_indices in kf.split(X_train):

    rf_model = RandomForestRegressor(n_jobs=-1, n_estimators=100).fit(X_train[train_indices], Y_train[train_indices])

    predictions = rf_model.predict(X_train[val_indices])

    actuals = Y_train[val_indices]

    root_mean_squared_errors.append(sqrt(mean_squared_error(actuals, predictions)))



rmse = np.mean(root_mean_squared_errors)

print("average of CV rmse score: {0:.0f}".format(rmse))
# Train XGBRegressor model with Cross-Validation on train set

root_mean_squared_errors = []



kf = KFold(n_splits=10)

for train_indices, val_indices in kf.split(X_train):

    xgb_model = xgb.XGBRegressor(n_jobs=-1, n_estimators=300).fit(X_train[train_indices], Y_train[train_indices])

    predictions = xgb_model.predict(X_train[val_indices])

    actuals = Y_train[val_indices]

    root_mean_squared_errors.append(sqrt(mean_squared_error(actuals, predictions)))



rmse = np.mean(root_mean_squared_errors)

print("average of CV rmse score: {0:.0f}".format(rmse))
# Train MLP model with Cross-Validation on train set



# Do rescaling as NN models like features from 0-1 range

scaler = MinMaxScaler(copy=True)

scaler.fit(X_train)



# Enscapsulate model training and evaluation in function

def train_mlp(X_train, Y_train, params={}, n_splits=10):

    root_mean_squared_errors_train = []

    root_mean_squared_errors = []



    kf = KFold(n_splits=n_splits)

    for train_indices, val_indices in kf.split(X_train):

        mlp_model = MLPRegressor(**params).fit(X_train[train_indices], Y_train[train_indices])

        predictions = mlp_model.predict(X_train[val_indices])

        actuals = Y_train[val_indices]

        root_mean_squared_errors_train.append(sqrt(mlp_model.loss_))

        root_mean_squared_errors.append(sqrt(mean_squared_error(actuals, predictions)))



    rmse_train = np.mean(root_mean_squared_errors_train)

    rmse_valid = np.mean(root_mean_squared_errors)

    

    return rmse_train, rmse_valid

    

X_train_scaled = scaler.transform(X_train)



rmse_train, rmse_valid = train_mlp(X_train_scaled, Y_train, n_splits=2)

print("average of CV rmse score on train set: {0:.0f}".format(rmse_train))

print("average of CV rmse score on val set: {0:.0f}".format(rmse_valid))
def objective(space):

    hidden_layers = tuple(int(space['hidden_layers']) * [int(space['hidden_neurons'])])

    

    rmse_train, rmse_valid = train_mlp(X_train_scaled, 

                                       Y_train, 

                                       params={'solver': 'adam',

                                               'hidden_layer_sizes': hidden_layers,

                                               'activation': space['activation'],

                                               'shuffle': True,

                                               'max_iter': int(space['max_iter']),

                                               'learning_rate_init': space['learning_rate_init'],

                                               'verbose': False},

                                       n_splits=2)



    return {'loss': rmse_valid, 'status': STATUS_OK}



activation_functions = ['relu', 'tanh', 'logistic']

space = {

    'activation': hp.choice('activation', activation_functions),

    'hidden_neurons': hp.quniform('hidden_neurons', 10, 50, 10),

    'hidden_layers': hp.quniform('hidden_layers', 2, 4, 1),

    'learning_rate_init': hp.loguniform('learning_rate_init', np.log(0.001), np.log(0.01)),

    'alpha': hp.loguniform('alpha', np.log(0.01), np.log(0.1)),

    'max_iter': hp.quniform('max_iter', 100, 750, 50)

}



trials = Trials()

best = fmin(fn=objective,

            space=space,

            algo=tpe.suggest,

            max_evals=15,

            trials=trials)
sns.lineplot(x=np.arange(0, len(trials.losses())), y=trials.losses());
best
best['activation'] = activation_functions[best['activation']]



hidden_layers = tuple(int(best['hidden_layers']) * [int(best['hidden_neurons'])])

rmse_train, rmse_valid = train_mlp(X_train_scaled, Y_train, 

                                   params={'solver': 'adam',

                                           'hidden_layer_sizes': hidden_layers,

                                           'activation': best['activation'],

                                           'shuffle': True,

                                           'max_iter': int(best['max_iter']),

                                           'learning_rate_init': best['learning_rate_init'],

                                           'verbose': False})



print("average of CV rmse score on train set: {0:.0f}".format(rmse_train))

print("average of CV rmse score on val set: {0:.0f}".format(rmse_valid))
# Train RandomForest model with Cross-Validation on train set

root_mean_squared_errors = []



rf_model = RandomForestRegressor(n_jobs=-1, n_estimators=300).fit(X_train, Y_train)

predictions = rf_model.predict(X_test)

actuals = Y_test

root_mean_squared_errors.append(sqrt(mean_squared_error(actuals, predictions)))



rmse = np.mean(root_mean_squared_errors)

print("average of CV rmse score on test set: {0:.0f}".format(rmse))