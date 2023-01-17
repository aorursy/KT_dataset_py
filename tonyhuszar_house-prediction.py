#import sys
#!{sys.executable} -m pip install catboost
# Standard imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Machine learning libraries
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
# Open the data
df = pd.read_csv("house-prices-advanced/train.csv")
df.head()
df.set_index('Id', inplace=True)
df.head()
df.info()
df['YrSold'].head()
len(df)
df.describe().T
df.isna().sum()
df.isnull().sum()
df.duplicated().sum()
df.head()
for label, content in df.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(pd.isnull(content).sum())
            print(label)
df['LotFrontage'].value_counts()
df['MasVnrArea'].value_counts()
df['GarageYrBlt']
for label, content in df.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # Add binary column to tell if data was missing
            df[label+"_is_missing"] = pd.isnull(content)
            # Fill missing numeric values with median
            df[label] = content.fillna(content.median())
# Checking for missing values again
for label, content in df.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(pd.isnull(content).sum())
            print(label)
# Now let's clean our string data
for label, content in df.items():
    if not pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(pd.isnull(content).sum())
            print(label)
df['PoolQC'].value_counts()
df['PoolQC']
pd.Categorical(df['Neighborhood']).codes
for label, content in df.items():
    if not pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # Add binary column
            df[label+"_is_missing"] = pd.isnull(content)
        # Transform strings into codes
        df[label] = pd.Categorical(content).codes+1
pd.isnull(df).sum()
# Now let's clean our string data
for label, content in df.items():
    if not pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(pd.isnull(content).sum())
            print(label)
df.head()
df['YrSold'].value_counts()
for label, content in df.items():
    if not pd.api.types.is_numeric_dtype(content):
        df[label] = content.astype("category").cat.as_ordered()
df.info()
# Let's split our data 
X = df.drop("SalePrice", axis=1)
y = df['SalePrice']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate model
model = XGBRegressor()

# Fit the model
model.fit(X_train, y_train)
# Score the model
model.score(X_test, y_test)
df_tmp = df.copy() # making a copy of the original data to avoid complications
df_tmp
# Let's create a function that is using all these models
# But first... import!
def model_testing():
    """We are testing 4 different models and progressing with the best 2."""
    np.random.seed(42)
    
    scores = {}
    
    models = {'XGBoost': XGBRegressor(),
             'RandomForestRegressor': RandomForestRegressor(),
             'LinearRegression': LinearRegression(),
             'CatBoostRegressor': CatBoostRegressor(),}
    
    for k, v in models.items():
        scores[k] = v.fit(X_train, y_train).score(X_test, y_test)
        
    return scores
model_testing()
# Hyperparameter grids
xgbr_grid = {"learning_rate" : [0.05, 0.15, 0.30],
 "max_depth": [3, 6, 15],
 "min_child_weight": [1, 4, 7],
 "gamma": [0.0, 0.2, 0.4],
 "colsample_bytree": [0.3, 0.5, 0.7],}

catboostr_grid = {"iterations": [100, 300, 500],
                 "learning_rate":[0.01, 0.02, 0.03],
                 "depth": [1, 3, 6],
                 "l2_leaf_reg": [1, 2, 3],
                 "border_count": [1, 16, 32],}
# Use RandomizedSearchCV for grids
xgbr_model = RandomizedSearchCV(XGBRegressor(),
                               param_distributions=xgbr_grid,
                               n_iter=20,
                               cv=5,
                               verbose=True)

catboostr_model = RandomizedSearchCV(CatBoostRegressor(),
                                    param_distributions=catboostr_grid,
                                    n_iter=20,
                                    cv=5,
                                    verbose=True)

#xgbr_model.fit(X_train, y_train)
#catboostr_model.fit(X_train, y_train)
#xgbr_model.best_params_
#catboostr_model.best_params_
# Let's compare XGBRegression vs CatBoostRegression model's best parameter.
%time
#print(xgbr_model.score(X_test, y_test))
#print(catboostr_model.score(X_test, y_test))
df.head()
# First off, let's create a copy of our data.
df_copy = df.copy()
df_copy.head()
df_copy['MSSubClass']
df_copy['SaleType'].value_counts()
df_copy['Utilities']
# It all seems to be fine to me. I am confused about what else to try.
# Let's do some graphing now maybe?
df_copy['PoolArea'].hist()
plt.xlim([-10, 80])
df_copy['PoolArea'].value_counts()
# Build a correlational matrix
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df_copy[:25].corr(),
                     linewidths=0.75,
                     fmt='.2f',
                     cmap='YlGnBu',
                     annot=True)
df_copy
# Let's predict on the test data.
test_df = pd.read_csv("house-prices-advanced/test.csv")
test_df.head()
test_df.set_index('Id', inplace=True)
# Now it works for some reason..
test_df.head()
#test_df['SalePrice'] it doesn't exist so that's good.
set(df_copy) - set(test_df)
def preprocess_data(df):
    # Fill numeric rows with the median
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                df[label+"_is_missing"] = pd.isnull(content)
                df[label] = content.fillna(content.median())
        
        # Turn categorical variables into numbers
        if not pd.api.types.is_numeric_dtype(content):
            df[label+"_is_missing"] = pd.isnull(content)
            
            # We add the +1 because pandas encodes missing categories as -1
            df[label] = pd.Categorical(content).codes+1
    
    return df
df_test = preprocess_data(test_df)
df_test.head()
set(df_test.columns) - set(df_copy.columns)
X_train.head()
df_tmp.head()
test_df.head()
# Something weird is happening. I think the test data was missing more than 
# train data. 
def missing_columns(df):
    total_missing = 0
    for label, content in df.items():
        if "_is_missing" in label:
            print("Column " + label + " had missing data.")
            total_missing += 1
    return f'{total_missing} columns had missing data.'
missing_columns(test_df)
missing_columns(df_copy)
# Make a function which adds missing columns to the train data or vice versa.
def balance_columns(test_data, train_data):
    for i in set(test_data) - set(train_data):
        train_data[i] = False
balance_columns(test_df, df_copy)
df_copy.head()
test_df.head()
# Make predictions on the test dataset using the best model
#test_preds = xgbr_model.predict(test_df)
# I am going to retrain my data.
np.random.seed(42)

X = df_copy.drop("SalePrice", axis=1)
y = df_copy['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

xgbr_model.fit(X_train, y_train)
xgbr_model.best_params_
ideal_model = XGBRegressor(min_child_weight=4,
                         max_depth=15,
                         learning_rate=0.05,
                         gamma=0.4,
                         colsample_bytree=0.5)

ideal_model.fit(X_train, y_train)
# Let's try to make the prediction again.
#test_preds = ideal_model.predict(test_df)
# Let's match the column names from both datasets
test_df.head()
df_copy.head()
set(df_copy) - set(test_df)
cols_when_model_builds = ideal_model.get_booster().feature_names
cols_when_model_builds
test_df = test_df[cols_when_model_builds]
test_df
# Let's try to make the prediction again. Part 2
test_preds = ideal_model.predict(test_df) # please work..
# Let's create a dataframe where we can store the results.
df_preds = pd.DataFrame()
df_preds['ID'] = test_df.index
df_preds['SalePrice'] = test_preds
df_preds.head()
# Export to csv..
df_preds.to_csv('house-prices-advanced/predictions_1.csv',
               index=False)
# Find feature importance of our besgt model
ideal_model.feature_importances_
# Helper function for plotting feature importance
def plot_features(columns, importances, n=20):
    df = (pd.DataFrame({"features": columns,
                       "feature_importance": importances})
         .sort_values("feature_importance", ascending=False)
         .reset_index(drop=True))
    
    sns.barplot(x="feature_importance",
               y='features',
               data=df[:n],
               orient="h")
plot_features(X_train.columns, ideal_model.feature_importances_)
sum(ideal_model.feature_importances_)
# Seems like GarageCars is one of the most important columns.
# GarageCars: Size of garage in car capacity.
# Wondering why that is the most important?
# Second place to it is OverallQual though.
