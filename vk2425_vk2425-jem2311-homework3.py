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
df_orig = pd.read_csv('/kaggle/input/craigslist-carstrucks-data/vehicles.csv')
df_orig
# Get subsample of data
df = df_orig.sample(frac=0.5,axis=0)
df.columns
# Paint color correlation
import seaborn as sb
plt.xticks(rotation=90)
ax = sb.scatterplot(x="paint_color", y="price", data=df_orig)
# Latitude correlation
ax = sb.scatterplot(x="lat", y="price", data=df_orig)
# Longitude correlation
ax = sb.scatterplot(x="long", y="price", data=df_orig)
# State correlation
plt.xticks(rotation=90)
ax = sb.scatterplot(x="state", y="price", data=df_orig)
# Region correlation
ax = sb.scatterplot(x="region", y="price", data=df_orig)

df = df.drop(columns = ['id',
                           'url', 
                           'region', 
                           'region_url',
                           'title_status', 
                           'size', 
                           'description', 
                           'vin', 
                           'lat', 
                           'long', 
                           'image_url',
                           'county',
                           'state',
                           'model',
                           'paint_color'])
df
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
# Drop where nan is present
df = df.dropna()
# Cut price outliers
df = df[df['price'] > 1000]
df = df[df['price'] < 40000]
# Feature names
continuous_features = ['year', 'odometer']
categorical_features = ['drive', 'type', 'fuel', 'transmission', 'manufacturer', 'condition', 'cylinders']
# X and y
y = df['price']
X = df.drop(columns=['price'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# Minimum Preprocessing
def train_and_eval_classifier(classifier, cname):
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)])
    
    pipe = make_pipeline(preprocessor, classifier)
    scores = cross_val_score(pipe, X_train, y_train, scoring="r2")
    print("CV Score for {}: {}".format(cname, np.mean(scores)))
# Minimum viable models
# Try a couple of basic classifiers:
train_and_eval_classifier(LinearRegression(), "LinearRegression")
train_and_eval_classifier(Lasso(max_iter=6000), "Lasso")
train_and_eval_classifier(ElasticNet(), "ElasticNet")
df = df_orig.sample(frac=0.5,axis=0)
df = df.drop(columns = ['id',
                   'url', 
                   'region', 
                   'region_url',
                   'title_status', 
                   'size', 
                   'description', 
                   'vin', 
                   'lat', 
                   'long', 
                   'image_url',
                   'county',
                   'state',
                   'model',
                   'paint_color'])
df
# Drop only null rows instead of entirely dropping null values
# We will fix null with imputation later
df = df.dropna(thresh=10)
# Cut continuous outliers on quantiles instead

pl = df.price.quantile(0.1)
pu = df.price.quantile(0.99)

ol = df.odometer.quantile(0.1)
ou = df.odometer.quantile(0.99)

yl = df.year.quantile(0.1)
yu = df.year.quantile(0.99)

df = df[df.price > pl]
df = df[df.price < pu]

df = df[df.odometer > ol]
df = df[df.odometer < ou]

df = df[df.year > yl]
df = df[df.year < yu]
# Feature names
continuous_features = ['year', 'odometer']
categorical_features = ['drive', 'type', 'fuel', 'transmission', 'manufacturer', 'condition', 'cylinders']
# Extract X and y
y = df['price']
X = df.drop(columns=['price'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# Preprocessing
# Added Polynomial features
# Added StandardScaler
# Added imputation for continuous and categorical

def train_and_eval_classifier(classifier, cname):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('polyfeatures', PolynomialFeatures()),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNK')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, continuous_features),
            ('cat', categorical_transformer, categorical_features)])
    
    pipe = make_pipeline(preprocessor, classifier)
    scores = cross_val_score(pipe, X_train, y_train, scoring="r2")
    print("CV Score for {}: {}".format(cname, np.mean(scores)))
# Try a couple of classifiers to see if we actually improved
train_and_eval_classifier(LinearRegression(), "LinearRegression")
train_and_eval_classifier(ElasticNet(), "ElasticNet")
import xgboost as xgb
# XGBoost
def train_and_eval_xgb():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('polyfeatures', PolynomialFeatures()),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNK')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, continuous_features),
            ('cat', categorical_transformer, categorical_features)])
    
    
    pipe = make_pipeline(preprocessor, xgb.XGBRegressor(objective='reg:squarederror')) 
    
    # Parameter tuning
    param_grid = {'xgbregressor__n_estimators': [100, 120, 140],
                  'xgbregressor__learning_rate': [0.01, 0.1],
                  'xgbregressor__max_depth': [5, 7]}
    
    xgb_grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, return_train_score=True, n_jobs=-1)
    xgb_grid.fit(X_train, y_train) 
    print("Best score: %0.3f" % xgb_grid.best_score_) 
    print("Best parameters set:", xgb_grid.best_params_)
train_and_eval_xgb()
from sklearn.feature_selection import SelectFromModel
def find_influential_features():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNK')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, continuous_features),
            ('cat', categorical_transformer, categorical_features)])
    
    x = xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.1, max_depth=7, n_estimators=140)
    pipe = make_pipeline(preprocessor, x)
    
    scores = cross_val_score(pipe, X_train, y_train, scoring="r2")
    print("CV Score for {}: {}".format("XGBoost", np.mean(scores)))
    
    feature_sel = SelectFromModel(pipe, 1e-5)
    feature_sel.fit(X_train, y_train)
    
    return feature_sel, pipe
# Inf features, baseline score for XGBoost
inf_features, pipe = find_influential_features()
# Get feature importances
important_features = inf_features.estimator_.named_steps['xgbregressor'].feature_importances_
# Get caregorical feature names
cat_feature_names = pipe.named_steps['columntransformer'].transformers[1][1].steps[1][1].fit(X_train[categorical_features], y_train).get_feature_names()
# Add continuous feature names
feature_names = np.concatenate((continuous_features, cat_feature_names))
feature_names
important_features
# Twenty most important features
top_twenty = important_features.argsort()[-20:][::-1]
tt_fv = [important_features[i] for i in top_twenty]
tt_fn = [feature_names[i] for i in top_twenty]
# Display 20 most important features
import matplotlib.pyplot as plt
plt.scatter(tt_fn, tt_fv)
plt.xticks(tt_fn, tt_fn, rotation='vertical')
plt.show()
# Create Preprocessor
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='UNK')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, continuous_features),
        ('cat', categorical_transformer, categorical_features)])

preprocessor.fit(X_train, y_train)
# Run preprocessing, and only select dataset of 50 top features.
top_50 = important_features.argsort()[-50:][::-1]
X_train_trans = pd.DataFrame(preprocessor.transform(X_train).toarray())
X_train_trans = X_train_trans[top_50]
X_train_trans
# Retrain and evaluate model on new dataset.
x = xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.1, max_depth=7, n_estimators=140)
np.mean(cross_val_score(x, X_train_trans, y_train, scoring="r2"))
# Best score, from part 4
# Compute the marginal score of features
best_score = 0.862

total_features = continuous_features + categorical_features
for i, feature in enumerate(total_features):
    total_features_n = total_features.copy()
    total_features_n.remove(feature)
    
    if i < 2:
        cont_n = [total_features_n[0]]
        cat_n = total_features_n[1:]
    else:
        cont_n = total_features_n[:2]
        cat_n = total_features_n[2:]
    
    X_train_cut_col = X_train[total_features_n]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, cont_n),
            ('cat', categorical_transformer, cat_n)])

    preprocessor.fit(X_train_cut_col, y_train)
    X_train_cut_col_trans = pd.DataFrame(preprocessor.transform(X_train_cut_col).toarray())

    x = xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.1, max_depth=7, n_estimators=140)
    score = np.mean(cross_val_score(x, X_train_cut_col_trans, y_train, scoring="r2"))
    
    marginal_score = best_score - score
    print("Marginal score of column {}: {}".format(feature, marginal_score))
# From previous exploration, we can see that certain columns are less relevant
X_train_cut_col = X_train[["cylinders", "fuel", "year", "odometer"]]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['year', 'odometer']),
        ('cat', categorical_transformer, ["cylinders", "fuel"])])

preprocessor.fit(X_train_cut_col, y_train)
X_train_cut_col_trans = pd.DataFrame(preprocessor.transform(X_train_cut_col).toarray())

x = xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.1, max_depth=7, n_estimators=140)
np.mean(cross_val_score(x, X_train_cut_col_trans, y_train, scoring="r2"))
minimum_model = X_train_cut_col_trans.shape[1]
print("MINIMUM IMPORTANT FEATURES/COEFFICIENTS NEEDED: {}".format(minimum_model))
from sklearn.ensemble import RandomForestRegressor
# Attempt a random forests regressor with only 15 leaves.
def evaluate_minimum_random_forests():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNK')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, continuous_features),
            ('cat', categorical_transformer, categorical_features)])


    pipe = make_pipeline(preprocessor, RandomForestRegressor(max_leaf_nodes=minimum_model))    
    
    scores = cross_val_score(pipe, X_train, y_train, scoring="r2")
    print("CV Score for {}: {}".format("Minimum Random Forests Regressor", np.mean(scores)))
evaluate_minimum_random_forests()