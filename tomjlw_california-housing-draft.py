import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt
def read_csv(path = "/kaggle/input/california-housing-prices/housing.csv"):

    return pd.read_csv(path)
housing = read_csv()
housing.describe()
housing.info()
%matplotlib inline

housing.hist(bins=50, figsize=(20,15))
np.random.seed(42)
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
test_set.head()
housing["median_income"].head()
housing["median_income"].hist()
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3., 4.5, 6., np.inf],

                              labels=[1, 2, 3, 4, 5])
housing["income_cat"].hist()
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):

    strat_train_set = housing.loc[train_index]

    strat_test_set = housing.loc[test_index]
strat_train_set.drop("income_cat", axis=1, inplace=True)

strat_test_set.drop("income_cat", axis=1, inplace=True)
housing = strat_train_set.copy()

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,

    s=housing["population"]/100, label="population", figsize=(10,7),

    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,

    sharex=False)

plt.legend()
corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)
from pandas.plotting import scatter_matrix



attributes = ["median_house_value", "median_income", "total_rooms",

              "housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12, 8))
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]

housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]

housing["population_per_household"]=housing["population"]/housing["households"]
corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)
housing.describe()
housing = strat_train_set.drop("median_house_value", axis=1).copy()

housing_labels = strat_train_set["median_house_value"].copy()
print(housing.isnull().any(axis=1))

sample_incomplete_rows = housing[housing.isnull().any(axis=1)]

sample_incomplete_rows["total_bedrooms"].fillna(housing["total_bedrooms"].median, inplace=True)
from sklearn.preprocessing import OneHotEncoder

housing_cat = housing[['ocean_proximity']]

housing_cat_new = OneHotEncoder(sparse=False).fit_transform(housing_cat)
print(housing)
from sklearn.base import BaseEstimator, TransformerMixin



# get the right column indices: safer than hard-coding indices 3, 4, 5, 6

rooms_ix, bedrooms_ix, population_ix, household_ix = [

    list(housing.columns).index(col)

    for col in ("total_rooms", "total_bedrooms", "population", "households")]



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room = True): # no *args or **kwargs

        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):

        return self  # nothing else to do

    def transform(self, X, y=None):

        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]

        population_per_household = X[:, population_ix] / X[:, household_ix]

        if self.add_bedrooms_per_room:

            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]

            return np.c_[X, rooms_per_household, population_per_household,

                         bedrooms_per_room]

        else:

            return np.c_[X, rooms_per_household, population_per_household]



attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)

housing_extra_attribs = attr_adder.transform(housing.values)
housing_extra_attribs = pd.DataFrame(

    housing_extra_attribs,

    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],

    index=housing.index)

housing_extra_attribs.head()
from sklearn.preprocessing import StandardScaler

#StandardScaler.fit_transform(housing_extra_attribs, housing_extra_attribs)
housing_num = housing.drop('ocean_proximity', axis=1)

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import FunctionTransformer



def add_extra_features(X, add_bedrooms_per_room=True):

    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]

    population_per_household = X[:, population_ix] / X[:, household_ix]

    if add_bedrooms_per_room:

        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]

        return np.c_[X, rooms_per_household, population_per_household,

                     bedrooms_per_room]

    else:

        return np.c_[X, rooms_per_household, population_per_household]



num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),

        ('std_scaler', StandardScaler()),

    ])



housing_num_tr = num_pipeline.fit_transform(housing_num)
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)

cat_attribs = ["ocean_proximity"]



full_pipeline = ColumnTransformer([

        ("num", num_pipeline, num_attribs),

        ("cat", OneHotEncoder(), cat_attribs),

    ])



housing_prepared = full_pipeline.fit_transform(housing)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(housing_prepared, housing_labels)
from sklearn.metrics import mean_squared_error



housing_predictions = lin_reg.predict(housing_prepared)

lin_mse = mean_squared_error(housing_labels, housing_predictions)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
from sklearn.metrics import mean_absolute_error



lin_mae = mean_absolute_error(housing_labels, housing_predictions)

lin_mae
from sklearn.tree import DecisionTreeRegressor



tree_reg = DecisionTreeRegressor(random_state=42)

tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)

tree_mse = mean_squared_error(housing_labels, housing_predictions)

tree_rmse = np.sqrt(tree_mse)

tree_rmse
from sklearn.model_selection import cross_val_score



scores = cross_val_score(tree_reg, housing_prepared, housing_labels,

                         scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):

    print("Scores:", scores)

    print("Mean:", scores.mean())

    print("Standard deviation:", scores.std())



display_scores(tree_rmse_scores)
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,

                             scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)

forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)

forest_mse = mean_squared_error(housing_labels, housing_predictions)

forest_rmse = np.sqrt(forest_mse)

forest_rmse
from sklearn.model_selection import cross_val_score



forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,

                                scoring="neg_mean_squared_error", cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)

display_scores(forest_rmse_scores)
from sklearn.svm import SVR



svm_reg = SVR(kernel="linear")

svm_reg.fit(housing_prepared, housing_labels)

housing_predictions = svm_reg.predict(housing_prepared)

svm_mse = mean_squared_error(housing_labels, housing_predictions)

svm_rmse = np.sqrt(svm_mse)

svm_rmse
from sklearn.model_selection import GridSearchCV



param_grid = [

    # try 12 (3×4) combinations of hyperparameters

    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},

    # then try 6 (2×3) combinations with bootstrap set as False

    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},

  ]



forest_reg = RandomForestRegressor(random_state=42)

# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,

                           scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)

grid_search.best_params_
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint



param_distribs = {

        'n_estimators': randint(low=1, high=200),

        'max_features': randint(low=1, high=8),

    }



forest_reg = RandomForestRegressor(random_state=42)

rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,

                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)

rnd_search.fit(housing_prepared, housing_labels)
cvres = rnd_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
feature_importances = grid_search.best_estimator_.feature_importances_

feature_importances
final_model = grid_search.best_estimator_



X_test = strat_test_set.drop("median_house_value", axis=1)

y_test = strat_test_set["median_house_value"].copy()



X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)



final_mse = mean_squared_error(y_test, final_predictions)

final_rmse = np.sqrt(final_mse)
final_rmse