import os

import numpy as np

import pandas as pd
def load_housing_data():

    DATA_PATH = '../input/hands-on-machine-learning-housing-dataset/housing.csv'

    data = pd.read_csv(DATA_PATH)

    return data
housing = load_housing_data()
housing.head()
housing.info()
housing['ocean_proximity'].value_counts()
housing.describe()
%matplotlib inline

import matplotlib.pyplot as plt

housing.hist(bins=50, figsize=(20,15))
def split_train_test(data, test_ratio):

    shuffled_indices = np.random.permutation(len(data)) #returns a shuffled numpy array

    test_set_size = int(len(data) * test_ratio)

    test_indices = shuffled_indices[:test_set_size]

    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]



##so the np.random.permutation shuffles the order of the rows in a DataFrame and returns an np array    

#df_test = pd.DataFrame({'column_1':[1,2,3,4], 'column_2':[5,6,7,8]})

#print(df_test)

#np.random.permutation(df_test)

#np.random.permutation(10)
#help(np.random.permutation)

#help(pd.DataFrame.iloc)
train_set, test_set = split_train_test(housing, 0.2)
len(train_set)
len(test_set)
from zlib import crc32

def test_set_check(identifier, test_ratio):

    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32



def split_train_test_by_id(data, test_ratio, id_column):

    ids = data[id_column]

    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))

    return data.loc[~in_test_set], data.loc[in_test_set] # ~ is used to compare binary numbers



## The lambda keyword is used to create small anonymous functions.

## A lambda function can take any number of arguments, but can only have one expression.

## The expression is evaluated and the result is returned.
#help(pd.DataFrame.loc) # Access a group of rows and columns by label(s) or a boolean array.

#help(pd.DataFrame.apply) #apply a function along the axis of a DataFrame
housing_with_id = housing.reset_index() #adds an 'index' column

train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'index')

housing_with_id
housing_with_id["id"] = housing["longitude"]*1000 + housing["latitude"]

train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

housing_with_id
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

print(len(train_set), "/",len(test_set))
#Function below converts bin values into discrete intervals

housing["income_cat"] = pd.cut(housing["median_income"],

                               bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf],

                               labels=[1, 2, 3, 4, 5])

housing
np.inf
#help(pd.cut)
housing["income_cat"].hist()
housing
from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):

    strat_train_set = housing.loc[train_index]

    strat_test_set = housing.loc[test_index]
strat_test_set['income_cat'].value_counts() / len(strat_test_set)
housing['income_cat'].value_counts() / len(housing)
strat_train_set
for set_ in (strat_train_set, strat_test_set):

    set_.drop(labels=["income_cat"], axis=1, inplace=True)
strat_train_set
exp_train_set = strat_train_set.copy() ###explore only the TRAIN set
exp_train_set.plot(kind="scatter", x="longitude", y="latitude")
###`alpha` creates a better visualization, wich highlights high density areas

exp_train_set.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
exp_train_set.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,

                   s=exp_train_set['population']/100, label='population', figsize=(10,7),

                   c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)

plt.legend()
#help(exp_train_set.plot)
corr_matrix = exp_train_set.corr()##returns a DataFrame

corr_matrix
corr_matrix['median_house_value'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix

##below are the most promissing attributes

attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']

scatter_matrix(exp_train_set[attributes], figsize=(12,8))## thiss is a pandas function
exp_train_set.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)
exp_train_set['rooms_per_household'] = exp_train_set['total_rooms'] / exp_train_set['households']

exp_train_set['bedrooms_per_rooms'] = exp_train_set['total_bedrooms'] / exp_train_set['total_rooms']

exp_train_set['population_per_household'] = exp_train_set['population'] / exp_train_set['households']
corr_matrix = exp_train_set.corr()

corr_matrix['median_house_value'].sort_values(ascending=False)
## drop() creates a copy of the data and does not affect strat_train_set

prep_train_set = strat_train_set.drop('median_house_value', axis=1)

prep_train_set_labels = strat_train_set['median_house_value'].copy()

prep_train_set
# Option 1:

# prep_train_set.dropna(subset=["total_bedrooms"])

#

# Option 2:

# prep_train_set.drop(total_bedrooms, axis=1)

#

# Option 3:

# median = prep_train_set['total_bedrooms'].median()

# prep_train_set['total_bedrooms'].fillna(median, inplace=True)
from sklearn.impute import SimpleImputer



imputer = SimpleImputer(strategy="median")



## the line below is necessary because we have to have only numeric values

prep_train_set_num = prep_train_set.drop('ocean_proximity', axis=1)



imputer.fit(prep_train_set_num)## THIS IS A "TRAINNED" IMPUTER

## imputer has simply computed the median of each attribute and stored the result in its statistics_ instance variable

imputer.statistics_



## now i can use this "trained" imputer to transform the training set by replacing missing values with the learned medians:

X = imputer.transform(prep_train_set_num)## The result is a plain numpy array containing the transformed features.

                                         ## If you want tto put it back into a pandas DataFrame, it's simple:



prep_train_set_tr = pd.DataFrame(X, columns=prep_train_set_num.columns,

                                 index=prep_train_set_num.index)





prep_train_set_num.median().values
prep_train_set_tr
imputer.strategy
# prep_train_set_cat = strat_train_set['ocean_proximity'] ## this line creates a Series

prep_train_set_cat = strat_train_set[['ocean_proximity']] ## this one creates a DataFrame

prep_train_set_cat.head(10)
type(prep_train_set_cat)
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()

prep_train_set_cat_encoded = ordinal_encoder.fit_transform(prep_train_set_cat)

prep_train_set_cat_encoded[:10]
ordinal_encoder.categories_
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()

prep_train_set_cat_1hot = cat_encoder.fit_transform(prep_train_set_cat)

prep_train_set_cat_1hot
prep_train_set_cat_1hot.toarray() ## this is a scipy module
# help(prep_train_set_cat_1hot.toarray)
### THE CODE IN THIS CELL I DID NOT UNDERSTAND VERY WELL

from sklearn.base import BaseEstimator, TransformerMixin



rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs

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

prep_train_set_extra_attribs = attr_adder.transform(prep_train_set.values)
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



num_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='median')),

    ('attribs_adder', CombinedAttributesAdder()),

    ('std_scaler', StandardScaler()),

])



prep_train_set_num_transformed = num_pipeline.fit_transform(prep_train_set_num)



########################## mass transformation

from sklearn.compose import ColumnTransformer



num_attribs = list(prep_train_set_num)

cat_attribs = ['ocean_proximity']



full_pipeline = ColumnTransformer([## this is the full pipeline for the data transformation

    ('num', num_pipeline, num_attribs),

    ('cat', OneHotEncoder(), cat_attribs)

])



prep_train_set_prepared = full_pipeline.fit_transform(prep_train_set)
prep_train_set_prepared
prep_train_set_labels
housing_prepared, housing_labels = prep_train_set_prepared, prep_train_set_labels

from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(housing_prepared, housing_labels)
some_data = housing.iloc[:5]

some_labels = housing_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)

print('Predictions: ', lin_reg.predict(some_data_prepared))

print('Labels:', list(some_labels))
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)

lin_mse = mean_squared_error(housing_labels, housing_predictions)

lin_rmse = np.sqrt(lin_mse)

print("Linear, Root Mean Squared Error:", lin_rmse)
from sklearn.tree import DecisionTreeRegressor



tree_reg = DecisionTreeRegressor()

tree_reg.fit(housing_prepared, housing_labels)



housing_predictions = tree_reg.predict(housing_prepared)

tree_mse = mean_squared_error(housing_labels, housing_predictions)

tree_rmse = np.sqrt(tree_mse)

print("Tree, Root Mean Squared Error:", tree_rmse)
from sklearn.model_selection import cross_val_score

tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels,

                         scoring="neg_mean_squared_error", cv=10) ## 10 folds

tree_rmse_scores = np.sqrt(-tree_scores)

tree_scores
def display_scores(scores):

    print('Scores:', scores)

    print('Mean:', scores.mean())

    print('Standard Deviation:', scores.std())



display_scores(tree_rmse_scores)
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,

                             scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()

forest_reg.fit(housing_prepared, housing_labels)



housing_predictions = forest_reg.predict(housing_prepared)

forest_mse = mean_squared_error(housing_labels, housing_predictions)

forest_rmse = np.sqrt(forest_mse)

print("Tree, Root Mean Squared Error:", forest_rmse)
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,

                         scoring="neg_mean_squared_error", cv=10) ## 10 folds

forest_rmse_scores = np.sqrt(-forest_scores)

display_scores(forest_rmse_scores)
import joblib
from sklearn.model_selection import GridSearchCV



param_grid = [

    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},

    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}

]



forest_reg = RandomForestRegressor()



grid_search = GridSearchCV(forest_reg, param_grid, cv=5,

                           scoring='neg_mean_squared_error',

                           return_train_score=True)



grid_search.fit(housing_prepared, housing_labels)
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):

    print(np.sqrt(-mean_score), params)
feature_importances = grid_search.best_estimator_.feature_importances_

feature_importances
extra_attribs = ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_rooms']

cat_encoder = full_pipeline.named_transformers_['cat']

cat_one_hot_attribs = list(cat_encoder.categories_[0])

attributes = num_attribs + extra_attribs + cat_one_hot_attribs

sorted(zip(feature_importances, attributes), reverse=True)
final_model = grid_search.best_estimator_



X_test = strat_test_set.drop('median_house_value', axis=1)

y_test = strat_test_set["median_house_value"].copy()



X_test_prepared = full_pipeline.transform(X_test)



final_predictions = final_model.predict(X_test_prepared)



final_mse = mean_squared_error(y_test, final_predictions)

final_rmse = np.sqrt(final_mse)

final_rmse
from scipy import stats

confidence = 0.95

squared_errors = (final_predictions - y_test) ** 2

np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,

                         loc=squared_errors.mean(),

                         scale=stats.sem(squared_errors)))