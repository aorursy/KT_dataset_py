import numpy as np

import pandas as pd



train_data_path = '/kaggle/input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(train_data_path)

home_data.describe()
corr_matrix = home_data.corr()

corr_matrix["SalePrice"].sort_values(ascending=False)
home_data = home_data.drop(['MSSubClass','OverallCond','YrSold','LowQualFinSF','Id','MiscVal','BsmtHalfBath','BsmtFinSF2','3SsnPorch','MoSold','PoolArea'],axis=1)
from sklearn.model_selection import train_test_split



train_set, test_set = train_test_split(home_data, test_size=0.2, random_state=42)
#train_X, train_y

train_X = train_set.drop("SalePrice", axis=1)

train_y = train_set["SalePrice"].copy()

#test_X, test_y

test_X = train_set.drop("SalePrice", axis=1)

test_y = train_set["SalePrice"].copy()
def get_numeric(data):

    num = [key for key in dict(data.dtypes)

                   if dict(data.dtypes)[key]

                       in ['float64','float32','int32','int64']]

    return data[num]
train_X_num = get_numeric(train_X)

train_X_num.head()
def get_cat(data):

    cat = [key for key in dict(data.dtypes)

             if dict(data.dtypes)[key] in ['object']]

    return data[cat]
train_X_cat = get_cat(train_X)

train_X_cat.head()
#pipeline for numerical values

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



num_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='median')),

    ('std_scaler', StandardScaler()),

])

housing_prepared = num_pipeline.fit_transform(train_X_num)

housing_labels = train_y
# #num/cat fullpipeline

# from sklearn.preprocessing import OrdinalEncoder

# from sklearn.compose import ColumnTransformer



# full_pipeline = ColumnTransformer([

#     ('num', num_pipeline, numeric_var),

#     ('cat_imputer', SimpleImputer(),)

#     ('cat', OrdinalEncoder(), cat_var),

# ])

# housing_prepared = full_pipeline.fit_transform(train_X)
from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor(random_state=1)

model.fit(housing_prepared, housing_labels)
from sklearn.metrics import mean_absolute_error



preds = model.predict(housing_prepared)

mae = mean_absolute_error(preds, housing_labels)

mae

from sklearn.model_selection import cross_val_score



scores = cross_val_score(model, housing_prepared, housing_labels, cv=10, scoring='neg_mean_absolute_error')
def display_scores(scores):

    print("scores:", scores)

    print("mean:", scores.mean())

    print("standard deviation:",scores.std())

display_scores(-scores)
# from sklearn.model_selection import GridSearchCV



# param_grid = [

#     {'n_estimators':[3, 10, 30,100], 'max_features': [2,4,6,8,16,26], 'max_leaf_nodes': [5,10,50,100]},

# ]

# model = RandomForestRegressor(random_state=1)

# grid_search = GridSearchCV(model, param_grid, cv=10,

#                           scoring='neg_mean_absolute_error',

#                           return_train_score=True)

# grid_search.fit(housing_prepared, housing_labels)
# grid_search.best_params_
# cvres = grid_search.cv_results_

# for mean_score, params in zip(cvres['mean_test_score'],cvres['params']):

#     print(-mean_score, params)
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint



param_distribs = {

        'n_estimators': randint(low=1, high=200),

        'max_features': randint(low=1, high=26),

        'max_leaf_nodes': randint(low=1, high=500),

    }



forest_reg = RandomForestRegressor(random_state=42)

rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,

                                n_iter=50, cv=10, scoring='neg_mean_absolute_error', random_state=42)

rnd_search.fit(housing_prepared, housing_labels)
rnd_search.best_params_
cvres = rnd_search.cv_results_

for mean_score, params in zip(cvres['mean_test_score'],cvres['params']):

    print(-mean_score, params)
test_data_path = '/kaggle/input/home-data-for-ml-course/test.csv'



full_test_data = pd.read_csv(test_data_path)

num_test_data = full_test_data[train_X_num.columns]

final_model = rnd_search.best_estimator_



housing_test = num_pipeline.transform(num_test_data)

final_predictions = final_model.predict(housing_test)

final_predictions





output = pd.DataFrame({'Id': full_test_data.Id,

                      'SalePrice': final_predictions})

output.to_csv('submission.csv', index=False)