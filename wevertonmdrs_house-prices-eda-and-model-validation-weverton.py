import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col = 'Id')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col = 'Id')
y = data.SalePrice #target data

X_full = data.copy()

X = data.drop(['SalePrice'], axis = 1)

X_test = test.copy()
X_full.head()
X_full.describe()
X_correlation = X_full.corr() # correlation function

best_correlation_variable = X_correlation.index[abs(X_correlation['SalePrice']) > 0.4]



plt.figure(figsize = (10,10))

graph = sns.heatmap(X_full[best_correlation_variable].corr(), annot = True, cmap="Blues")
plt.figure(figsize = (12,10))



plt.xlabel("OverallQual",fontsize = 14)

plt.ylabel("SalePrice",fontsize = 14)

sns.barplot(X_full.OverallQual,X_full.SalePrice)

plt.show()
X_train_full, X_valid_full, y_train_full, y_valid_full = train_test_split(X, y, 

                                                        train_size=0.8, test_size=0.2,

                                                      random_state=0)

# All categorical columns

object_cols = [col for col in X_train_full.columns if X_train_full[col].dtype == "object"]



# Columns that can be safely label encoded

good_label_cols = [col for col in object_cols if 

                   set(X_train_full[col]) == set(X_valid_full[col])]

        

# Problematic columns that will be dropped from the dataset

bad_label_cols = list(set(object_cols)-set(good_label_cols))



X_train_full.drop(bad_label_cols, axis = 1, inplace = True)

X_valid_full.drop(bad_label_cols, axis = 1, inplace = True)



# Remove missing columns

cols_with_missing = [col for col in X_train_full.columns 

                     if X_train_full[col].isnull().sum() > 0.1*X_full.shape[0]]



X_train_full.drop(cols_with_missing, axis = 1, inplace = True)

X_valid_full.drop(cols_with_missing, axis = 1, inplace = True)



# Columns that will be one-hot encoded

low_cardinality_cols = [cname for cname in X_train_full.columns 

                        if X_train_full[cname].nunique() < 10 

                        and X_train_full[cname].dtype == 'object']



# Columns of numerical data

nume_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype

                                                    in ['int64', 'float64']]



my_cols = low_cardinality_cols + nume_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = test[my_cols].copy()
def model_definition(nume_cols, low_cardinality_cols, si_numerical_strategy, 

                     si_categorical_strategy, 

                     n_estimators, random_state):



    numerical_transformer = SimpleImputer(strategy = si_numerical_strategy)



    categorical_transformer = Pipeline(steps = [

        ('imputer', SimpleImputer(strategy = si_categorical_strategy)),

        ('onehot', OneHotEncoder(handle_unknown = 'ignore'))

    ])



    preprocessor = ColumnTransformer(

        transformers = [

            ('num', numerical_transformer, nume_cols),

            ('cat', categorical_transformer, low_cardinality_cols)

        ])

    

    model_RFR = RandomForestRegressor(n_estimators = n_estimators, 

                                      random_state = random_state)

      

    my_pipeline_RFR = Pipeline(steps = [('preprocessor', preprocessor), ('model', model_RFR)

                                       ])



    return my_pipeline_RFR





def cross_validation_scores(my_pipeline, X, y):

    # Predicting the data

    mae_score = -1 * cross_val_score(my_pipeline, X, y,

                              cv=5,

                              scoring='neg_mean_absolute_error')



    return mae_score.mean()



def get_XGB(X_train, X_valid, X_test, y_train, y_valid, learning_rate, 

            n_estimator, n_jobs):

    

    X_train = pd.get_dummies(X_train)

    X_valid = pd.get_dummies(X_valid)

    X_test = pd.get_dummies(X_test)

    X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

    X_train, X_test = X_train.align(X_test, join='left', axis=1)

    

    model_xgb = XGBRegressor(n_estimators = n_estimator, learning_rate = learning_rate, 

                         n_jobs = n_jobs)

    model_xgb.fit(X_train, y_train, 

             early_stopping_rounds = 5, 

             eval_set = [(X_valid, y_valid)], 

             verbose = False)

    

    prediction = model_xgb.predict(X_valid)

    mae_xgb = mean_absolute_error(prediction, y_valid)

    

    return model_xgb, mae_xgb
si_categorical_strategy = ['most_frequent', 'constant']



n_estimators = []

results_mae = {} # dictionary for the results

for estimator in range(100,1000,100):

    pipeline_RFR = model_definition(nume_cols, low_cardinality_cols, 'mean', 

                                        'most_frequent', estimator, 0)

    results_mae[estimator] = cross_validation_scores(pipeline_RFR, X, y)

    n_estimators.append(estimator)
plt.figure(figsize = (20,12))

plt.xlabel("Estimators Number",fontsize = 16)

plt.ylabel("Mean Absolute Error (MAE)",fontsize  =16)

plt.title("Number of RandomForest Estimators vs Mean Absolute Error (MAE)",fontsize  =16)

plt.plot(list(results_mae.keys()), list(results_mae.values()), label = 'Mean absolute error')

plt.show()
maes_XGB = []

models_XGB = []



for learn_r in np.arange(0.01,0.11,0.01):

    

    model_xgb, mae_xgb = get_XGB(X_train, X_valid, X_test, y_train_full, y_valid_full, 

                                 learn_r, 900, 2)

    

    models_XGB.append(model_xgb)

    maes_XGB.append(mae_xgb)
plt.figure(figsize=(20,10))

plt.xlabel("Learning rate value",fontsize=16)

plt.ylabel("Mean Absolute Error (MAE)",fontsize=16)

plt.title("Learning rate vs MAE (n_estimators = 900)",fontsize=16)

plt.plot(np.arange(0.01,0.11,0.01),maes_XGB)

plt.legend()

plt.show()