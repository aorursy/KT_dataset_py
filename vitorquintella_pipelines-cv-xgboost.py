# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex4 import *

print("Setup Complete")
import pandas as pd

from sklearn.model_selection import train_test_split



# Read the data

X_full = pd.read_csv('../input/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X_full.SalePrice

X_full.drop(['SalePrice'], axis=1, inplace=True)



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 

                                                                train_size=0.8, test_size=0.2,

                                                                random_state=0)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_train_full.columns if

                    X_train_full[cname].nunique() < 10 and 

                    X_train_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if 

                X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X = X_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
X_train.head()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='median')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



# Define model

model = RandomForestRegressor(n_estimators=100, random_state=0)



# Bundle preprocessing and modeling code in a pipeline

clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ])



# Preprocessing of training data, fit model 

clf.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = clf.predict(X_valid)



print('MAE:', mean_absolute_error(y_valid, preds))
print('MAE:', mean_absolute_error(y_valid, preds))
# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='median') # Your code here



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

]) # Your code here



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



# Define model

model = RandomForestRegressor(n_estimators=100, random_state=0) # Your code here



# Check your answer

step_1.a.check()
# Lines below will give you a hint or solution code

step_1.a.hint()

# step_1.a.solution()
# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])



# Preprocessing of training data, fit model 

my_pipeline.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = my_pipeline.predict(X_valid)



# Evaluate the model

score = mean_absolute_error(y_valid, preds)

print('MAE:', score)



# Check your answer

step_1.b.check()
# Line below will give you a hint

# step_1.b.hint()
def taylored_pipeline(n_estimators):

    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                                  ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))

                                 ])

    return my_pipeline

    
from sklearn.model_selection import cross_val_score

def get_score(n_estimators):

    """Return the average MAE over 3 CV folds of random forest model.

    

    Keyword argument:

    n_estimators -- the number of trees in the forest

    """

    # Replace this body with your own code

    my_pipeline = taylored_pipeline(n_estimators)



    

    # Multiply by -1 since sklearn calculates *negative* MAE

    scores = -1 * cross_val_score(my_pipeline, X, y,

                                  cv=3,

                                  scoring='neg_mean_absolute_error')

    return scores.mean()

    
import numpy as np

candidate_trees =  np.arange(start = 50, stop = 400+1, step = 50) #+1 to inlcude last value

results = {n_trees: get_score(n_trees) for n_trees in candidate_trees}



display(results)
import matplotlib.pyplot as plt



plt.plot(results.keys(), results.values())

plt.show()



n_estimators_best = min(results, key=results.get)

print("best n_estimators is = ", n_estimators_best, "; with CV mean MAE =",results.get(n_estimators_best))
# fit on train&valid with best parameters

my_pipeline = taylored_pipeline(n_estimators_best)

my_pipeline.fit(X, y)







# Preprocessing of test data, fit model



preds_test = my_pipeline.predict(X_test)

# Check your answer

step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

# step_2.solution()
# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)