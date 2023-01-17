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

#X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 

#                                                                train_size=0.8, test_size=0.2,

#                                                                random_state=0)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

#categorical_cols = [cname for cname in X_train_full.columns if

#                    X_train_full[cname].nunique() < 10 and 

#                    X_train_full[cname].dtype == "object"]

categorical_cols = [cname for cname in X_full.columns if

                    X_full[cname].nunique() < 10 and 

                    X_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_full.columns if 

                X_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X = X_full[my_cols].copy()

#X_train = X_train_full[my_cols].copy()

#X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant')



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

# model = XGBRegressor(n_estimators=1000, learning_rate=.05, random_state=0)
from sklearn.model_selection import cross_val_score



def get_score(n_estimators):

    my_pipeline = Pipeline(steps=[

        ('preprocessor', preprocessor),

        ('model', XGBRegressor(n_estimators=n_estimators, learning_rate=.05, random_state=0))

    ])

    

    scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')

    

    return scores.mean()



# results = {i: get_score(i) for i in range(50, 1001, 50)}



import matplotlib.pyplot as plt

%matplotlib inline



#plt.plot(results.keys(), results.values())

#plt.show()
my_pipeline = Pipeline(steps=[

        ('preprocessor', preprocessor),

        ('model', XGBRegressor(n_estimators=500, learning_rate=.05, random_state=0))

    ])



# Preprocessing of training data, fit model 

my_pipeline.fit(X, y)



# Preprocessing of validation data, get predictions

#preds = my_pipeline.predict(X_valid)



# Evaluate the model

#score = mean_absolute_error(y_valid, preds)

#print('MAE:', score)
preds_test = my_pipeline.predict(X_test)



# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)