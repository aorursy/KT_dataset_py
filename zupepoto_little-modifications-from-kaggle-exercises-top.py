import pandas as pd

from sklearn.model_selection import train_test_split



# Read the data

X_full = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')



X_full.head()
# print some info

X_full.info()
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

X_test = X_test_full[my_cols].copy()



X_train.head()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import mean_absolute_error



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='median') # Your code here



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

]) # Your code here



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



# Transform the data

X_t = preprocessor.fit_transform(X_train)

X_v = preprocessor.transform(X_valid)
import xgboost



model_xgb = xgboost.XGBRegressor(colsample_bytree=0.4,

                 gamma=0,                 

                 learning_rate=0.07,

                 max_depth=3,

                 min_child_weight=1.5,

                 n_estimators=10000,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.6,

                 seed=42)





import lightgbm as lgb



model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
# XGB Model

# Bundle preprocessing and modeling code in a pipeline

my_pipeline_xgb = Pipeline(steps=[('preprocessor', preprocessor),

                                ('model', model_xgb)

                               ])



# Preprocessing of training data, fit model 

my_pipeline_xgb.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds_xgb = my_pipeline_xgb.predict(X_valid)





# LGB Model

# Bundle preprocessing and modeling code in a pipeline

my_pipeline_lgb = Pipeline(steps=[('preprocessor', preprocessor),

                                  ('model', model_lgb)

                                 ])



# Preprocessing of training data, fit model 

my_pipeline_lgb.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds_lgb = my_pipeline_lgb.predict(X_valid)



# Average the predictions

preds = (preds_xgb + preds_lgb)/2



# Evaluate the model

score = mean_absolute_error(y_valid, preds)

print('MAE:', score)
# Preprocessing of test data, fit model

my_pipeline_xgb.fit(X_full[my_cols], y)



preds_test_xgb = my_pipeline_xgb.predict(X_test) # Your code here



my_pipeline_lgb.fit(X_full[my_cols], y)



preds_test_lgb = my_pipeline_lgb.predict(X_test) # Your code here



preds_test = (preds_test_xgb + preds_test_lgb)/2



output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)