# to handle datasets

import pandas as pd

import numpy as np



# for plotting

import matplotlib.pyplot as plt

%matplotlib inline



# to divide train and test set

from sklearn.model_selection import train_test_split



# feature scaling

from sklearn.preprocessing import MinMaxScaler



# to build the models

from sklearn.linear_model import Lasso



# to evaluate the models

from sklearn.metrics import mean_squared_error

from math import sqrt



# to build the models

from sklearn.linear_model import LinearRegression, Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

import xgboost as xgb



pd.pandas.set_option('display.max_columns', None)



import warnings

warnings.filterwarnings('ignore')



from feature_engine import missing_data_imputers as msi

from feature_engine import discretisers as dsc

from feature_engine import categorical_encoders as ce



# sklearn pipeline to put it all together

from sklearn.pipeline import Pipeline as pipe
# load dataset

data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

print(data.shape)

data.head()
# Load the dataset for submission (the one on which our model will be evaluated by Kaggle)

# it contains exactly the same variables, but not the target



submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

submission.head()
# categorical variables

categorical = [var for var in data.columns if data[var].dtype=='O']



# make a list of all numerical variables first

numerical = [var for var in data.columns if data[var].dtype!='O']



# temporal variables

year_vars = [var for var in numerical if 'Yr' in var or 'Year' in var]



# discrete variables

discrete = [var for var in numerical if var not in year_vars and len(data[var].unique())<20]



# continuous vars

numerical = [var for var in numerical if var not in discrete and var not in ['Id', 'SalePrice'] and var not in year_vars]



print('There are {} categorical variables'.format(len(categorical)))

print('There are {} discrete variables'.format(len(discrete)))

print('There are {} numerical and continuous variables'.format(len(numerical)))
# Let's separate into train and test set



X_train, X_test, y_train, y_test = train_test_split(data, data.SalePrice, test_size=0.1,

                                                    random_state=0)

X_train.shape, X_test.shape
# function to calculate elapsed time



def elapsed_years(df, var):

    # capture difference between year variable and year the house was sold

    df[var] = df['YrSold'] - df[var]

    return df
for var in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:

    X_train = elapsed_years(X_train, var)

    X_test = elapsed_years(X_test, var)

    submission = elapsed_years(submission, var)
X_train[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']].head()
# drop YrSold

X_train.drop('YrSold', axis=1, inplace=True)

X_test.drop('YrSold', axis=1, inplace=True)

submission.drop('YrSold', axis=1, inplace=True)



# remove YrSold from our temporal var list

year_vars.remove('YrSold')
price_pipe = pipe([

    # add a binary variable to indicate missing information for the 2 variables below

    ('continuous_var_imputer', msi.AddNaNBinaryImputer(variables = ['LotFrontage', 'GarageYrBlt'])),

     

    # replace NA by the median in the 3 variables below, they are numerical

    ('continuous_var_median_imputer', msi.MeanMedianImputer(imputation_method='median', variables = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea'])),

     

    # replace NA by adding the label "Missing" in categorical variables (transformer will skip those variables where there is no NA)

    ('categorical_imputer', msi.CategoricalVariableImputer(variables = categorical)),

     

    # there were a few variables in the submission dataset that showed NA, but these variables did not show NA in the train set.

    # to handle those, I will add an additional step here

    ('additional_median_imputer', msi.MeanMedianImputer(imputation_method='median', variables = numerical)),



    # disretise numerical variables using trees

    ('numerical_tree_discretiser', dsc.DecisionTreeDiscretiser(cv = 3, scoring='neg_mean_squared_error', variables = numerical, regression=True)),

     

    # remove rare labels in categorical and discrete variables

    ('rare_label_encoder', ce.RareLabelCategoricalEncoder(tol = 0.03, n_categories=1, variables = categorical+discrete)),

     

    # encode categorical variables using the target mean 

    ('categorical_encoder', ce.MeanCategoricalEncoder(variables = categorical+discrete))

     ])
# the following vars in the submission dataset are encoded in different types

# so first I cast them as int, like in the train set



for var in ['BsmtFullBath', 'BsmtHalfBath', 'GarageCars']:

    submission[var] = submission[var].astype('float')
## the categorical encoders only work with categorical variables

# therefore we need to cast the discrete variables into categorical



X_train[discrete]= X_train[discrete].astype('O')

X_test[discrete]= X_test[discrete].astype('O')

submission[discrete]= submission[discrete].astype('O')
# let's create a list of the training variables

training_vars = [var for var in X_train.columns if var not in ['Id', 'SalePrice']]



print('total number of variables to use for training: ', len(training_vars))
price_pipe.fit(X_train[training_vars], y_train)
# let's capture the id to add it later to our submission

submission_id = submission['Id']
X_train = price_pipe.transform(X_train[training_vars])

X_test = price_pipe.transform(X_test[training_vars])

submission = price_pipe.transform(submission[training_vars])
# let's check that we didn't introduce NA

len([var for var in training_vars if X_train[var].isnull().sum()>0])
# let's check that we didn't introduce NA

len([var for var in training_vars if X_test[var].isnull().sum()>0])
# let's check that we didn't introduce NA

len([var for var in training_vars if submission[var].isnull().sum()>0])
training_vars = [var for var in X_train.columns]

len(training_vars)
# these are the binary variables that we introduced during feature engineering

[ var for var in training_vars if '_na' in var]
# fit scaler

scaler = MinMaxScaler() # create an instance

scaler.fit(X_train[training_vars]) #  fit  the scaler to the train set for later use
xgb_model = xgb.XGBRegressor()



eval_set = [(X_test[training_vars], np.log(y_test))]

xgb_model.fit(X_train[training_vars], np.log(y_train), eval_set=eval_set, verbose=False)



pred = xgb_model.predict(X_train[training_vars])

print('xgb train mse: {}'.format(mean_squared_error(y_train, np.exp(pred))))

print('xgb train rmse: {}'.format(sqrt(mean_squared_error(y_train, np.exp(pred)))))

print()

pred = xgb_model.predict(X_test[training_vars])

print('xgb test mse: {}'.format(mean_squared_error(y_test, np.exp(pred))))

print('xgb test rmse: {}'.format(sqrt(mean_squared_error(y_test, np.exp(pred)))))
rf_model = RandomForestRegressor(n_estimators=800, max_depth=6)

rf_model.fit(X_train[training_vars], np.log(y_train))



pred = rf_model.predict(X_train[training_vars])

print('rf train mse: {}'.format(mean_squared_error(y_train, np.exp(pred))))

print('rf train rmse: {}'.format(sqrt(mean_squared_error(y_train, np.exp(pred)))))



print()

pred = rf_model.predict(X_test[training_vars])

print('rf test mse: {}'.format(mean_squared_error(y_test, np.exp(pred))))

print('rf test rmse: {}'.format(sqrt(mean_squared_error(y_test, np.exp(pred)))))
SVR_model = SVR()

SVR_model.fit(scaler.transform(X_train[training_vars]), np.log(y_train))



pred = SVR_model.predict(X_train[training_vars])

print('SVR train mse: {}'.format(mean_squared_error(y_train, np.exp(pred))))

print('SVR train rmse: {}'.format(sqrt(mean_squared_error(y_train, np.exp(pred)))))



print()

pred = SVR_model.predict(X_test[training_vars])

print('SVR test mse: {}'.format(mean_squared_error(y_test, np.exp(pred))))

print('SVR test rmse: {}'.format(sqrt(mean_squared_error(y_test, np.exp(pred)))))
lin_model = Lasso(random_state=2909, alpha=0.005)

lin_model.fit(scaler.transform(X_train[training_vars]), np.log(y_train))



pred = lin_model.predict(scaler.transform(X_train[training_vars]))

print('Lasso Linear Model train mse: {}'.format(mean_squared_error(y_train, np.exp(pred))))

print('Lasso Linear Model train rmse: {}'.format(sqrt(mean_squared_error(y_train, np.exp(pred)))))



print()

pred = lin_model.predict(scaler.transform(X_test[training_vars]))

print('Lasso Linear Model test mse: {}'.format(mean_squared_error(y_test, np.exp(pred))))

print('Lasso Linear Model test rmse: {}'.format(sqrt(mean_squared_error(y_test, np.exp(pred)))))
# make predictions for the submission dataset

final_pred = pred = lin_model.predict(scaler.transform(submission[training_vars]))
temp = pd.concat([submission_id, pd.Series(np.exp(final_pred))], axis=1)

temp.columns = ['Id', 'SalePrice']

temp.head()
temp.to_csv('submit_housesale.csv', index=False)