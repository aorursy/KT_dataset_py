# learning from https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset

# Imports
import pandas as pd 
import numpy as np

# Standardize, Label Encode, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder 
from sklearn.impute import SimpleImputer

# Model
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, make_scorer
from xgboost import XGBRegressor

# Graph
from scipy.stats import skew
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

# Definitions
pd.set_option('display.float_format', lambda x: '%.3f' % x)
%matplotlib inline

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


train_data_source = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
print(train_data_source.shape)

test_data_source = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print(test_data_source.shape)
#######
# RUN #
#######

#make copy
train_data = train_data_source
y_source = train_data_source['SalePrice']

#log transform SalePrice
train_data['SalePrice'] = np.log(y_source)

print(train_data['SalePrice'])
#######
# RUN #
#######

#count missing 
MissingCount = train_data.isnull().sum().sort_values(ascending=False)
percent = MissingCount/train_data.shape[0]

missing_data = pd.concat([MissingCount,percent],axis=1,keys=['count','percent'])

#drop missing columns on train and test
cols_drop = (missing_data[missing_data['count']>1]).index
train_data_dropped = train_data.drop(cols_drop,axis=1)
test_data_dropped = test_data_source.drop(cols_drop,axis=1)

#drop 'Id'
train_data_dropped = train_data_dropped.drop('Id',axis=1)
test_data_dropped = test_data_dropped.drop('Id',axis=1)

#drop one row that has 1 missing value
train_data_dropped = train_data_dropped.drop(train_data_dropped[train_data_dropped['Electrical'].isnull()].index)

#drop outlier GrLivArea

train_data_dropped = train_data_dropped.drop(1298,axis=0)
train_data_dropped = train_data_dropped.drop(523,axis=0)

train_data = train_data_dropped
test_data = test_data_dropped

print('train shape:', train_data.shape)
print('test shape:', test_data.shape)
#make a copy of test file
test_x = test_data

#get train_x and train_y
y = train_data['SalePrice']
train = train_data.drop('SalePrice',axis=1)

#partition into train and validation set
train_x, valid_x, train_y, valid_y = train_test_split(train, y, test_size = 0.3, random_state = 0)

print('train x:', train_x.shape)
print('train y:', train_y.shape)
print('valid x:', valid_x.shape)
print('valid y:', valid_y.shape)
print('test x:', test_x.shape)


#list of type of variables
categorical_feature = train.select_dtypes(include = ['object']).columns
numerical_feature = train.select_dtypes(exclude = ['object']).columns

print('catgorical:', categorical_feature, '\n')
print('numerical:', numerical_feature, '\n')

#############
# NUMERICAL #
#############

#standardize
sc = StandardScaler()

train_x[numerical_feature] = sc.fit_transform(train_x[numerical_feature])
valid_x[numerical_feature] = sc.transform(valid_x[numerical_feature])
test_x[numerical_feature] = sc.transform(test_x[numerical_feature])

###############
# CATEGORICAL #
###############

#columns in common
good_label_cols = [col for col in categorical_feature if 
                   set(train_x[col]) == set(valid_x[col]) == set(test_x[col]) ]
      
# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(categorical_feature)-set(good_label_cols))
        
print('Categorical columns that will be label encoded:', good_label_cols)
print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)
# Drop categorical columns that will not be encoded
train_x = train_x.drop(bad_label_cols, axis=1)
valid_x = valid_x.drop(bad_label_cols, axis=1)
test_x = test_x.drop(bad_label_cols, axis=1)

# Apply label encoder 
label_encoder = LabelEncoder() # Your code here

for col in good_label_cols:
    train_x[col] = label_encoder.fit_transform(train_x[col])
    valid_x[col] = label_encoder.transform(valid_x[col])
    test_x[col] = label_encoder.transform(test_x[col])
    
print('train shape:', train_x.shape)
print('valid shape:', valid_x.shape)
print('test shape:', test_x.shape)
#missing value in test_x
#make copy
test_x_copy = test_x

#imputer
impt = SimpleImputer()
aa = pd.DataFrame(impt.fit_transform(train_x))
test_x_copy = pd.DataFrame(impt.transform(test_x_copy))

#assign back column names
test_x_copy.columns = test_x.columns

# Define error measure for official scoring : MAE

def mae_cv_train(model):
    mae = -cross_val_score(model, train_x, train_y, scoring = 'neg_mean_absolute_error', cv = 10)
    return mae.mean()

def mae_cv_test(model):
    mae = -cross_val_score(model, valid_x, valid_y, scoring = 'neg_mean_absolute_error', cv = 10)
    return mae.mean()
# Linear Regression
lr = LinearRegression()
lr.fit(train_x, train_y)

# Look at predictions on training and validation set
print("mae on Training set :", mae_cv_train(lr))
print("mae on Test set :", mae_cv_test(lr))

#fit prediction
pred_train = lr.predict(train_x)
pred_valid = lr.predict(valid_x)

#residual plot
plt.figure(figsize=(8,6))
plt.scatter(x=pred_train,y=pred_train-train_y, c = 'green', marker='s', label='Training')
plt.scatter(x=pred_valid,y=pred_valid-valid_y, c = 'yellow', marker='s', label='Validation')
plt.title('Residual vs Predicted')
plt.xlabel('Prediction')
plt.ylabel('Residual')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = 10.5, xmax = 14, color='red')
plt.show()

#real vs predicted
plt.figure(figsize=(8,6))
plt.scatter(x=pred_train,y=train_y, c = 'green', marker='s', label='Training')
plt.scatter(x=pred_valid,y=valid_y, c = 'yellow', marker='s', label='Validation')
plt.title('Real vs Predicted')
plt.xlabel('Prediction')
plt.ylabel('Real Value')
plt.legend(loc = 'upper left')
plt.show()


# Ridge Regression
ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
ridge.fit(train_x, train_y)
alpha = ridge.alpha_
print("Best alpha :", alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                cv = 10)
ridge.fit(train_x, train_y)
alpha = ridge.alpha_
print("Best alpha :", alpha)

print("Ridge mae on Training set :", mae_cv_train(ridge).mean())
print("Ridge mae on Test set :", mae_cv_test(ridge).mean())

#fit prediction
pred_train = ridge.predict(train_x)
pred_valid = ridge.predict(valid_x)

#residual plot
plt.figure(figsize=(8,6))
plt.scatter(x=pred_train,y=pred_train-train_y, c = 'green', marker='s', label='Training')
plt.scatter(x=pred_valid,y=pred_valid-valid_y, c = 'yellow', marker='s', label='Validation')
plt.title('Residual vs Predicted')
plt.xlabel('Prediction')
plt.ylabel('Residual')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = 10.5, xmax = 14, color='red')
plt.show()

#real vs predicted
plt.figure(figsize=(8,6))
plt.scatter(x=pred_train,y=train_y, c = 'green', marker='s', label='Training')
plt.scatter(x=pred_valid,y=valid_y, c = 'yellow', marker='s', label='Validation')
plt.title('Real vs Predicted')
plt.xlabel('Prediction')
plt.ylabel('Real Value')
plt.legend(loc = 'upper left')
plt.show()


xgb = XGBRegressor(n_estimators = 100)
xgb.fit(train_x, train_y, early_stopping_rounds=5, eval_set=[(valid_x, valid_y)], verbose=False)

print("mae on Training set :", mae_cv_train(xgb))
print("mae on Test set :", mae_cv_test(xgb))

#fit prediction
pred_train = xgb.predict(train_x)
pred_valid = xgb.predict(valid_x)

#residual plot
plt.figure(figsize=(8,6))
plt.scatter(x=pred_train,y=pred_train-train_y, c = 'green', marker='s', label='Training')
plt.scatter(x=pred_valid,y=pred_valid-valid_y, c = 'yellow', marker='s', label='Validation')
plt.title('Residual vs Predicted')
plt.xlabel('Prediction')
plt.ylabel('Residual')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = 10.5, xmax = 14, color='red')
plt.show()

#real vs predicted
plt.figure(figsize=(8,6))
plt.scatter(x=pred_train,y=train_y, c = 'green', marker='s', label='Training')
plt.scatter(x=pred_valid,y=valid_y, c = 'yellow', marker='s', label='Validation')
plt.title('Real vs Predicted')
plt.xlabel('Prediction')
plt.ylabel('Real Value')
plt.legend(loc = 'upper left')
plt.show()
# lasso
lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1], 
                max_iter = 50000, cv = 10)
lasso.fit(train_x, train_y)
alpha = lasso.alpha_
print("Best alpha :", alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 
                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 
                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 
                          alpha * 1.4], 
                max_iter = 50000, cv = 10)
lasso.fit(train_x, train_y)
alpha = lasso.alpha_
print("Best alpha :", alpha)

print("LASSO mae on Training set :", mae_cv_train(lasso).mean())
print("LASSO mae on Test set :", mae_cv_test(lasso).mean())

#fit prediction
pred_train = lasso.predict(train_x)
pred_valid = lasso.predict(valid_x)

#residual plot
plt.figure(figsize=(8,6))
plt.scatter(x=pred_train,y=pred_train-train_y, c = 'green', marker='s', label='Training')
plt.scatter(x=pred_valid,y=pred_valid-valid_y, c = 'yellow', marker='s', label='Validation')
plt.title('Residual vs Predicted')
plt.xlabel('Prediction')
plt.ylabel('Residual')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = 10.5, xmax = 14, color='red')
plt.show()

#real vs predicted
plt.figure(figsize=(8,6))
plt.scatter(x=pred_train,y=train_y, c = 'green', marker='s', label='Training')
plt.scatter(x=pred_valid,y=valid_y, c = 'yellow', marker='s', label='Validation')
plt.title('Real vs Predicted')
plt.xlabel('Prediction')
plt.ylabel('Real Value')
plt.legend(loc = 'upper left')
plt.show()

#linear prediction on test
predictions = np.exp(lr.predict(test_x_copy))
predictions
#ridge prediction on test
predictions = np.exp(ridge.predict(test_x_copy))
predictions
#XGBoost prediction on test
predictions = np.exp(xgb.predict(test_x_copy))
predictions
#LASSO prediction on test
predictions = np.exp(lasso.predict(test_x_copy))
predictions
submission = pd.DataFrame({'Id':test_data_source.Id, 'SalePrice':predictions})

submission.to_csv('submission.csv', index=False)