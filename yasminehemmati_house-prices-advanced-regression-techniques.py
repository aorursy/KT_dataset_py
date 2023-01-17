



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



 

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
#GOAL: Predict Housing Sales Prices (SalePrice)



train.info()

train.head()
####DROP FEAUTURES WITH MAJORITY OF ELEMENTS MISSING

majority_missing = []

for col in train.columns:

    if train[col].isnull().sum() >(train.shape[0])/2:

        majority_missing.append(col)

    

reduced_train = train.drop(majority_missing, axis=1)

reduced_test = test.drop(majority_missing, axis=1)



test_nan = reduced_test.isnull().sum().sort_values(ascending = False) 

len(test_nan[test_nan>=1])

#lets merge test and train to handle nan values at same time



df_all = pd.concat([reduced_train,reduced_test ],ignore_index=True)

df_all.shape, reduced_train.shape, reduced_test.shape #NOTE: first 1460 rows from training data last 1459 from test 
# replace all the categorical variable with their mode value and numerical variables with their median

df_all["Electrical"].dtype, df_all["SalePrice"].dtype # so categorical variables would have a data type of "dtype(0)" and integers "dtype('float64')"
has_null = df_all.isnull().sum()

has_null[has_null>0]
# to find categorical variable with missing features

str_missing = []

for col in df_all:

    if df_all[col].dtype == df_all["Electrical"].dtype:

        if df_all[col].isnull().sum() >0:

            str_missing.append(col)

str_missing
#in the data description for FireplaceQu Na means it does not have that feature

df_all['FireplaceQu'] = df_all['FireplaceQu'].fillna('None')
 # this will give us a list of all the columns that non nan inputs are a string since 



for col in df_all:

    if df_all[col].dtype == df_all["Electrical"].dtype:

        if df_all[col].isnull().sum() >0:

            df_all[col] = df_all[col].fillna(df_all[col].mode()[0])

            



for col in df_all:

    if df_all[col].dtype == df_all['GarageArea'].dtype and col != 'SalePrice':

        if df_all[col].isnull().sum() >0:

            mean = df_all[col].mean()

            df_all[col] = df_all[col].fillna(mean)

            
def categorical(data):

    cat_var = []

    for col in data:

        if df_all[col].dtype == df_all["Electrical"].dtype:

            cat_var.append(col)

    return cat_var        
#now we need to hand categorical variables

cat_var = categorical(df_all)

df_all = pd.get_dummies(df_all, columns = cat_var, drop_first = True)

        
#As noted previously first 1460 columns (inclusive) are from the training data and rest are from test data

train_final = df_all.iloc[:1460,:]

test_final = df_all.iloc[1460:,:]

test_final = test_final.drop('SalePrice',axis=1) #since test data originally did not have the SalePrice feature, this is what we are estimating

#Check that no more columns with missing values

train_final.isnull().sum().sort_values(ascending = True), test_final.isnull().sum().sort_values(ascending = True)
#from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

import xgboost as xgb

model_xgb = xgb.XGBRegressor(learning_rate = 0.11)

Y= train_final['SalePrice']

X= train_final.drop('SalePrice', axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state = 0)

model_xgb.fit(X_train,Y_train)

xgb_pred = model_xgb.predict(X_test)

r2_score(Y_test, xgb_pred)


# Use the model to make predictions

predicted_prices = model_xgb.predict(test_final)

# We will look at the predicted prices to ensure we have something sensible.

print(predicted_prices)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)