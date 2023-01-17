%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import copy



from datetime import datetime



from sklearn.model_selection import KFold, cross_val_score

from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_squared_error



from sklearn import linear_model as lm



from math import sqrt
pd.set_option('display.max_columns', None) #to see all the columns in the output
def missing_values_table(df):

        

        #https://stackoverflow.com/questions/26266362/how-to-count-the-nan-values-in-a-column-in-pandas-dataframe

        

        #mis_label = pd.DataFrame(df.columns.tolist()).iloc[:,0]

        mis_val = df.isnull().sum()

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        mis_val_dtype = df.dtypes

        

        mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis=1)

        

        mis_val_table_ren_columns = mis_val_table.rename(columns = {

                                                                    0 : 'Missing Values', 

                                                                    1 : 'Percentage',

                                                                    2 : 'Data Types'})

        

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        'Percentage', ascending=False).round(1)

        

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        return mis_val_table_ren_columns
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
print(train.shape)

print(test.shape)
#copy the dataframe to keep original unchanged

train_cp = train.copy()

train_columns = train.columns.tolist() ##get as a list of all the column names



test_cp = test.copy()

test_columns = test.columns.tolist() ##get as a list of all the column names



id_column = 'Id' #id column name

target_column = 'SalePrice' #target column name
train_cp.drop(['Id'], axis=1, inplace=True)

test_cp.drop(['Id'], axis=1, inplace=True)
train_cp = train_cp[train_cp.GrLivArea < 4500]

train_cp.reset_index(drop=True, inplace=True)

train_cp["SalePrice"] = np.log1p(train_cp["SalePrice"])

y = train_cp['SalePrice'].reset_index(drop=True)
train_cp.head()
train_features = train_cp.drop(['SalePrice'], axis=1)

test_features = test_cp

features = pd.concat([train_features, test_features],sort=True).reset_index(drop=True)
features.shape
missing_columns_df = missing_values_table(features)

missing_columns_df
missing_columns_df['Missing Values'].plot.bar()
for row, col in missing_columns_df.iterrows():     

    if col[2] == 'object':

        features[row] = features[row].fillna("None")

    else:

        features[row] = features[row].fillna(0)
features.tail()
missing_columns_df = missing_values_table(features)

missing_columns_df
final_features = pd.get_dummies(features).reset_index(drop=True)

final_features.shape
X = final_features.iloc[:len(y), :] #get train set by using the SalePrice dataframe length

X_test = final_features.iloc[len(y):, :] #get test set



#train,  SalePrice in Train, test data accordingly

X.shape, y.shape, X_test.shape
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
house_price_reg = lm.LinearRegression()

house_price_reg.fit(X, y)

#print('Intercept: \n', house_price_reg.intercept_)

#print('Coefficients: \n', house_price_reg.coef_)
house_price_predictions = house_price_reg.predict(X_test)

len(house_price_predictions)
house_price_predictions
print(rmsle(y, house_price_reg.predict(X)))
print('Predict submission', datetime.now(),)

submission = pd.read_csv("../input/sample_submission.csv")

submission.iloc[:,1] = np.floor(house_price_predictions)