#Import Required Libraries

import warnings

warnings.filterwarnings("ignore")



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv') 

test  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.shape
train.head()
import pandas_profiling

profile_report = pandas_profiling.ProfileReport(train)

profile_report
# Dropping rows where the target is missing

Target = 'SalePrice'

train.dropna(axis=0, subset=[Target], inplace=True)
# Combine Test and Training sets to maintain consistancy.

data=pd.concat([train.iloc[:,:-1],test],axis=0)



print('train df has {} rows and {} features'.format(train.shape[0],train.shape[1]))

print('test df has {} rows and {} features'.format(test.shape[0],test.shape[1]))

print('Combined df has {} rows and {} features'.format(data.shape[0],data.shape[1]))
data.head()
# Dropping unwanted columns

data = data.drop(columns=['Id'],axis=1)
# Looking for Missing Values



def missingValuesInfo(df):

    total = df.isnull().sum().sort_values(ascending = False)

    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100, 2)

    temp = pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent'])

    return temp.loc[(temp['Total'] > 0)]



missingValuesInfo(train)
# Missing Value Handling



def HandleMissingValues(df):

    # for Object columns fill using 'UNKOWN'

    # for Numeric columns fill using median

    num_cols = [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64']]

    cat_cols = [cname for cname in df.columns if df[cname].dtype == "object"]

    values = {}

    for a in cat_cols:

        values[a] = 'UNKOWN'



    for a in num_cols:

        values[a] = df[a].median()

        

    df.fillna(value=values,inplace=True)

    

    

HandleMissingValues(data)

data.head()
# Check for any missing values

data.isnull().sum().sum()
#Categorical Feature Encoding



def getObjectColumnsList(df):

    return [cname for cname in df.columns if df[cname].dtype == "object"]



def PerformOneHotEncoding(df,columnsToEncode):

    return pd.get_dummies(df,columns = columnsToEncode)



cat_cols = getObjectColumnsList(data)

data = PerformOneHotEncoding(data,cat_cols)

data.head()
data.shape
#spliting the data into train and test datasets

train_data=data.iloc[:1460,:]

test_data=data.iloc[1460:,:]

print(train_data.shape)

test_data.shape
# Get X,y for modelling

X=train_data

y=train.loc[:,'SalePrice']
from sklearn.linear_model import RidgeCV



ridge_cv = RidgeCV(alphas=(0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))

ridge_cv.fit(X, y)

ridge_cv_preds=ridge_cv.predict(test_data)
import xgboost as xgb



model_xgb = xgb.XGBRegressor(n_estimators=340, max_depth=2, learning_rate=0.2)

model_xgb.fit(X, y)

xgb_preds=model_xgb.predict(test_data)
predictions = ( ridge_cv_preds + xgb_preds )/2
#make the submission data frame

submission = {

    'Id': test.Id.values,

    'SalePrice': predictions

}

solution = pd.DataFrame(submission)

solution.head()
#make the submission file

solution.to_csv('submission.csv',index=False)