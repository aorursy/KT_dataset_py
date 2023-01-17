# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
# Importing the train data.
housing_train=pd.read_csv('train.csv')
housing_train.head()
# Displaying data shape.
housing_train.shape
# Describe my numeric data (Statistics)
numerical_columns = [col for col in housing_train.columns if (housing_train[col].dtype=='int64' or housing_train[col].dtype=='float64')]
housing_train[numerical_columns].describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max'], :]
# data statistics for countinious variables (mean,std, Q1, Q3.......)
# Display correlation matrix to help in understanding any multicolinearity,correlation between idependent
# and dependent variables.
plt.figure(figsize=(30,20))
corrMatrix = housing_train.corr()
sns.heatmap(corrMatrix, annot=True, cmap='coolwarm')
### There are some variables that have high correlation(Multicolinearity issue), We will deal with this issue later
# Visualizing the missing data (Heat map)
sns.heatmap(housing_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# Train data variables, values and type.
housing_train.info()
housing_train.drop(['Id'],axis=1,inplace=True)
housing_train.drop(['Alley'],axis=1,inplace=True)
housing_train.drop(['PoolQC','Fence','MiscFeature', 'GarageArea'],axis=1,inplace=True)
housing_train.drop(['GarageYrBlt'],axis=1,inplace=True)
housing_train.drop(['1stFlrSF'],axis=1,inplace=True)
## Fill Missing Values for numeric variables.
housing_train['LotFrontage']=housing_train['LotFrontage'].fillna(housing_train['LotFrontage'].mean())
# Fill Missing Values for categorical variables.
housing_train['BsmtCond']=housing_train['BsmtCond'].fillna(housing_train['BsmtCond'].mode()[0])
housing_train['BsmtQual']=housing_train['BsmtQual'].fillna(housing_train['BsmtQual'].mode()[0])
housing_train['FireplaceQu']=housing_train['FireplaceQu'].fillna(housing_train['FireplaceQu'].mode()[0])
housing_train['GarageType']=housing_train['GarageType'].fillna(housing_train['GarageType'].mode()[0])
housing_train['GarageFinish']=housing_train['GarageFinish'].fillna(housing_train['GarageFinish'].mode()[0])
housing_train['GarageQual']=housing_train['GarageQual'].fillna(housing_train['GarageQual'].mode()[0])
housing_train['GarageCond']=housing_train['GarageCond'].fillna(housing_train['GarageCond'].mode()[0])
housing_train['MasVnrType']=housing_train['MasVnrType'].fillna(housing_train['MasVnrType'].mode()[0])
housing_train['MasVnrArea']=housing_train['MasVnrArea'].fillna(housing_train['MasVnrArea'].mode()[0])
housing_train['BsmtFinType2']=housing_train['BsmtFinType2'].fillna(housing_train['BsmtFinType2'].mode()[0])
housing_train['BsmtExposure']=housing_train['BsmtExposure'].fillna(housing_train['BsmtExposure'].mode()[0])
housing_train.isnull().sum().sum()
# Only 38 missing variables remained, I will drop them.
housing_train.dropna(inplace=True)
housing_train.isna().sum().sum()
## Making sure no missing data anymore using a heatmap.
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# Reading my data
housing_test= pd.read_csv('test.csv')
# Saving the Id variable for later.
Id= housing_test['Id']
# Displaying first 5 rows in test data.
housing_test.head(5)
# Test data shape.
housing_test.shape
# Displaying test data information ( variables, value counts, variable type)
housing_test.info()
# Displaying the missing data
housing_test.isna().sum().sum()
# Visualizing the missing data
sns.heatmap(housing_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# Test data information
housing_test.info()
#These variables have more than 50% missing values that is why we have decided to drop them.
housing_test.drop(['Id', 'Alley', 'MiscFeature', 'Fence', 'PoolQC', 'GarageYrBlt', 'GarageArea','1stFlrSF'], axis=1, inplace=True)
# Imputing the missing data in numeric variables( mean).
housing_test.LotFrontage.fillna(housing_test.LotFrontage.mean(), inplace=True)
# Impute the missing data with the mode.
housing_test.MSZoning .fillna(housing_test.MSZoning.mode()[0], inplace= True)
housing_test.Utilities.fillna(housing_test.Utilities.mode()[0], inplace= True)
housing_test.MasVnrType.fillna(housing_test.MasVnrType.mode()[0], inplace= True)
housing_test.BsmtQual.fillna(housing_test.BsmtQual.mode()[0], inplace= True)
housing_test.BsmtCond.fillna(housing_test.BsmtCond.mode()[0], inplace= True)
housing_test.BsmtExposure.fillna(housing_test.BsmtExposure.mode()[0], inplace= True)
housing_test.BsmtFinType1.fillna(housing_test.BsmtFinType1.mode()[0], inplace= True)
housing_test.BsmtFinType2.fillna(housing_test.BsmtFinType2.mode()[0], inplace= True)
housing_test.FireplaceQu.fillna(housing_test.FireplaceQu.mode()[0], inplace= True)
housing_test.GarageType.fillna(housing_test.GarageType.mode()[0], inplace= True)
housing_test.GarageFinish.fillna(housing_test.GarageFinish.mode()[0], inplace= True) 
housing_test.GarageQual.fillna(housing_test.GarageQual.mode()[0], inplace= True)
housing_test.GarageCond.fillna(housing_test.GarageCond.mode()[0], inplace= True)
housing_test['GarageCars'].fillna(housing_test['GarageCars'].mean())
housing_test['BsmtUnfSF'].fillna(housing_test['BsmtUnfSF'].mean())
housing_test['BsmtFinSF1'].fillna(housing_test['BsmtFinSF1'].mean())
housing_test['BsmtFinSF2'].fillna(housing_test['BsmtFinSF2'].mean())
# Test data shape.
housing_test.shape
# Exporting the clean test data into CSV file.
housing_test.to_csv('housing_testclean.csv', index=False)
# These are all the categorical variables.
columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']
# Defining a function that will be used to convert categorical data into dummies.
def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final
# Importing the clean test data.
test_df=pd.read_csv('housing_testclean.csv')
# Clean Test shape.
test_df.shape
# Displaying first 5 rows from the test data
test_df.head()
# Concating test and train data
final_df=pd.concat([housing_train,test_df],axis=0)
# Displaying the shape of the final data
final_df.shape
# Using the function to convert categorical data into dummies
final_df=category_onehot_multcols(columns)
# Final data shape
final_df.shape
# Eliminating the duplicates.
final_df =final_df.loc[:,~final_df.columns.duplicated()]
# Data shape after eliminating duplicates.
final_df.shape
final_df.head()
# Train, test data split.
df_Train=final_df.iloc[:1422,:]
df_Test=final_df.iloc[1422:,:]
y_test= df_Test['SalePrice']
df_Test.drop(['SalePrice'],axis=1,inplace=True)
X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']
X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']
# X_train, y_train, test shape.
print(X_train.shape)
print(y_train.shape)
print(df_Test.shape)
# Fill NAs with 0.
df_Test.fillna(0, inplace= True)
# RidgeCV
from sklearn.linear_model import RidgeCV
ridge_model = RidgeCV(alphas=(0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))
ridge_model.fit(X_train, y_train)
ridge_model_preds = ridge_model.predict(df_Test)
ridge_model_preds
#xgboost model
import xgboost as xgb
xgb_model = xgb.XGBRegressor(n_estimators=340, max_depth=2, learning_rate=0.2)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(df_Test)
predictions = ( ridge_model_preds + xgb_preds )/2
predictions
submission = {'Id': Id.values,'SalePrice': predictions}
final_submission = pd.DataFrame(submission)
# Exporting the final prediction into a csv
final_submission.to_csv('submission_house_price.csv', index=False)
final_submission.head()