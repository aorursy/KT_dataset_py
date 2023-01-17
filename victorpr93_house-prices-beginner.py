# Import packages



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

pd.set_option('display.max_columns', None)

warnings.filterwarnings("ignore")



sns.set(style="white", font_scale=1.2)

# Load dataframes



df_train = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')

df_test = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')
df_train.head()
df_test.head()
df_train.describe()
df_train.describe(include='O')
print('df_train shape:', df_train.shape)

print('df_test shape:', df_test.shape)
# Function to check missing values in each dataframe



def check_missing_values(df, df_name):

    print(f'{df_name} - Missing values:')

    print('-'*30)

    columns = df.columns



    for column in columns:

        count_missing_values = df[column].isnull().sum()

        missing_values = (count_missing_values / len(df[column])) * 100

    

        if missing_values !=0:

            print(f'{column} --> {count_missing_values} values | {missing_values:.2f}%')
check_missing_values(df_train, 'DF TRAIN')
check_missing_values(df_test, 'DF TEST')
plt.figure(figsize=(20,5))

sns.heatmap(df_train.isnull(), cmap='viridis', cbar=False, yticklabels=False)

plt.title('MISSING VALUES')
na_means_donthave = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType',

                 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
# Function to drop columns with a threshold amount of missing values if that column is not an exception



columns_to_drop = []

def drop_columns_with_n_missing_values(df, threshold, exceptions):

    '''

    df_list: list of dataframes

    threshold: percentage of missing values threshold

    exceptions: don't delete if it's from that column

    '''

    

    for col in df.columns:

        if col in exceptions:

            continue

        else:

            if ((df[col].isnull().sum() / len(df[col]))*100  >= threshold):

                columns_to_drop.append(col)            
drop_columns_with_n_missing_values(df_train, 40, na_means_donthave)
print('Columns to drop:', columns_to_drop)
df_train.drop(columns_to_drop, axis=1, inplace=True)

df_test.drop(columns_to_drop, axis=1, inplace=True)
#check_missing_values(df_train, 'DF TRAIN')
#check_missing_values(df_test, 'DF TEST')
categoric_var = df_train.select_dtypes(include = ["object"]).columns

numeric_var = df_train.select_dtypes(exclude = ["object"]).columns

numeric_var = numeric_var.drop("SalePrice")



print("Numeric variables : " + str(len(numeric_var)))

print("Categoric variables : " + str(len(categoric_var)))
corr_mat = df_train.corr()



corr_mat['SalePrice'].sort_values(ascending=False)
plt.figure(figsize=(20,10))

sns.heatmap(corr_mat, linecolor='white', linewidths=0.1)
top_corr_mat = corr_mat.index[abs(corr_mat["SalePrice"])>0.5]



plt.figure(figsize=(10,10))

sns.heatmap(df_train[top_corr_mat].corr(), annot=True, cmap='coolwarm')
top_corr_features = list(top_corr_mat)
sns.pairplot(df_train[top_corr_features])
fig, axes = plt.subplots(1,2, figsize=(12,4), sharey=True)



sns.regplot(x='YearBuilt', y='SalePrice', data=df_train, ax=axes[0])

sns.regplot(x='YearRemodAdd', y='SalePrice', data=df_train, ax=axes[1])



plt.tight_layout()
fig, axes = plt.subplots(1,2, figsize=(12,4), sharey=True)



sns.regplot(x='TotalBsmtSF', y='SalePrice', data=df_train, ax=axes[0])

sns.regplot(x='1stFlrSF', y='SalePrice', data=df_train, ax=axes[1])



plt.tight_layout()
fig, axes = plt.subplots(1,2, figsize=(12,4), sharey=True)



sns.regplot(x='GrLivArea', y='SalePrice', data=df_train, ax=axes[0])

sns.regplot(x='GarageArea', y='SalePrice', data=df_train, ax=axes[1])



plt.tight_layout()
fig, axes = plt.subplots(1,2, figsize=(12,4), sharey=True)



sns.barplot(x='OverallQual', y='SalePrice', data=df_train, ax=axes[0])

sns.barplot(x='FullBath', y='SalePrice', data=df_train, ax=axes[1])



plt.tight_layout()
fig, axes = plt.subplots(1,2, figsize=(12,4), sharey=True)



sns.barplot(x='TotRmsAbvGrd', y='SalePrice', data=df_train, ax=axes[0])

sns.barplot(x='GarageCars', y='SalePrice', data=df_train, ax=axes[1])



plt.tight_layout()
# Train DF

for col in na_means_donthave:

    df_train[col].fillna('Not', inplace=True)
# Test DF

for col in na_means_donthave:

    df_test[col].fillna('Not', inplace=True)
#check_missing_values(df_train, 'DF TRAIN')
#check_missing_values(df_test, 'DF TEST')
df_train[categoric_var].head()
for col in categoric_var:

    df_train[col].fillna(df_train[col].mode()[0], inplace=True)
for col in categoric_var:

    df_test[col].fillna(df_test[col].mode()[0], inplace=True)
df_train[numeric_var].head()
for col in numeric_var:

    df_train[col].fillna(df_train[col].median(), inplace=True)
for col in numeric_var:

    df_test[col].fillna(df_test[col].median(), inplace=True)
check_missing_values(df_train, 'DF TRAIN')
check_missing_values(df_test, 'DF TEST')
df_full = pd.concat((df_train, df_test), axis=0)
df_full = pd.get_dummies(df_full, drop_first=True)
df_full['SalePrice']
df_train = df_full[df_full['SalePrice'].isnull() == False]
df_test = df_full[df_full['SalePrice'].isnull() == True]



df_test.drop('SalePrice', axis=1, inplace=True)
print('df_train shape:', df_train.shape)

print('df_test shape:', df_test.shape)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor



from sklearn.preprocessing import StandardScaler



from sklearn.metrics import mean_squared_error
# dictionary to append the RMSE results of each model

results = {}
X = df_train.drop('SalePrice', axis=1)

y = df_train['SalePrice']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
reg = LinearRegression()



reg.fit(X_train, y_train)



reg_pred = reg.predict(X_test)
reg_rmse = np.sqrt(mean_squared_error(y_test, reg_pred))

print('Linear Regression (simple) RMSE:', reg_rmse)



results['Linear Regression (simple)'] = reg_rmse
rf = RandomForestRegressor(random_state=1)



rf.fit(X_train, y_train)



rf_pred = rf.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print('Random Forest RMSE:', rf_rmse)



results['Random Forest'] = rf_rmse
xgb = XGBRegressor(objective='reg:squarederror')



xgb.fit(X_train, y_train)



xgb_pred = xgb.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

print('XGB RMSE:', xgb_rmse)



results['XGB'] = xgb_rmse
pd.options.display.float_format = '{:.4f}'.format

df_results = pd.DataFrame(results.items(), columns=['Algorithm', 'RMSE'])



df_results.sort_values('RMSE')
xgb.fit(X, y)



xgb_pred = xgb.predict(df_test)



submission_xgb = pd.DataFrame({'Id':df_test.index, 'SalePrice':xgb_pred})



submission_xgb.to_csv('submission_xgb.csv', index=False)