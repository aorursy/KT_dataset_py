import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv(r'../input/train.csv')
df.head()
pd.options.display.max_columns = 100
df.head()
df.shape
df.info()
df_null=(df.isnull().sum()/len(df))*100
df_null.sort_values(ascending=False)
df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'], axis=1, inplace=True)
df_null=(df.isnull().sum()/len(df))*100
df_null.sort_values(ascending=False)
df.info()
df.head()
df['GarageCond'].fillna(df['GarageCond'].mode()[0], inplace=True)
df['GarageCond'].astype('category').value_counts()
df_null=(df.isnull().sum()/len(df))*100
df_null.sort_values(ascending=False).head()
df['GarageType'].fillna(df['GarageType'].mode()[0], inplace=True)
df['GarageType'].astype('category').value_counts()
df_null=(df.isnull().sum()/len(df))*100

df_null.sort_values(ascending=False).head()
df['GarageQual'].fillna(df['GarageQual'].mode()[0], inplace=True)

df['GarageQual'].astype('category').value_counts()
df_null=(df.isnull().sum()/len(df))*100

df_null.sort_values(ascending=False).head()
df['GarageYrBlt'].fillna(df['GarageYrBlt'].mode()[0], inplace=True)

df['GarageYrBlt'].astype('category').value_counts().head()
df_null=(df.isnull().sum()/len(df))*100

df_null.sort_values(ascending=False).head()
df['GarageFinish'].fillna(df['GarageFinish'].mode()[0], inplace=True)

df['GarageFinish'].astype('category').value_counts()
df_null=(df.isnull().sum()/len(df))*100

df_null.sort_values(ascending=False).head()
df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0], inplace=True)

df['BsmtFinType2'].astype('category').value_counts()
df_null=(df.isnull().sum()/len(df))*100

df_null.sort_values(ascending=False).head()
df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0], inplace=True)

df['BsmtExposure'].astype('category').value_counts()
df_null=(df.isnull().sum()/len(df))*100

df_null.sort_values(ascending=False).head()
df['BsmtQual'].fillna(df['BsmtQual'].mode()[0], inplace=True)

df['BsmtQual'].astype('category').value_counts()
df_null=(df.isnull().sum()/len(df))*100

df_null.sort_values(ascending=False).head()
df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0], inplace=True)

df['BsmtFinType1'].astype('category').value_counts()
df_null=(df.isnull().sum()/len(df))*100

df_null.sort_values(ascending=False).head()
df['BsmtCond'].fillna(df['BsmtCond'].mode()[0], inplace=True)

df['BsmtCond'].astype('category').value_counts()
df_null=(df.isnull().sum()/len(df))*100

df_null.sort_values(ascending=False).head()
df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0], inplace=True)

df['MasVnrArea'].astype('category').value_counts().head()
df_null=(df.isnull().sum()/len(df))*100

df_null.sort_values(ascending=False).head()
df['MasVnrType'].fillna(df['MasVnrType'].mode()[0], inplace=True)

df['MasVnrType'].astype('category').value_counts()
df_null=(df.isnull().sum()/len(df))*100

df_null.sort_values(ascending=False).head()
df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)

df['Electrical'].astype('category').value_counts()
df_null=(df.isnull().sum()/len(df))*100

df_null.sort_values(ascending=False).head()
df.info()
sns.barplot(x="OverallQual", y="Id", data=df)
sns.barplot(x="OverallCond", y="Id", data=df)
df['OverallQual'].astype('category').value_counts()
df['OverallCond'].astype('category').value_counts()
df[['OverallQual','OverallCond']] = df[['OverallQual','OverallCond']].astype(str)
df.info()
df['YearBuilt_diff'] = df['YrSold'] - df['YearBuilt']

df['YearRemodAdd_diff'] = df['YrSold'] - df['YearRemodAdd']

df['GarageYrBlt_diff'] = df['YrSold'] - df['GarageYrBlt']
df.drop(['YearBuilt', 'YearRemodAdd','GarageYrBlt', 'YrSold', 'MoSold'], axis=1, inplace = True )
df_1 = df.select_dtypes(include=['float64', 'int64'])
df_1.head()
df_1=df
df_1.drop(['Id'], axis=1, inplace=True)
df_1.head()
X=df_1
y = X.pop('SalePrice')
X.head()
df_categ = X.select_dtypes(include=['object'])

df_categ.head()
df_dummies = pd.get_dummies(df_categ, drop_first = True)
df_dummies.shape
df_categ.columns
df_dummies.head()
X = X.drop(list(df_categ.columns), axis=1)
X = pd.concat([X, df_dummies], axis=1)
X.shape
from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit_transform(X)
X.columns
X.head()
cols=X.columns
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.7,test_size = 0.3, random_state=100)
# Possibles values of alpha to tune

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 

 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 

 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0, 50.0, 100.0,

 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]}
from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV
ridge = Ridge()
folds = 5

model_cv = GridSearchCV(estimator = ridge, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            

model_cv.fit(X_train, y_train) 
model_cv.cv_results_
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results.head()
cv_results = cv_results[cv_results['param_alpha']<=1000]
cv_results.head()
cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')



# plotting

plt.figure(figsize=(8,5))

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.grid(True)

plt.show()
alpha = 15

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)
ridge_coeff = list(zip(X_train.columns, sorted(abs(ridge.coef_), reverse=True)))
#from sklearn.metrics import r2_score

#r2_score(y_test, y_pred)
ridge_coeff
pd.DataFrame(ridge_coeff, columns={'Feature','Coefficient'}).head(5)
from sklearn.linear_model import Lasso
lasso = Lasso()
model_cv = GridSearchCV(estimator = lasso, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1) 
model_cv.fit(X_train, y_train) 
import warnings

warnings.filterwarnings('ignore')
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results
cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')
plt.figure(figsize=(8,5))

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')



plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.grid(True)

plt.show()
alpha =15

lasso = Lasso(alpha=alpha)

lasso.fit(X_train, y_train) 
lasso_coeff = list(zip(X_train.columns, sorted(abs(lasso.coef_),reverse=True)))
#from sklearn.metrics import r2_score

#r2_score(y_test, y_pred_model_4)
y_pred=lasso.predict(X_test)
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
pd.DataFrame(lasso_coeff, columns={'Feature','Coefficient'}).head(10)
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor
def calculate_VIF(data_frame):

    vif = pd.DataFrame(columns = ['Features', 'VIF'])

    vif['Features'] = data_frame.columns

    vif['VIF'] = [variance_inflation_factor(data_frame.values, i) for i in range(data_frame.shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.sort_values(by = "VIF", ascending = False)

    return vif
X_train_copy1 = X_train
X_train_copy1 = sm.add_constant(X_train_copy1)
lr_model_1= sm.OLS(y_train, X_train_copy1).fit()
lr_model_1.params
lr_model_1.summary()
vif_model_1 = calculate_VIF(X_train)

vif_model_1
drop_cols= ['Exterior2nd_Other', 'ExterCond_Po', 'Electrical_Mix']
X_train.drop(drop_cols, axis=1, inplace = True)
drop_columns = vif_model_1[vif_model_1['VIF'] >= 1.7]['Features']
drop_columns
for col in drop_columns:

    X_train.drop([col], axis=1, inplace = True)
X_train.shape
X_train_2 = X_train
X_train_2 = sm.add_constant(X_train_2)
lr_model_2 = sm.OLS(y_train, X_train_2).fit()
lr_model_2.params
lr_model_2.summary()
vif_model2_col = calculate_VIF(X_train)

vif_model2_col
cols_drop = ['LotConfig_FR3', 'LotConfig_FR2']

X_train.drop(cols_drop, axis=1, inplace = True)
X_train_3 = X_train
X_train_3 = sm.add_constant(X_train_3)
lr_model_3 = sm.OLS(y_train, X_train_3).fit()
lr_model_3.params
lr_model_3.summary()
vif_model3_cols = calculate_VIF(X_train)

vif_model3_cols
cols_drop = ['SaleType_ConLw', 'Foundation_Wood', 'Neighborhood_Blueste', 'Utilities_NoSeWa', 'Condition1_RRNe']

X_train.drop(cols_drop, axis=1, inplace = True)
X_train_4 = X_train
X_train_4 = sm.add_constant(X_train_4)
lr_model_4 = sm.OLS(y_train, X_train_4).fit()
lr_model_4.params
lr_model_4.summary()
vif_model4_cols = calculate_VIF(X_train)

vif_model4_cols
y_train_price = lr_model_4.predict(X_train_4)
fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 50)
X_train.columns
X_test_model_4=X_test[['BsmtHalfBath', 'WoodDeckSF', '3SsnPorch', 'ScreenPorch', 'MiscVal',

       'LotShape_IR2', 'HeatingQC_Po', 'SaleType_CWD', 'SaleType_Con',

       'SaleType_ConLI', 'SaleType_Oth', 'SaleCondition_Family']]
X_test_model_4.head()
X_test_model_4 = sm.add_constant(X_test_model_4)
y_pred_model_4 = lr_model_4.predict(X_test_model_4)
y_pred_model_4.shape
y_test.shape
c = [i for i in range(1,439,1)]



fig = plt.figure()

plt.figure(figsize = (25, 10))



plt.plot(c,y_test, color="red", linewidth=2, linestyle="-")     

plt.plot(c,y_pred_model_4, color="green",  linewidth=2, linestyle="-")  



fig.suptitle('Actual vs Predicted', fontsize=15)              



plt.xlabel('Index', fontsize=15)                              

plt.ylabel('SalePrice', fontsize=15) 
y_pred_model_4.head()
from sklearn.metrics import r2_score

r2_score(y_test, y_pred_model_4)
y_test
y_pred_model_4