# Importing the required Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV



from sklearn.metrics import r2_score



import math



import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df.head()
# Checking the shape of the dataframe

df.shape
df.info()
df.describe()
# Checking the percentage of misisng values in each column

round(100*df.isnull().sum()/len(df),2)
# Definning a function to plot barchart

import math

def func_bar(*args,data_df):                        

    m=math.ceil(len(args)/2)  # getting the length of arguments to determine the shape of subplots                   

    fig,axes = plt.subplots(m,2,squeeze=False, figsize = (16, 6*m))

    ax_li = axes.flatten()       # flattening the numpy array returned by subplots

    i=0

    for col in args:

        sns.countplot(x=col, data=data_df,ax=ax_li[i], order = data_df[col].value_counts().index)

        ax_li[i].set_title(col)

        ax_li[i].set_yscale('log')

        plt.tight_layout()

        i=i+1
cat_list=df.select_dtypes('object').columns.tolist()  # creating a lsit of categorical columns

func_bar(*cat_list,data_df=df)   # plotting the caregorical variables
# Filling nan values

def na_fill(*args,data):

    for col in args:

        data[col]=data[col].fillna('NA')
col_fillna=['BsmtQual', 'BsmtCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','GarageType',

            'GarageFinish', 'GarageQual', 'GarageCond','FireplaceQu', 'Fence']

na_fill(*col_fillna,data=df)
df.drop(['Alley','MiscFeature','PoolQC'],axis=1,inplace=True)

df.shape
col=df.columns[df.isnull().any()].tolist()

df[col].isnull().sum()/len(df)
df['LotFrontage'].describe()
# Since both the mean and median are almost equal, the variable lot frontage is imputed with mean

df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
# change year to date

df['YearBuilt'] = pd.to_datetime(df['YearBuilt'],format='%Y').dt.year

df['YearRemodAdd']=pd.to_datetime(df['YearRemodAdd'],format='%Y').dt.year

df['GarageYrBlt']=pd.to_datetime(df['GarageYrBlt'],format='%Y',errors = 'coerce').dt.year

df['Selling_Time'] = pd.to_datetime(df['YrSold'].astype(str)  + df['MoSold'].astype(str),format='%Y%m').dt.year
# Derived parameters

import datetime as dt

current_year=dt.date.today().year

df['Age of Building']=current_year-df['YearBuilt']

df['Age of Garage']=current_year-df['GarageYrBlt']

df['Remodelled']=current_year-df['YearRemodAdd']

df['Years after selling']=current_year-df['Selling_Time']
#dropping original columns

df.drop(['YearBuilt','YearRemodAdd','GarageYrBlt','MoSold','YrSold','Selling_Time'],axis=1,inplace=True)
# Imputing missing values in Age of Garage column. The missing values are beacuse garage was not built at all

df['Age of Garage'].fillna(value=-1,inplace=True)  # impute missing value in Age of Garage with a negative value to indicate no garage
col=df.columns[df.isnull().any()].tolist()

df[col].isnull().sum()/len(df)
df['MasVnrType'].fillna(df['MasVnrType'].mode().iloc[0],inplace=True)

df['Electrical'].fillna(df['Electrical'].mode().iloc[0],inplace=True)

df['MasVnrArea'].fillna(df['MasVnrArea'].median(),inplace=True)
# Dropping rows having misisng values

#df.dropna(inplace=True)

#df.shape
def uniq(*args,data):

    for col in args:

        print(data[col].value_counts())
col = ['Street','Utilities', 'CentralAir']

uniq(*col,data=df)
df.drop(col,axis=1,inplace=True)

df.shape
col_bx=['LotFrontage', 'LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF',

        'GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','ScreenPorch','SalePrice','Age of Building',

         'Age of Garage','Remodelled']  # list of columns whose distribution is to be checked
num_list=df.select_dtypes(exclude=['object']).columns.tolist()  # creating a list of numeric columns
nl=[col for col in num_list if col not in col_bx] # list of columns to be plotted using bar chart

func_bar(*nl[1:],data_df=df)
#function for box plot

def bx_plot(*args,data): 

    

    m=math.ceil(len(args)/2)  # getting the length f arguments to determine the shape of subplots                   

    fig,axes = plt.subplots(m,2,squeeze=False, figsize = (16, 3*m))

    ax_li = axes.flatten()       # flattening the numpy array returned by subplots

    i=0

    for col in args:

        sns.boxplot(data[col],ax=ax_li[i])  # plotting the box plot

        ax_li[i].set_title(col)

        plt.tight_layout()

        i=i+1
bx_plot(*col_bx,data=df)
# Outlier treatment is done only for specific numerical variables

q=df['LotFrontage'].quantile(0.999)

df=df[df['LotFrontage']<=q]



q=df['LotArea'].quantile(0.995)

df=df[df['LotArea']<=q]





q=df['GrLivArea'].quantile(0.999)

df=df[df['GrLivArea']<=q]



q=df['TotalBsmtSF'].quantile(0.99)

df=df[df['TotalBsmtSF']<=q]



q=df['SalePrice'].quantile(0.99)

df=df[df['SalePrice']<=q]

df.shape
sns.distplot(df['SalePrice'])
df['SalePrice']=np.log(df['SalePrice'])

sns.distplot(df['SalePrice'])
# Label encode ordered categorical variables

cols_label=['LotShape','LandSlope','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

      'HeatingQC','KitchenQual','FireplaceQu','GarageFinish','GarageQual','GarageCond','Fence']

le=LabelEncoder() 

df[cols_label]= df[cols_label].apply(lambda col: le.fit_transform(col))
col_oh=['MSZoning','LandContour', 'LotConfig', 'Neighborhood','Condition1','Condition2','BldgType','HouseStyle',

        'RoofStyle','Exterior1st','Exterior2nd','Foundation','Heating','Electrical','Functional','GarageType','GarageFinish',

        'PavedDrive','SaleType','SaleCondition','RoofMatl','MasVnrType']



df_dummy = pd.get_dummies(df[col_oh])
df_dummy.drop(['MSZoning_C (all)','LandContour_Low','LotConfig_FR3', 'Neighborhood_Blueste','Condition1_RRNe',

               'Condition2_RRAe','BldgType_2fmCon','HouseStyle_2.5Fin','RoofStyle_Shed','Exterior1st_CBlock','Exterior2nd_Other',

               'Foundation_Wood','Heating_Floor','Electrical_Mix','Functional_Sev','GarageType_2Types','PavedDrive_P', 

               'SaleType_Con','SaleCondition_AdjLand','MasVnrType_BrkCmn','RoofMatl_Metal' ],axis=1,inplace=True)
col_fin=df_dummy.columns.tolist()

df = pd.concat([df, df_dummy], axis=1)# Adding the results to the master dataframe

df.drop(col_oh,axis=1,inplace=True)# Dropping original columns

plt.figure(figsize = (20,10))        # Size of the figure

sns.heatmap(df.corr())

plt.show()
# function to eliminate redundant pairs

def redundant_pairs(df):

    '''Get diagonal and lower triangular pairs of correlation matrix'''

    pairs_to_drop = set()   # to ensure that duplicate pairs are not present

    cols = df.columns       # getting list of all solumns in dataframe  

    for i in range(0, df.shape[1]):

        for j in range(0, i+1):

            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop



# function to get highly correlated pairs

def get_top_abs_correlations(df, n):     

    cor = df.corr().abs().unstack()  # getting the absolute value of all correlation coefficienrs

    labels_to_drop = redundant_pairs(df)

    cor = cor.drop(labels=labels_to_drop).sort_values(ascending=False)

    return cor[0:n]

print("Top Absolute Correlations")

print(get_top_abs_correlations(df[df.columns[2:]], 32))
col_dr=['SaleType_New','Exterior1st_VinylSd','Exterior1st_CemntBd','Exterior1st_MetalSd','RoofStyle_Gable',

        'Exterior1st_HdBoard','GarageCars','Exterior1st_Wd Sdng','MSZoning_FV','Electrical_FuseA',

        'PavedDrive_N','Exterior1st_AsbShng','RoofMatl_Tar&Grv' ,'TotRmsAbvGrd','TotalBsmtSF','MSZoning_RL','2ndFlrSF',

        'MasVnrType_None','2ndFlrSF','Exterior1st_Stucco','Foundation_CBlock','SaleType_WD','Exterior1st_Plywood',

        'LotConfig_Corner','Heating_GasA','GarageType_Attchd','BsmtFinType2']

df.drop(col_dr,axis=1,inplace=True)

df.shape
var_to_scale=[col for col in num_list if col not in col_dr]

var_to_scale.pop(0)   # Removing Id variable

len(var_to_scale)
var_to_scale.remove('SalePrice')
# Split the data into train and test

df_train,df_test = train_test_split(df, train_size=0.7, test_size=0.3, random_state=100)
scaler = StandardScaler()

df_train[var_to_scale] = scaler.fit_transform(df_train[var_to_scale])
y_train = df_train.pop('SalePrice')

X_train = df_train[df_train.columns[1:]] # excluding the Id column
X_train.head()
y_train.head()
# list of alphas to tune

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 

 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 

 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}





ridge = Ridge()



# cross validation

folds = 5

model_cv = GridSearchCV(estimator = ridge, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            

model_cv.fit(X_train, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results = cv_results[cv_results['param_alpha']<=200]

cv_results.head()
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')



# plotting

plt.figure(figsize = (20,10))

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')

plt.grid(b=True,which='major', color='#666666', linestyle='-',alpha=0.5)

plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

plt.minorticks_on()

plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper right')

plt.show()
model_cv.best_params_
# Based on the above results alpha is ch0sen as 20

ridge = Ridge(alpha=10)

ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_train)

round(r2_score(y_train, y_pred_ridge),2)
ridge_coef = pd.Series(ridge.coef_, index = X_train.columns) 

print("The number of variables with non-zero coefficients found using Ridge regression is " + str(sum(ridge_coef != 0)) + " out of " +  str(len(ridge_coef)) + " variables")
imp_coef_ridge = abs(ridge_coef).sort_values(ascending=False) # Sorting the coefficients

plt.figure(figsize = (20,15))

import matplotlib

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef_ridge[:10].plot(kind = "barh")   # Plotting value of coefficients for most important 10 variables

plt.title("Feature importance using Ridge  Model")
imp_coef_ridge[:10]  # Displaying the coefficients of first 30 variables 
df_test[var_to_scale] = scaler.transform(df_test[var_to_scale])

y_test = df_test.pop('SalePrice')

X_test = df_test[df_test.columns[1:]]
y_test_pred_ridge = ridge.predict(X_test)

round(r2_score(y_test, y_test_pred_ridge),2)
lasso = Lasso()

# cross validation

model_cv = GridSearchCV(estimator = lasso, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            



model_cv.fit(X_train, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results.head()
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')



# plotting

plt.figure(figsize = (20,10))

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlim(0,.01)

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')

plt.grid(b=True,which='major', color='#666666', linestyle='-',alpha=0.5)

plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

plt.minorticks_on()

plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='lower left')

plt.show()
model_cv.best_params_
lasso = Lasso(alpha=.001)

lasso.fit(X_train, y_train) 
y_pred_lasso = lasso.predict(X_train)

round(r2_score(y_train, y_pred_lasso),2)
y_test_pred_lasso = lasso.predict(X_test)

round(r2_score(y_test, y_test_pred_lasso),2)
lasso_coef = pd.Series(lasso.coef_, index = X_train.columns) 

print("Lasso picked " + str(sum(lasso_coef != 0)) + " variables and eliminated the other " +  str(sum(lasso_coef == 0)) + " variables")
plt.figure(figsize = (20,15))

imp_coef_lasso = abs(lasso_coef).sort_values(ascending=False)

import matplotlib

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef_lasso[:10].plot(kind = "barh")

plt.title("Feature importance using Lasso Model")
# Reading the test data

test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test_df.shape
# Filling nan values

na_fill(*col_fillna,data=test_df)
test_df.drop(['Alley','MiscFeature','PoolQC','Street','Utilities', 'CentralAir'],axis=1,inplace=True)

test_df.shape
# Since both the mean and median are almost equal, the variable lot frontage is imputed with mean

test_df['LotFrontage']=test_df['LotFrontage'].fillna(test_df['LotFrontage'].mean())
test_df['MasVnrType'].fillna(test_df['MasVnrType'].mode().iloc[0],inplace=True)

test_df['Electrical'].fillna(test_df['Electrical'].mode().iloc[0],inplace=True)

test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].median(),inplace=True)
# change year to date

test_df['YearBuilt'] = pd.to_datetime(test_df['YearBuilt'],format='%Y').dt.year

test_df['YearRemodAdd']=pd.to_datetime(test_df['YearRemodAdd'],format='%Y').dt.year

test_df['GarageYrBlt']=pd.to_datetime(test_df['GarageYrBlt'],format='%Y',errors = 'coerce').dt.year

test_df['Selling_Time'] = pd.to_datetime(test_df['YrSold'].astype(str)  + test_df['MoSold'].astype(str),format='%Y%m').dt.year



# Derived parameters

current_year=dt.date.today().year

test_df['Age of Building']=current_year-test_df['YearBuilt']

test_df['Age of Garage']=current_year-test_df['GarageYrBlt']

test_df['Remodelled']=current_year-test_df['YearRemodAdd']

test_df['Years after selling']=current_year-test_df['Selling_Time']



#dropping original columns

test_df.drop(['YearBuilt','YearRemodAdd','GarageYrBlt','MoSold','YrSold','Selling_Time'],axis=1,inplace=True)



# Imputing missing values in Age of Garage column. The missing values are beacuse garage was not built at all

test_df['Age of Garage'].fillna(value=-1,inplace=True)  # impute missing value in Age of Garage with a negative value to indicate no garage



col=test_df.columns[test_df.isnull().any()].tolist()

test_df[col].isnull().sum()/len(test_df)



# Dropping rows having misisng values

test_df.shape
col=test_df.columns[test_df.isnull().any()].tolist()

test_df[col].isnull().sum()/len(test_df)

test_df[col].info()
cat_col = test_df[col].select_dtypes('object').columns.tolist()

num_col=test_df[col].select_dtypes(exclude=['object']).columns.tolist()

#imputing missing values in other columns

test_df[cat_col]=test_df[cat_col].fillna(test_df.mode().iloc[0])

test_df[num_col]=test_df[num_col].fillna(test_df.median())
#### Label Encoding



# Label encode ordered categorical variables

cols_label=['LotShape','LandSlope','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

      'HeatingQC','KitchenQual','FireplaceQu','GarageFinish','GarageQual','GarageCond','Fence']



le=LabelEncoder() 

test_df[cols_label]= test_df[cols_label].apply(lambda col: le.fit_transform(col))



#### One Hot Encoding



col_oh=['MSZoning','LandContour', 'LotConfig', 'Neighborhood','Condition1','Condition2','BldgType','HouseStyle',

        'RoofStyle','Exterior1st','Exterior2nd','Foundation','Heating','Electrical','Functional','GarageType','GarageFinish',

        'PavedDrive','SaleType','SaleCondition','RoofMatl','MasVnrType']

test_df_dummy = pd.get_dummies(test_df[col_oh])
col_to_drop =set(test_df_dummy.columns) -set(col_fin)

col_to_drop
test_df_dummy.drop(col_to_drop,axis=1,inplace=True)

test_df_dummy.shape
test_df = pd.concat([test_df, test_df_dummy], axis=1)# Adding the results to the master dataframe

test_df.drop(col_oh,axis=1,inplace=True)# Dropping original columns



test_df.shape
missing_cols = set(df.columns ) - set( test_df.columns )

# Add a missing column in test set with default value equal to 0

for c in missing_cols:

    test_df[c] = 0

# Ensure the order of column in the test set is in the same order than in train set

test_df = test_df[df.columns]
# Dropping the Sale price column that is added manually in the previous stage

test_df.drop('SalePrice',axis=1,inplace=True) 

test_df.shape
test_df[var_to_scale] = scaler.transform(test_df[var_to_scale])
X = test_df[test_df.columns[1:]] # excluding the Id column

X.head()
# prediction

y_pred=lasso.predict(X)

y_pred
y_pred= np.exp(y_pred)  # Converting the normalised data to original scale
sub_df=pd.DataFrame({'Id':test_df.Id,'SalePrice':y_pred})

sub_df.head()
sub_df.to_csv('submission.csv', index=False)