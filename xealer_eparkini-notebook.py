#data manipulating libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

pd.set_option('display.max_columns', 81)

plt.style.use(style='ggplot')

plt.rcParams['figure.figsize'] = (10,8)



#statistics libraries

from scipy import stats

from scipy.stats import skew,norm, kurtosis

from scipy.stats.stats import pearsonr

#Call the datasets

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

test_df1 = pd.read_csv('../input/test.csv')
train_df.head()
#remove IDs from train_df and test_df

train_df.drop("Id", axis = 1,inplace = True)

test_df.drop("Id",axis = 1, inplace= True)
train_df.head()
train_df.info()
train_df['SalePrice'].describe()
sns.distplot(train_df.SalePrice);

plt.ylabel('Frequency');

plt.title('SalePrice Distribution');

print("skewness: %f" % train_df['SalePrice'].skew())

print("kurtosis: %f" % train_df['SalePrice'].kurtosis())
#log transform the target 

train_df["SalePrice"] = np.log(train_df["SalePrice"])



#Kernel Density plot

sns.distplot(train_df.SalePrice, fit=norm);

plt.ylabel('Frequency')

plt.title('SalePrice distribution');
print("skewness: %f" % train_df['SalePrice'].skew())

print("kurtosis: %f" % train_df['SalePrice'].kurtosis())
corrmap = train_df.corr()

top_corr_features = corrmap.index[abs(corrmap["SalePrice"])>0.5]

g = sns.heatmap(train_df[top_corr_features].corr(),annot=True,cmap="PiYG")
#Checking correlation against the SalePrice

col ='TotalBsmtSF'

data = pd.concat([train_df['SalePrice'],train_df[col]],axis=1)

data.plot.scatter(x=col, y='SalePrice', color='Brown');

plt.show()
first_quartile = train_df['TotalBsmtSF'].describe()['25%']

third_quartile = train_df['TotalBsmtSF'].describe()['75%']



# Interquartile range

iqr = third_quartile - first_quartile



# Remove outliers

train_df = train_df[(train_df['TotalBsmtSF'] > (first_quartile - 3 * iqr)) &

            (train_df['TotalBsmtSF'] < (third_quartile + 3 * iqr))]
col ='TotalBsmtSF'

data = pd.concat([train_df['SalePrice'],train_df[col]],axis=1)

data.plot.scatter(x=col, y='SalePrice', color='Brown');
col ='GrLivArea'

data =pd.concat([train_df['SalePrice'], train_df[col]], axis=1)

data.plot.scatter(x=col, y='SalePrice', color='brown');
#scatter plot GarageArea/SalePrice

col = 'GarageArea'

res =pd.concat([train_df['SalePrice'], train_df[col]], axis=1)

res.plot.scatter(x=col,y='SalePrice', color='brown');

house_df = pd.concat((train_df.loc[:, 'MSSubClass': 'SaleCondition'], test_df.loc[:,'MSSubClass':'SaleCondition']))



# Function to calculate missing values by column

def missing_values_table(df):

        # Total missing values

        mis_val = house_df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * house_df.isnull().sum() / len(house_df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("This dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns
missing_values_table(house_df)
total = house_df.isnull().sum().sort_values(ascending=False)

percent = (house_df.isnull().sum()/house_df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



percent_data = percent.head(20)

percent_data.plot(kind="bar", figsize = (8,6), fontsize = 10)

plt.xlabel("Columns", fontsize = 20)

plt.ylabel("Count", fontsize = 20)

plt.title("Total Missing Value (%)", fontsize = 20)
#Checking on missing values - Categorical variables

house_df.select_dtypes(include='object').isnull().sum()[house_df.select_dtypes(include='object').isnull().sum()>0]
house_df.shape
for column in ('Alley', 'Utilities', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

           'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 

            'MiscFeature', 'MSSubClass'):

    house_df[column] = house_df[column].fillna('None')
for column in ('MSZoning', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'SaleType', 'Electrical'):

    house_df[column] = house_df[column].fillna(house_df[column].mode()[0])
#functional: NA is typical

house_df["Functional"] = house_df["Functional"].fillna('Typ')
#fill missing value with median Lot frontage of all the neighboorhood

house_df["LotFrontage"] = house_df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
#Numerical variables

house_df.select_dtypes(include = ['float', 'int']).isnull().sum()[house_df.select_dtypes(include = ['int', 'float']).isnull()

                                                                 .sum() > 0]
#Some "NAs" means "None" (which I will fill with 0) or means "Not Available" (which I will fill with mean)

for col in ('MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt',

            'GarageCars', 'GarageArea'):

    house_df[col] = house_df[col].fillna(0)
print(house_df.isnull().sum().sum())
#house_df = house_df.drop(['Utilities'], axis=1)
#Transforming required numerical features to categorical 

house_df['MSSubClass']= house_df['MSSubClass'].apply(str)

house_df['OverallCond'] =house_df['OverallCond'].astype(str)

house_df['YrSold'] = house_df['YrSold'].astype(str)

house_df['MoSold'] = house_df['MoSold'].astype(str)
from sklearn.preprocessing import LabelEncoder
#Label Encoding some categorical variables

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



#apply LabelEncoder to categorical features

for c in cols:

    label = LabelEncoder()

    label.fit(list(house_df[c].values))

    house_df[c] = label.transform(list(house_df[c].values))
print(house_df.shape)
house_df['TotalSF'] = house_df['TotalBsmtSF'] + house_df['1stFlrSF'] + house_df['2ndFlrSF']
# skewed numeric features 

numeric_features = house_df.dtypes[house_df.dtypes != "object"].index



skewed_features = house_df[numeric_features].apply(lambda x : skew (x.dropna())).sort_values(ascending=False)

#compute skewness

print ("\nskew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_features})   

skewness.head(7)
#Box cox transforming

skewness = skewness[abs(skewness) > 0.75]

print ("There are {} skewed numerical features to box cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p 

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    house_df[feat] = boxcox1p(house_df[feat], lam)
#adding dummies to categorical features

house_df = pd.get_dummies(house_df)

print(house_df.shape)
house_df.head()
ntrain_df = train_df.shape[0]

ntest_df = test_df.shape[0]

y_train= train_df.SalePrice.values

train_df = house_df[:ntrain_df]

test_df = house_df[ntrain_df:]
train_df.head()
 # Import the necessary modules

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split



y = y_train

X = house_df[:ntrain_df].values



# Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)



# Create the regressor: reg_all

reg_all = LinearRegression()



# Fit the regressor to the training data

reg_all.fit(X_train, y_train)



# Predict on the test data: y_pred

y_pred = reg_all.predict(X_test)



# Compute and print R^2 and RMSE

#commonly used metric to evaluate regression models



print("Linear Regression Test Set:")

print("R^2: {}".format(reg_all.score(X_test, y_test)))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Root Mean Squared Error: {}".format(rmse))
print("Linear Regression Train Set:")

y_pred = reg_all.predict(X_train)

print("R^2: {}".format(reg_all.score(X_train, y_train)))

rmse = np.sqrt(mean_squared_error(y_train, y_pred))

print("Root Mean Squared Error: {}".format(rmse))
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(X_train,y_train)
ridgee = ridge.predict(X_test)

# Compute and print R^2 and RMSE

#commonly used metric to evaluate regression models

print("Ridge Regression Test Set:")

print("R^2: {}".format(ridge.score(X_test, y_test)))

rmse = np.sqrt(mean_squared_error(y_test, ridgee))

print("Root Mean Squared Error: {}".format(rmse))
print("Ridge Regression Train Set:")

y_pred = ridge.predict(X_train)

print("R^2: {}".format(ridge.score(X_train, y_train)))

rmse = np.sqrt(mean_squared_error(y_train, y_pred))

print("Root Mean Squared Error: {}".format(rmse))
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.001, normalize=True)
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)

# Compute and print R^2 and RMSE

#commonly used metric to evaluate regression models

print("Lasso Regression Test Set:")

print("R^2: {}".format(lasso.score(X_test, y_test)))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Root Mean Squared Error: {}".format(rmse))
print("Lasso Regression Train Set:")

y_pred = lasso.predict(X_train)

print("R^2: {}".format(lasso.score(X_train, y_train)))

rmse = np.sqrt(mean_squared_error(y_train, y_pred))

print("Root Mean Squared Error: {}".format(rmse))
from sklearn import tree

from sklearn.tree import DecisionTreeRegressor
regr_tree = DecisionTreeRegressor(max_depth=2)

regr_tree.fit(X_train, y_train)

y_pred = regr_tree.predict(X_test)

# Compute and print R^2 and RMSE

#commonly used metric to evaluate regression models

print("Decision Trees Test Set:")

print("R^2: {}".format(regr_tree.score(X_test, y_test)))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Root Mean Squared Error: {}".format(rmse))
print("Decision Trees Train Set:")

y_pred = regr_tree.predict(X_train)

print("R^2: {}".format(regr_tree.score(X_train, y_train)))

rmse = np.sqrt(mean_squared_error(y_train, y_pred))

print("Root Mean Squared Error: {}".format(rmse))
from sklearn.ensemble import RandomForestRegressor

random = RandomForestRegressor()



random.fit(X_train, y_train)

y_pred1 = random.predict(X_test)

# Compute and print R^2 and RMSE

#commonly used metric to evaluate regression models

print("Random Forest Test Set:")

print("R^2: {}".format(random.score(X_test, y_test)))

rmse = np.sqrt(mean_squared_error(y_test, y_pred1))

print("Root Mean Squared Error: {}".format(rmse))
print("Random Forest Train Set:")

y_pred = random.predict(X_train)

print("R^2: {}".format(random.score(X_train, y_train)))

rmse = np.sqrt(mean_squared_error(y_train, y_pred))

print("Root Mean Squared Error: {}".format(rmse))
simple_avg = (reg_all.predict(test_df.values) + ridge.predict(test_df.values) + lasso.predict(test_df.values) + random.predict(test_df.values) ) / 4

# shape to export

from pandas import DataFrame

my_pred = np.expm1(simple_avg)



new_res = my_pred *1

result = pd.DataFrame({'Id':test_df1.Id, 'SalePrice':new_res})

result.to_csv('Kwazi_Sample_submission.csv', index=False)

result.head()