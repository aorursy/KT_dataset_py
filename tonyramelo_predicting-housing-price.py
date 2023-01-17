# This first set of packages include Pandas, for data manipulation, numpy for mathematical computation and matplotlib & seaborn, for visualisation.

import pandas as pd

import numpy as np

from IPython.display import display

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set(style='white', context='notebook', palette='deep')

print('Data Manipulation, Mathematical Computation and Visualisation packages imported!')



# Stats Package

from scipy import stats

from scipy.stats import skew, norm

from scipy.special import boxcox1p

from scipy.stats.stats import pearsonr

print('Stats Package imported!')



# Metrics used for measuring the accuracy and performance of the models

from sklearn import metrics

from sklearn.metrics import mean_squared_error

print('Metrics packages imported!')



# Algorithms used for modeling

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.kernel_ridge import KernelRidge

import xgboost as xgb

print('Algorithm packages imported!')



# Pipeline and scaling preprocessing will be used for models that are sensitive

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import SelectFromModel

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

print('Pipeline and Preprocessing Packages imported!')



# Model selection packages used for sampling dataset and optimising parameters

from sklearn import model_selection

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import ShuffleSplit

print('Model Packages imported!')



# Set visualisation colours

mycols = ["#66c2ff", "#5cd6d6", "#00cc99", "#85e085", "#ffd966", "#ffb366", "#ffb3b3", "#dab3ff", "#c2c2d6"]

sns.set_palette(palette = mycols, n_colors = 4)

print('My colours are ready! :)')



# To ignore annoying warning

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

warnings.filterwarnings("ignore", category=DeprecationWarning)

print('Deprecation warning will be ignored!')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print('Training and test data have been imported!')



#Save the 'Id' column

train_ID = train['Id']

test_ID = test['Id']



#Drop Id from train and test data

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)



print("\nTraining shape: {}".format(train.shape))

print("Test shape: {}".format(test.shape))

train.head()
# Count the column types

train.dtypes.value_counts()
train['SalePrice'].describe()
#histogram of SalePrice

sns.distplot(train['SalePrice']);
#saleprice correlation matrix

correlation = train.corr()

k = 10 #number of variables for heatmap

corr = correlation.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[corr].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cmap="Greens", cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=corr.values, xticklabels=corr.values)

plt.show()
#OverallQual

var = 'OverallQual'

vis = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=vis)

fig.axis(ymin=0, ymax=800000);



#GrLivArea

fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()





#GarageCars

var = 'GarageCars'

vis = pd.concat([train['SalePrice'], train[var]], axis=1)

fig = sns.boxplot(x=var, y="SalePrice", data=vis)

fig.axis(ymin=0, ymax=800000);
plt.subplots(figsize=(15, 5))



plt.subplot(1, 2, 1)

g = sns.regplot(x=train['GrLivArea'], y=train['SalePrice'], fit_reg=False).set_title("Before")



# Delete outliers

plt.subplot(1, 2, 2)                                                                                

train = train.drop(train[(train['GrLivArea']>4000)].index)

g = sns.regplot(x=train['GrLivArea'], y=train['SalePrice'], fit_reg=False).set_title("After")
#Save the length of train and test data so we could split later when we split later when we do modelling at the end!

ntrain = train.shape[0]

ntest = test.shape[0]



# Also save the target value, as we will remove this

targetval = train.SalePrice.values



# concatenate training and test data into data

data = pd.concat((train, test)).reset_index(drop=True)

data.drop(['SalePrice'], axis=1, inplace=True)



print("Concated Data Shape: {}".format(data.shape))
# Using data description, fill these missing values with "None"

for col in ("PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",

           "GarageType", "GarageFinish", "GarageQual", "GarageCond",

           "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",

            "BsmtFinType2", "MSSubClass", "MasVnrType"):

    data[col] = data[col].fillna("None")

print("'None' - treated...")



# The area of the lot out front is likely to be similar to the houses in the local neighbourhood

# Therefore, let's use the median value of the houses in the neighbourhood to fill this feature

data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))

print("'LotFrontage' - treated...")



# Using data description, fill these missing values with 0 

for col in ("GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", 

           "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "MasVnrArea",

           "BsmtFullBath", "BsmtHalfBath"):

    data[col] = data[col].fillna(0)

print("'0' - treated...")





# Fill these features with their mode, the most commonly occuring value. This is okay since there are a low number of missing values for these features

data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])

data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])

data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])

data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])

data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])

data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])

data["Functional"] = data["Functional"].fillna(data['Functional'].mode()[0])

print("'mode' - treated...")



data_na = data.isnull().sum()

print("Features with missing values: ", data_na.drop(data_na[data_na == 0].index))
# From inspection, we can remove Utilities

data = data.drop(['Utilities'], axis=1)



data_na = data.isnull().sum()

print("Features with missing values: ", len(data_na.drop(data_na[data_na == 0].index)))
#All the features that are in metrics:



# Quadratic

data["OverallQual-2"] = data["OverallQual"] ** 2

data["GrLivArea-2"] = data["GrLivArea"] ** 2

data["GarageCars-2"] = data["GarageCars"] ** 2

data["GarageArea-2"] = data["GarageArea"] ** 2

data["TotalBsmtSF-2"] = data["TotalBsmtSF"] ** 2

data["1stFlrSF-2"] = data["1stFlrSF"] ** 2

data["FullBath-2"] = data["FullBath"] ** 2

data["TotRmsAbvGrd-2"] = data["TotRmsAbvGrd"] ** 2

data["Fireplaces-2"] = data["Fireplaces"] ** 2

data["MasVnrArea-2"] = data["MasVnrArea"] ** 2

data["BsmtFinSF1-2"] = data["BsmtFinSF1"] ** 2

data["LotFrontage-2"] = data["LotFrontage"] ** 2

data["WoodDeckSF-2"] = data["WoodDeckSF"] ** 2

data["OpenPorchSF-2"] = data["OpenPorchSF"] ** 2

data["2ndFlrSF-2"] = data["2ndFlrSF"] ** 2

print("Quadratics done!...")



# Cubic

data["OverallQual-3"] = data["OverallQual"] ** 3

data["GrLivArea-3"] = data["GrLivArea"] ** 3

data["GarageCars-3"] = data["GarageCars"] ** 3

data["GarageArea-3"] = data["GarageArea"] ** 3

data["TotalBsmtSF-3"] = data["TotalBsmtSF"] ** 3

data["1stFlrSF-3"] = data["1stFlrSF"] ** 3

data["FullBath-3"] = data["FullBath"] ** 3

data["TotRmsAbvGrd-3"] = data["TotRmsAbvGrd"] ** 3

data["Fireplaces-3"] = data["Fireplaces"] ** 3

data["MasVnrArea-3"] = data["MasVnrArea"] ** 3

data["BsmtFinSF1-3"] = data["BsmtFinSF1"] ** 3

data["LotFrontage-3"] = data["LotFrontage"] ** 3

data["WoodDeckSF-3"] = data["WoodDeckSF"] ** 3

data["OpenPorchSF-3"] = data["OpenPorchSF"] ** 3

data["2ndFlrSF-3"] = data["2ndFlrSF"] ** 3

print("Cubics done!...")



# Square Root

data["OverallQual-Sq"] = np.sqrt(data["OverallQual"])

data["GrLivArea-Sq"] = np.sqrt(data["GrLivArea"])

data["GarageCars-Sq"] = np.sqrt(data["GarageCars"])

data["GarageArea-Sq"] = np.sqrt(data["GarageArea"])

data["TotalBsmtSF-Sq"] = np.sqrt(data["TotalBsmtSF"])

data["1stFlrSF-Sq"] = np.sqrt(data["1stFlrSF"])

data["FullBath-Sq"] = np.sqrt(data["FullBath"])

data["TotRmsAbvGrd-Sq"] = np.sqrt(data["TotRmsAbvGrd"])

data["Fireplaces-Sq"] = np.sqrt(data["Fireplaces"])

data["MasVnrArea-Sq"] = np.sqrt(data["MasVnrArea"])

data["BsmtFinSF1-Sq"] = np.sqrt(data["BsmtFinSF1"])

data["LotFrontage-Sq"] = np.sqrt(data["LotFrontage"])

data["WoodDeckSF-Sq"] = np.sqrt(data["WoodDeckSF"])

data["OpenPorchSF-Sq"] = np.sqrt(data["OpenPorchSF"])

data["2ndFlrSF-Sq"] = np.sqrt(data["2ndFlrSF"])

print("Roots done!...")
#Basement Quality

data['BsmtQual'] = data['BsmtQual'].map({"None":0, "Fa":1, "TA":2, "Gd":3, "Ex":4})

data['BsmtQual'].unique()
#Basement Condition

data['BsmtCond'] = data['BsmtCond'].map({"None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})

data['BsmtCond'].unique()
#Basement Exposure

data['BsmtExposure'] = data['BsmtExposure'].map({"None":0, "No":1, "Mn":2, "Av":3, "Gd":4})

data['BsmtExposure'].unique()
#BsmtFinType

data = pd.get_dummies(data, columns = ["BsmtFinType1"], prefix="BsmtFinType1")
#BsmtFinSF1_Band

data['BsmtFinSF1_Band'] = pd.cut(data['BsmtFinSF1'], 4)

data['BsmtFinSF1_Band'].unique()
#BsmtFinSF1

data.loc[data['BsmtFinSF1']<=1002.5, 'BsmtFinSF1'] = 1

data.loc[(data['BsmtFinSF1']>1002.5) & (data['BsmtFinSF1']<=2005), 'BsmtFinSF1'] = 2

data.loc[(data['BsmtFinSF1']>2005) & (data['BsmtFinSF1']<=3007.5), 'BsmtFinSF1'] = 3

data.loc[data['BsmtFinSF1']>3007.5, 'BsmtFinSF1'] = 4

data['BsmtFinSF1'] = data['BsmtFinSF1'].astype(int)

data.drop('BsmtFinSF1_Band', axis=1, inplace=True)

data = pd.get_dummies(data, columns = ["BsmtFinSF1"], prefix="BsmtFinSF1")
#BsmtFinType2

data = pd.get_dummies(data, columns = ["BsmtFinType2"], prefix="BsmtFinType2")

data['BsmtFinSf2_Flag'] = data['BsmtFinSF2'].map(lambda x:0 if x==0 else 1)

data.drop('BsmtFinSF2', axis=1, inplace=True)
#BsmtUnfSF_Band

data['BsmtUnfSF_Band'] = pd.cut(data['BsmtUnfSF'], 3)

data['BsmtUnfSF_Band'].unique()
data.loc[data['BsmtUnfSF']<=778.667, 'BsmtUnfSF'] = 1

data.loc[(data['BsmtUnfSF']>778.667) & (data['BsmtUnfSF']<=1557.333), 'BsmtUnfSF'] = 2

data.loc[data['BsmtUnfSF']>1557.333, 'BsmtUnfSF'] = 3

data['BsmtUnfSF'] = data['BsmtUnfSF'].astype(int)



data.drop('BsmtUnfSF_Band', axis=1, inplace=True)



data = pd.get_dummies(data, columns = ["BsmtUnfSF"], prefix="BsmtUnfSF")
#totalBsmtSF_Band

data['TotalBsmtSF_Band'] = pd.cut(data['TotalBsmtSF'], 10)

data['TotalBsmtSF_Band'].unique()

data.loc[data['TotalBsmtSF']<=509.5, 'TotalBsmtSF'] = 1

data.loc[(data['TotalBsmtSF']>509.5) & (data['TotalBsmtSF']<=1019), 'TotalBsmtSF'] = 2

data.loc[(data['TotalBsmtSF']>1019) & (data['TotalBsmtSF']<=1528.5), 'TotalBsmtSF'] = 3

data.loc[(data['TotalBsmtSF']>1528.5) & (data['TotalBsmtSF']<=2038), 'TotalBsmtSF'] = 4

data.loc[(data['TotalBsmtSF']>2038) & (data['TotalBsmtSF']<=2547.5), 'TotalBsmtSF'] = 5

data.loc[(data['TotalBsmtSF']>2547.5) & (data['TotalBsmtSF']<=3057), 'TotalBsmtSF'] = 6

data.loc[(data['TotalBsmtSF']>3057) & (data['TotalBsmtSF']<=3566.5), 'TotalBsmtSF'] = 7

data.loc[data['TotalBsmtSF']>3566.5, 'TotalBsmtSF'] = 8

data['TotalBsmtSF'] = data['TotalBsmtSF'].astype(int)



data.drop('TotalBsmtSF_Band', axis=1, inplace=True)



data = pd.get_dummies(data, columns = ["TotalBsmtSF"], prefix="TotalBsmtSF")

#1stFlrSF_Band'

data['1stFlrSF_Band'] = pd.cut(data['1stFlrSF'], 6)

data['1stFlrSF_Band'].unique()
data.loc[data['1stFlrSF']<=1127.5, '1stFlrSF'] = 1

data.loc[(data['1stFlrSF']>1127.5) & (data['1stFlrSF']<=1921), '1stFlrSF'] = 2

data.loc[(data['1stFlrSF']>1921) & (data['1stFlrSF']<=2714.5), '1stFlrSF'] = 3

data.loc[(data['1stFlrSF']>2714.5) & (data['1stFlrSF']<=3508), '1stFlrSF'] = 4

data.loc[(data['1stFlrSF']>3508) & (data['1stFlrSF']<=4301.5), '1stFlrSF'] = 5

data.loc[data['1stFlrSF']>4301.5, '1stFlrSF'] = 6

data['1stFlrSF'] = data['1stFlrSF'].astype(int)



data.drop('1stFlrSF_Band', axis=1, inplace=True)



data = pd.get_dummies(data, columns = ["1stFlrSF"], prefix="1stFlrSF")
#2ndFlrSF_Band'

data['2ndFlrSF_Band'] = pd.cut(data['2ndFlrSF'], 6)

data['2ndFlrSF_Band'].unique()
data.loc[data['2ndFlrSF']<=310.333, '2ndFlrSF'] = 1

data.loc[(data['2ndFlrSF']>310.333) & (data['2ndFlrSF']<=620.667), '2ndFlrSF'] = 2

data.loc[(data['2ndFlrSF']>620.667) & (data['2ndFlrSF']<=931), '2ndFlrSF'] = 3

data.loc[(data['2ndFlrSF']>931) & (data['2ndFlrSF']<=1241.333), '2ndFlrSF'] = 4

data.loc[(data['2ndFlrSF']>1241.333) & (data['2ndFlrSF']<=1551.667), '2ndFlrSF'] = 5

data.loc[data['2ndFlrSF']>1551.667, '2ndFlrSF'] = 6

data['2ndFlrSF'] = data['2ndFlrSF'].astype(int)



data.drop('2ndFlrSF_Band', axis=1, inplace=True)



data = pd.get_dummies(data, columns = ["2ndFlrSF"], prefix="2ndFlrSF")
#LowQualFinSF

data['LowQualFinSF_Flag'] = data['LowQualFinSF'].map(lambda x:0 if x==0 else 1)

data.drop('LowQualFinSF', axis=1, inplace=True)
#Number of baths

data['TotalBathrooms'] = data['BsmtHalfBath'] + data['BsmtFullBath'] + data['HalfBath'] + data['FullBath']



columns = ['BsmtHalfBath', 'BsmtFullBath', 'HalfBath', 'FullBath']

data.drop(columns, axis=1, inplace=True)
#Kitchen Quality

data['KitchenQual'] = data['KitchenQual'].map({"Fa":1, "TA":2, "Gd":3, "Ex":4})

data['KitchenQual'].unique()
#Fire Place Quality

data['FireplaceQu'] = data['FireplaceQu'].map({"None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})

data['FireplaceQu'].unique()
#GrivArea

data['GrLivArea_Band'] = pd.cut(data['GrLivArea'], 6)

data['GrLivArea_Band'].unique()
data.loc[data['GrLivArea']<=1127.5, 'GrLivArea'] = 1

data.loc[(data['GrLivArea']>1127.5) & (data['GrLivArea']<=1921), 'GrLivArea'] = 2

data.loc[(data['GrLivArea']>1921) & (data['GrLivArea']<=2714.5), 'GrLivArea'] = 3

data.loc[(data['GrLivArea']>2714.5) & (data['GrLivArea']<=3508), 'GrLivArea'] = 4

data.loc[(data['GrLivArea']>3508) & (data['GrLivArea']<=4301.5), 'GrLivArea'] = 5

data.loc[data['GrLivArea']>4301.5, 'GrLivArea'] = 6

data['GrLivArea'] = data['GrLivArea'].astype(int)



data.drop('GrLivArea_Band', axis=1, inplace=True)

data = pd.get_dummies(data, columns = ["GrLivArea"], prefix="GrLivArea")
#MSSubClass

data['MSSubClass'] = data['MSSubClass'].astype(str)

data = pd.get_dummies(data, columns = ["MSSubClass"], prefix="MSSubClass")

#BldgType

data['BldgType'] = data['BldgType'].astype(str)

data = pd.get_dummies(data, columns = ["BldgType"], prefix="BldgType")
#HouseStyle

data['HouseStyle'] = data['HouseStyle'].map({"2Story":"2Story", "1Story":"1Story", "1.5Fin":"1.5Story", "1.5Unf":"1.5Story", 

                                                     "SFoyer":"SFoyer", "SLvl":"SLvl", "2.5Unf":"2.5Story", "2.5Fin":"2.5Story"})

data = pd.get_dummies(data, columns = ["HouseStyle"], prefix="HouseStyle")
#Remodelling to Categorical

train['Remod_Diff'] = train['YearRemodAdd'] - train['YearBuilt']

data['Remod_Diff'] = data['YearRemodAdd'] - data['YearBuilt']

data.drop('YearRemodAdd', axis=1, inplace=True)
#Year Built

data['YearBuilt_Band'] = pd.cut(data['YearBuilt'], 7)

data['YearBuilt_Band'].unique()

data['YearBuilt_Band'] = pd.cut(data['YearBuilt'], 7)

data['YearBuilt_Band'].unique()

data.loc[data['YearBuilt']<=1892, 'YearBuilt'] = 1

data.loc[(data['YearBuilt']>1892) & (data['YearBuilt']<=1911), 'YearBuilt'] = 2

data.loc[(data['YearBuilt']>1911) & (data['YearBuilt']<=1931), 'YearBuilt'] = 3

data.loc[(data['YearBuilt']>1931) & (data['YearBuilt']<=1951), 'YearBuilt'] = 4

data.loc[(data['YearBuilt']>1951) & (data['YearBuilt']<=1971), 'YearBuilt'] = 5

data.loc[(data['YearBuilt']>1971) & (data['YearBuilt']<=1990), 'YearBuilt'] = 6

data.loc[data['YearBuilt']>1990, 'YearBuilt'] = 7

data['YearBuilt'] = data['YearBuilt'].astype(int)



data.drop('YearBuilt_Band', axis=1, inplace=True)

data = pd.get_dummies(data, columns = ["YearBuilt"], prefix="YearBuilt")
#Foundation

data = pd.get_dummies(data, columns = ["Foundation"], prefix="Foundation")
#Functional

data['Functional'] = data['Functional'].map({"Sev":1, "Maj2":2, "Maj1":3, "Mod":4, "Min2":5, "Min1":6, "Typ":7})

data['Functional'].unique()
#Roofstyle

data = pd.get_dummies(data, columns = ["RoofStyle"], prefix="RoofStyle")
#Roof Material

data = pd.get_dummies(data, columns = ["RoofMatl"], prefix="RoofMatl")
#Exterior1st and 2nd floor

def Exter2(col):

    if col['Exterior2nd'] == col['Exterior1st']:

        return 1

    else:

        return 0

    

data['ExteriorMatch_Flag'] = data.apply(Exter2, axis=1)

data.drop('Exterior2nd', axis=1, inplace=True)



data = pd.get_dummies(data, columns = ["Exterior1st"], prefix="Exterior1st")
#Masonry veneer type

data = pd.get_dummies(data, columns = ["MasVnrType"], prefix="MasVnrType")
#MasVnrArea - No correlation to the SalePrice

data.drop('MasVnrArea', axis=1, inplace=True)
#External Quality

data['ExterQual'] = data['ExterQual'].map({"Fa":1, "TA":2, "Gd":3, "Ex":4})

data['ExterQual'].unique()
#External Condition

data = pd.get_dummies(data, columns = ["ExterCond"], prefix="ExterCond")
#GarageType

data = pd.get_dummies(data, columns = ["GarageType"], prefix="GarageType")
#Year garage was built

data['GarageYrBlt_Band'] = pd.qcut(data['GarageYrBlt'], 3)

data['GarageYrBlt_Band'].unique()



data.loc[data['GarageYrBlt']<=1964, 'GarageYrBlt'] = 1

data.loc[(data['GarageYrBlt']>1964) & (data['GarageYrBlt']<=1996), 'GarageYrBlt'] = 2

data.loc[data['GarageYrBlt']>1996, 'GarageYrBlt'] = 3

data['GarageYrBlt'] = data['GarageYrBlt'].astype(int)



data.drop('GarageYrBlt_Band', axis=1, inplace=True)

data = pd.get_dummies(data, columns = ["GarageYrBlt"], prefix="GarageYrBlt")
#Garage Finish

data = pd.get_dummies(data, columns = ["GarageFinish"], prefix="GarageFinish")
#GarageArea_Band

data['GarageArea_Band'] = pd.cut(data['GarageArea'], 3)

data['GarageArea_Band'].unique()



data.loc[data['GarageArea']<=496, 'GarageArea'] = 1

data.loc[(data['GarageArea']>496) & (data['GarageArea']<=992), 'GarageArea'] = 2

data.loc[data['GarageArea']>992, 'GarageArea'] = 3

data['GarageArea'] = data['GarageArea'].astype(int)



data.drop('GarageArea_Band', axis=1, inplace=True)



data = pd.get_dummies(data, columns = ["GarageArea"], prefix="GarageArea")
#Garage Quality

data['GarageQual'] = data['GarageQual'].map({"None":"None", "Po":"Low", "Fa":"Low", "TA":"TA", "Gd":"High", "Ex":"High"})

data['GarageQual'].unique()

data = pd.get_dummies(data, columns = ["GarageQual"], prefix="GarageQual")
#Garage Condition

data['GarageCond'] = data['GarageCond'].map({"None":"None", "Po":"Low", "Fa":"Low", "TA":"TA", "Gd":"High", "Ex":"High"})

data['GarageCond'].unique()

data = pd.get_dummies(data, columns = ["GarageCond"], prefix="GarageCond")
#Wooddeck

def WoodDeckFlag(col):

    if col['WoodDeckSF'] == 0:

        return 1

    else:

        return 0

    

data['NoWoodDeck_Flag'] = data.apply(WoodDeckFlag, axis=1)



data['WoodDeckSF_Band'] = pd.cut(data['WoodDeckSF'], 4)



data.loc[data['WoodDeckSF']<=356, 'WoodDeckSF'] = 1

data.loc[(data['WoodDeckSF']>356) & (data['WoodDeckSF']<=712), 'WoodDeckSF'] = 2

data.loc[(data['WoodDeckSF']>712) & (data['WoodDeckSF']<=1068), 'WoodDeckSF'] = 3

data.loc[data['WoodDeckSF']>1068, 'WoodDeckSF'] = 4

data['WoodDeckSF'] = data['WoodDeckSF'].astype(int)

data.drop('WoodDeckSF_Band', axis=1, inplace=True)



data = pd.get_dummies(data, columns = ["WoodDeckSF"], prefix="WoodDeckSF")
#Total Surface Area of Porch



data['TotalPorchSF'] = data['OpenPorchSF'] + data['OpenPorchSF'] + data['EnclosedPorch'] + data['3SsnPorch'] + data['ScreenPorch'] 

train['TotalPorchSF'] = train['OpenPorchSF'] + train['OpenPorchSF'] + train['EnclosedPorch'] + train['3SsnPorch'] + train['ScreenPorch']



def PorchFlag(col):

    if col['TotalPorchSF'] == 0:

        return 1

    else:

        return 0

    

data['NoPorch_Flag'] = data.apply(PorchFlag, axis=1)



data['TotalPorchSF_Band'] = pd.cut(data['TotalPorchSF'], 4)

data['TotalPorchSF_Band'].unique()



data.loc[data['TotalPorchSF']<=431, 'TotalPorchSF'] = 1

data.loc[(data['TotalPorchSF']>431) & (data['TotalPorchSF']<=862), 'TotalPorchSF'] = 2

data.loc[(data['TotalPorchSF']>862) & (data['TotalPorchSF']<=1293), 'TotalPorchSF'] = 3

data.loc[data['TotalPorchSF']>1293, 'TotalPorchSF'] = 4

data['TotalPorchSF'] = data['TotalPorchSF'].astype(int)



data.drop('TotalPorchSF_Band', axis=1, inplace=True)

data = pd.get_dummies(data, columns = ["TotalPorchSF"], prefix="TotalPorchSF")
#Pool Area

def PoolFlag(col):

    if col['PoolArea'] == 0:

        return 0

    else:

        return 1

data['HasPool_Flag'] = data.apply(PoolFlag, axis=1)

data.drop('PoolArea', axis=1, inplace=True)
#PoolQC - Not correlated to Price will drop

data.drop('PoolQC', axis=1, inplace=True)
#Fence

data = pd.get_dummies(data, columns = ["Fence"], prefix="Fence")
#Zoning Classification

data = pd.get_dummies(data, columns = ["MSZoning"], prefix="MSZoning")
#Neightborhood

data = pd.get_dummies(data, columns = ["Neighborhood"], prefix="Neighborhood")
#Condition

data['Condition1'] = data['Condition1'].map({"Norm":"Norm", "Feedr":"Street", "PosN":"Pos", "Artery":"Street", "RRAe":"Train",

                                                    "RRNn":"Train", "RRAn":"Train", "PosA":"Pos", "RRNe":"Train"})

data['Condition2'] = data['Condition2'].map({"Norm":"Norm", "Feedr":"Street", "PosN":"Pos", "Artery":"Street", "RRAe":"Train",

                                                    "RRNn":"Train", "RRAn":"Train", "PosA":"Pos", "RRNe":"Train"})

def ConditionMatch(col):

    if col['Condition1'] == col['Condition2']:

        return 0

    else:

        return 1

    

data['Diff2ndCondition_Flag'] = data.apply(ConditionMatch, axis=1)

data.drop('Condition2', axis=1, inplace=True)



data = pd.get_dummies(data, columns = ["Condition1"], prefix="Condition1")
#Lot ARea

data['LotArea_Band'] = pd.qcut(data['LotArea'], 8)

data['LotArea_Band'].unique()



data.loc[data['LotArea']<=5684.75, 'LotArea'] = 1

data.loc[(data['LotArea']>5684.75) & (data['LotArea']<=7474), 'LotArea'] = 2

data.loc[(data['LotArea']>7474) & (data['LotArea']<=8520), 'LotArea'] = 3

data.loc[(data['LotArea']>8520) & (data['LotArea']<=9450), 'LotArea'] = 4

data.loc[(data['LotArea']>9450) & (data['LotArea']<=10355.25), 'LotArea'] = 5

data.loc[(data['LotArea']>10355.25) & (data['LotArea']<=11554.25), 'LotArea'] = 6

data.loc[(data['LotArea']>11554.25) & (data['LotArea']<=13613), 'LotArea'] = 7

data.loc[data['LotArea']>13613, 'LotArea'] = 8

data['LotArea'] = data['LotArea'].astype(int)



data.drop('LotArea_Band', axis=1, inplace=True)

data = pd.get_dummies(data, columns = ["LotArea"], prefix="LotArea")
#LotShape

data = pd.get_dummies(data, columns = ["LotShape"], prefix="LotShape")
#Land Contour

data = pd.get_dummies(data, columns = ["LandContour"], prefix="LandContour")
#LotConfig

data['LotConfig'] = data['LotConfig'].map({"Inside":"Inside", "FR2":"FR", "Corner":"Corner", "CulDSac":"CulDSac", "FR3":"FR"})

data = pd.get_dummies(data, columns = ["LotConfig"], prefix="LotConfig")
#Landslope

data['LandSlope'] = data['LandSlope'].map({"Gtl":1, "Mod":2, "Sev":2})



def Slope(col):

    if col['LandSlope'] == 1:

        return 1

    else:

        return 0

    

data['GentleSlope_Flag'] = data.apply(Slope, axis=1)

data.drop('LandSlope', axis=1, inplace=True)
#Street - No Correlation

data.drop('Street', axis=1, inplace=True) 
#Alley

data = pd.get_dummies(data, columns = ["Alley"], prefix="Alley")
#Paved Driveway

data = pd.get_dummies(data, columns = ["PavedDrive"], prefix="PavedDrive")
#Heating

data['GasA_Flag'] = data['Heating'].map({"GasA":1, "GasW":0, "Grav":0, "Wall":0, "OthW":0, "Floor":0})

data.drop('Heating', axis=1, inplace=True)
#Heating Quality

data['HeatingQC'] = data['HeatingQC'].map({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})

data['HeatingQC'].unique()
#Central Air

data['CentralAir'] = data['CentralAir'].map({"Y":1, "N":0})

data['CentralAir'].unique()
#Electrical

data['Electrical'] = data['Electrical'].map({"SBrkr":"SBrkr", "FuseF":"Fuse", "FuseA":"Fuse", "FuseP":"Fuse", "Mix":"Mix"})

data = pd.get_dummies(data, columns = ["Electrical"], prefix="Electrical")
#MiscFEature - drop no correlation

columns=['MiscFeature', 'MiscVal']

data.drop(columns, axis=1, inplace=True)
#MoSold

data = pd.get_dummies(data, columns = ["MoSold"], prefix="MoSold")
#YearSold

data = pd.get_dummies(data, columns = ["YrSold"], prefix="YrSold")
#Saletype

data['SaleType'] = data['SaleType'].map({"WD":"WD", "New":"New", "COD":"COD", "CWD":"CWD", "ConLD":"Oth", "ConLI":"Oth", 

                                                 "ConLw":"Oth", "Con":"Oth", "Oth":"Oth"})

data = pd.get_dummies(data, columns = ["SaleType"], prefix="SaleType")

#Sale Condition

data = pd.get_dummies(data, columns = ["SaleCondition"], prefix="SaleCondition")
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

train["SalePrice"] = np.log1p(train["SalePrice"])

targetval = train["SalePrice"]



#Check the new distribution 

plt.subplots(figsize=(15, 10))

g = sns.distplot(train['SalePrice'], fit=norm, label = "Skewness : %.2f"%(train['SalePrice'].skew()));

g = g.legend(loc="best")
# First lets single out the numeric features

numeric_feats = data.dtypes[data.dtypes != "object"].index



# Check how skewed they are

skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)



plt.subplots(figsize =(65, 20))

skewed_feats.plot(kind='bar');
skewness = skewed_feats[abs(skewed_feats) > 0.5]



skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    data[feat] = boxcox1p(data[feat], lam)



print(skewness.shape[0],  "skewed numerical features have been Box-Cox transformed")
# First, re-create the training and test datasets

train = data[:ntrain]

test = data[ntrain:]



print(train.shape)

print(test.shape)
import xgboost as xgb



model = xgb.XGBRegressor()

model.fit(train, targetval)



# Sort feature importances from GBC model trained earlier

indices = np.argsort(model.feature_importances_)[::-1]

indices = indices[:75]



# Visualise these with a barplot

plt.subplots(figsize=(20, 15))

g = sns.barplot(y=train.columns[indices], x = model.feature_importances_[indices], orient='h', palette = mycols)

g.set_xlabel("Relative importance",fontsize=12)

g.set_ylabel("Features",fontsize=12)

g.tick_params(labelsize=9)

g.set_title("XGB feature importance");
xgb_train = train.copy()

xgb_test = test.copy()



import xgboost as xgb

model = xgb.XGBRegressor()

model.fit(xgb_train, targetval)



# Allow the feature importances attribute to select the most important features

xgb_feat_red = SelectFromModel(model, prefit = True)



# Reduce estimation, validation and test datasets

xgb_train = xgb_feat_red.transform(xgb_train)

xgb_test = xgb_feat_red.transform(xgb_test)





print("Results of 'feature_importances_':")

print('X_train: ', xgb_train.shape, '\nX_test: ', xgb_test.shape)
# Next we want to sample our training data to test for performance of robustness ans accuracy, before applying to the test data

X_train, X_test, targetval, Y_test = model_selection.train_test_split(xgb_train, targetval, test_size=0.3, random_state=42)



print('X_train: ', X_train.shape, '\nX_test: ', X_test.shape, '\ntargetval: ', targetval.shape, '\nY_test: ', Y_test.shape)
import xgboost as xgb

#Machine Learning Algorithm (MLA) Selection and Initialization

models = [KernelRidge(), ElasticNet(), Lasso(), BayesianRidge(), RandomForestRegressor(), xgb.XGBRegressor()]



# First I will use ShuffleSplit as a way of randomising the cross validation samples.

shuff = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)



#create table to compare MLA metrics

columns = ['Name', 'Parameters', 'Train Accuracy Mean', 'Test Accuracy']

before_model_compare = pd.DataFrame(columns = columns)



#index through models and save performance to table

row_index = 0

for alg in models:



    #set name and parameters

    model_name = alg.__class__.__name__

    before_model_compare.loc[row_index, 'Name'] = model_name

    before_model_compare.loc[row_index, 'Parameters'] = str(alg.get_params())

    

    alg.fit(X_train, targetval)

    

    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate

    training_results = np.sqrt((-cross_val_score(alg, X_train, targetval, cv = shuff, scoring= 'neg_mean_squared_error')).mean())

    test_results = np.sqrt(((Y_test-alg.predict(X_test))**2).mean())

    

    before_model_compare.loc[row_index, 'Train Accuracy Mean'] = (training_results)*100

    before_model_compare.loc[row_index, 'Test Accuracy'] = (test_results)*100

    

    row_index+=1

    print(row_index, alg.__class__.__name__, 'trained...')



decimals = 3

before_model_compare['Train Accuracy Mean'] = before_model_compare['Train Accuracy Mean'].apply(lambda x: round(x, decimals))

before_model_compare['Test Accuracy'] = before_model_compare['Test Accuracy'].apply(lambda x: round(x, decimals))

before_model_compare
models = [KernelRidge(), ElasticNet(), Lasso(), GradientBoostingRegressor(), BayesianRidge(), LassoLarsIC(), RandomForestRegressor(), xgb.XGBRegressor()]



KR_param_grid = {'alpha': [0.1], 'coef0': [100], 'degree': [1], 'gamma': [None], 'kernel': ['polynomial']}

EN_param_grid = {'alpha': [0.001], 'copy_X': [True], 'l1_ratio': [0.6], 'fit_intercept': [True], 'normalize': [False], 

                         'precompute': [False], 'max_iter': [300], 'tol': [0.001], 'selection': ['random'], 'random_state': [None]}

LASS_param_grid = {'alpha': [0.0005], 'copy_X': [True], 'fit_intercept': [True], 'normalize': [False], 'precompute': [False], 

                    'max_iter': [300], 'tol': [0.01], 'selection': ['random'], 'random_state': [None]}

GB_param_grid = {'loss': ['huber'], 'learning_rate': [0.1], 'n_estimators': [300], 'max_depth': [3], 

                                        'min_samples_split': [0.0025], 'min_samples_leaf': [5]}

BR_param_grid = {'n_iter': [200], 'tol': [0.00001], 'alpha_1': [0.00000001], 'alpha_2': [0.000005], 'lambda_1': [0.000005], 

                 'lambda_2': [0.00000001], 'copy_X': [True]}

LL_param_grid = {'criterion': ['aic'], 'normalize': [True], 'max_iter': [100], 'copy_X': [True], 'precompute': ['auto'], 'eps': [0.000001]}

RFR_param_grid = {'n_estimators': [50], 'max_features': ['auto'], 'max_depth': [None], 'min_samples_split': [5], 'min_samples_leaf': [2]}

XGB_param_grid = {'max_depth': [3], 'learning_rate': [0.1], 'n_estimators': [300], 'booster': ['gbtree'], 'gamma': [0], 'reg_alpha': [0.1],

                  'reg_lambda': [0.7], 'max_delta_step': [0], 'min_child_weight': [1], 'colsample_bytree': [0.5], 'colsample_bylevel': [0.2],

                  'scale_pos_weight': [1]}

params_grid = [KR_param_grid, EN_param_grid, LASS_param_grid, GB_param_grid, BR_param_grid, LL_param_grid, RFR_param_grid, XGB_param_grid]



after_model_compare = pd.DataFrame(columns = columns)



row_index = 0

for alg in models:

    

    gs_alg = GridSearchCV(alg, param_grid = params_grid[0], cv = shuff, scoring = 'neg_mean_squared_error', n_jobs=-1)

    params_grid.pop(0)



    #set name and parameters

    model_name = alg.__class__.__name__

    after_model_compare.loc[row_index, 'Name'] = model_name

    

    gs_alg.fit(X_train, targetval)

    gs_best = gs_alg.best_estimator_

    after_model_compare.loc[row_index, 'Parameters'] = str(gs_alg.best_params_)

    

    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate

    after_training_results = np.sqrt(-gs_alg.best_score_)

    after_test_results = np.sqrt(((Y_test-gs_alg.predict(X_test))**2).mean())

    

    after_model_compare.loc[row_index, 'Train Accuracy Mean'] = (after_training_results)*100

    after_model_compare.loc[row_index, 'Test Accuracy'] = (after_test_results)*100

    

    row_index+=1

    print(row_index, alg.__class__.__name__, 'trained...')



decimals = 3

after_model_compare['Train Accuracy Mean'] = after_model_compare['Train Accuracy Mean'].apply(lambda x: round(x, decimals))

after_model_compare['Test Accuracy'] = after_model_compare['Test Accuracy'].apply(lambda x: round(x, decimals))

after_model_compare
models = [KernelRidge(), ElasticNet(), Lasso(), GradientBoostingRegressor(), BayesianRidge(), LassoLarsIC(), RandomForestRegressor(), xgb.XGBRegressor()]

names = ['KernelRidge', 'ElasticNet', 'Lasso', 'Gradient Boosting', 'Bayesian Ridge', 'Lasso Lars IC', 'Random Forest', 'XGBoost']

params_grid = [KR_param_grid, EN_param_grid, LASS_param_grid, GB_param_grid, BR_param_grid, LL_param_grid, RFR_param_grid, XGB_param_grid]

stacked_validation_train = pd.DataFrame()

stacked_test_train = pd.DataFrame()



row_index=0



for alg in models:

    

    gs_alg = GridSearchCV(alg, param_grid = params_grid[0], cv = shuff, scoring = 'neg_mean_squared_error', n_jobs=-1)

    params_grid.pop(0)

    

    gs_alg.fit(X_train, targetval)

    gs_best = gs_alg.best_estimator_

    stacked_validation_train.insert(loc = row_index, column = names[0], value = gs_best.predict(X_test))

    print(row_index+1, alg.__class__.__name__, 'predictions added to stacking validation dataset...')

    

    stacked_test_train.insert(loc = row_index, column = names[0], value = gs_best.predict(xgb_test))

    print(row_index+1, alg.__class__.__name__, 'predictions added to stacking test dataset...')

    print("-"*50)

    names.pop(0)

    

    row_index+=1

    

print('Done')
stacked_validation_train.head()
stacked_test_train.head()
# First drop the Lasso results from the table, as we will be using Lasso as the meta-model

drop = ['Lasso']

stacked_validation_train.drop(drop, axis=1, inplace=True)

stacked_test_train.drop(drop, axis=1, inplace=True)



# Now fit the meta model and generate predictions

meta_model = make_pipeline(RobustScaler(), Lasso(alpha=0.00001, copy_X = True, fit_intercept = True,

                                              normalize = False, precompute = False, max_iter = 10000,

                                              tol = 0.0001, selection = 'random', random_state = None))

meta_model.fit(stacked_validation_train, Y_test)



meta_model_pred = np.expm1(meta_model.predict(stacked_test_train))

print("Meta-model trained and applied!...")
models = [KernelRidge(), ElasticNet(), Lasso(), GradientBoostingRegressor(), BayesianRidge(), LassoLarsIC(), RandomForestRegressor(), xgb.XGBRegressor()]

names = ['KernelRidge', 'ElasticNet', 'Lasso', 'Gradient Boosting', 'Bayesian Ridge', 'Lasso Lars IC', 'Random Forest', 'XGBoost']

params_grid = [KR_param_grid, EN_param_grid, LASS_param_grid, GB_param_grid, BR_param_grid, LL_param_grid, RFR_param_grid, XGB_param_grid]

final_predictions = pd.DataFrame()



row_index=0



for alg in models:

    

    gs_alg = GridSearchCV(alg, param_grid = params_grid[0], cv = shuff, scoring = 'neg_mean_squared_error', n_jobs=-1)

    params_grid.pop(0)

    

    gs_alg.fit(stacked_validation_train, Y_test)

    gs_best = gs_alg.best_estimator_

    final_predictions.insert(loc = row_index, column = names[0], value = np.expm1(gs_best.predict(stacked_test_train)))

    print(row_index+1, alg.__class__.__name__, 'final results predicted added to table...')

    names.pop(0)

    

    row_index+=1



print("-"*50)

print("Done")

    

final_predictions.head()
ensemble = meta_model_pred*(1/10) + final_predictions['XGBoost']*(1.5/10) + final_predictions['Gradient Boosting']*(2/10) + final_predictions['Bayesian Ridge']*(1/10) + final_predictions['Lasso']*(1/10) + final_predictions['KernelRidge']*(1/10) + final_predictions['Lasso Lars IC']*(1/10) + final_predictions['Random Forest']*(1.5/10)



submission = pd.DataFrame()

submission['Id'] = test_ID

submission['SalePrice'] = ensemble

#submission.to_csv('final_submission.csv',index=False)

print("Submission file, created!")