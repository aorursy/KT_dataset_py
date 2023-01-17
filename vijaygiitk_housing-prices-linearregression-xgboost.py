import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



from scipy import stats

from scipy.stats import norm



%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# Print some useful info for the train set

print(f'Train size is: {train.shape}')

print(f'Test size is: {test.shape}')
train['SalePrice'].describe()
def distribution_plot_and_qqplot(data):

    # Get the fitted parameters used by the function

    (mu, sigma) = norm.fit(data)

    print('mu = {:.2f} and sigma = {:.2f}'.format(mu, sigma))



    # Plot the distribution

    g = sns.distplot(data, fit=norm)

    legend1 = plt.legend(['Skewness : {:.2f}'.format(data.skew())], loc=4)

    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

    plt.gca().add_artist(legend1)

    plt.ylabel('Frequency')

    plt.title(f'{data.name} distribution')



    # Get also the QQ-plot

    fig = plt.figure()

    res = stats.probplot(data, plot=plt)

    plt.show()

    

distribution_plot_and_qqplot(train['SalePrice'])
f,ax = plt.subplots(figsize = (15,15))

sns.heatmap(train.corr(), annot = True, linewidths=.5, fmt='.1f', ax=ax)

plt.show()
corre = train.corr()

top_corr_features = corre.index[abs(corre['SalePrice'])>0.5]

g = sns.heatmap(train[top_corr_features].corr(),annot=True)
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], size = 2.5)

plt.show()
# Before plotting let's create a useful function to use it again later

def plot_scatter(x, y):

    fig, ax = plt.subplots()

    ax.scatter(x=x, y=y)

    plt.xlabel(x.name, fontsize=12)

    plt.ylabel(y.name, fontsize=12)

    plt.show()

    
plot_scatter(train['GrLivArea'], train['SalePrice'])

plot_scatter(train['TotalBsmtSF'], train['SalePrice'])

plot_scatter(train['1stFlrSF'], train['SalePrice'])

plot_scatter(train['OverallQual'], train['SalePrice'])

plot_scatter(train['GarageCars'], train['SalePrice'])

train[train.GrLivArea>4500]

train[train.TotalBsmtSF>4000]

train[train['1stFlrSF']>4000]
train = train.drop(train[train['Id']==524].index)

train = train.drop(train[train['Id']==1299].index)

train.shape
df = pd.concat([train,test])



pd.set_option('display.max_rows',5000)

pd.set_option('display.max_columns',500)

df = df.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd'], axis =1)

#checkig the columns for categorical and numerical values

print(df.select_dtypes(include = ['int64','float64']).columns)

print(df.select_dtypes(include = ['object']).columns)

df = df.set_index('Id')
#missing data

total_miss = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.shape[0]*100).sort_values(ascending=False)

missing_data = pd.concat([total_miss,percent], axis=1, keys=['Total','Percent'])



missing_data.head(35)
columns_drop =percent[percent > 20].keys()



columns_drop
df = df.drop(columns_drop, axis = 1)



print(df.shape)

df.describe(include = 'all')
missing_cols = df.columns[df.isnull().any()]



missing_cols
bsmt_cols = ['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1',

       'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtUnfSF','TotalBsmtSF']



bsmt_feat = df[bsmt_cols]

bsmt_feat.info()
bsmt_feat = bsmt_feat[bsmt_feat.isnull().any(axis=1)]



#print(bsmt_feat)

print(bsmt_feat.shape)
bsmt_feat_all_nan = bsmt_feat[(bsmt_feat.isnull() | bsmt_feat.isin([0])).all(1)]



#print(bsmt_feat_all_nan)

print(bsmt_feat_all_nan.shape)
qual = list(df.loc[:,df.dtypes=='object'].columns.values)



for i in bsmt_cols:

    if i in qual:

        bsmt_feat_all_nan[i] = bsmt_feat_all_nan[i].replace(np.nan,'NA')

    else:

        bsmt_feat_all_nan[i] = bsmt_feat_all_nan[i].replace(np.nan,0)



bsmt_feat.update(bsmt_feat_all_nan)

df.update(bsmt_feat_all_nan)
#Finding remaining rows which have null columns



bsmt_feat = bsmt_feat[bsmt_feat.isin([np.nan]).any(axis=1)]



#print(bsmt_feat)

print(bsmt_feat.shape)
#Bucket the continuous columns

print(df['BsmtFinSF2'].max())

print(df['BsmtFinSF2'].min())



#Bucket this  range in 5 buckets.

#pd.cut(range(0,1526),5)
df_slice = df[(df['BsmtFinSF2'] >= 305) & (df['BsmtFinSF2'] <= 610)]



#Impute this particular row

bsmt_feat.at[333,'BsmtFinType2'] = df_slice['BsmtFinType2'].mode()[0]
#Impute the missing BsmtExposure value with the slice of BsmtExposure when BsmtQual is Gd.

bsmt_feat['BsmtExposure'] = bsmt_feat['BsmtExposure'].replace(np.nan, df[df['BsmtQual'] == 'Gd']['BsmtExposure'].mode()[0])



#Similarily

bsmt_feat['BsmtCond'] = bsmt_feat['BsmtCond'].replace(np.nan, df['BsmtCond'].mode()[0])

bsmt_feat['BsmtQual'] = bsmt_feat['BsmtQual'].replace(np.nan, df['BsmtQual'].mode()[0])

df.update(bsmt_feat)



df.columns[df.isnull().any()]
#Now impute the missing values in Garage Features.



garage_cols = ['GarageCars', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType',

       'GarageYrBlt']



gar_feat = df[garage_cols]

gar_feat.info()
gar_feat = gar_feat[gar_feat.isnull().any(axis=1)]



#print(gar_feat)

print(gar_feat.shape)
gar_feat_all_nan = gar_feat[(gar_feat.isnull() | gar_feat.isin([0])).all(1)]



#print(gar_feat_all_nan)

print(gar_feat_all_nan.shape)
for i in garage_cols:

    if i in qual:

        gar_feat_all_nan[i] = gar_feat_all_nan[i].replace(np.nan,'NA')

    else:

        gar_feat_all_nan[i] = gar_feat_all_nan[i].replace(np.nan,0)

gar_feat.update(gar_feat_all_nan)

df.update(gar_feat_all_nan)
gar_feat = gar_feat[gar_feat.isnull().any(axis=1)]



#gar_feat
for i in garage_cols:

    gar_feat[i] = gar_feat[i].replace(np.nan, df[df['GarageType'] == 'Detchd'][i].mode()[0])



#gar_feat
df.update(gar_feat)



df.columns[df.isnull().any()]
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])



df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])



df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])



df['Functional'] = df['Functional'].fillna(df['Functional'].mode()[0])



df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])



df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])



df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])



df['Utilities'] = df['Utilities'].fillna(df['Utilities'].mode()[0])



df['MasVnrType'] = df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df[df['MasVnrArea'].isnull() == True]['MasVnrType'].unique()
df.loc[(df['MasVnrType'] == 'None') & (df['MasVnrArea'].isnull() == True), 'MasVnrArea'] = 0
#print(df['MasVnrArea'].isnull().sum())

#print(df['MasVnrType'].isnull().sum())

#print(df.columns[df.isnull().any()])
lotconfig = ['Corner','Inside','CulDSac','FR2','FR3']



for i in lotconfig:

    df['LotFrontage'] = pd.np.where((df['LotFrontage'].isnull() == True) & (df['LotConfig'] == i), df[df['LotConfig'] == i]['LotFrontage'].mean(),df['LotFrontage'])



df.isnull().sum().max()
#Few Features are in numerical in nature but actually are of Categorical



cat_con_columns = ['MSSubClass', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt','YrSold']

for i in cat_con_columns:

    df[i] = df[i].astype(str)
import calendar

df['MoSold'] = df['MoSold'].apply(lambda x : calendar.month_abbr[x])



df['MoSold'].unique()
quan = list(df.loc[:,df.dtypes != 'object'].columns.values)

# Ordered Data

from pandas.api.types import CategoricalDtype



df['BsmtCond'] = df['BsmtCond'].astype(CategoricalDtype(categories=['NA','Po','Fa','TA','Gd','Ex'], ordered = True)).cat.codes



df['BsmtExposure'] = df['BsmtExposure'].astype(CategoricalDtype(categories=['NA','No','Mn','Av','Gd'], ordered = True)).cat.codes



df['BsmtFinType1'] = df['BsmtFinType1'].astype(CategoricalDtype(categories=['NA','Unf','LwQ','Rec','BLQ','ALQ','GLQ'], ordered = True)).cat.codes



df['BsmtFinType2'] = df['BsmtFinType2'].astype(CategoricalDtype(categories=['NA','Unf','LwQ','Rec','BLQ','ALQ','GLQ'], ordered = True)).cat.codes



df['BsmtQual'] = df['BsmtQual'].astype(CategoricalDtype(categories=['NA','Po','Fa','TA','Gd','Ex'], ordered = True)).cat.codes



df['ExterQual'] = df['BsmtQual'].astype(CategoricalDtype(categories=['Po','Fa','TA','Gd','Ex'], ordered = True)).cat.codes



df['ExterCond'] = df['ExterCond'].astype(CategoricalDtype(categories=['Po','Fa','TA','Gd','Ex'], ordered = True)).cat.codes



df['Functional'] = df['Functional'].astype(CategoricalDtype(categories=['Sal','Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'], ordered = True)).cat.codes



df['GarageCond'] = df['GarageCond'].astype(CategoricalDtype(categories=['NA','Po','Fa','TA','Gd','Ex'], ordered = True)).cat.codes



df['GarageQual'] = df['GarageQual'].astype(CategoricalDtype(categories=['NA','Po','Fa','TA','Gd','Ex'], ordered = True)).cat.codes



df['GarageFinish'] = df['GarageFinish'].astype(CategoricalDtype(categories=['NA','Unf','RFn','Fin'], ordered = True)).cat.codes



df['HeatingQC'] = df['HeatingQC'].astype(CategoricalDtype(categories=['Po','Fa','TA','Gd','Ex'], ordered = True)).cat.codes



df['KitchenQual'] = df['KitchenQual'].astype(CategoricalDtype(categories=['Po','Fa','TA','Gd','Ex'], ordered = True)).cat.codes



df['PavedDrive'] = df['PavedDrive'].astype(CategoricalDtype(categories=['N','P','Y'], ordered = True)).cat.codes



df['Utilities'] = df['Utilities'].astype(CategoricalDtype(categories=['ELO','NoSeWa','NoSewr','AllPub'], ordered = True)).cat.codes

skewed_features = ['2ndFlrSF','3SsnPorch',

 'BedroomAbvGr','BsmtFinSF1','BsmtFinSF2',

 'BsmtFullBath','BsmtHalfBath','BsmtUnfSF',

 'EnclosedPorch','Fireplaces','FullBath',

 'GarageCars','GrLivArea', 'HalfBath',

 'KitchenAbvGr','LotArea','LotFrontage',

 'LowQualFinSF','MasVnrArea','MiscVal',

 'OpenPorchSF','PoolArea','ScreenPorch',

 'TotalBsmtSF','WoodDeckSF']
## Remove Skewness from the data

for i in skewed_features:

    df[i] = np.log1p(df[i])



log_SalePrice = np.log1p(train['SalePrice'])



distribution_plot_and_qqplot(log_SalePrice)
# Create Dummies for all non ordinal categorical data

qual1 = list(df.loc[:,df.dtypes == 'object'].columns.values)

print(len(qual1))



df_with_dummies = pd.get_dummies(df, columns=qual1, drop_first=True)

df_with_dummies.shape

##Normalize



df_inputs = df_with_dummies.copy()

targets = log_SalePrice.copy()



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df_inputs)

df_inputs_scaled = scaler.transform(df_inputs)
#Segregate data into original train and test

train_len = len(train)

train_scaled = df_inputs_scaled[:train_len]

test_scaled = df_inputs_scaled[train_len:]



print(train_scaled.shape)



print(test_scaled.shape)

# Train Test Split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_scaled, targets, test_size=0.2, random_state=365)

import xgboost

regressor = xgboost.XGBRegressor(learning_rate = 0.06, max_depth= 3, n_estimators = 350, random_state= 0)

regressor.fit(x_train,y_train)
y_hat = regressor.predict(x_train)



plt.scatter(y_train, y_hat, alpha = 0.2)

plt.xlabel('Targets (y_train)',size=18)

plt.ylabel('Predictions (y_hat)',size=18)

plt.show()
regressor.score(x_train,y_train)
##Testing

y_hat_test = regressor.predict(x_test)





plt.scatter(y_test, y_hat_test, alpha=0.2)

plt.xlabel('Targets (y_test)',size=18)

plt.ylabel('Predictions (y_hat_test)',size=18)

plt.show()
y_predict = regressor.predict(test_scaled)

y_predict = np.expm1(y_predict)
## k-Fold cross validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = regressor, X = x_train, y = y_train, cv = 10)

print(accuracies.mean())

print(accuracies.std())