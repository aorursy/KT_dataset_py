import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')
sns.set_style('whitegrid')
house_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
house_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

len(house_train), len(house_test)
house_train.head()
house_train.info()
house_train.describe().T
house_test.head()
house_test.info()
house_test.describe().T
house_train.isnull().sum().value_counts()
plt.figure(figsize=(17,7))
sns.heatmap(house_train.isnull(), yticklabels=False, cbar=False, cmap='YlGnBu')
house_test.isnull().sum().value_counts()
plt.figure(figsize=(17,7))
sns.heatmap(house_test.isnull(), yticklabels=False, cbar=False, cmap='YlGnBu')
null_train = ['LotFrontage','Alley','MasVnrType','MasVnrArea','BsmtQual','BsmtCond',
             'BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical','FireplaceQu','GarageType',
             'GarageYrBlt','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']

null_test = ['MSZoning','LotFrontage','Alley','Utilities','Exterior1st','Exterior2nd',
            'MasVnrType','MasVnrArea','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
            'BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath',
            'BsmtHalfBath','KitchenQual','Functional','FireplaceQu','GarageType','GarageYrBlt',
            'GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond','PoolQC','Fence',
            'MiscFeature','SaleType']
sns.distplot(house_train['SalePrice'])
house_train['YearBuilt'].plot.hist()
house_train['YearRemodAdd'].plot.hist()
corr_mat = house_train.corr(method='spearman').sort_values(by='SalePrice', ascending=False).sort_values(by='SalePrice', ascending=False, axis=1)

plt.figure(figsize=(15,12))
sns.heatmap(corr_mat.iloc[:20,:20], annot=True, lw=.2)
sns.lmplot('GrLivArea','SalePrice',data=house_train,hue='CentralAir')
sns.lmplot('GarageArea','SalePrice',data=house_train,hue='CentralAir')
sns.lmplot('TotalBsmtSF','SalePrice',data=house_train,hue='CentralAir')
sns.lmplot('1stFlrSF','SalePrice',data=house_train,hue='CentralAir')
sns.lmplot('OpenPorchSF','SalePrice',data=house_train,hue='CentralAir')
sns.lmplot('LotFrontage','SalePrice',data=house_train,hue='CentralAir')
from sklearn.preprocessing import LabelEncoder, RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
rScaler = RobustScaler() 
lenc = LabelEncoder()
ac = lenc.fit_transform(house_train['SaleCondition'])
dff = house_train.copy()
dff.head()
dff['Alley'].unique()
df = lenc.fit_transform(dff['Alley'].astype(str))
dff['BsmtQual'].fillna('0',inplace=True)
ad = lenc.fit_transform(dff['BsmtQual'])
lenc.fit(['NA','Po','Fa','TA','Gd','Ex']) # Testing
dff['FireplaceQu'].fillna('NA',inplace=True)
edd = lenc.transform(dff['FireplaceQu'])
def scaleImputer(col):
    '''
    Base on the data in features used is an ordinal type which has 6 common scales,
    * Ex: Excellent
    * Gd: Good
    * TA: Typical/Average
    * Fa: Fair
    * Po: Poor 
    * NA: Nan or No Feature Exist
    so we made a manual function, actually we can use LabelEncoder but the number labelled
    would be random!!
    '''
    if col == 'Ex':
        return 5
    elif col == 'Gd':
        return 4
    elif col == 'TA':
        return 3
    elif col == 'Fa':
        return 2
    elif col == 'Po':
        return 1
    return 0

def bsmtExpImputer(col):
    if col == 'No':
        return 0
    elif col == 'Gd':
        return 3
    elif col == 'Av':
        return 2
    elif col == 'Mn':
        return 1
    return 0

def bsmtFinTpImputer(col):
    if col == 'GLQ':
        return 6
    elif col == 'ALQ':
        return 5
    elif col == 'BLQ':
        return 4
    elif col == 'Rec':
        return 3
    elif col == 'LwQ':
        return 2
    elif col == 'Unf':
        return 1
    return 0

def fenceImputer(col):
    if col == 'GdPrv':
        return 4
    elif col == 'MnPrv':
        return 3
    elif col == 'GdWo':
        return 2
    elif col == 'MnWw':
        return 1
    return 0

def garFinImputer(col):
    if col == 'Fin':
        return 3
    elif col == 'RFn':
        return 2
    elif col == 'Unf':
        return 1
    return 0

def funcImputer(col):
    if col == 'Typ':
        return 7
    elif col == 'Min1':
        return 6
    elif col == 'Min2':
        return 5
    elif col == 'Mod':
        return 4
    elif col == 'Maj1':
        return 3
    elif col == 'Maj2':
        return 2
    elif col == 'Sev':
        return 1
    return 0
house_train['Functional'].unique()
house_train['Alley'].fillna('0',inplace=True)
house_train['Alley'] = lenc.fit_transform(house_train['Alley'])

house_train['BsmtQual'].fillna('0',inplace=True)
house_train['BsmtQual'] = house_train['BsmtQual'].apply(scaleImputer)

house_train['BsmtCond'].fillna('0',inplace=True)
house_train['BsmtCond'] = house_train['BsmtCond'].apply(scaleImputer)

house_train['BsmtExposure'].fillna('No',inplace=True)
house_train['BsmtExposure'] = house_train['BsmtExposure'].apply(bsmtExpImputer)

house_train['BsmtFinType1'].fillna('No',inplace=True)
house_train['BsmtFinType1'] = house_train['BsmtFinType1'].apply(bsmtFinTpImputer)

house_train['BsmtFinType2'].fillna('No',inplace=True)
house_train['BsmtFinType2'] = house_train['BsmtFinType2'].apply(bsmtFinTpImputer)

house_train['FireplaceQu'].fillna('No',inplace=True)
house_train['FireplaceQu'] = house_train['FireplaceQu'].apply(scaleImputer)

house_train['GarageYrBlt'].fillna(0,inplace=True)

house_train['GarageQual'].fillna('0',inplace=True)
house_train['GarageQual'] = house_train['GarageQual'].apply(scaleImputer)

house_train['GarageCond'].fillna('0',inplace=True)
house_train['GarageCond'] = house_train['GarageCond'].apply(scaleImputer)

house_train['PoolQC'].fillna('0',inplace=True)
house_train['PoolQC'] = house_train['PoolQC'].apply(scaleImputer)

house_train['MiscFeature'].fillna('0',inplace=True)
lenc.fit(['0','TenC','Shed','Othr','Gar2','Elev'])
house_train['MiscFeature'] = lenc.transform(house_train['MiscFeature'])

house_train['GarageType'].fillna('0',inplace=True)
lenc.fit(['0', 'Detchd', 'CarPort', 'BuiltIn', 'Basment','Attchd', '2Types'])
house_train['GarageType'] = lenc.transform(house_train['GarageType'])

house_train['Fence'].fillna('0',inplace=True)
house_train['Fence'] = house_train['Fence'].apply(fenceImputer)

house_train['GarageFinish'].fillna('0',inplace=True)
house_train['GarageFinish'] = house_train['GarageFinish'].apply(garFinImputer)

house_train['Electrical'].fillna('SBrkr',inplace=True)

house_train['MasVnrType'].fillna('None',inplace=True)
lenc.fit(['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing',
       'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'BrkComm', 'AsphShn',
       'Stone', 'ImStucc', 'CBlock', 'Other'])
house_train['Exterior1st'] = lenc.transform(house_train['Exterior1st'])

house_train['ExterQual'] = house_train['ExterQual'].apply(scaleImputer)
house_train['ExterCond'] = house_train['ExterCond'].apply(scaleImputer)
house_train['HeatingQC'] = house_train['HeatingQC'].apply(scaleImputer)
house_train['KitchenQual'] = house_train['KitchenQual'].apply(scaleImputer)
house_train['Functional'] = house_train['Functional'].apply(funcImputer)
#For Modelling Part Later...

X = house_train.drop(['Id','SalePrice'],axis=1)
y = house_train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train['LotFrontage'].fillna(X_train['LotFrontage'].median(),inplace=True)
X_test['LotFrontage'].fillna(X_test['LotFrontage'].median(),inplace=True)
X_train['MasVnrArea'].fillna(X_train['MasVnrArea'].median(),inplace=True)
X_test['MasVnrArea'].fillna(X_test['MasVnrArea'].median(),inplace=True)
# For the predictions on the real test data...
house_train['LotFrontage'].fillna(house_train['LotFrontage'].median(),inplace=True)
house_train['MasVnrArea'].fillna(house_train['MasVnrArea'].median(),inplace=True)
ob_feat = list(house_train.select_dtypes(include=['object']).columns)

feat_trans = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoding', OneHotEncoder(handle_unknown='ignore'))])

prep_train = ColumnTransformer(transformers=[('cat', feat_trans, ob_feat)],remainder='passthrough')
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
house_test['MSZoning'].fillna('RL',inplace=True)

house_test['Alley'].fillna('0',inplace=True)
house_test['Alley'] = lenc.fit_transform(house_test['Alley'])

house_test['Utilities'].fillna('AllPub',inplace=True)

house_test['Exterior1st'].fillna('Other',inplace=True)

house_test['Exterior2nd'].fillna('Other',inplace=True)

house_test['BsmtQual'].fillna('0',inplace=True)
house_test['BsmtQual'] = house_test['BsmtQual'].apply(scaleImputer)

house_test['BsmtCond'].fillna('0',inplace=True)
house_test['BsmtCond'] = house_test['BsmtCond'].apply(scaleImputer)

house_test['BsmtExposure'].fillna('No',inplace=True)
house_test['BsmtExposure'] = house_test['BsmtExposure'].apply(bsmtExpImputer)

house_test['BsmtFinType1'].fillna('No',inplace=True)
house_test['BsmtFinType1'] = house_test['BsmtFinType1'].apply(bsmtFinTpImputer)

house_test['BsmtFinType2'].fillna('No',inplace=True)
house_test['BsmtFinType2'] = house_test['BsmtFinType2'].apply(bsmtFinTpImputer)

house_test['FireplaceQu'].fillna('No',inplace=True)
house_test['FireplaceQu'] = house_test['FireplaceQu'].apply(scaleImputer)

house_test['GarageYrBlt'].fillna(0,inplace=True)

house_test['GarageQual'].fillna('0',inplace=True)
house_test['GarageQual'] = house_test['GarageQual'].apply(scaleImputer)

house_test['GarageCond'].fillna('0',inplace=True)
house_test['GarageCond'] = house_test['GarageCond'].apply(scaleImputer)

house_test['PoolQC'].fillna('0',inplace=True)
house_test['PoolQC'] = house_test['PoolQC'].apply(scaleImputer)

house_test['MiscFeature'].fillna('0',inplace=True)
lenc.fit(['0','TenC','Shed','Othr','Gar2','Elev'])
house_test['MiscFeature'] = lenc.transform(house_test['MiscFeature'])

house_test['GarageType'].fillna('0',inplace=True)
lenc.fit(['0', 'Detchd', 'CarPort', 'BuiltIn', 'Basment','Attchd', '2Types'])
house_test['GarageType'] = lenc.transform(house_test['GarageType'])

house_test['Fence'].fillna('0',inplace=True)
house_test['Fence'] = house_test['Fence'].apply(fenceImputer)

house_test['GarageFinish'].fillna('0',inplace=True)
house_test['GarageFinish'] = house_test['GarageFinish'].apply(garFinImputer)

house_test['BsmtFinSF1'].fillna(house_test['BsmtFinSF1'].median(),inplace=True)
house_test['BsmtFinSF2'].fillna(house_test['BsmtFinSF2'].median(),inplace=True)
house_test['BsmtUnfSF'].fillna(house_test['BsmtUnfSF'].median(),inplace=True)
house_test['BsmtFullBath'].fillna(house_test['BsmtFullBath'].median(),inplace=True)
house_test['BsmtHalfBath'].fillna(house_test['BsmtHalfBath'].median(),inplace=True)

house_test['GarageArea'].fillna(384,inplace=True)
house_test['GarageCars'].fillna(1,inplace=True)
house_test['GarageYrBlt'].fillna(house_test['GarageYrBlt'].median(),inplace=True)

house_test['SaleType'].fillna('Oth',inplace=True)
house_test['TotalBsmtSF'].fillna(house_test['BsmtUnfSF']+house_test['BsmtFinSF1']+house_test['BsmtFinSF2'],inplace=True)
house_test['LotFrontage'].fillna(house_test['LotFrontage'].median(),inplace=True)
house_test['MasVnrArea'].fillna(house_test['MasVnrArea'].median(),inplace=True)
house_test['MasVnrType'].fillna('None',inplace=True)
house_test['KitchenQual'].fillna('TA',inplace=True)
house_test['Functional'].fillna('Typ',inplace=True)
lenc.fit(['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing',
       'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'BrkComm', 'AsphShn',
       'Stone', 'ImStucc', 'CBlock', 'Other'])
house_test['Exterior1st'] = lenc.transform(house_test['Exterior1st'])

house_test['ExterQual'] = house_test['ExterQual'].apply(scaleImputer)
house_test['ExterCond'] = house_test['ExterCond'].apply(scaleImputer)
house_test['HeatingQC'] = house_test['HeatingQC'].apply(scaleImputer)
house_test['KitchenQual'] = house_test['KitchenQual'].apply(scaleImputer)
house_test['Functional'] = house_test['Functional'].apply(funcImputer)
house_test[house_test['KitchenQual']=='None']
house_train.iloc[100][['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF', 'TotalBsmtSF']]
house_test[house_test['SaleType'].isnull()][['GarageYrBlt',
 'GarageCars','GarageType',
 'GarageArea']]
house_test[house_test['GarageType']=='Detchd']['GarageYrBlt'].median()
house_test[house_test['KitchenQual'].isnull()]['KitchenAbvGr']
ob_feat1 = list(house_test.select_dtypes(include=['object']).columns)
# for i in range(len(ob_feat)):
#     print(ob_feat[i],i,'\n',house_train[ob_feat[i]].unique())
#     print(house_test[ob_feat1[i]].unique(),'\n')
# sep_feat = ob_feat[14], ob_feat[17], ob_feat[18], ob_feat[21], ob_feat[24], ob_feat[25]
prep_test = ColumnTransformer(transformers=[('cat1', feat_trans, ob_feat1)],remainder='passthrough')
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge, ElasticNet, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
linR = Pipeline(steps=[('preprocessor', prep_train),
                      ('linear_reg', LinearRegression())]) 
linR.fit(X_train,y_train)
rdg = Pipeline(steps=[('preprocessor', prep_train),
                      ('linear_reg', Ridge())])
rdg.fit(X_train,y_train)
eNet = Pipeline(steps=[('preprocessor', prep_train),
                      ('linear_reg', ElasticNet(max_iter=15000))])
eNet.fit(X_train,y_train)
lass = Pipeline(steps=[('preprocessor', prep_train),
                      ('linear_reg', Lasso(max_iter=3000))])
lass.fit(X_train,y_train)
lgbR = Pipeline(steps=[('preprocessor', prep_train),
                      ('linear_reg', LGBMRegressor())])
lgbR.fit(X_train,y_train)
linR.score(X_test,y_test), rdg.score(X_test,y_test), lass.score(X_test,y_test), eNet.score(X_test,y_test), lgbR.score(X_test,y_test)
lr_pred = linR.predict(X_test)
rdg_pred = rdg.predict(X_test)
las_pred = lass.predict(X_test)
eNt_pred = eNet.predict(X_test)
lgb_pred = lgbR.predict(X_test)
print(f'RMSE: {np.sqrt(mean_squared_error(y_test,lr_pred))}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test,rdg_pred))}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test,las_pred))}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test,eNt_pred))}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test,lgb_pred))}')
X = house_train.drop(['Id','SalePrice'],axis=1)
y = house_train['SalePrice']
X_real = house_test.drop('Id',axis=1)
lgbR.fit(X, y)
y_pred = lgbR.predict(X_real)
output_df = pd.DataFrame()
output_df['Id'] = house_test['Id']
output_df['SalePrice'] = y_pred
output_df.tail()
output_df.to_csv('advanced_house_reg.csv',index=False)