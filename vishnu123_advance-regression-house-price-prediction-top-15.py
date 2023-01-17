import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np
from scipy.stats import norm
from scipy import stats
plt.style.use("fivethirtyeight")
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
sample,features = train.shape[0],train.shape[1]
print(f"Train data contains {sample} rows and {features} columns")
train.head()
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
sample,features = test.shape[0],test.shape[1]
print(f"Test data contains {sample} rows and {features} columns")
test.head()
plt.figure(figsize = (12,8))
sns.distplot(train['SalePrice']
                ,color = 'r')
#plt.xlabel("Sale Price distribution across train dataset")
plt.ylabel("Frequency")
skewness = np.round(train['SalePrice'].skew(),2)
kurtosis = np.round(train['SalePrice'].kurt(),2)
plt.axvline(np.percentile(train['SalePrice'],80),color = 'blue',label = "80% percentile")
plt.suptitle(f"Sale Price distribution across train dataset")
plt.legend()
plt.show()

print(f"Above Distribution has {skewness} skewness values")
print(f"Above Distribution has {kurtosis} kurtosis values")
plt.figure(figsize = (16,8))
plt.subplot(1,2,1)
sns.distplot(np.log(train['SalePrice'])
                ,color = 'r',label = "Actual Distribution",fit = norm)
#plt.xlabel("Sale Price distribution across train dataset")
plt.ylabel("Frequency")
plt.suptitle(f"log(SalePrice) distribution across train dataset")
plt.legend()

plt.subplot(1,2,2)
stats.probplot(np.log(train['SalePrice']),plot = plt)

plt.show()
train['log(Price)'] = np.log(train['SalePrice']) # added one new feature which is going to be very useful in future sections 
plt.figure(figsize = (24,8))
plt.subplot(1,3,1)
sns.distplot(train['1stFlrSF']
                ,color = 'r',label = "Actual Distribution")

plt.axvline(np.percentile(train['1stFlrSF'],80),color = 'blue',label = "80% percentile")

skewness = np.round(train['1stFlrSF'].skew(),2)
kurtosis = np.round(train['1stFlrSF'].kurt(),2)

plt.ylabel("Frequency")
plt.suptitle(f"log(1stFlrSF) distribution across train dataset")
plt.legend()

plt.subplot(1,3,2)
sns.distplot(np.log(train['1stFlrSF'])
                ,color = 'r',label = "Normal Distribution",fit = norm)
# stats.probplot(np.log(train['1stFlrSF']),plot = plt)

plt.subplot(1,3,3)

sns.scatterplot(train['log(Price)'],np.log(train['1stFlrSF']))
plt.xlabel("log(SalePrice)")
plt.ylabel("log(1stFlrSF)")
cor = stats.pearsonr(train['log(Price)'],np.log(train['1stFlrSF']))
spear_cor = stats.spearmanr(train['log(Price)'],np.log(train['1stFlrSF']))
plt.title("Price vs 1stFlr Area")

plt.show()


print(f"Above Distribution has {skewness} skewness values")
print(f"Above Distribution has {kurtosis} kurtosis values")
print(f"Pearson Correlation between Price and 1stFlr Area is {cor[0]}")
print(f"Spearman Correlation between Price and 1stFlr Area is {spear_cor[0]}")
plt.figure(figsize = (24,8))
plt.subplot(1,3,1)
sns.distplot(train['2ndFlrSF']
                ,color = 'r',label = "Actual Distribution")

skewness = np.round(train['2ndFlrSF'].skew(),2)
kurtosis = np.round(train['2ndFlrSF'].kurt(),2)

plt.ylabel("Frequency")
plt.suptitle(f"log(2ndFlrSF) distribution across train dataset")
plt.legend()

plt.subplot(1,3,2)
sns.distplot(np.log1p(train['2ndFlrSF'])
                ,color = 'r',label = "Normal Distribution",fit = norm)
# stats.probplot(np.log(train['1stFlrSF']),plot = plt)

plt.subplot(1,3,3)

sns.scatterplot(train['log(Price)'],np.log(train['2ndFlrSF']))
plt.xlabel("log(SalePrice)")
plt.ylabel("log(2ndFlrSF)")
cor = stats.pearsonr(train['log(Price)'],np.log1p(train['2ndFlrSF']))
spear_cor = stats.spearmanr(train['log(Price)'],np.log1p(train['2ndFlrSF']))
plt.title("Price vs 2ndFlrSF Area")

plt.show()


print(f"Above Distribution has {skewness} skewness values")
print(f"Above Distribution has {kurtosis} kurtosis values")
print(f"Pearson Correlation between Price and 2ndFlr Area is {cor[0]}")
print(f"Spearman Correlation between Price and 2ndFlr Area is {spear_cor[0]}")
sample = train[train['2ndFlrSF']!=0]

plt.figure(figsize = (24,8))
plt.subplot(1,3,1)
sns.distplot(sample['2ndFlrSF']
                ,color = 'r',label = "Actual Distribution")

skewness = np.round(sample['2ndFlrSF'].skew(),2)
kurtosis = np.round(sample['2ndFlrSF'].kurt(),2)

plt.ylabel("Frequency")
plt.suptitle(f"log(2ndFlrSF) distribution across train dataset")
plt.legend()

plt.subplot(1,3,2)
sns.distplot(np.log1p(sample['2ndFlrSF'])
                ,color = 'r',label = "Normal Distribution",fit = norm)
# stats.probplot(np.log(train['1stFlrSF']),plot = plt)

plt.subplot(1,3,3)

sns.scatterplot(sample['log(Price)'],np.log(sample['2ndFlrSF']))
plt.xlabel("log(SalePrice)")
plt.ylabel("log(2ndFlrSF)")
cor = stats.pearsonr(sample['log(Price)'],np.log1p(sample['2ndFlrSF']))
spear_cor = stats.spearmanr(sample['log(Price)'],np.log1p(sample['2ndFlrSF']))
plt.title("Price vs 2ndFlrSF Area")

plt.show()


print(f"Above Distribution has {skewness} skewness values")
print(f"Above Distribution has {kurtosis} kurtosis values")
print(f"Pearson Correlation between Price and 2ndFlr Area is {cor[0]}")
print(f"Spearman Correlation between Price and 2ndFlr Area is {spear_cor[0]}")
plt.figure(figsize = (24,8))
plt.subplot(1,3,1)
sns.distplot(train['GrLivArea']
                ,color = 'r',label = "Actual Distribution")
plt.axvline(np.percentile(train['GrLivArea'],80),color = 'blue',label = "80% percentile")

skewness = np.round(train['GrLivArea'].skew(),2)
kurtosis = np.round(train['GrLivArea'].kurt(),2)

plt.ylabel("Frequency")
plt.suptitle(f"log(GrLivArea) distribution across train dataset")
plt.legend()

plt.subplot(1,3,2)
sns.distplot(np.log(train['GrLivArea'])
                ,color = 'r',label = "Normal Distribution",fit = norm)
# stats.probplot(np.log(train['1stFlrSF']),plot = plt)

plt.subplot(1,3,3)

sns.scatterplot(train['log(Price)'],np.log(train['GrLivArea']))
plt.xlabel("log(SalePrice)")
plt.ylabel("log(GrLivArea)")
cor = stats.pearsonr(train['log(Price)'],np.log(train['GrLivArea']))
spear_cor = stats.spearmanr(train['log(Price)'],np.log(train['GrLivArea']))
plt.title("Price vs GrLivArea Area")

plt.show()



print(f"Above Distribution has {skewness} skewness values")
print(f"Above Distribution has {kurtosis} kurtosis values")
print(f"Pearson Correlation between Price and GrLivArea is {cor[0]}")
print(f"Spearman Correlation between Price and GrLivArea is {spear_cor[0]}")
#saleprice correlation matrix

col_drop = ["log(Price)"]
plt.figure(figsize = (24,12))
k = 10 #number of variables for heatmap

cols = train.drop(col_drop,axis=1).corr().nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train.drop(col_drop,axis=1)[cols].values.T)
sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
fig,ax = plt.subplots(figsize = (24,8))

temp = train.groupby('YearBuilt')['SalePrice'].median()
sns.boxenplot(x = train['YearBuilt'],y = train['SalePrice'],ax = ax)
#sns.lineplot(y = temp.values,x = temp.index,ax = ax)
ax.plot(temp.values)
plt.xticks(rotation = 90)
plt.suptitle("Year Built vs SalePrice")
plt.show()
fig,ax = plt.subplots(figsize = (24,8))

temp = train.groupby('OverallQual')['SalePrice'].median()
sns.boxenplot(x = train['OverallQual'],y = train['SalePrice'],ax = ax)
#sns.lineplot(y = temp.values,x = temp.index,ax = ax)
ax.plot(temp.values)
plt.xticks(rotation = 90)
plt.suptitle("Overall Quality vs SalePrice")
plt.show()
fig,ax = plt.subplots(figsize = (24,8))

temp = train.groupby('OverallCond')['SalePrice'].median()
sns.boxenplot(x = train['OverallCond'],y = train['SalePrice'],ax = ax)
#sns.lineplot(y = temp.values,x = temp.index,ax = ax)
ax.plot(temp.values)
plt.xticks(rotation = 90)
plt.suptitle("Overall Condition vs SalePrice")
plt.show()
fig,ax = plt.subplots(figsize = (24,8))

temp = train.groupby('FullBath')['SalePrice'].median()
sns.boxenplot(x = train['FullBath'],y = train['SalePrice'],ax = ax)
#sns.lineplot(y = temp.values,x = temp.index,ax = ax)
ax.plot(temp.values)
plt.xticks(rotation = 90)
plt.suptitle("FullBath vs SalePrice")
plt.show()
def percent_missing(df):
    data = pd.DataFrame(df)
    df_cols = list(pd.DataFrame(data))
    dict_x = {}
    for i in range(0, len(df_cols)):
        dict_x.update({df_cols[i]: round(data[df_cols[i]].isnull().mean()*100,2)})
    
    return dict_x

missing = percent_missing(train)
df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)
keys,value = [],[]
for i in range(len(df_miss[:10])):
    keys.append(df_miss[i][0])
    value.append(df_miss[i][1])
# let's see the distribution of missing values

plt.figure(figsize = (16,12))
ax = sns.barplot(x = keys,y = value)

rects = ax.patches

for rect in rects:
    height = rect.get_height()
    ax.text(rect.get_x() + 0.1,height + 2, str (np.round(height,0)), ha = 'left',va = 'top')



plt.xlabel("Top 10 Null Features",fontsize = 20)
plt.ylabel("Frequency",fontsize = 20)
plt.show()
def convert_dtype(data):
    data['MSSubClass'] = data['MSSubClass'].apply(str)
    data['YrSold'] = data['YrSold'].astype(str)
    data['MoSold'] = data['MoSold'].astype(str)
    return(data)
train = convert_dtype(train)
test = convert_dtype(test)
print(train.shape)
print(test.shape)
def handle_missing(data):
    
    # the data description states that NA refers to typical ('Typ') values
    data['Functional'] = data['Functional'].fillna('Typ')
    
    # Replace the missing values in each of the columns below with their mode
    data['Electrical'] = data['Electrical'].fillna("SBrkr")
    data['KitchenQual'] = data['KitchenQual'].fillna("TA")
    data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])
    data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])
    data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])
    data['MSZoning'] = data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
    
    
    # the data description stats that NA refers to "No Pool"
    data["PoolQC"] = data["PoolQC"].fillna("None")
    
    
    # Replacing the missing values with 0, since no garage = no cars in garage
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        data[col] = data[col].fillna(0)
    
    # Replacing the missing values with None
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        data[col] = data[col].fillna('None')
    
    # NaN values for these categorical basement features, means there's no basement
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        data[col] = data[col].fillna('None')
        
    # Group the by neighborhoods, and fill in missing value by the median LotFrontage of the neighborhood
    data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    # We have no particular intuition around how to fill in the rest of the categorical features
    
    # So we replace their missing values with None
    objects = []
    for i in data.columns:
        if data[i].dtype == object:
            objects.append(i)
    data.update(data[objects].fillna('None'))
        
    # And we do the same thing for numerical features, but this time with 0s
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = []
    for i in data.columns:
        if data[i].dtype in numeric_dtypes:
            numeric.append(i)
    data.update(data[numeric].fillna(0))    
    return data

train = handle_missing(train)
test = handle_missing(test)
print(train.shape)
print(test.shape)
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in train.drop(['Id','SalePrice','log(Price)'],axis=1).columns:
    if train[i].dtype in numeric_dtypes:
        numeric.append(i)
        
def find_skewness(data):
    skew = {}
    for col in data.columns:
        skewness = np.round(data[col].skew(),3)
        skew[col] = skewness
    return(skew)

skew_data = find_skewness(train[numeric]) 
skew_data = dict(sorted(skew_data.items(), key=lambda x: x[1], reverse=True))

f, ax = plt.subplots(figsize=(16, 8))
ax.set_yscale("log")
ax = sns.boxenplot(data = train[numeric], orient="v", palette="Set2")
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
plt.xticks(rotation = 90)
sns.despine(trim=True, left=True)
for i in list(skew_data.keys())[:5]:
    train[i] = stats.boxcox(train[i]+1
                            , stats.boxcox_normmax(train[i] + 1))
skew_data = find_skewness(train[numeric]) 
skew_data = dict(sorted(skew_data.items(), key=lambda x: x[1], reverse=True))

f, ax = plt.subplots(figsize=(16, 8))
ax.set_yscale("log")
ax = sns.boxenplot(data = train[numeric], orient="v", palette="Set2")
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features (After BoxCox Transformation)")
plt.xticks(rotation = 90)
sns.despine(trim=True, left=True)
numeric = []
for i in test.drop(['Id'],axis=1).columns:
    if test[i].dtype in numeric_dtypes:
        numeric.append(i)
skew_data = find_skewness(test[numeric]) 
skew_data = dict(sorted(skew_data.items(), key=lambda x: x[1], reverse=True))
for i in list(skew_data.keys())[:5]:
    test[i] = stats.boxcox(test[i]+1
                            , stats.boxcox_normmax(test[i] + 1))
    
print(train.shape)
print(test.shape)
def new_features(data):
    
    data['BsmtFinType1_Unf'] = 1*(data['BsmtFinType1'] == 'Unf')
    data['HasWoodDeck'] = (data['WoodDeckSF'] == 0) * 1
    data['HasOpenPorch'] = (data['OpenPorchSF'] == 0) * 1
    data['HasEnclosedPorch'] = (data['EnclosedPorch'] == 0) * 1
    data['Has3SsnPorch'] = (data['3SsnPorch'] == 0) * 1
    data['HasScreenPorch'] = (data['ScreenPorch'] == 0) * 1
    data['YearsSinceRemodel'] = data['YrSold'].astype(int) - data['YearRemodAdd'].astype(int)
    data['Total_Home_Quality'] = data['OverallQual'] + data['OverallCond']
    data = data.drop(['Utilities', 'Street', 'PoolQC',], axis=1)
    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
    data['YrBltAndRemod'] = data['YearBuilt'] + data['YearRemodAdd']

    data['Total_sqr_footage'] = (data['BsmtFinSF1'] + data['BsmtFinSF2'] +
                                     data['1stFlrSF'] + data['2ndFlrSF'])
    data['Total_Bathrooms'] = (data['FullBath'] + (0.5 * data['HalfBath']) +
                                   data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath']))
    data['Total_porch_sf'] = (data['OpenPorchSF'] + data['3SsnPorch'] +
                                  data['EnclosedPorch'] + data['ScreenPorch'] +
                                  data['WoodDeckSF'])
    
    
    # below transformation is for each bimodal distribution 
    
    data['TotalBsmtSF'] = data['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
    data['2ndFlrSF'] = data['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
    data['GarageArea'] = data['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
    data['GarageCars'] = data['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)
    data['LotFrontage'] = data['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)
    data['MasVnrArea'] = data['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)
    data['BsmtFinSF1'] = data['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
    
    # boolean features

    data['haspool'] = data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    data['has2ndfloor'] = data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    data['hasgarage'] = data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    data['hasbsmt'] = data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    data['hasfireplace'] = data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    return(data)
train = new_features(train)
test = new_features(test)
print(train.shape)
print(test.shape)
def log_square_features(data):
    log_features = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                     'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
                     'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
                     'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',
                     'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearRemodAdd','TotalSF']

    for col in log_features:
        data['log('+col+')'] = np.log(1.01+data[col])

    squared_features = ['log(LotFrontage)', 
                  'log(TotalBsmtSF)', 'log(1stFlrSF)', 'log(2ndFlrSF)', 'log(GrLivArea)',
                  'log(GarageCars)', 'log(GarageArea)']

    for col in squared_features:
        data['Square_'+col] = data[col]*data[col]
    return(data)

train = log_square_features(train)
test = log_square_features(test)
print(train.shape)
print(test.shape)
train.drop(['Id'],axis=1,inplace = True)
test.drop(['Id'],axis=1,inplace = True)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(train.drop(['SalePrice','log(Price)'],axis=1),train[['SalePrice','log(Price)']],test_size = 0.05,random_state = 42)
print(f"Train data has shape {X_train.shape}")
print(f"Test data has shape {X_test.shape}")
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
obj_col = []
num_col = []
for col in X_train.columns:
    if X_train[col].dtype=='O':
        obj_col.append(col)
    else:
        num_col.append(col)

temp1 = pd.DataFrame()
temp2 = pd.DataFrame()
temp3 = pd.DataFrame()
error = []       
for i in tqdm(range(len(obj_col))):
    try:
        vec = CountVectorizer(stop_words=[])
        vec.fit(X_train[obj_col[i]])
        X1 = vec.transform(X_train[obj_col[i]])
        X2 = vec.transform(X_test[obj_col[i]])
        X3 = vec.transform(test[obj_col[i]])
    except: 
        error.append(obj_col[i])
        continue
    
    feature_name = []
    for f in vec.get_feature_names():
        feature_name.append('f'+str(i)+'_'+f)
        
    X1 = pd.DataFrame(X1.toarray(),columns = feature_name)
    X2 = pd.DataFrame(X2.toarray(),columns = feature_name)
    X3 = pd.DataFrame(X3.toarray(),columns = feature_name)
    temp1 = pd.concat([temp1,X1],axis=1)
    temp2 = pd.concat([temp2,X2],axis=1)
    temp3 = pd.concat([temp3,X3],axis=1)
X_train = X_train[num_col]
X_test = X_test[num_col]
test = test[num_col]
print(X_train.shape)
print(X_test.shape)
print(test.shape)
X_train = pd.concat([X_train,temp1],axis=1)
X_test = pd.concat([X_test,temp2],axis=1)
test = pd.concat([test,temp3],axis=1)
print(X_train.shape)
print(X_test.shape)
print(test.shape)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from mlxtend.regressor import StackingRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error
# Misc
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
def rmse(y_true,y_pred):
    return(np.sqrt(mean_squared_error(y_true,y_pred)))
robust_scaler = RobustScaler()
robust_scaler.fit(X_train)
X_tr = robust_scaler.transform(X_train)
X_ts = robust_scaler.transform(X_test)
test = robust_scaler.transform(test)
%%time
ridge = Ridge()
param_dist = {'alpha':[1e-5,9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5,10,12,15,20,30,50],
              'solver':['auto','svd','lsqr','cholesky','saga']}

clf = GridSearchCV(ridge,
                   param_grid=param_dist,
                   cv=5,
                   return_train_score=True,
                   scoring = 'neg_mean_squared_error')

clf.fit(X_tr,y_train['log(Price)'].values)
ridge = clf.best_estimator_ #Ridge(alpha = 12,solver = 'svd')
ridge.fit(X_tr,y_train['log(Price)'].values)
pred = ridge.predict(X_ts)

print("RMSE score for test data is :",rmse(y_test['log(Price)'].values,pred))
rfg = RandomForestRegressor(n_estimators=500,
                          max_depth=30,
                          min_samples_split=2,
                          min_samples_leaf=2,
                          max_features=None,
                          oob_score=True,
                          random_state=42)
rfg.fit(X_tr,y_train['log(Price)'].values)
pred = rfg.predict(X_ts)
print("RMSE score for test data is :",rmse(y_test['log(Price)'].values,pred))
lightgbm = LGBMRegressor(objective='regression', 
                       num_leaves=6,
                       learning_rate=0.002, 
                       n_estimators=10000,
                       max_bin=200, 
                       bagging_fraction=0.8,
                       bagging_freq=4, 
                       bagging_seed=8,
                       feature_fraction=0.2,
                       feature_fraction_seed=8,
                       min_sum_hessian_in_leaf = 11,
                       verbose=-1,
                       random_state=42)

lightgbm.fit(X_tr,y_train['log(Price)'].values)
pred = lightgbm.predict(X_ts)
print("RMSE score for test data is :",rmse(y_test['log(Price)'].values,pred))
gbr = GradientBoostingRegressor(n_estimators=2000,
                                learning_rate=0.005,
                                max_depth=5,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=42)

gbr.fit(X_tr,y_train['log(Price)'].values)
pred = gbr.predict(X_ts)
print("RMSE score for test data is :",rmse(y_test['log(Price)'].values,pred))
stack_gen = StackingRegressor(regressors=(lightgbm, gbr, rfg),
                                meta_regressor=gbr)
stack_gen.fit(X_tr,y_train['log(Price)'].values)
pred = stack_gen.predict(X_ts)
print("RMSE score for test data is :",rmse(y_test['log(Price)'].values,pred))
def ensemble(data):
    return( 0.3*gbr.predict(data) + 
          0.2*lightgbm.predict(data) + 
         0.1*stack_gen.predict(data) + 0.1*rfg.predict(data)+ 0.3*ridge.predict(data))

rmse(y_test['log(Price)'].values,ensemble(X_ts))
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
# submission['SalePrice'] = (np.exp(gbr.predict(test)) + np.exp(lightgbm.predict(test)) + np.exp(stack_gen.predict(test)) + np.exp(rfg.predict(test))+ np.exp(ridge.predict(test)))/5
submission['SalePrice'] = 0.20*np.exp(gbr.predict(test)+0.0) +  0.25*np.exp(lightgbm.predict(test)) +  0.35*np.exp(stack_gen.predict(test)) + 0.2*np.exp(rfg.predict(test))