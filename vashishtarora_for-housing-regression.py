# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Thanks to Monthep https://www.kaggle.com/monthepp/house-prices-advanced-regression-techniques/notebook

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
test_df.head(3)

train_df.shape
#test_df.head(3)
test_df.shape
test_df.insert(80,'SalePrice',0)

test_df.shape

test_df.head(2)
train_df['type'],test_df['type']= 'training','testing'
import seaborn as sns
#sns.distplot(train_df['SalePrice']);
#it is a case of positive skewness
train_df['SalePrice'] = np.log1p(train_df.loc[train_df['type']=='training',['SalePrice']])
sns.distplot(train_df['SalePrice']);
#test_df.head(4)
combine_df=pd.concat([train_df,test_df])
combine_df.shape
total = combine_df.isnull().sum().sort_values(ascending=False)
percent = (combine_df.isnull().sum()/combine_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis = 1,keys = ['total','percent'])
missing_data.head(50)
# Dropping these columns as these are not of much use as they ahve many missing values
combine_df = combine_df.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage'],axis = 1)
#combine_df = combine_df.drop(['LotFrontage'],axis = 1)

combine_df.shape
#combine_df.info()
col_list = combine_df.select_dtypes(['number']).columns.tolist()
print(col_list)
print(len(col_list))
col_list_obj = combine_df.select_dtypes(['object']).columns.tolist()
print(col_list_obj)
print(len(col_list_obj))
total = combine_df.isnull().sum().sort_values(ascending=False)
percent = (combine_df.isnull().sum()/combine_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis = 1,keys = ['total','percent'])
missing_data.head(28)
#for numerical columns we will use scatter plots
sns.distplot(combine_df['SalePrice']  )
print('skewness is {0}'.format(combine_df['SalePrice'].skew()))


#combine_df.loc[combine_df['type']=='training','LotShape']
combine_df = combine_df.drop(['GarageType','GarageYrBlt','GarageFinish'],axis = 1)
col_list = combine_df.select_dtypes(['number']).columns.tolist()
print(col_list)
print(len(col_list))
col_list_obj = combine_df.select_dtypes(['object']).columns.tolist()
print(col_list_obj)
print(len(col_list_obj))
def scatter_plo(var1):
    fig, axes = plt.subplots(figsize=(14, 12), ncols=4, nrows=3, sharey=True)
    axes = axes.flatten()
    for i,v in enumerate(var1):
        #data = pd.concat([combine_df['SalePrice'], combine_df[v]], axis=1)
        sns.scatterplot(ax =axes[i],x=v, y='SalePrice',data = combine_df[combine_df['type']=='training']);
def box_plo(var1):
    fig, axes = plt.subplots(figsize=(16, 9), ncols=4, nrows=3, sharey=True)
    axes = axes.flatten()
    print(type(var1))
    for i,v in enumerate(var1):
        #data = pd.concat([combine_df['SalePrice'], combine_df[combine_df['type']=='training']], axis=1)
        #f, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(ax=axes[i],x=v, y="SalePrice", data=combine_df[combine_df['type']=='training'])
        #fig.axis(ymin=0, ymax=1000);
box_plo(col_list_obj[:12])
    
#scatter_plo(col_list[:12])
#scatter_plo(col_list[12:24])
#scatter_plo(col_list[24:36])


# it is having 4 missing values
combine_df['MSZoning'] = combine_df['MSZoning'].fillna(combine_df['MSZoning'].value_counts().idxmax())
combine_df['BsmtCond'] = combine_df['BsmtCond'].fillna(combine_df['BsmtCond'].value_counts().idxmax())
combine_df[combine_df['type'] =='training'].sort_values(by = 'Utilities', ascending = False)[:1]
combine_df = combine_df.drop(combine_df[combine_df['Id'] == 945].index)
combine_df = combine_df.drop(combine_df.loc[combine_df['Electrical'].isnull()].index)
#combine_df = combine_df.drop(combine_df[combine_df['Id'] == 945].index)
total = combine_df[combine_df['type']=='training'].isnull().sum().sort_values(ascending=False)
percent = (combine_df[combine_df['type']=='training'].isnull().sum()/combine_df[combine_df['type']=='training'].isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis = 1,keys = ['total','percent'])
missing_data.head(28)
combine_df.loc[(combine_df['LotArea'] > 100000) & (combine_df['type'] == 'training'), 'type'] = 'excluded'
combine_df.loc[(combine_df['MasVnrArea'] > 1200) & (combine_df['type'] == 'training'), 'type'] = 'excluded'
combine_df.loc[(combine_df['TotalBsmtSF'] > 4000) & (combine_df['type'] == 'training'), 'type'] = 'excluded'
combine_df.loc[(combine_df['1stFlrSF'] > 3000) & (combine_df['type'] == 'training'), 'type'] = 'excluded'
combine_df.loc[(combine_df['GrLivArea'] > 4000) & (combine_df['type'] == 'training'), 'type'] = 'excluded'
combine_df.loc[(combine_df['BedroomAbvGr'] > 6) & (combine_df['type'] == 'training'), 'type'] = 'excluded'
combine_df.loc[(combine_df['KitchenAbvGr'] > 2) & (combine_df['type'] == 'training'), 'type'] = 'excluded'






combine_df.loc[(combine_df['EnclosedPorch'] > 400) & (combine_df['type'] == 'training'), 'type'] = 'excluded'

combine_df.loc[(combine_df['WoodDeckSF'] > 800) & (combine_df['type'] == 'training'), 'type'] = 'excluded'

combine_df.loc[(combine_df['OpenPorchSF'] > 400) & (combine_df['type'] == 'training'), 'type'] = 'excluded'

combine_df.loc[(combine_df['3SsnPorch'] > 400) & (combine_df['type'] == 'training'), 'type'] = 'excluded'
combine_df.loc[(combine_df['MiscVal'] > 5000) & (combine_df['type'] == 'training'), 'type'] = 'excluded'
combine_df[combine_df['type']=='training'].shape
combine_df['OverallQualCond'] = combine_df['OverallQual'] + combine_df['OverallCond']
col = ['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','GarageArea','TotalBsmtSF','GrLivArea']
combine_df[col]=combine_df[col].fillna(0)
combine_df['totalarea'] = combine_df['OpenPorchSF'] + combine_df['EnclosedPorch'] + combine_df['3SsnPorch']+ combine_df['ScreenPorch']+ combine_df['GarageArea']+ combine_df['TotalBsmtSF']+ combine_df['GrLivArea'] 
#combine_df['ExterQual'] =pd.concat([train_df['ExterQual'],test_df['ExterQual']], ignore_index=True)
#combine_df['ExterCond'] = pd.concat([train_df['ExterCond'],test_df['ExterCond']], ignore_index=True)
combine_df[combine_df['type']=='training'].shape
combine_df['ExterQual'] = combine_df['ExterQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3,'Fa': 2,'Po': 1})
combine_df['ExterCond'] = combine_df['ExterCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3,'Fa': 2,'Po': 1})
combine_df[combine_df['type']=='training'].shape
test_df.shape
total = combine_df.isnull().sum().sort_values(ascending=False)
percent = (combine_df.isnull().sum()/combine_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis = 1,keys = ['total','percent'])
missing_data.head(19)
combine_df['ExterQualCond'] = combine_df['ExterQual'] + combine_df['ExterCond']

combine_df.head(2)
combine_df[combine_df['type']=='testing'].shape
combine_df[combine_df['type']=='training'].shape
#combine_df['BsmtQual'] =pd.concat([train_df['BsmtQual'],test_df['BsmtQual']], ignore_index=True)
#combine_df['BsmtCond'] = pd.concat([train_df['BsmtCond'],test_df['BsmtCond']], ignore_index=True)
combine_df['BsmtQual'] = combine_df['BsmtQual'].map({'NA':0,'Ex': 5, 'Gd': 4, 'TA': 3,'Fa': 2,'Po': 1})
combine_df['BsmtCond'] = combine_df['BsmtCond'].map({'NA':0,'Ex': 5, 'Gd': 4, 'TA': 3,'Fa': 2,'Po': 1})
col = ['BsmtQual','BsmtCond']
combine_df[col] = combine_df[col].fillna(0)
combine_df[combine_df['type']=='testing'].shape

combine_df[combine_df['type']=='training'].shape
combine_df['BsmtQualCond'] = combine_df['BsmtQual'] + combine_df['BsmtCond']
#combine_df['GarageQual'] =pd.concat([train_df['GarageQual'],test_df['GarageQual']], ignore_index=True)
#combine_df['GarageCond'] = pd.concat([train_df['GarageCond'],test_df['GarageCond']], ignore_index=True)
combine_df['GarageQual'] = combine_df['GarageQual'].map({'NA':0,'Ex': 5, 'Gd': 4, 'TA': 3,'Fa': 2,'Po': 1})
combine_df['GarageCond'] = combine_df['GarageCond'].map({'NA':0,'Ex': 5, 'Gd': 4, 'TA': 3,'Fa': 2,'Po': 1})
col = ['GarageQual','GarageCond']
combine_df[col] = combine_df[col].fillna(0)
combine_df['GarageQualCond'] = combine_df['GarageQual'] + combine_df['GarageCond']
combine_df[combine_df['type']=='testing'].shape
combine_df[combine_df['type']=='training'].shape
combine_df['totalarea'].head(2)
col = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd']
combine_df[col] = combine_df[col].fillna(0)
combine_df['TotalBsmtBath'] = combine_df['BsmtFullBath'] + combine_df['BsmtHalfBath'] * 0.5
combine_df['TotalAbvGrBath'] = combine_df['FullBath'] + combine_df['HalfBath'] * 0.5
combine_df['TotalAbvGrRooms'] = combine_df['TotRmsAbvGrd'] + combine_df['TotalAbvGrBath']
combine_df['TotalRooms'] = combine_df['TotRmsAbvGrd'] + combine_df['TotalBsmtBath'] + combine_df['TotalAbvGrBath']
combine_df.loc[:, col].describe(include='all')
from scipy.stats import skew
for col in combine_df.select_dtypes(include=['number']).columns.tolist():
    skewness = skew(combine_df[col].dropna())
    if skewness > 1.0: combine_df[col] = np.log1p(combine_df[col])
    combine_df[col] = combine_df[col].fillna(0)
for col in combine_df.select_dtypes(include=['object']).columns.tolist():
    combine_df[col] = combine_df[col].fillna('None')
combine_df[combine_df['type']=='testing'].shape
combine_df[combine_df['type']=='training'].shape
combine_df.head()
combine_df['type'] = combine_df['type'].map({'training':1,'testing':2,'excluded':0})
#combine_df.info()
#combine_df['type'].unique()
#abc =combine_df[combine_df['type']=='excluded'].sum()
combine_df.groupby('type')['type'].value_counts()
#combine_df.info()
#print(abc)

combine_df[combine_df['type']==1].shape
combine_df[combine_df['type']== 2].shape
objects_col = combine_df.select_dtypes('object').columns.tolist()
print(objects_col)

for col in objects_col:
    combine_df[col] =      combine_df[col].astype('category')
#combine_df[objects_col] = combine_df[objects_col].astype('category')
#combine_df.info()
                                                  
col_objects = combine_df.select_dtypes(['category']).columns
combine_df[col_objects] = combine_df[col_objects].apply(lambda x:x.cat.codes)
combine_df.info()

combine_df[combine_df['type']== 1].shape
combine_df[combine_df['type']== 2].shape
correlationmap = combine_df[combine_df['type'] ==1].corr()
fig,ax = plt.subplots(figsize=(30,20))
heatmap = sns.heatmap(correlationmap,annot = True,cmap = plt.cm.RdBu,fmt='.1f',square= True)
x = combine_df[combine_df['type'] == 1].drop(['Id','SalePrice', 'type','GarageQual','GarageCond', 'BsmtQual','BsmtCond','ExterQual','ExterCond','OverallQual','OverallCond'], axis=1)
y = combine_df[combine_df['type'] == 1]['SalePrice']

combine_df[combine_df['type']== 1].shape
x.head(4)
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
forestreg = RandomForestRegressor(max_depth=99, n_estimators=2000, random_state=0).fit(x, y)
feat = pd.DataFrame(data=forestreg.feature_importances_, index=x.columns, columns=['FeatureImportances']).sort_values(['FeatureImportances'], ascending=False)
# list feature importances
feat[feat['FeatureImportances'] > 0.0001].shape
x = combine_df[combine_df['type'] == 1][feat[feat['FeatureImportances'] > 0.0001].index]
y = combine_df[combine_df['type'] == 1]['SalePrice']
from sklearn.preprocessing import RobustScaler
x.shape
feat[feat['FeatureImportances'] > 0.0001].shape
scaler = RobustScaler()
x = scaler.fit_transform(x)
feat[feat['FeatureImportances'] > 0.0001].index
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
x_train, x_validate, y_train, y_validate = train_test_split(x, y, random_state=0, test_size=0.25)
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
#from sklearn.ensemble import DecisionTreeRegressor,RandomForestRegressor

def rmse(model,x,y):
    return(np.sqrt(np.abs(cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv = 5))))
#alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
#cv_ridge = [rmse(Ridge(alpha = alpha),X,y).mean() 
           # for alpha in alphas]
linearreg = LinearRegression().fit(x_train,y_train)
linearreg_predict = linearreg.predict(x_validate)
linreg_cvscores = rmse(linearreg, x, y)
print('linear regression  and cross validation score is {0} '.format(linreg_cvscores.mean() *10))

Ridgereg = Ridge(alpha=10).fit(x_train,y_train)
Ridgereg_predict = Ridgereg.predict(x_validate)
Ridgereg_cvscores = rmse(Ridgereg, x, y)
print('Ridge regression  and cross validation score is {0}'.format(Ridgereg_cvscores.mean()*10))

Lassoreg = Lasso(alpha=0.01).fit(x_train,y_train)
Lassoreg_predict = Lassoreg.predict(x_validate)
Lassoreg_cvscores = rmse(Lassoreg, x, y)
print('Lasso regression  and cross validation score is {0}'.format(Lassoreg_cvscores.mean()*10))

treereg = DecisionTreeRegressor(max_depth=20, min_samples_split=5, splitter='best').fit(x_train, y_train)
treereg_ypredict = treereg.predict(x_validate)
treereg_cvscores = rmse(treereg, x, y)
print('Tree regression  and cross validation score is {0}'.format(treereg_cvscores.mean()*10))

forestreg = RandomForestRegressor(max_depth=20, min_samples_split=5, n_estimators=250, random_state=0).fit(x_train, y_train)
forestreg_ypredict = forestreg.predict(x_validate)
forestreg_cvscores = rmse(forestreg, x, y)
print('Random forest regression  and cross validation score is {0}'.format(forestreg_cvscores.mean()*10))
#x_test.shape
#x_test = combine_df[combine_df['type'] == 2].drop(['Id','SalePrice', 'type','GarageQual','GarageCond', 'BsmtQual','BsmtCond','ExterQual','ExterCond','OverallQual','OverallCond'], axis=1)
#x_test.shape
x_test = combine_df[combine_df['type'] == 2].drop(['Id','SalePrice', 'type','GarageQual','GarageCond', 'BsmtQual','BsmtCond','ExterQual','ExterCond','OverallQual','OverallCond'], axis=1)
x_test = x_test[feat[feat['FeatureImportances'] > 0.0001].index]
#x_test.shape
x_test = scaler.transform(x_test)
#y_test = pd.DataFrame(np.expm1(Ridgereg.predict(x_test)), columns=['SalePrice'])

y_test.head(10)
#out = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': y_test['SalePrice']})
#out.to_csv('submission.csv', index=False)
#coef = pd.Series(Lassoreg.coef_, index = x_train.columns)
import xgboost
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
xgb.fit(x_train,y_train)
#predictions = np.exp(xgb.predict(x_validate))

xgb_cvscores = rmse(xgb, x, y)
print(xgb_cvscores.mean())
y_test = pd.DataFrame(np.expm1(Ridgereg.predict(x_test)), columns=['SalePrice'])
out = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': y_test['SalePrice']})
out.to_csv('submission.csv', index=False)
#predictions.head(5)
#xgb.fit(x_train,y_train)
#x_test = combine_df[combine_df['type'] == 2][feat[feat['FeatureImportances'] > 0.01].index]
#y_test = pd.DataFrame(xgb.predict(x_test), columns=['Saleprice'])
#x_test.head(5)
#y_test.head(5)
#XGBreg_cvscores = rmse(xgb, X, y)
#print('XGB  regression  and cross validation score is {0}'.format(XGBreg_cvscores.mean()*10))
# As per this above, we will use XGB model with grid search CV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

xg_reg = xgboost.XGBRegressor()
parameters_xg = {"objective" : ["reg:linear"], "n_estimators" : [5, 10, 15, 20]}

grid_xg = GridSearchCV(xg_reg, parameters_xg, verbose=1 , scoring="r2")
print("Best XGB Model: " + str(grid_xg.best_estimator_))
print("Best Score: " + str(grid_xg.best_score_))