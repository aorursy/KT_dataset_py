# Import Libraries
import numpy as np
import pandas as pd
import scipy.stats as sp
# Import Training Data
ames = pd.read_csv('../input/train.csv',index_col=['YrSold','MoSold','Id'])
ames.sort_index(inplace=True)
ames.shape
# Import Training Data
test = pd.read_csv('../input/test.csv',index_col=['YrSold','MoSold','Id'])
test.sort_index(inplace=True)
test.shape
with pd.option_context('display.max_columns', len(ames.columns)-1):
    display(ames.drop('SalePrice',axis=1).describe())
discrete = ['FullBath', 'HalfBath', 'Fireplaces', 'GarageCars']

nominal = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig', 
           'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
           'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Foundation', 
           'CentralAir', 'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition',
           'BedroomAbvGr', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageFinish', 
           'Heating', 'KitchenAbvGr','MasVnrType']

continuous = ['YearBuilt', 'YearRemodAdd', 'LotFrontage', 'LotArea', 'GarageArea', 
              'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
             '1stFlrSF', '2ndFlrSF','BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF',
             'WoodDeckSF', 'GarageYrBlt', 'GrLivArea', 'BsmtUnfSF', 'LowQualFinSF',
             'MasVnrArea','OpenPorchSF']

ordinal = ['LotShape', 'TotRmsAbvGrd', 'Utilities', 'LandSlope', 'OverallQual', 
           'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
           'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageQual', 
           'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'BsmtFullBath', 'BsmtHalfBath']
ames.SalePrice.describe().to_frame().style.format("{:,.0f}")
_ = ames.SalePrice.value_counts().sort_index().plot()
ts = ames.groupby(level=['YrSold','MoSold']).SalePrice.aggregate(['sum','size'])
ts.index = ts.index.map(lambda x: x[0] + x[1]/12)
_ = ts.plot(secondary_y='size',grid=True)
ts = ames.groupby(level=['YrSold','MoSold']).SalePrice.median()
ts.index = ts.index.map(lambda x: x[0] + x[1]/12)
_ = ts.plot(title='SalePrice (Month Median)',grid=True)
for col in nominal:
    groups = ames.groupby(col)
    f_test = sp.f_oneway(*[g.SalePrice.values for _,g in groups])
    out = 'Predictor {} ONE WAY ANOVA RE SalePrice: F {:6.1f}  P-value {:.6f}'.format(col,f_test.statistic,f_test.pvalue)
    print(out)
    display(groups.SalePrice.aggregate(['mean','count']).style.\
            format({'mean':"${:,.0f}",'size':"{:,d}"}).background_gradient(axis=0,cmap='RdYlGn'))
for col in ordinal:
    groups = ames.groupby(col)
    f_test = sp.f_oneway(*[g.SalePrice.values for _,g in groups])
    out = 'Predictor {} ONE WAY ANOVA RE SalePrice: F {:6.1f}  P-value {:.6f}'.format(col,f_test.statistic,f_test.pvalue)
    print(out)
    display(groups.SalePrice.aggregate(['mean','count']).style.\
            format({'mean':"${:,.0f}",'size':"{:,d}"}).background_gradient(axis=0,cmap='RdYlGn'))
for col in continuous[:15]:
    x = ames[[col,'SalePrice']]
    out = '{:25} Correlation to SalePrice = {:8.4}'.format(col, x.corr().iloc[1,0])
    print(out)
    x.plot.scatter(col,'SalePrice',logy=True,grid=True)
for col in continuous[15:]:
    x = ames[[col,'SalePrice']]
    out = '{:25} Correlation to SalePrice = {:8.4}'.format(col, x.corr().iloc[1,0])
    print(out)
    x.plot.scatter(col,'SalePrice',logy=True,grid=True)
for col in discrete:
    groups = ames.groupby(col)
    f_test = sp.f_oneway(*[g.SalePrice.values for _,g in groups])
    out = 'Predictor {} ONE WAY ANOVA RE SalePrice: F {:6.1f}  P-value {:.6f}'.format(col,f_test.statistic,f_test.pvalue)
    print(out)
    ames[[col,'SalePrice']].dropna().boxplot('SalePrice', by=col, figsize=(8,8))
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn import tree

X = pd.concat([ames.reset_index().drop('SalePrice',axis=1),test.reset_index()])
object_types = X.select_dtypes('O').columns
X = pd.get_dummies(X,columns=object_types).drop('Id',axis=1)
preds = X.columns
y = np.array([0]*len(ames) + [1]*len(test))

impute = SimpleImputer(strategy='median')
X = impute.fit_transform(X)

rough = DecisionTreeClassifier(max_depth=3,min_samples_leaf=.02)

rough.fit(X,y)

fi = pd.Series(rough.feature_importances_,index=preds)
fi[fi>0].plot(kind='bar',title='Important Features')
print('WEAK MODEL (Accuracy {:.2%})'.format(rough.score(X,y)))
dot_data = tree.export_graphviz(rough,out_file=None,
                              impurity = False,
                              feature_names = preds,
                              class_names = ['Train', 'Test'],
                              rounded = True,
                              filled= True )
import graphviz 
graphviz.Source(dot_data) 