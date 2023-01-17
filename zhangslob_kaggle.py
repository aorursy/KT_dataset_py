import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn import metrics

%matplotlib inline

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sns.distplot(train['SalePrice'])
train['SalePrice'].describe()
#skewness and kurtosis
print("偏度: %f" % train['SalePrice'].skew())
print("峰度: %f" % train['SalePrice'].kurt())
# 查看热图
data = train.corr()
sns.heatmap(data)
data['SalePrice'].sort_values()
sns.lmplot(x='GrLivArea', y='SalePrice', data=train, fit_reg=False, scatter=True)
train = train[-((train.SalePrice < 200000) &  (train.GrLivArea > 4000))]
for col in train.columns:
    if train[col].isnull().sum() > 0:
        print(col, train[col].isnull().sum())
#一半是删除过多空值的属性，一半是删除无关联的属性 
train = train.drop(["MiscFeature", "Id", "PoolQC", "Alley", "Fence","GarageFinish", "KitchenAbvGr", "EnclosedPorch", "MSSubClass", "OverallCond", "YrSold", "LowQualFinSF", "MiscVal", "BsmtHalfBath", "BsmtFinSF2", "3SsnPorch", "MoSold", "PoolArea"], axis=1)

test = test.drop(["MiscFeature", "PoolQC", "Alley", "Fence","GarageFinish", "KitchenAbvGr", "EnclosedPorch", "MSSubClass", "OverallCond", "YrSold", "LowQualFinSF", "MiscVal", "BsmtHalfBath", "BsmtFinSF2", "3SsnPorch", "MoSold", "PoolArea"], axis=1)

#汇总train和test的数据
all_data = pd.concat((train, test))
#如果数字，填写均值。如果字符串，填写次数最多的
for col in train.columns:
    if train[col].isnull().sum() > 0:
        if train[col].dtypes == 'object':
            val = all_data[col].dropna().value_counts().idxmax()
            train[col] = train[col].fillna(val)
        else:
            val = all_data[col].dropna().mean()
            train[col] = train[col].fillna(val)
            
for col in test.columns:
    if test[col].isnull().sum() > 0:
        if test[col].dtypes == 'object':
            val = all_data[col].dropna().value_counts().idxmax()
            test[col] = test[col].fillna(val)
        else:
            val = all_data[col].dropna().mean()
            test[col] = test[col].fillna(val)
#综合处理，转值类型
for col in all_data.select_dtypes(include = [object]).columns:
    train[col] = train[col].astype('category', categories = all_data[col].dropna().unique())
    test[col] = test[col].astype('category', categories = all_data[col].dropna().unique())

for col in train.columns:
    if train[col].dtype.name == 'category':
        tmp = pd.get_dummies(train[col], prefix = col)
        train = train.join(tmp)
        train = train.drop(col, axis=1)

for col in test.columns:
    if test[col].dtype.name == 'category':
        tmp = pd.get_dummies(test[col], prefix = col)
        test = test.join(tmp)
        test = test.drop(col, axis=1)
lr = linear_model.LinearRegression()
X = train.drop("SalePrice", axis=1)
y = np.log(train["SalePrice"])

lr = lr.fit(X, y)
results = lr.predict(test.drop("Id", axis = 1))
final = np.exp(results)

submission = pd.DataFrame()
submission['Id'] = test.Id
submission['SalePrice'] = final

submission.head()
submission.to_csv("submission1.csv", index= False)
