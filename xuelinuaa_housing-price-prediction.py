# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import warnings
warnings.filterwarnings('ignore')
data_train=pd.read_csv('../input/train.csv')
data_test=pd.read_csv('../input/test.csv')
data_train
data_train.shape,data_test.shape
data_train.info()
#删除第一列['Id']
data_train.drop(['Id'],axis =1)
#求相关性矩阵，筛选与SalePrice具有最大系数的相关变量
corr_matrix=data_train.corr()
f,ax=plt.subplots(figsize=(20,9))
sns.heatmap(corr_matrix,vmax=1,annot=True)
#选取具有最大相关性features，top_corr_feature
top_corr_feature=corr_matrix.index[abs(corr_matrix["SalePrice"])>0.5]
print(top_corr_feature)
#进一步缩小相关矩阵,10个具有最大相关性的features
top_corr_matrix=data_train[top_corr_feature].corr()
f,ax=plt.subplots(figsize=(15,10))
sns.heatmap(top_corr_matrix,vmax=1,annot=True)
#求数据两两之间关系，绘制关系点图
cols=['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars']
sns.pairplot(data_train[cols],size=4)
plt.show()
data_train[cols].isnull().sum()
data_test[cols].isnull().sum()
#np.median(data_test['TotalBsmtSF']) 
print(data_test['TotalBsmtSF'].median())         #TotalBsmtSF中位数
print(data_test['GarageCars'].median())          #GarageCars中位数

#对test数据中的缺失值进行填充
#由于缺失值较少，选取用中位数进行填充
data_test['TotalBsmtSF'].fillna(988, inplace = True)
data_test['GarageCars'].fillna(2,inplace=True)
data_test[cols].isnull().sum()
#计算缺失值所占比例，允许missing_data的数目，先定25%
#参考资料：https://discuss.analyticsvidhya.com/t/what-should-be-the-allowed-percentage-of-missing-values/2456

train_nas=data_train.isnull().sum().sort_values(ascending=False)
percent=(data_train.isnull().sum()/data_train.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([train_nas,percent],axis=1,keys=['train_nas','percent'])
missing_data.head(20)
#对预测数据SalePrice进行可视化
data_train['SalePrice'].describe()
from scipy import stats
from scipy.stats import norm,skew

sns.distplot(data_train['SalePrice'],fit=norm)
(mu,sigma)=norm.fit(data_train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
fig = plt.figure(figsize=(8,5))
res = stats.probplot(data_train['SalePrice'], plot=plt)
#正态概率图，用来检验数据是否服从正态分布，如果是一条直线的话，表示服从正态分布
plt.show()
print("Skewness:%f"%data_train['SalePrice'].skew())
print("Kurtosis:%f"%data_train['SalePrice'].kurt())
data_train.SalePrice=np.log1p(data_train.SalePrice)#取对数，使其符合正态分布
fig=plt.figure(figsize=(15,5))
plt.subplot(121)
y=data_train.SalePrice
sns.distplot(y,fit=norm)
plt.ylabel('Frequency')
plt.xlabel('SalePrice')
plt.title('SalePrice Distribution')

plt.subplot(122)
res=stats.probplot(data_train['SalePrice'],plot=plt)
#用线性模型进行预测：LinearRegression,Ridge,Lasso
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score #模型准确度评价指标 R2
from sklearn.cross_validation import train_test_split

cols=['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars']
x = data_train[cols].values
y = data_train['SalePrice'].values

X_train1,X_test1, y_train1, y_test1 = train_test_split(x,y, test_size=0.33, random_state=42)

Regs={
    'LinearRegression':LinearRegression(),
    'ridge':Ridge(),
    'Lasso':Lasso()
}
for Reg in Regs:
    try:
        Regs[Reg].fit(X_train1,y_train1)
        y_pred1=Regs[Reg].predict(X_test1)
        print(Reg+" cost:"+str(r2_score(y_test1,y_pred1)))
    except Exception as e:
        print(Reg+"Error")
        print(str(e)) 

'''
#也可以使用最简单的分步计算
line.fit(X_train1,y_train1)
ridge.fit(X_train1,y_train1)
lasso.fit(X_train1,y_train1)
#预测数据
line_y_pre=line.predict(x_test1)
ridge_y_pre=ridge.predict(x_test1)
lasso_y_pre=lasso.predict(x_test1)
'''
'''
line_score=r2_score(y_test,line_y_pre)
ridge_score=r2_score(y_test,ridge_y_pre)
lasso_score=r2_score(y_test,lasso_y_pre)
display(line_score,ridge_score,lasso_score)
'''
Reg1=Ridge()
Reg1.fit(X_train1,y_train1)
y_pred1=Reg1.predict(X_test1)
print(y_pred1)
y_pred_L=np.expm1(y_pred1)
y_pred_L.shape()
prediction1 = pd.DataFrame(y_pred_L, columns=['SalePrice'])
result1 = pd.concat([data_test['Id'], prediction1], axis=1)
#result1
from sklearn import preprocessing
from sklearn import linear_model,svm,gaussian_process
from sklearn.ensemble import RandomForestRegressor

import numpy as np
x_scaler = preprocessing.StandardScaler().fit_transform(x)
y_scaler = preprocessing.StandardScaler().fit_transform(y.reshape(-1,1))
X_train,X_test, y_train, y_test = train_test_split(x_scaler, y_scaler, test_size=0.33, random_state=42)
clfs={
    'svm':svm.SVR(),
    'RandomForestRegressor':RandomForestRegressor(n_estimators=400),
    'BayesianRidge':linear_model.BayesianRidge()
}
for clf in clfs:
    try:
        clfs[clf].fit(X_train,y_train)
        y_pred=clfs[clf].predict(X_test)
        print(clf+"cost:"+str(np.sum(y_pred-y_test)/len(y_pred)))
    except Exception as e:
        print(clf+"Error")
        print(str(e))
X_train,X_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)
clf=RandomForestRegressor(n_estimators=400)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(y_pred)
sum(abs(y_pred-y_test)/len(y_pred))
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

rfr = clf
x = data_test[cols].values
y_rfr_pred = rfr.predict(x)
print(y_rfr_pred)

print(y_rfr_pred.shape)
#预测得到的数据需要inverse一下
y_rfr_pred1=np.expm1(y_rfr_pred)

prediction = pd.DataFrame(y_rfr_pred1, columns=['SalePrice'])
result = pd.concat([ data_test['Id'], prediction], axis=1)
result.columns
result
result.to_csv('./Predictions.csv', index=False)