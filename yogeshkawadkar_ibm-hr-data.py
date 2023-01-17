# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
import warnings

warnings.filterwarnings("ignore")
data=pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

data.head()
data.drop('EmployeeNumber',axis=1,inplace=True)
for i in data.columns:

    if data[i].nunique()==1:

        data.drop(i,axis=1,inplace=True)
data.info()
for i in data.columns:

    if data[i].dtype=='O':

        print(i)

        print(data[i].unique())

        print()
for col in data.columns:

    if data[col].dtype=='O':

        un = data[col].unique()

        var=0

        for i in un:

            data[col].replace(i,var,inplace=True)

            var+=1
data.head()
cat=['Attrition','BusinessTravel','Department','Education','EducationField','EnvironmentSatisfaction',

     'Gender','JobInvolvement','JobLevel','JobRole','JobSatisfaction','MaritalStatus','OverTime','WorkLifeBalance',

     'StockOptionLevel','RelationshipSatisfaction','PerformanceRating'] 
cat_data= data[cat]

cat_data.head()
fig_dims = (20,15)

fig, ax = plt.subplots(figsize=fig_dims)



#mask=np.triu(np.ones_like(cat_data.corr(),dtype=bool))



cmap=sns.diverging_palette(h_neg=15,h_pos=240,as_cmap=True)

sns.heatmap(cat_data.corr(),center=0,cmap=cmap,linewidths=1,annot=True,fmt='.2f',ax=ax);
for a in cat_data.columns:

    sns.countplot(cat_data[a])

    plt.show()
quan=data.columns.to_list()

for i in cat:

    quan.remove(i)
quan_data=data[quan]

quan_data.head()
fig_dims = (20,15)

fig, ax = plt.subplots(figsize=fig_dims)



#mask=np.triu(np.ones_like(quan_data.corr(),dtype=bool))



cmap=sns.diverging_palette(h_neg=15,h_pos=240,as_cmap=True)

sns.heatmap(quan_data.corr(),center=0,cmap=cmap,linewidths=1,annot=True,fmt='.2f',ax=ax);
sns.pairplot(quan_data);
for a in quan_data.columns:

    sns.distplot(quan_data[a])

    plt.show()
def ecdf(data):

    n = len(data)

    x = np.sort(data)

    y = np.arange(1, n+1) / n

    return x, y
for a in quan_data.columns:

    x1,y1=ecdf(quan_data[a])

    x2,y2=ecdf(np.random.normal(np.mean(quan_data[a]),np.std(quan_data[a]),size=10000))

    plt.plot(x1,y1,marker='.',linestyle=None)

   

    plt.xlabel(a)

    plt.plot(x2,y2)

    plt.legend(['Real', 'Theory'])

    plt.show()
MonthlyIncome=quan_data['MonthlyIncome']

quan_data.drop('MonthlyIncome',axis=1,inplace=True)
from sklearn.preprocessing import StandardScaler

ss=StandardScaler().fit(quan_data)



quan_data_col_name=quan_data.columns

quan_data_col_name
quan_pre_data=pd.DataFrame(ss.transform(quan_data),columns=quan_data_col_name)
for a in quan_pre_data.columns:

    sns.distplot(quan_pre_data[a])

    plt.show()
quan_pre_data.head()
cat_data.head()
MonthlyIncome.head()
from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.linear_model import LassoCV

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn.feature_selection import RFE

from sklearn.metrics import mean_squared_error

from math import sqrt
y=MonthlyIncome

x=pd.concat([quan_pre_data,cat_data],axis=1)



X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=12)

X_train.iloc[:,:13]
lcv=LassoCV().fit(X_train,y_train)

lcv.alpha_
print(lcv.score(X_train,y_train))

print(lcv.score(X_test,y_test))
lcv_mask=lcv.coef_!=0

lcv_mask
rfr=RandomForestRegressor().fit(X_train,y_train)

gbr=GradientBoostingRegressor().fit(X_train,y_train)
print(rfr.score(X_train,y_train))

print(rfr.score(X_test,y_test))

print()

print(gbr.score(X_train,y_train))

print(gbr.score(X_test,y_test))
rfr.feature_importances_
importances_rf=pd.Series(rfr.feature_importances_,index=X_train.columns)

importances_rf_sort=importances_rf.sort_values()

importances_rf_sort.plot(kind='barh',figsize=(10,10));
importances_gbr=pd.Series(gbr.feature_importances_,index=X_train.columns)

importances_gbr_sort=importances_gbr.sort_values()

importances_gbr_sort.plot(kind='barh',figsize=(10,10),color='red');
importances_lcv=pd.Series(lcv.coef_,index=X_train.columns)

importances_lcv_sort=importances_lcv.sort_values()

importances_lcv_sort.plot(kind='barh',figsize=(10,10),color='green');
rfe_rfr=RFE(estimator=RandomForestRegressor(), n_features_to_select=15, step=3, verbose=1).fit(X_train,y_train)
print(rfe_rfr.score(X_train,y_train))

print(rfe_rfr.score(X_test,y_test))
sqrt(mean_squared_error(y_test,rfe_rfr.predict(X_test)))
rfr_mask=rfe_rfr.support_
rfe_gbr=RFE(estimator=GradientBoostingRegressor(), n_features_to_select=15, step=3, verbose=1)

rfe_gbr.fit(X_train,y_train)
print(rfe_gbr.score(X_train,y_train))

print(rfe_gbr.score(X_test,y_test))
gbr_mask=rfe_gbr.support_

votes=np.sum([lcv_mask,gbr_mask,rfr_mask],axis=0)

votes
mask = votes>=2

mask
mask_data=X_train.loc[:,mask]

mask_data.head()
from sklearn.decomposition import PCA

pca=PCA().fit(mask_data)
pca=PCA().fit(mask_data)



print(pca.explained_variance_ratio_)

print()

print(print(pca.explained_variance_ratio_.cumsum()))
len(pca.components_)
plt.plot(pca.explained_variance_ratio_)
del data,quan_data,cat_data,quan_pre_data
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint



param_dist={'loss':['ls', 'lad', 'huber', 'quantile'],

           'n_estimators':randint(100,200),

           'max_depth':randint(1,5)

           }

cv=GradientBoostingRegressor()



final_modelCV=RandomizedSearchCV(cv,param_dist,cv=10,verbose=True,n_jobs=-1)
final_modelCV.fit(mask_data,y_train)
final_modelCV.score(mask_data,y_train)
final_modelCV.best_params_
final_modelCV.score(mask_data,y_train)
final_modelCV.score(X_test.loc[:,mask],y_test)
sqrt(mean_squared_error(y_train,final_modelCV.predict(X_train.loc[:,mask])))