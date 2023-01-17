import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

pd.options.display.max_columns=500

pd.options.display.max_rows=100
train=pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')

test=pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')

sub=pd.read_csv('/kaggle/input/home-data-for-ml-course/sample_submission.csv')
display(train, test) # display columns of train and test
train.corr()['SalePrice'].sort_values(ascending=False)[:7]
fig=plt.figure(figsize=(15,7))



ax1=fig.add_subplot(221)

ax2=fig.add_subplot(222)

ax3=fig.add_subplot(223)

ax4=fig.add_subplot(224)



sns.scatterplot(train['OverallQual'], train['SalePrice'], ax=ax1)

sns.scatterplot(train['GrLivArea'], train['SalePrice'], hue=train['ExterQual'],ax=ax2)

sns.scatterplot(train['GarageCars'], train['SalePrice'], ax=ax3)

sns.scatterplot(train['GarageArea'], train['SalePrice'], ax=ax4)



plt.show()
train=train.loc[(train['SalePrice']>300000) | (train['GrLivArea']<4000)] # outlier delete
train.corr()['SalePrice'].sort_values(ascending=False)
alldata=pd.concat([train,test],sort=False)

alldata=alldata.drop(['Id','SalePrice'],axis=1)
alldata_2=pd.get_dummies(alldata)

alldata_2=alldata_2.fillna(-1) # missing value replacement with -1
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

alldata_3=ss.fit_transform(alldata_2)
train_2=alldata_3[:len(train)] #data set split

test_2=alldata_3[len(train):]
from sklearn.linear_model import Ridge, RidgeCV

from sklearn.model_selection import cross_val_score



ridgecv=RidgeCV(alphas=[1]) # 기본값 규제를 주지 않았다.

np.sqrt(-cross_val_score(ridgecv , train_2, train['SalePrice'], n_jobs=-1, cv=10, scoring='neg_mean_squared_error').mean())
rmse=[]

alphas=[1,5, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 700, 750, 900, 1000] 

for i in alphas :

    ridgecv=RidgeCV(alphas=[i])

    result=np.sqrt(-cross_val_score(ridgecv , train_2, train['SalePrice'], n_jobs=-1, cv=10, scoring='neg_mean_squared_error').mean())

    rmse.append(result)

    print(result)
plt.figure(figsize=(8,3))

sns.lineplot(x=alphas, y=rmse)

plt.suptitle('RMSE by alpha of ridge regression')

plt.xlabel('alpha')

plt.ylabel('RMSE')

plt.show()
ridgecv=RidgeCV(alphas=[1]) # 기본값 규제를 주지 않았다.

ridgecv.fit(train_2,np.log(train['SalePrice']))



ridgecv_2=RidgeCV(alphas=[100]) # 기본값 규제를 주지 않았다.

ridgecv_2.fit(train_2,np.log(train['SalePrice']))





fig=plt.figure(figsize=(25,70))



ax1=fig.add_subplot(121)

ax2=fig.add_subplot(122)



#x=alldata_2.columns.tolist(),

chart_1=sns.barplot(y=alldata_2.columns.tolist(), x=ridgecv.coef_.tolist(), ax=ax1)

chart_2=sns.barplot(y=alldata_2.columns.tolist(), x=ridgecv_2.coef_.tolist(), ax=ax2)



ax1.set_xlim(-0.02, 0.05)

ax2.set_xlim(-0.02, 0.05)



ax1.set_title('coefficient with alpha 1')

ax2.set_title('coefficient with alpha 100')





plt.show()

fig=plt.figure(figsize=(15,4))



ax1=fig.add_subplot(121)

ax2=fig.add_subplot(122)



sns.distplot(train['SalePrice'],ax=ax1)

sns.distplot(np.log(train['SalePrice']),ax=ax2)



bdict = {'facecolor' : 'r', 'alpha' : 0.5, 'boxstyle' : 'rarrow', 'linewidth' : 2}





fig.suptitle('Target distribution from unlogged to logged', bbox=bdict, color='black')

ax1.set_title('Un-logged')

ax2.set_title('Logged')



plt.show()
ridgecv=RidgeCV(alphas=np.linspace(1,1000,1000)) #np.linspace(1,1000,1000)

ridgecv.fit(train_2,np.log(train['SalePrice']))

result=ridgecv.predict(test_2)

sub['SalePrice']=np.exp(result)

sub.to_csv('result.csv',index=False)