# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import(

    LinearRegression,

    Ridge,

    Lasso



)





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



train['WhatIsData'] = 'Train'

test['WhatIsData'] = 'Test'

test['SalePrise'] = 9999999999

alldata = pd.concat([train,test],axis = 0).reset_index(drop = True)

print('The size of train is : ' + str(train.shape))

print('The size of test is : ' + str(test.shape))
train.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False)
test.isnull().sum()[test.isnull().sum()>0].sort_values(ascending=False)
na_col_list = alldata.isnull().sum()[alldata.isnull().sum()>0].index.tolist() # 欠損を含むカラムをリスト化

alldata[na_col_list].dtypes.sort_values() #データ型
na_float_cols = alldata[na_col_list].dtypes[alldata[na_col_list].dtypes=='float64'].index.tolist() #float64

na_obj_cols = alldata[na_col_list].dtypes[alldata[na_col_list].dtypes=='object'].index.tolist() #object



for na_float_col in na_float_cols:

    alldata.loc[alldata[na_float_col].isnull(),na_float_col] = 0.0

    

for na_obj_col in na_obj_cols:

    alldata.loc[alldata[na_obj_col].isnull(),na_obj_col] = 'NA'    
alldata.isnull().sum()[alldata.isnull().sum()>0].sort_values(ascending=False)
cat_cols = alldata.dtypes[alldata.dtypes=='object'].index.tolist()



num_cols = alldata.dtypes[alldata.dtypes!='object'].index.tolist()



other_cols = ['Id','WhatIsData']



cat_cols.remove('WhatIsData')

num_cols.remove('Id') 



alldata_cat = pd.get_dummies(alldata[cat_cols])



all_data = pd.concat([alldata[other_cols],alldata[num_cols],alldata_cat],axis=1)
sns.distplot(train['SalePrice'])
sns.distplot(np.log(train['SalePrice']))
train_ = all_data[all_data['WhatIsData']=='Train'].drop(['WhatIsData','Id'], axis=1).reset_index(drop=True)

test_ = all_data[all_data['WhatIsData']=='Test'].drop(['WhatIsData','SalePrice'], axis=1).reset_index(drop=True)



train_x = train_.drop('SalePrice',axis=1)

train_y = np.log(train_['SalePrice'])



test_id = test_['Id']

test_data = test_.drop('Id',axis=1)
scaler = StandardScaler()  #スケーリング

param_grid = [0.001, 0.01, 0.1, 1.0, 10.0,100.0,1000.0] #パラメータグリッド

cnt = 0

for alpha in param_grid:

    ls = Lasso(alpha=alpha) #Lasso回帰モデル

    pipeline = make_pipeline(scaler, ls) #パイプライン生成

    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=0)

    pipeline.fit(X_train,y_train)

    train_rmse = np.sqrt(mean_squared_error(y_train, pipeline.predict(X_train)))

    test_rmse = np.sqrt(mean_squared_error(y_test, pipeline.predict(X_test)))

    if cnt == 0:

        best_score = test_rmse

        best_estimator = pipeline

        best_param = alpha

    elif best_score > test_rmse:

        best_score = test_rmse

        best_estimator = pipeline

        best_param = alpha

    else:

        pass

    cnt = cnt + 1

    

print('alpha : ' + str(best_param))

print('test score is : ' +str(best_score))
plt.subplots_adjust(wspace=0.4)

plt.subplot(121)

plt.scatter(np.exp(y_train),np.exp(best_estimator.predict(X_train)))

plt.subplot(122)

plt.scatter(np.exp(y_test),np.exp(best_estimator.predict(X_test)))
ls = Lasso(alpha = 0.01)

pipeline = make_pipeline(scaler, ls)

pipeline.fit(train_x,train_y)

test_SalePrice = pd.DataFrame(np.exp(pipeline.predict(test_data)),columns=['SalePrice'])

test_Id = pd.DataFrame(test_id,columns=['Id'])

pd.concat([test_Id, test_SalePrice],axis=1).to_csv('submission_lasso.csv',index=False)