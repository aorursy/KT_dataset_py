# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



from scipy import stats

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns





    







# Any results you write to the current directory are saved as output.
data1=pd.read_csv('../input/auditcsv/audit_risk.csv')

data2=pd.read_csv('../input/trailcsv/trial (1).csv')
data1.head()

data2.sample(7)
data1.dtypes
len(data1.columns)

data1.columns
len(data2.columns)

data2.columns
data1.isnull().sum()

data2.isnull().sum()
data2.shape
data1.shape
data1.describe()

data2.describe()

data1.nunique()
data2.nunique()
mod_data1=data1.fillna(data1["Money_Value"].mean())
mod_data1.isnull().sum()

mod_data2=data2.fillna(data2["Money_Value"].mean())
mod_data2.isnull().sum()
data1['Money_Value']

result1=pd.merge(mod_data1,mod_data2 , on=('Sector_score', 'PARA_A','PARA_B','TOTAL','numbers','Money_Value','Risk'))

result1.shape
result2=result1.drop_duplicates()

print(result2.shape)
result1.nunique()
result=pd.merge(mod_data1,mod_data2, left_index=True, right_index=True)
result.head()
result.shape
result.shape
result.isnull().sum()

result_data=mod_data1.merge(mod_data2)
result_data.shape
result_data.columns
result_data_sn1=result_data.drop_duplicates()

result_data_sn1.shape
result_data_sn1["LOCATION_ID"].nunique()
result_data_sn1.dtypes
result_data_sn1["LOCATION_ID"].unique()
final_result=result_data_sn1.copy()
final_result
final_result[["LOCATION_ID","Risk"]]=final_result[["LOCATION_ID","Risk"]].astype("category")
final_result.dtypes
final_result["LOCATION_ID"]=final_result["LOCATION_ID"].replace("LOHARU", 45)

final_result["LOCATION_ID"]=final_result["LOCATION_ID"].replace("NUH", 46)

final_result["LOCATION_ID"]=final_result["LOCATION_ID"].replace("SAFIDON", 47)

final_result["LOCATION_ID"].unique()



plt.boxplot(final_result["Risk_B"])
final_result.describe()
final_result[final_result['PARA_B']==1264.630000]
final_outlier=final_result[final_result['PARA_B']!=1264.630000]
final_outlier[final_outlier['PARA_B']==1264.630000]
plt.boxplot(final_outlier['PARA_B'])
final_outlier.head()
final_outlier[['Money_Value','Risk_D']].describe()
final_outlier[final_outlier["Money_Value"]==935.030000]
final_outlier.describe()
final_table=final_outlier[(final_outlier["Money_Value"]!=935.030000)&(final_outlier["Risk_D"]!=561.018000)&(final_outlier["Inherent_Risk"]!=622.838000)&(final_outlier["TOTAL"]!=191.360000)]
final_table.describe()
plt.boxplot(final_table["Inherent_Risk"])
final_table.shape
plt.boxplot(final_table["TOTAL"])
final_result.columns
y_final_reg1=final_table["Audit_Risk"]

x_final_reg1=final_table.drop(["Audit_Risk"],axis=1)

x_final_reg1.shape

y_final_reg1.shape

#np.reshape((579,1))

y_final_reg1.shape

from sklearn.preprocessing import MinMaxScaler, StandardScaler
mm_final=MinMaxScaler()

sc_final=StandardScaler()

y_final_reg=final_table["Audit_Risk"]

y_final_clas=final_table["Risk"]

final_1=final_table.drop(["Audit_Risk","Risk"],axis=1)

cols=['Sector_score', 'LOCATION_ID','PARA_A', 'Score_A', 'Risk_A', 'PARA_B',

       'Score_B', 'Risk_B', 'TOTAL', 'numbers', 'Score_B.1', 'Risk_C',

       'Money_Value', 'Score_MV', 'Risk_D', 'District_Loss', 'PROB', 'RiSk_E',

       'History', 'Prob', 'Risk_F', 'Score', 'Inherent_Risk', 'CONTROL_RISK',

       'Detection_Risk', 'SCORE_A', 'SCORE_B', 'Marks',

       'MONEY_Marks', 'District', 'Loss', 'LOSS_SCORE', 'History_score']

mm_final1=final_1.copy()

sc_final1=final_1.copy()

mm_final1[cols]=mm_final.fit_transform(mm_final1[cols])

sc_final1[cols]=sc_final.fit_transform(sc_final1[cols])

final_1.sample(5)
final_1[final_1["LOSS_SCORE"]+final_1["History"]!=final_1["History_score"]].sample(10)
import seaborn as sns

sns.set(style="white")

correln=final_1.corr()

f,ax = plt.subplots(figsize=(20,15))

cmap = sns.diverging_palette(220,10, as_cmap=True)

sns.heatmap(correln,cmap=cmap, vmax=.3,linewidths=.5, cbar_kws={"shrink": .7})

correln.describe()
y_final_reg.shape
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(sc_final1,y_final_reg,test_size=0.14,random_state=0)

print("no of x train values : {} no of x test values :  {} no of y train values:    {}  ".format(x_train.shape[0],x_test.shape[0],y_train.shape[0]))
from sklearn.linear_model import LinearRegression

lreg=LinearRegression()

lreg.fit(x_train,y_train)

print("coefficient of determination:",lreg.score(x_train,y_train))

#print(lreg.score(x_test,y_test))

lreg_pred= lreg.predict(x_train)

lreg_pred1=lreg.predict(x_test)

#print(lreg_pred)

from sklearn.metrics import mean_squared_error

from math import sqrt

tr_mse=mean_squared_error(y_train,lreg_pred)

tr_rmse=sqrt(tr_mse)

ts_mse=mean_squared_error(y_test,lreg_pred1)

ts_rmse=sqrt(ts_mse)

print("mean squared error for train: %4f" %tr_mse)

print("root mean squred error for train: %5f" %tr_rmse)

print("mean squared error for test: %6f"%ts_mse)

print("root mean squared error for test: %6f"%ts_rmse)



from sklearn.linear_model import Ridge

x_range=[0.01,0.1,0.5,1,10]

train_list=[]

test_list=[]

for alpha in x_range:

    ridge=Ridge(alpha)

ridge.fit(x_train,y_train)

train_list.append(ridge.score(x_train,y_train))

test_list.append(ridge.score(x_test,y_test))

print(train_list) 

print(test_list)