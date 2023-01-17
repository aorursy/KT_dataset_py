# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns

from scipy import stats

from datetime import datetime

import math

import matplotlib.pyplot as plt

%matplotlib inline
data1 = pd.read_csv('../input/train.csv',parse_dates=['timestamp'])

macro=pd.read_csv('../input/macro.csv',parse_dates=['timestamp'])

test =pd.read_csv('../input/test.csv',parse_dates=['timestamp'])

test_size = test.shape[0]



# merge two dataset into one



datatemp = pd.merge(data1, macro,how='left',on='timestamp',sort=True)

testtemp = pd.merge(test, macro,how='left',on='timestamp',sort=True)

#datatemp.dropna(subset=['id'],inplace=True)
class find_info(object):

    def __init__(self,df):

        """

        Please give the parameters: df is the data frame and id is the unique column.

        """

        self.df = df

    

    def func_desc(self,id):

        

        meanvar=np.round(self.df.mean(),1)

        minvar=self.df.min()

        maxvar=self.df.max()

        stdvar=np.round(self.df.std(),1)

        countvar=self.df.count()

        totalvar=self.df.id.count()

        missing_pct=np.round(100*(totalvar-countvar)/totalvar,1)

        statistics=pd.concat([meanvar, minvar, maxvar, stdvar, countvar,missing_pct], axis=1) 

        stat_df = pd.DataFrame(statistics).reset_index()

        orname=list(stat_df.columns.values)

        stat_df.rename(columns={orname[0]: 'variable', orname[1]: 'mean', 

                    orname[2]: 'min', orname[3]: 'max',

                    orname[4]: 'std',orname[5]: 'non_missing',

                    orname[6]: 'missing_pct'},                  

                    inplace = True)

        stat_df.sort_values(by='missing_pct',ascending=True,inplace=True)

        return stat_df



    def find_conlumns_by_type(self,type):

        objtype = type

        objdf = self.df.select_dtypes(include=objtype)

        return objdf
# Using create function to look into datatemp



find_information_temp = find_info(data1)



des_temp = find_information_temp.func_desc('id')

des_temp.head()
#sns.distplot(data1['price_doc'],bins=1000)
from datetime import datetime as dt



data2=pd.DataFrame()

data2['price'] = data1.price_doc

data2['year'] = data1.timestamp.dt.year

datatemp['year'] = data1.timestamp.dt.year
sns.boxplot(x=data2.year,y=data2.price,data=data2)
#We will delete the price greater than 25000000 for now 9.8% of the datatemp

for year in [2011,2012,2013,2014,2015]:

    if 2011:

        data=datatemp.drop(datatemp.price_doc>=20000000,axis=0)

    if 2012:

        data=datatemp.drop(datatemp.price_doc>=22000000,axis=0)

    if 2013:

        data=datatemp.drop(datatemp.price_doc>=22000000,axis=0)

    if 2014:

        data=datatemp.drop(datatemp.price_doc>=27000000,axis=0)

    if 2015:

        data=datatemp.drop(datatemp.price_doc>=30000000,axis=0)

    else:

        print("check data")

data.drop('year',axis=1,inplace=True)

data.shape
# Using distince value counts to find out stange value in data

# Found old_education_build_share and modern_education_share has value of 8,12345

# child_on_acc_pre_school has "#!"

obj_name_list = data.select_dtypes(include=['object']).columns

for distin in obj_name_list:

    print("Column name "+distin+" distinct values:")

    print(data[distin].value_counts())
# Then come back to clean up this column

import re

from collections import Counter



def clean_comma(line):

    c=Counter(list(line))

    if c[',']<=0:

        pass

    else:

        line_num = int(re.sub(',','',line))

        return line_num

    

columns2clean = ['old_education_build_share', 'modern_education_share','child_on_acc_pre_school']

for col in columns2clean:

    col_new = []

    for item in data[col]:

        if pd.isnull(item) or item.strip() == '#!':

            col_new.append(np.nan)

        else:

            col_new.append(clean_comma(str(item)))

    data[col] = col_new

    print(col_new[-10:])
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder
combined = pd.concat([datatemp.drop('price_doc',axis=1),testtemp],axis=0).reset_index()

print(combined.shape)

print(sum(combined.index.value_counts()>1))
num = combined.select_dtypes(include=['int16','int32','int64','float16','float32','float64'])

obj = combined.select_dtypes(include=['object'])

head = pd.DataFrame(pd.isnull(num).sum(),columns=['cnt'])

missing = head[head.cnt>0].index.values

missing[:20]
combined.drop(obj.columns,axis=1,inplace=True)
for name in missing:

    combined[name+"_missing"] = pd.isnull(combined[name])+0

    combined[name].fillna(combined[name].mean(),inplace=True)

obj.fillna(value='missing',inplace=True)

pd.isnull(obj).sum()    
le = LabelEncoder()

obj_2 = le.fit_transform(np.array(obj)[:,0])



for i in range(1, np.array(obj).shape[1]):

        enc_label = LabelEncoder()

        obj_2 = np.column_stack((obj_2, enc_label.fit_transform(np.array(obj)[:,i])))

train_categorical_values = obj_2.astype(float)
enc_onehot = OneHotEncoder()

train_cat_data = enc_onehot.fit_transform(train_categorical_values)



cols = [obj.columns[i] + '_' + str(j) for i in range(0,len(obj.columns)) for j in range(0,enc_onehot.n_values_[i]) ]

train_cat_data_df = pd.DataFrame(train_cat_data.toarray(),columns=cols)



data_cleaned = pd.concat([combined, train_cat_data_df],axis=1)

id_list=data_cleaned.id[-test_size:]
data_cleaned.drop(['index','id','timestamp'],axis=1,inplace=True)
train2model = data_cleaned[:-test_size]

test2model = data_cleaned[-test_size:]



# Try log the price

y = data1['price_doc']

#y = [np.log(y) for y in data1['price_doc']]

X = train2model
from sklearn.cross_validation import train_test_split

from sklearn.feature_selection import SelectKBest
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
def rmsle(preds, actual):

    labels = list(actual)

    preds = list(preds)

    terms_to_sum = [(math.log(labels[i] + 1) - math.log(max(0, preds[i]) + 1)) ** 2.0 for i, pred in enumerate(labels)]

    return 'rmsle', (sum(terms_to_sum) * (1.0 / len(preds))) ** 0.5
from sklearn.linear_model import LinearRegression

from sklearn.metrics import roc_curve, auc, r2_score
linear_model = LinearRegression()

linear = linear_model.fit(X_train, y_train)

pred_train=linear.predict(X_train)

pred_linear=linear.predict(X_test)



print(linear.score(X_train, y_train))

print(r2_score(y_test,pred_linear))
rmsle(pred_linear, y_test)
# Residual plot

train = plt.scatter(pred_train,(pred_train-y_train),c='r',alpha=0.5)

test = plt.scatter(pred_linear,(pred_linear-y_test),c='b',alpha=0.5)

plt.hlines(y=0,xmin=-10,xmax=10)

plt.legend((train,test),('Training Resisual','Testing Resisual'),loc='lower left')

plt.title('Residual Plot')
from sklearn.ensemble import GradientBoostingRegressor
r=GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=5,loss='ls')

fit_gd = r.fit(X_train, y_train)

pred_gd = fit_gd.predict(X_test)



print(fit_gd.score(X_train, y_train))

print(r2_score(y_test,pred_gd))
pred_gd_train = fit_gd.predict(X_train)

train = plt.scatter(pred_gd_train,(pred_gd_train-y_train),c='r',alpha=0.5)

test = plt.scatter(pred_gd,(pred_gd-y_test),c='b',alpha=0.5)

plt.hlines(y=0,xmin=-10,xmax=10)

plt.legend((train,test),('Training Resisual','Testing Resisual'),loc='lower left')

plt.title('Residual Plot for XG Boost')
# Price predicting

result_gd = fit_gd.predict(test2model)

result_gd_final = pd.DataFrame()

result_gd_final['id'] = id_list

result_gd_final['price'] = result_gd

result_gd_final.head()
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

rf_model = rf.fit(X_train, y_train)

pred_rf_train = rf_model.predict(X_train)

pred_rf_test = rf_model.predict(X_test)



print(rf_model.score(X_train, y_train))

print(r2_score(y_test,pred_rf_test))
train = plt.scatter(pred_rf_train,(pred_rf_train-y_train),c='r',alpha=0.5)

test = plt.scatter(pred_rf_test,(pred_rf_test-y_test),c='b',alpha=0.5)

plt.hlines(y=0,xmin=-10,xmax=10)

plt.legend((train,test),('Training Resisual','Testing Resisual'),loc='lower left')

plt.title('Residual Plot for Random Forest')
# Price predicting

result_rf = rf_model.predict(test2model)



result_rf_final = pd.DataFrame()

result_rf_final['id'] = id_list

result_rf_final['price'] = result_rf