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
folder = '/kaggle/input/house-prices-advanced-regression-techniques/'

train = pd.read_csv(f'{folder}train.csv')

test = pd.read_csv(f'{folder}test.csv')   
for col in train.columns:

    if train[col].dtype == 'object':

        labels, uniques = pd.factorize(train[col], sort=True)

        train[col] = pd.Series(labels, dtype = 'category')

        

for col in test.columns:

    if test[col].dtype == 'object':

        labels, uniques = pd.factorize(test[col], sort=True)

        test[col] = pd.Series(labels, dtype = 'category')



train = train.fillna(0)

test = test.fillna(0)
 
#Feature selection using Pearson Correlation

cor = train.corr()

#sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

#plt.show()



# columns selected based on correlation

cor_cols = list(cor['SalePrice'][(cor['SalePrice'] > .50) | (cor['SalePrice'] < -0.50)].index)







from sklearn.model_selection import train_test_split

 



import xgboost as xgb

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor  



regr = xgb.XGBRegressor(  learning_rate=0.01,n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)



#regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)



#for add_col in add_cols:

add_cols = list(train.columns)

for col in cor_cols:

    add_cols.remove(col)

add_cols.remove('Id')     

 

y = 'SalePrice'

cor_cols.remove(y)



dic = {}



x1, x2, y1, y2 = train_test_split(train[cor_cols], train[y], random_state = 42)



regr.fit(x1, y1)



preds = regr.predict(x2)



baseline = np.sqrt(mean_squared_error((y2), preds))



# check which continuous columns increases prediction above the cor_cols

for add_col in add_cols:

 



        

        

    df_cols = cor_cols + [add_col] + [y]  

    df = train[df_cols]

    

    if str(train[add_col].dtype) == 'category':      

        # Get one hot encoding of column

        one_hot = pd.get_dummies(df[add_col])

        # Drop column B as it is now encoded

        df = df.drop(add_col,axis = 1)

        # Join the encoded df

        df = df.join(one_hot)

        

        train_cols = cor_cols + list(one_hot.columns)

    else:

        train_cols = cor_cols + [add_col]

    

    x1, x2, y1, y2 = train_test_split(df[train_cols], df[y], random_state = 42)

    #try: 

    regr.fit(x1, y1)



    preds = regr.predict(x2)



    rmse = np.sqrt(mean_squared_error((y2), preds))

    dic[add_col] =   ( baseline - rmse)

   # except:

      #  print()





 
potential_cols = []

for key in dic:

    if dic[key] > ((baseline / 100) * 5):

        potential_cols.append(key)



# amount of features         

len(potential_cols + cor_cols)

 
from heapq import nlargest

regr_cols = nlargest(4, dic, key=dic.get)
from heapq import nlargest

from sklearn.preprocessing import MinMaxScaler

regr_cols = nlargest(5, dic, key=dic.get)



df_cols = set(cor_cols + regr_cols + ['labels'])



train['labels'] = 'train'

test['labels'] = 'test'

df = train[df_cols]

df = pd.concat([df[df_cols],test[df_cols]])



labels = df['labels']

df_cols = set(cor_cols + regr_cols)

df = df[df_cols]



 # Import libraries and download example data

from sklearn.preprocessing import StandardScaler, OneHotEncoder

 



# Define which columns should be encoded vs scaled

columns_to_encode = []

for col in df.columns:

    if str(df[col].dtype) == 'category':    

        columns_to_encode.append(col)

columns_to_encode       

 

columns_to_scale = []

for col in df.columns:

    if str(df[col].dtype) != 'category':    

        columns_to_scale.append(col)





# Instantiate encoder/scaler

scaler = StandardScaler()



ohe    = OneHotEncoder(sparse=False, categories='auto')



# Scale and Encode Separate Columns

scaled_columns  = scaler.fit_transform(df[columns_to_scale]) 

encoded_columns =    ohe.fit_transform(df[columns_to_encode])



# Concatenate (Column-Bind) Processed Columns Back Together

df = np.concatenate([scaled_columns, encoded_columns], axis=1)

df = pd.DataFrame(df)

df['labels'] = labels.values       



#scaler = MinMaxScaler()

 

#df = pd.DataFrame(scaler.fit_transform(df))      

#df['labels'] = labels

train_trans = df[df['labels'] == 'train'].drop(['labels'], axis = 1)

train_trans['SalePrice'] = np.log( train['SalePrice'])

test_trans = df[df['labels'] == 'test'].drop(['labels'], axis = 1)



#regr = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,

                #max_depth = 5, n_estimators = 200)        



#x1, x2, y1, y2 = train_test_split(df , y)

 

#regr.fit(x1, y1)



#preds = regr.predict(x2)

 



#rmse = np.sqrt(mean_squared_error((y2), preds))



#print( 'regr+cor model: ', rmse, 'versus baseline: ', baseline)

 
regr = xgb.XGBRegressor(  learning_rate=0.01,n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)



x1, x2, y1, y2 = train_test_split(train_trans.iloc[:,:-1], train_trans.iloc[:,-1:], random_state = 42)



regr.fit(x1, y1)



preds = regr.predict(x2)

preds =  np.exp(preds)

baseline = np.sqrt(mean_squared_error((y2), preds))

baseline
regr.fit(train_trans.iloc[:,:-1], train_trans.iloc[:,-1:])



preds = regr.predict(test_trans)



test['SalePrice'] = np.exp( preds)



sub = test[['Id', 'SalePrice']]

 

sub.to_csv('submission-houses.csv',index=False)

 
 