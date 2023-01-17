import pandas as pd 

import numpy as np

import gc

from scipy import sparse
from datetime import datetime as dt

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib

#matplotlib.use('Qt5Agg')

sns.set_style('whitegrid')



%matplotlib inline
def add_feature(df):

    df['year'] = pd.DatetimeIndex(df['date']).year

    df['month'] = pd.DatetimeIndex(df['date']).month

#    df['week'] = pd.DatetimeIndex(df['date']).week

    df['day'] = pd.DatetimeIndex(df['date']).day

    

    df['year'] = df['year'].astype(np.uint16)

    df['month'] = df['month'].astype(np.uint8)

#    df['week'] = df['week'].astype(np.uint8)

    df['day'] = df['day'].astype(np.uint8)

    return df
coltype = {

    'date':object,

    'onpromotion':bool

}

ttl_chunks = []

for chunk in pd.read_csv('../input/train.csv',dtype=coltype,chunksize=1000000):

    chunk['onpromotion'].fillna(0,inplace=True)

    chunk['onpromotion'] = chunk['onpromotion'].astype(np.uint8)

    chunk = add_feature(chunk)

    ttl_chunks.append(chunk)
ttl_sales = pd.DataFrame([],columns=['year','month','unit_sales'])

for chunk in ttl_chunks:

    sales = chunk.groupby(['year','month'])['unit_sales'].sum().reset_index()

    ttl_sales = pd.concat([ttl_sales,sales],axis=0,ignore_index=True)

ttl_sales = ttl_sales.groupby(['year','month'])['unit_sales'].sum().reset_index()

ttl_sales['next_year'] = ttl_sales['unit_sales'].shift(-12)

ttl_sales['yoy'] = ttl_sales['next_year']/ttl_sales['unit_sales']-1

weight = ttl_sales.groupby('year').mean().reset_index()[['year','yoy']]

weight = weight[weight['year'] != 2017]

weight['year'] = weight['year']+1

weight
num_train_chunks = len(ttl_chunks)

chunks=ttl_chunks[76:86]
#del ttl_chunks;gc.collect();
print("Train dataset chunk numbers is "+str(len(chunks)))
test_chunks = []

for test_chunk in pd.read_csv("../input/test.csv",dtype=coltype,chunksize=1000000):

    test_chunk = add_feature(test_chunk)

    test_chunk['onpromotion'].fillna(0,inplace=True) 

    test_chunk['onpromotion'] = test_chunk['onpromotion'].astype(np.uint8)

    test_chunk['unit_sales'] = np.nan

    test_chunks.append(test_chunk)



print("Testing dataset chunk numbers is "+str(len(test_chunks)))

test = pd.DataFrame([],columns=test_chunks[0].columns)

for df in test_chunks:

    test = pd.concat([test,df],axis=0)

num_test = len([test])

test_id = test.id

len_test = test.shape[0]

print("Test dataset info: " + str(test.shape))
# Make test shorter record

#test = test.iloc[:10000]
all_chunks = chunks+[test]

del chunk; gc.collect()
rows_test = test.shape[0]

rows_test
def add_more_features(chunks,df_2_add,add_key=None):

    new_chunks = []

    for chunk in chunks:

        new_chunk = chunk.merge(df_2_add,how='left',on=add_key)

        new_chunks.append(new_chunk)

    del new_chunk,df_2_add;gc.collect()

    return new_chunks
stores = pd.read_csv('../input/stores.csv',usecols=[0,4])

all_chunks = add_more_features(all_chunks,stores,'store_nbr')

del stores;gc.collect();
items = pd.read_csv('../input/items.csv',usecols=[0,3])

all_chunks = add_more_features(all_chunks,items,'item_nbr')

del items;gc.collect();
oil = pd.read_csv('../input/oil.csv',usecols=[0,1])

all_chunks = add_more_features(all_chunks,oil,'date')

del oil;gc.collect();
for chunk in all_chunks:

    chunk['dcoilwtico'][0] = np.mean(chunk['dcoilwtico'])

    chunk['dcoilwtico'].fillna(method='ffill',inplace=True) 

    chunk['dcoilwtico'] = chunk['dcoilwtico'].astype(np.float32)
holidays = pd.read_csv('../input/holidays_events.csv',usecols=[0,1,5])

holidays['type'] = holidays['type'].apply(lambda x:0 if x=='Work Day' else 1)

holidays['transferred'] = holidays['transferred'].apply(lambda x:1 if x=='True' else 0)

all_chunks = add_more_features(all_chunks,holidays,'date')



del holidays;gc.collect();
for chunk in all_chunks:

    chunk['type'].fillna(0,inplace=True) 

    chunk['type'] = chunk['type'].astype(np.uint8)

    chunk['transferred'].fillna(0,inplace=True) 

    chunk['transferred'] = chunk['transferred'].astype(np.uint8)
from sklearn.preprocessing import OneHotEncoder

from scipy import sparse
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer
chunks = all_chunks[:-1]

test = all_chunks[-1]

comb = pd.DataFrame([],columns=all_chunks[0].columns)

for i in range(len(chunks)):

    if i == 0:

        comb = chunks[i]

    else:

        comb = pd.concat([comb,chunks[i]],axis=0)

#del all_chunks

#gc.collect()
# train dataset unique value in very column combine with test dataset unique value of very column

# Append all combination of above combined column unique values difference before label endcorder

# Choose chunks 0 for now for quick work



comb_fornow = comb[['year','month','day','store_nbr','item_nbr','cluster','onpromotion','perishable','transferred','type','dcoilwtico']]

y = comb.unit_sales.values

cat_cols=['year','month','day','store_nbr','item_nbr','cluster']

num_cols=['onpromotion','perishable','type','transferred','dcoilwtico']

del comb;gc.collect();
cat_train = comb_fornow[cat_cols]

cat_test = test[cat_cols]

num_train = comb_fornow[num_cols]

num_test = test[num_cols]


def TrainTestSameDimension(cat_train,cat_test):

    # train_fill_values is the value fillna in created dataset. train_fill_values IS THE last row of 

    # Categroy data set

    train_fill_values = dict(zip(cat_train.columns,cat_train.iloc[-1].values))

    test_fill_values = dict(zip(cat_test.columns,cat_test.iloc[-1].values))

    temp_train = []

    temp_test = []

    if cat_train.shape[1] != cat_test.shape[1]:

        print("Please remove target variable!")

    else:

        for col in cat_train.columns:

            globals() [col+'_train'] = set(cat_test[col])-set(cat_train[col])

            globals() [col+'_test'] = set(cat_train[col])-set(cat_test[col])

            temp_train.append(pd.Series(list(globals() [col+'_train']),name=col))

            temp_test.append(pd.Series(list(globals() [col+'_test']),name=col))

        train_df = pd.DataFrame(temp_train).T

        train_df.fillna(value=train_fill_values,inplace=True)

        test_df = pd.DataFrame(temp_test).T

        test_df.fillna(value=test_fill_values,inplace=True)

    return {'train':train_df,'test':test_df}
#del comb_fornow ;gc.collect();
train_added2cat_train=TrainTestSameDimension(cat_train[cat_cols],cat_test[cat_cols])['train']

test_added2cat_test=TrainTestSameDimension(cat_train[cat_cols],cat_test[cat_cols])['test']
train_final = pd.concat([comb_fornow,train_added2cat_train],axis=0)

train_final.fillna(0,inplace=True)

cat_test_concat = pd.concat([test,test_added2cat_test],axis=0)

#del train_added2cat_train ;gc.collect();

#del test_added2cat_test ;gc.collect();
# fill number columns' na with 0

#train_final = cat_train_concat.merge(num_train,how='left',left_index=True,right_index=True)

#train_final.fillna(0,inplace=True)

#del cat_train_concat ;gc.collect();

del num_train ;gc.collect();

print("Wait for above step finished!")
test_final = pd.concat([comb_fornow,train_added2cat_train],axis=0)

test_final.fillna(0,inplace=True)

#del cat_test_concat ;gc.collect();

del num_test ;gc.collect();
X_cat = train_final[cat_cols]

X_num = train_final[num_cols]
test_cat = test_final[cat_cols]

test_num = test_final[num_cols]
def get_sparse(X_cat,X_num=None):

    ohe_list = []

    for cat in X_cat.columns:

        ohe = OneHotEncoder(sparse=True)

        ohe_list.append(ohe.fit_transform(X_cat[cat].values.reshape(len(X_cat),1)))

    

    for i in range(len(ohe_list)):

        if i ==0:

            temp = ohe_list[0]

        else:

            X_cat_new = sparse.hstack([temp,ohe_list[i]])

    if sparse.issparse(X_num):

        dct = X_cat_new

    else:

        dct = sparse.hstack([X_cat_new.astype(float),X_num])

    return dct
#def generate_datasets(dataset,cat_cols,num_cols):

##    train_chunk = chunk[:(len(chunk)-num_test)]

##    comb = pd.concat([train_chunk,test],axis=0)

#    comb = dataset

#    comb.fillna(0,inplace=True)

#    y=comb.unit_sales.values

#    X_cat=comb[cat_cols]

#    X_num=comb[num_cols]

#    

#    from sklearn.preprocessing import OneHotEncoder

#    from scipy import sparse

#    

#    X_sparse = get_sparse(X_cat,X_num)['sparse']

##    row_train = train_chunks.shape[0]

##    row_test = test.shape[0]

#    #return {'sparse':X_sparse,'y':y,'id':id,'row_train':row_train,'row_test':row_test}

#    return {'sparse':X_sparse}
X_sparse = get_sparse(X_cat,X_num)

test = get_sparse(test_cat,test_num)

y=np.array(y.tolist()+[0]*(X_sparse.shape[0]-len(y)))

print(X_sparse.shape)

print(len(y))
#import pickle

#outfile_name = "pickle_66-75.pkl"

#d = {'train':X_sparse,'test':test,'y':y}

#with open(outfile_name, 'wb')  as fn:

#    pickle.dump(d,fn,protocol=pickle.HIGHEST_PROTOCOL)

#

#with open("pickle1.pkl", 'rb')  as fn:

#    d = pickle.load(fn)

#

#X=d['train']

#test=d['test']

#y=d['y']
X=X_sparse
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

lg = LinearRegression()

pred = lg.fit(X_train,y_train).predict(X_test)

r2 = lg.score(X_test,y_test)

intercept = lg.intercept_

coef_list = lg.coef_

print(len(coef_list))
r2
#from sklearn.tree import DecisionTreeRegressor

#dt = DecisionTreeRegressor()

#pred = dt.fit(X_train,y_train).predict(X_test)

#dt_r2 = dt.score(X_test,y_test)
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, y_train)

dvalid = xgb.DMatrix(X_test, y_test)
params = {"objective": "reg:linear",

          "booster" : "gbtree",

          "eta": 0.3,

          "max_depth": 10,

          "subsample": 0.9,

          "colsample_bytree": 0.7,

          "silent": 1,

          "seed": 1301

          }

num_boost_round = 30

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
def rmspe(y, yhat):

    return np.sqrt(np.mean((yhat/y-1) ** 2))



def rmspe_xg(yhat, y):

    y = np.expm1(y.get_label())

    yhat = np.expm1(yhat)

    return "rmspe", rmspe(y,yhat)
#gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,early_stopping_rounds=20, feval=rmspe_xg, verbose_eval=True)

gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,early_stopping_rounds=20, verbose_eval=True)

print("Validating")

yhat = gbm.predict(xgb.DMatrix(X_test))

error = rmspe(y_test, yhat)

print('RMSPE: {:.6f}'.format(error))

from sklearn.metrics import r2_score

r2=r2_score(y_test, yhat)

r2
dtest = xgb.DMatrix(test)
pred_xgb = gbm.predict(dtest)




#result3 = pd.DataFrame({"id": test_id, 'unit_sales': pred_xgb[:len_test]})
result1 = pd.DataFrame({"id": test_id, 'unit_sales': pred_xgb[:len_test]})
#result2 = pd.DataFrame({"id": test_id, 'unit_sales': pred_xgb[:len_test]})
result1.to_csv('submittion_1.csv',index=False)
result