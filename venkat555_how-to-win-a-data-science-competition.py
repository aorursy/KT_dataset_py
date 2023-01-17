# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import gc

import matplotlib.pyplot as plt

%matplotlib inline 

import pickle



pd.set_option('display.max_rows', 600)

pd.set_option('display.max_columns', 50)



import lightgbm as lgb

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from tqdm import tqdm_notebook



from itertools import product

import seaborn as sns

!pip install bayesian-optimization

import shap

def downcast_dtypes(df):

    '''

        Changes column types in the dataframe: 

                

                `float64` type to `float32`

                `int64`   type to `int32`

    '''

    

    # Select columns to downcast

    float_cols = [c for c in df if df[c].dtype == "float64"]

    int_cols =   [c for c in df if df[c].dtype == "int64"]

    

    # Downcast

    df[float_cols] = df[float_cols].astype(np.float32)

    df[int_cols]   = df[int_cols].astype(np.int32)

    

    return df



def clip40(x):

    if x>40:

        return 40

    elif x<0:

        return 0

    else:

        return x

    

def clip20(x):

    if x>20:

        return 20

    elif x<0:

        return 0

    else:

        return x

    

def remove_from_list( master_list , to_remove):

    for elm in to_remove:

        if elm in master_list: master_list.remove(elm)

    

    return master_list



from sklearn.model_selection import learning_curve



def over_underfit_plot(model, X, y):

    gc.collect()

    plt.figure()



    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1)

    train_scores_mean = np.mean(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    del train_scores,test_scores

    plt.grid()



    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")

    plt.legend(loc="best")

    plt.yticks(sorted(set(np.append(train_scores_mean, test_scores_mean))))

    

    





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import multiprocessing



multiprocessing.cpu_count()
DATA_FOLDER = '/kaggle/input/competitive-data-science-predict-future-sales/'

sales    = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv'))

items           = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))

item_categories = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))

test           = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv'))

shops = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))

submissions           = pd.read_csv(os.path.join(DATA_FOLDER, 'sample_submission.csv'))

test_block = sales['date_block_num'].iloc[-1] + 1

test['date_block_num'] = test_block
item_categories.head(2).T
sales.head(3)
sales.shape
items.head(1)
submissions.head(1)
shops.head(1)
test.head(1)
# convert date into year , month , day to aid in feature generation later 

sales['time'] = pd.to_datetime(sales['date'], format='%d.%m.%Y')   

sales['year'] = sales.time.dt.year

sales['month'] = sales.time.dt.month

sales['day'] = sales.time.dt.day
# printing some stats about data 

# number of unique shop , items , shop-item combos in train/test 

print('Number of unique train values in column "shop_id" of the dataframe : ',sales['shop_id'].nunique())

print('Number of unique train values in column "item_id" of the dataframe : ',sales['item_id'].nunique())



print('Number of unique test values in column "shop_id" of the dataframe : ',test['shop_id'].nunique())

print('Number of unique test values in column "item_id" of the dataframe : ',test['item_id'].nunique())

item_categories.head(1).T
from sklearn.preprocessing import LabelEncoder

item_categories['split'] = item_categories['item_category_name'].str.split('-')

item_categories['type'] = item_categories['split'].map(lambda x: x[0].strip())

item_categories['type_code'] = LabelEncoder().fit_transform(item_categories['type'])

item_categories['subtype'] = item_categories['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

item_categories['subtype_code'] = LabelEncoder().fit_transform(item_categories['subtype'])

item_categories = item_categories[['item_category_id','type_code', 'subtype_code' , 'item_category_name']]
import seaborn as sns

# plot shops with num items sold

gb = sales.filter(['shop_id' , 'item_id' ] , axis=1 )

gb= gb.groupby(['shop_id'] , as_index=False).agg({'item_id' : { 'num_unique_items': pd.Series.nunique}})

gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]



# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(50, 15))

# Plot the total crashes

sns.set_color_codes("pastel" )

sns.barplot(x="shop_id", y="num_unique_items", data=gb,

            label="Total", color="b")
# plot shops with total num sold

# plot shops with num items sold

gb = sales.filter(['shop_id' , 'item_cnt_day'] , axis=1 )

gb= gb.groupby(['shop_id'] , as_index=False).agg({'item_cnt_day' : { 'total_num_items': 'sum'}})

gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]



# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(50, 15))

# Plot the total crashes

sns.set_color_codes("pastel")

sns.barplot(x="shop_id", y="total_num_items", data=gb,

            label="Total", color="b")
# to get around the memory constraint , i have decided to get rid of shop's selling less than 25k 

gb_threshold = gb[gb['total_num_items']>=50000]

# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(10, 5))

# Plot the total crashes

sns.set_color_codes("pastel")

sns.barplot(x="shop_id", y="total_num_items", data=gb_threshold,

            label="Total", color="b")
# filter of sales to only shops satisfying the threshold in terms of total num items sold 

#sales = sales[sales['shop_id'].isin(gb_threshold['shop_id'])]

sales.shape
print(' printing max of sales dateframe')

sales.max()
print(' printing min of sales dateframe')

sales.min()
# check how many rows are with negative price 

zero_df = sales[sales['item_cnt_day']<0]

print( 'number of rows with negative item_cnt_day %f' % (zero_df['item_cnt_day'].count()))

zero_df = sales[sales['item_price']<0]

print( 'number of rows with negative item_price %f' % (zero_df['item_price'].count()))
print(' setting -ve item_cnt_day to zero ')

sales['item_cnt_day'] = sales['item_cnt_day'].apply(lambda x: x if x >=0 else 0)

print(' removing -ve item_price items ')

sales['item_price'] = sales['item_price'].apply(lambda x: x if x >=0 else 0)
print(' checking for sanity of item_cnt_day')

sales.min()
plt.figure(figsize=(10,4))

plt.xlim(-100, 3000)

sns.boxplot(x=sales.item_cnt_day)



plt.figure(figsize=(10,4))

plt.xlim(sales.item_price.min(), sales.item_price.max()*1.1)

sns.boxplot(x=sales.item_price)
# remove item_price and item_cnt_day outliers

sales = sales[sales.item_price<100000]

sales = sales[sales.item_cnt_day<=1500]
sales['revenue'] = sales['item_cnt_day']*sales['item_price']
sales.head(3)
sales = pd.merge(sales, items , how='left', on='item_id')

sales = pd.merge(sales, item_categories , how='left', on='item_category_id')
item_categories.head(1).T
sales.head(2).T
def lag_feature(df, lags, col):

    tmp = df[['date_block_num','shop_id','item_id', col]]

    for i in lags:

        shifted = tmp.copy()

        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]

        shifted['date_block_num'] += i

        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')

    return df
%%time

gc.collect()

# Create "grid" with columns

index_cols = ['shop_id', 'item_id', 'date_block_num']



# For every month we create a grid from all shops/items combinations from that month

grid = [] 

for block_num in sales['date_block_num'].unique():

    cur_shops = sales.loc[sales['date_block_num'] == block_num, 'shop_id'].unique()

    cur_items = sales.loc[sales['date_block_num'] == block_num, 'item_id'].unique()

    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))



# Turn the grid into a dataframe

grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)

grid = pd.concat([grid, test])



# Groupby data to get shop-item-month aggregates

print(" .. \ shop-item-month aggregates ")

gb = sales.groupby(index_cols,as_index=False).agg({'item_cnt_day':{'target':'sum'}})

# Fix column names

gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values] 

# Join it to the grid

all_data = pd.merge(grid, gb, how='left', on=index_cols).fillna(0)

all_data = lag_feature( all_data , [1,6,12] , 'target')

#all_data.drop(['target'], axis=1, inplace=True)



# Same as above but with shop-month aggregates

print(" .. \ shop month sum ")

gb = sales.groupby(['shop_id', 'date_block_num'],as_index=False).agg({'item_cnt_day':{'target_shop':'sum'}})

gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]

all_data = pd.merge(all_data, gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)

all_data = lag_feature( all_data , [1,6,12] , 'target_shop')

all_data.drop(['target_shop'], axis=1, inplace=True)



# Same as above but with item-month aggregates

print(" .. \ item month sum" )

gb = sales.groupby(['item_id', 'date_block_num'],as_index=False).agg({'item_cnt_day':{'target_item':'sum'}})

gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]

all_data = pd.merge(all_data, gb, how='left', on=['item_id', 'date_block_num']).fillna(0)

all_data = lag_feature( all_data , [1,6,12] , 'target_item')

all_data.drop(['target_item'], axis=1, inplace=True)



# # date_block_num averages 

print(" .. \ date_block_num mean")

gb = sales.groupby(['date_block_num'],as_index=False).agg({'item_cnt_day':{'d_mean':'mean'}})

gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]

all_data = pd.merge(all_data, gb, how='left', on=['date_block_num']).fillna(0)

all_data = lag_feature( all_data , [1] , 'd_mean')

all_data.drop(['d_mean'], axis=1, inplace=True)



# commenting this out we are adding holiday seasonality later 

#days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])

#all_data['days_of_month']= all_data['date_block_num'].apply(lambda x: days[x%12]) 



# Category for each item

item_category_mapping = items[['item_id','item_category_id']].drop_duplicates()

all_data = pd.merge(all_data, item_category_mapping, how='left', on='item_id')

all_data = pd.merge(all_data, item_categories, how='left', on='item_category_id')



print(" .. \ date_block_num item_id agg ")

gb = sales.groupby(['date_block_num','item_id'],as_index=False).agg({'item_cnt_day':{'did_mean':'mean'}})

gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]

all_data = pd.merge(all_data, gb, how='left', on=['date_block_num','item_id']).fillna(0)

all_data = lag_feature( all_data , [1] , 'did_mean')

all_data.drop(['did_mean'], axis=1, inplace=True)



print(" .. \ date_block_num shop_id mean")

gb = sales.groupby(['date_block_num','shop_id'],as_index=False).agg({'item_cnt_day':{'dsid_mean':'mean'}})

gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]

all_data = pd.merge(all_data, gb, how='left', on=['date_block_num','shop_id']).fillna(0)

all_data = lag_feature( all_data , [1] , 'dsid_mean')

all_data.drop(['dsid_mean'], axis=1, inplace=True)



print(" .. \ date_block_num shop_id item_id mean")

gb = sales.groupby(['date_block_num','shop_id' , 'item_id'],as_index=False).agg({'item_cnt_day':{'dsiid_mean':'mean'}})

gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]

all_data = pd.merge(all_data, gb, how='left', on=['date_block_num','shop_id' , 'item_id']).fillna(0)

all_data = lag_feature( all_data , [1] , 'dsiid_mean')

all_data.drop(['dsiid_mean'], axis=1, inplace=True)



print(" .. \ date_block_num item_category_id mean")

gb = sales.groupby(['date_block_num','item_category_id'],as_index=False).agg({'item_cnt_day':{'dic_mean':'mean'}})

gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]

all_data = pd.merge(all_data, gb, how='left', on=['date_block_num','item_category_id']).fillna(0)

all_data = lag_feature( all_data , [1] , 'dic_mean')

all_data.drop(['dic_mean'], axis=1, inplace=True)



print(" .. \ date_block_num shop_id item_category_id mean")

gb = sales.groupby(['date_block_num', 'shop_id' , 'item_category_id'],as_index=False).agg({'item_cnt_day':{'dsic_mean':'mean'}})

gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]

all_data = pd.merge(all_data, gb, how='left', on=['date_block_num','shop_id' , 'item_category_id']).fillna(0)

all_data = lag_feature( all_data , [1] , 'dsic_mean')

all_data.drop(['dsic_mean'], axis=1, inplace=True)



print(" .. \ date_block_num item_id item_category_id mean")

gb = sales.groupby(['date_block_num', 'item_id' , 'item_category_id'],as_index=False).agg({'item_cnt_day':{'diic_mean':'mean'}})

gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]

all_data = pd.merge(all_data, gb, how='left', on=['date_block_num','item_id' , 'item_category_id']).fillna(0)

all_data = lag_feature( all_data , [1] , 'diic_mean')

all_data.drop(['diic_mean'], axis=1, inplace=True)



print(" .. \ date_block_num shop_id item_id item_category_id mean")

gb = sales.groupby(['date_block_num', 'shop_id' , 'item_id' , 'item_category_id'],as_index=False).agg({'item_cnt_day':{'dsiic_mean':'mean'}})

gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]

all_data = pd.merge(all_data, gb, how='left', on=['date_block_num','shop_id','item_id' , 'item_category_id']).fillna(0)

all_data = lag_feature( all_data , [1] , 'dsiic_mean')

all_data.drop(['dsiic_mean'], axis=1, inplace=True)



print(" .. \ item_id iip_mean mean")

gb = sales.groupby([ 'item_id'],as_index=False).agg({'item_price':{'iip_mean':'mean'}})

gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]

all_data = pd.merge(all_data, gb, how='left', on=['item_id']).fillna(0)

all_data = lag_feature( all_data , [1] , 'iip_mean')

all_data.drop(['iip_mean'], axis=1, inplace=True)



print(" .. \ shop_id sip_mean  mean")

gb = sales.groupby([ 'shop_id'],as_index=False).agg({'item_price':{'sip_mean':'mean'}})

gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]

all_data = pd.merge(all_data, gb, how='left', on=['shop_id']).fillna(0)

all_data = lag_feature( all_data , [1] , 'sip_mean')

all_data.drop(['sip_mean'], axis=1, inplace=True)



print(" .. \ date_block_num item_id diip_mean  mean")

gb = sales.groupby([ 'date_block_num','item_id'],as_index=False).agg({'item_price':{'diip_mean':'mean'}})

gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]

all_data = pd.merge(all_data, gb, how='left', on=['date_block_num','item_id']).fillna(0)

all_data = lag_feature( all_data , [1] , 'diip_mean')

all_data.drop(['diip_mean'], axis=1, inplace=True)



print(" .. \ date_block_num shop_id dsip_mean  mean")

gb = sales.groupby([ 'date_block_num','shop_id'],as_index=False).agg({'item_price':{'dsip_mean':'mean'}})

gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]

all_data = pd.merge(all_data, gb, how='left', on=['date_block_num','shop_id']).fillna(0)

all_data = lag_feature( all_data , [1] , 'dsip_mean')

all_data.drop(['dsip_mean'], axis=1, inplace=True)



print(" .. \ date_block_num shop_id item_id  dsiip_mean")

gb = sales.groupby([ 'date_block_num','shop_id', 'item_id'],as_index=False).agg({'item_price':{'dsiip_mean':'mean'}})

gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]

all_data = pd.merge(all_data, gb, how='left', on=['date_block_num','shop_id' ,'item_id']).fillna(0)

all_data = lag_feature( all_data , [1] , 'dsiip_mean')

all_data.drop(['dsiip_mean'], axis=1, inplace=True)



# mean encoding for categorical features

print(" .. \ si_mean ")

gb = sales.groupby(['shop_id'],as_index=False).agg({'item_cnt_day':{'si_mean':'mean'}})

gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]

all_data = pd.merge(all_data, gb, how='left', on=['shop_id']).fillna(0)



print(" .. \ ii_mean ")

gb = sales.groupby(['item_id'],as_index=False).agg({'item_cnt_day':{'ii_mean':'mean'}})

gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]

all_data = pd.merge(all_data, gb, how='left', on=['item_id']).fillna(0)



print(" .. \ ic_mean ")

gb = sales.groupby(['item_category_id'],as_index=False).agg({'item_cnt_day':{'ic_mean':'mean'}})

gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]

all_data = pd.merge(all_data, gb, how='left', on=['item_category_id']).fillna(0)



print(" .. \ item_category type_code ")

gb = sales.groupby(['type_code'],as_index=False).agg({'item_cnt_day':{'tc_mean':'mean'}})

gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]

all_data = pd.merge(all_data, gb, how='left', on=['type_code']).fillna(0)



print(" .. \ item_category subtype_code ")

gb = sales.groupby(['subtype_code'],as_index=False).agg({'item_cnt_day':{'stc_mean':'mean'}})

gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]

all_data = pd.merge(all_data, gb, how='left', on=['subtype_code']).fillna(0)



# gb = sales.groupby([ 'date_block_num','shop_id'],as_index=False).agg({'revenue':{'dsrev_mean':'mean'}})

# gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]

# all_data = pd.merge(all_data, gb, how='left', on=['date_block_num','shop_id']).fillna(0)



# gb = sales.groupby(['shop_id'],as_index=False).agg({'revenue':{'srev_mean':'mean'}})

# gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]

# all_data = pd.merge(all_data, gb, how='left', on=['shop_id']).fillna(0)



# Downcast dtypes from 64 to 32 bit to save memory

all_data = downcast_dtypes(all_data)

del grid,gb

gc.collect();
all_data[all_data['date_block_num']==33].head(1)
all_data[all_data['date_block_num']==33].describe()
# set the values from lagged_proxy into lagged_submissions 

lagged_submissions = pd.merge(submissions , test , on='ID' ).drop(columns=['item_cnt_month'])

lagged_submissions = pd.merge(lagged_submissions, all_data[all_data['date_block_num']==33] , how="left" ,on=['shop_id' , 'item_id'])

lagged_submissions = lagged_submissions.filter(['ID','target_lag_1'])

values = {'target_lag_1': 0}

lagged_submissions = lagged_submissions.fillna(value=values)

lagged_submissions['item_cnt_month'] = lagged_submissions['target_lag_1'].clip(0,20)



# create a submission file for baseline 

lagged_submissions.filter(['ID' , 'item_cnt_month']).to_csv( 'sample_submissions.csv' , index=False  )
#https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

# tfidf features for item names 

import sklearn.feature_extraction as feature_extraction

feature_cnt = 5

tfidf = feature_extraction.text.TfidfVectorizer(max_features=feature_cnt )

items['item_name_len'] = items['item_name'].map(len) 

items['item_name_wc'] = items['item_name'].map(lambda x: len(str(x).split(' ')))

txtFeatures = pd.DataFrame(tfidf.fit_transform(items['item_name']).toarray())

cols = txtFeatures.columns

for i in range(feature_cnt):

    items['item_name_tfidf_' + str(i)] = txtFeatures[cols[i]]

#items.drop(columns='item_name', inplace=True)

#items.head()
#tfidf features for item category names

feature_cnt = 5

tfidf = feature_extraction.text.TfidfVectorizer(max_features=feature_cnt)

item_categories['item_cat_name_len'] = item_categories['item_category_name'].map(len)  #Lenth of Item Category Description

item_categories['item_cat_name_wc'] = item_categories['item_category_name'].map(lambda x: len(str(x).split(' '))) #Item Category Description Word Count

txtFeatures = pd.DataFrame(tfidf.fit_transform(item_categories['item_category_name']).toarray())

cols = txtFeatures.columns

for i in range(feature_cnt):

    item_categories['item_cat_name_tfidf_' + str(i)] = txtFeatures[cols[i]]

#item_categories.drop(columns='item_category_name', inplace=True)

#item_categories.head()
items.head(2)
# merge all the manufactured features onto the main data frame

all_data = pd.merge(all_data, items[['item_id','item_category_id' , 'item_name_tfidf_0' , 'item_name_tfidf_1',

                                    'item_name_tfidf_2','item_name_tfidf_3','item_name_tfidf_4']], how='left', on=['item_id','item_category_id'])

all_data = pd.merge(all_data, item_categories[['item_category_id' , 'item_cat_name_tfidf_0' , 'item_cat_name_tfidf_1',

                                    'item_cat_name_tfidf_2','item_cat_name_tfidf_3','item_cat_name_tfidf_4' ]], how='left', on=['item_category_id'])
all_data[all_data['date_block_num'] ==33 ].head(3).T
# initialize list of russian holidays 

data = [[0, 8], [1,2] , [2,2], [3,0] , [4,6] , [5,3] , [6,1] , [7,2] , [8,2] , [9,1] , [10,2] , [11,3] , [12,9] ,

       [13,3] , [14,4] , [15, 1] , [16,8] , [17,7] , [18,2] , [19,0] , [20,2] , [21,2] , [22, 4] , [23,1] , [24,10],

       [25,3] , [26,3] , [27,1] , [28,5] , [29,5] , [30,2] , [31,0] , [32 ,2] , [33,1], [34,3]] 

  

# Create the pandas DataFrame 

holidays = pd.DataFrame(data, columns = ['date_block_num', 'num_hols']) 



# russian holiday season is typically dec , jan 

holidays['holiday_season'] = holidays['date_block_num'].apply(lambda x: 1 if x in ([0,11,12,23,24]) else 0)
all_data = pd.merge(all_data, holidays, how='left', on='date_block_num')

all_data = downcast_dtypes(all_data)
all_data[all_data['date_block_num']==34].head(15)
pp_df = all_data[all_data['shop_id']==7].head(1000)

corrmatt = pp_df.corr()

f,ax = plt.subplots(figsize=(35,30))

sns.heatmap(corrmatt, annot=True , square=True)
#Correlation with output variable

cor_target = abs(corrmatt["target"])

#Selecting highly correlated features

relevant_features = cor_target[cor_target>0.5]

relevant_features
corrmatt
import pickle

all_data.to_pickle('feature_data.pkl')

del all_data, pp_df, sales
all_data = pd.read_pickle('feature_data.pkl')
all_data[all_data['date_block_num']==34].head(1).T
# Don't use old data from year 2013

all_data = all_data[all_data['date_block_num'] >= 12] 



shift_range= [1, 2, 3, 4, 5, 6,9,  12]



# List of all lagged features

fit_cols = [col for col in all_data.columns if col[-1] in [str(item) for item in shift_range]] 



for col in all_data.columns:

    if "mean" in col:

        fit_cols.append(col)

        

for col in all_data.columns:

    if "tfidf" in col:

        fit_cols.append(col)



# We will drop these at fitting stage

to_drop_cols = list(set(list(all_data.columns)) - (set(fit_cols)|set(index_cols))) + ['date_block_num' , 'shop_id' , 'item_id' ]

gc.collect();
# Save `date_block_num`, as we can't use them as features, but will need them to split the dataset into parts 

dates = all_data['date_block_num']



last_block = dates.max()

print('Test `date_block_num` is %d' % last_block)
all_data[all_data['date_block_num']==34].head(1)
dates = all_data['date_block_num']

print('Test `date_block_num` is %d' % test_block)

dates_train  = dates[dates <  test_block-1]

dates_val = dates[dates == test_block-1]

dates_test  = dates[dates == test_block]



X_train = all_data.loc[dates <  test_block-1].drop(to_drop_cols, axis=1)

X_test =  all_data.loc[dates == test_block].drop(to_drop_cols, axis=1)

X_val = all_data.loc[dates ==  test_block-1].drop(to_drop_cols, axis=1)



y_train = all_data.loc[dates <  test_block-1, 'target'].values.clip(0,40)

y_test = all_data.loc[dates == test_block, 'target'].values.clip(0,40)

y_val =  all_data.loc[dates == test_block-1, 'target'].values.clip(0,40)

target_range=(0,20)



# create data for gridsearchcv

X_cv_train = all_data.loc[((dates>28) & (dates<32)) ].drop(to_drop_cols, axis=1)

y_cv_train = all_data.loc[((dates>28) & (dates<32)), 'target'].values.clip(0,40)

X_train.head(1).T
X_train.isna().sum().sum()
X_test.isna().sum().sum()
# make sure that none of the values are negative

#sum(n < 0 for n in all_data.values.flatten())
print( ' Training a plain linear regression model ')

gc.collect()

# would be good to use logisticregression and tune hyperparameters using 

# bayesian-optimization 

from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression

#https://stackoverflow.com/questions/39163354/evaluating-logistic-regression-with-cross-validation

target_range=(0,20)



lr = LinearRegression()

lr.fit(X_train.values, y_train)
print(' lr model coefficients are ' ) 

lr.coef_
pred_lr_val = lr.predict(X_val)

from sklearn.metrics import mean_squared_error

from math import sqrt

print('Val RMSE for lr is %f' % sqrt(mean_squared_error(y_val, pred_lr_val.clip(0,20))))

print('R^2 score for lr is %f' % (r2_score(y_val, pred_lr_val.clip(0,20))))
over_underfit_plot(lr, X_train, y_train)
pickle.dump(lr,open("firstlevel_lr_model" , "wb"))
pred_lr = lr.predict(X_test.values).clip(*target_range)



print(' pred_lr min %f , pred_lr max %f ' , (pred_lr.min(),pred_lr.max()))
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)

X_train_std = scaler.transform(X_train)

X_valid_std = scaler.transform(X_val)



model_lr_scaled = LinearRegression(fit_intercept=True, normalize=True)

model_lr_scaled.fit(X_train_std, y_train)

print(np.sqrt(mean_squared_error(y_val, model_lr_scaled.predict(X_valid_std))))

model_lr_scaled.predict(X_valid_std).min()

over_underfit_plot(model_lr_scaled, X_train_std, y_train)
print('Val RMSE for lr is %f' % sqrt(mean_squared_error(y_val, model_lr_scaled.predict(X_val))))

print('R^2 score for lr is %f' % (r2_score(y_val, model_lr_scaled.predict(X_val))))
print('Val RMSE for lr is %f' % sqrt(mean_squared_error(y_val, model_lr_scaled.predict(X_val).clip(0,20))))

print('R^2 score for lr is %f' % (r2_score(y_val, model_lr_scaled.predict(X_val).clip(0,20))))
pred_lr_scaled = model_lr_scaled.predict(X_test.values).clip(*target_range)
print(' pred_lr_scaled min %f , pred_lr_scaled max %f ' , (pred_lr_scaled.min(),pred_lr_scaled.max()))
# from sklearn import metrics

# from sklearn.model_selection import cross_validate

# from sklearn.model_selection import cross_val_score

# predicted = cross_val_score(lr, X_train, y_train, cv=10)

# predicted_mean = predicted.mean()

# print('CV score of LR Model %f' % (predicted_mean))
# lr.coef_

# from matplotlib import pyplot

# importance = lr.coef_

# # summarize feature importance

# for i,v in enumerate(importance):

#     print('Feature: %0d, Score: %.5f' % (i,v))

# # plot feature importance

# pyplot.bar([x for x in range(len(importance))], importance)

# pyplot.show()
#https://stackoverflow.com/questions/24255723/sklearn-logistic-regression-important-features

feature_importance = abs(lr.coef_)

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .8



featfig = plt.figure(figsize=(20,10))

featax = featfig.add_subplot(1, 1, 1)

featax.barh(pos, feature_importance[sorted_idx], align='center')

featax.set_yticks(pos)

featax.set_yticklabels(np.array(X_train.columns)[sorted_idx], fontsize=10)

featax.set_xlabel('Relative CV Model Feature Importance')



plt.tight_layout()   

plt.show()
feat_importances = pd.Series(lr.coef_, index=X_train.columns)

feat_importances.nlargest(20).plot(kind='barh')
# from sklearn.pipeline import Pipeline

# from sklearn.preprocessing import StandardScaler

# from sklearn.feature_selection import SelectFromModel

# from sklearn.ensemble import RandomForestClassifier



# lrp_model = Pipeline(

#     [

#         ('select', SelectFromModel(LinearRegression(), threshold=0.05)),

#         ('scaler', StandardScaler()),

#         ('final', LinearRegression())

#     ]

# )





# lrp_model.fit(X_train, y_train)

# pred = lrp_model.predict(X_val)

# print('score on training set:', lrp_model.score(X_train, y_train))

# print('score on test set:', lrp_model.score(X_val, y_val))
# pred_lrp_val = lrp_model.predict(X_val)

# from sklearn.metrics import mean_squared_error

# from math import sqrt

# print('Val RMSE for lrp_model is %f' % sqrt(mean_squared_error(y_val, pred_lrp_val.clip(0,20))))

# print('R^2 score for lrp_model is %f' % (r2_score(y_val, pred_lrp_val.clip(0,20))))
# over_underfit_plot(lrp_model, X_train, y_train)
# pred_lrp = lrp_model.predict(X_test.values).clip(*target_range)
# print(' pred_lrp_scaled min %f , pred_lrp_scaled max %f ' , (pred_lrp.min(),pred_lrp.max()))
#https://slundberg.github.io/shap/notebooks/linear_explainer/Sentiment%20Analysis%20with%20Logistic%20Regression.html

# import shap



# shap.initjs()

# explainer = shap.LinearExplainer(lr, X_train, feature_dependence="independent")

# shap_values = explainer.shap_values(X_test)

# X_test_array = X_test # we need to pass a dense version for the plotting functions

# shap.summary_plot(shap_values, X_test_array, feature_names=X_train.columns)


# # gc.collect()

# from bayes_opt import BayesianOptimization

# from sklearn.metrics import mean_squared_error

# def bayesion_opt_lgbm(X, y, init_iter=3, n_iters=7, random_state=11, seed = 101, num_iterations = 100 ):

#     dtrain = lgb.Dataset(data=X, label=y)

#     def lgb_r2_score(preds, dtrain):

#         labels = dtrain.get_label()

#         return 'r2', r2_score(labels, preds), True

    

#     # Objective Function

#     def hyp_lgbm_r2(num_leaves, feature_fraction, bagging_fraction, max_depth, min_split_gain, min_child_weight , learning_rate):



#         params = {'application':'regression','num_iterations': num_iterations,

#                 'learning_rate':learning_rate, 'early_stopping_round':50,

#                 'metric':'lgb_r2_score'} # Default parameters

#         params["num_leaves"] = int(round(num_leaves))

#         params['feature_fraction'] = max(min(feature_fraction, 1), 0)

#         params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)

#         params['max_depth'] = int(round(max_depth))

#         params['min_split_gain'] = min_split_gain

#         params['min_child_weight'] = min_child_weight

#         cv_results = lgb.cv(params, dtrain, nfold=5, seed=seed,categorical_feature=[], stratified=False,

#                           verbose_eval =None ,feval=lgb_r2_score)

#         # print(cv_results)

#         return np.max(cv_results['r2-mean'])

    

#     def hyp_lgbm_l1(num_leaves, feature_fraction, bagging_fraction, max_depth, min_split_gain, min_child_weight,learning_rate):

      

#         params = {'application':'regression','num_iterations': num_iterations,

#                   'learning_rate':learning_rate, 'early_stopping_round':50,

#                   'metric':'l1'} # Default parameters

#         params["num_leaves"] = int(round(num_leaves))

#         params['feature_fraction'] = max(min(feature_fraction, 1), 0)

#         params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)

#         params['max_depth'] = int(round(max_depth))

#         params['min_split_gain'] = min_split_gain

#         params['min_child_weight'] = min_child_weight

#         cv_result = lgb.cv(params, dtrain, nfold=5, seed=random_seed, stratified=False, verbose_eval =None, metrics=['l1'])



#         return -np.min(cv_result['l1-mean'])

    

#     def hyp_lgbm_l2(num_leaves, feature_fraction, bagging_fraction, max_depth, min_split_gain, min_child_weight,learning_rate):



#         params = {'application':'regression','num_iterations': num_iterations,

#                   'learning_rate':learning_rate, 'early_stopping_round':50,

#                   'metric':'l2'} # Default parameters

#         params["num_leaves"] = int(round(num_leaves))

#         params['feature_fraction'] = max(min(feature_fraction, 1), 0)

#         params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)

#         params['max_depth'] = int(round(max_depth))

#         params['min_split_gain'] = min_split_gain

#         params['min_child_weight'] = min_child_weight

#         #cv_result = lgb.cv(params, dtrain, nfold=5, seed=seed, stratified=False, verbose_eval =None, metrics=['l2'])

#         cv_result = lgb.cv(params, dtrain, nfold=5, seed=seed, stratified=False, metrics=['l2'])



#         return -np.min(cv_result['l2-mean'])

    

    

#     # Domain space-- Range of hyperparameters 

#     pds = {'num_leaves': (80, 100),

#             'feature_fraction': (0.1, 0.9),

#             'bagging_fraction': (0.8, 1),

#             'max_depth': (5, 15),

#             'min_split_gain': (0.001, 0.1),

#             'min_child_weight': (10, 25),

#             'learning_rate' : (0.02 , 0.05)

#             }



#     # Surrogate model

#     optimizer = BayesianOptimization(hyp_lgbm_l2, pds, random_state=random_state)



#     # Optimize

#     optimizer.maximize(init_points=init_iter, n_iter=n_iters)

    

#     return optimizer.max['params']



# opt_hyp_params = bayesion_opt_lgbm(X_train, y_train, init_iter=5, n_iters=10, random_state=77, seed = 101, num_iterations = 100)

# print(" Tuned Hyperparameters for LGBM regressor \n")

# opt_hyp_params
# 100 iteration bayesian optimization parameters

# opt_hyp_params = {'bagging_fraction': 1.0,

#  'feature_fraction': 0.9,

#  'learning_rate': 0.05,

#  'max_depth': 15.0,

#  'min_child_weight': 19.167083237894754,

#  'min_split_gain': 0.1,

#  'num_leaves': 80.0}

# Original set

# opt_hyp_params = {

#     'boosting_type': 'dart',

#     'metric': 'l2_root', # RMSE

#     'verbose': 1,

#     'seed': 0,

#     'max_depth': 8,

#     'learning_rate': 0.1,

#     'reg_lambda': 2.0,

#     'reg_alpha': 2.0,

#     'subsample': 0.7,

#     'num_leaves': 20,

#     'feature_fraction': 0.8,

#     'drop_rate': 0.2,

#     'bagging_fraction': 0.75,

#     'min_split_gain': 0.1,

#     'min_child_weight': 19   

# }



# from bayesian optimization 

opt_hyp_params = {



    'verbose': 1,

    'seed': 0,

    'max_depth': 8,

    'learning_rate': 0.1,

    'reg_lambda': 2.0,

    'reg_alpha': 2.0,

    'subsample': 0.7,

    'num_leaves': 20,

    'feature_fraction': 0.8,

    'drop_rate': 0.2,

    'bagging_fraction': 0.75,

    'min_split_gain': 0.1,

    'min_child_weight': 19   

}



int(opt_hyp_params['num_leaves'])
# from gridsearchcv

evals_result = {}  # to record eval results for plotting

lgb_params = {

                'metric': 'rmse', # RMSE

                'nthread':1, 

                'num_leaves' : 50,

                'learning_rate' : 0.01,

                'max_depth':7,

                'bagging_fraction':0.6,

                'feature_fraction': 0.8,

                'min_child_samples':20, 

                'min_child_weight': 0.001,

                'reg_alpha':0.5,

                'reg_lambda':0.001,

                'min_data_in_leaf': 2**7, 

                'bagging_seed': 2**7, 

                'bagging_freq':1,

                'verbose':0 ,

                'subsample': 0.7

              }

lgb_train = lgb.Dataset(X_train, y_train)

lgb_valid = lgb.Dataset(X_val, y_val)

lgb_cv_train = lgb.Dataset(X_cv_train, y_cv_train)



cat_features = [ 'si_mean' ,  'ii_mean' , 'ic_mean' , 'tc_mean' , 'stc_mean'  ]

# %%time

# cv_results = lgb.cv(lgb_params, lgb_train, num_boost_round=500, nfold=3, stratified=False, 

#                     shuffle=True, metrics='rmse',early_stopping_rounds=50, verbose_eval=50, show_stdv=True, 

#                     seed=0,categorical_feature=cat_features,eval_train_metric=True)

# print('best n_estimators:', len(cv_results['train rmse-mean']))

# print('best cv score:', cv_results['train rmse-mean'][-1])
# params = {    'boosting_type': 'gbdt', 

#     'objective': 'regression', 

#     'learning_rate': 0.1, 

#     'num_leaves': 50, 

#     'max_depth': 7,  

#     'subsample': 0.8, 

#     'colsample_bytree': 0.8, 

#     }



# cv1_results = lgb.cv(

#     params, lgb_cv_train, num_boost_round=200, nfold=5, stratified=False, shuffle=True, metrics='rmse',

#     early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0,categorical_feature=cat_features,eval_train_metric=True)



# print('best n_estimators:', len(cv1_results['train rmse-mean']))

# print('best cv score:', cv1_results['train rmse-mean'][-1])
# tried out as per https://programmer.ink/think/5d45867ef3982.html

# {'max_depth': [5, 6, 7, 8], 'num_leaves': [50]}

# {'max_depth': [5, 6, 7, 8], 'num_leaves': [100]} 

# {'max_depth': [7],    'num_leaves':[50,60,70,80,100 ]

#from sklearn.model_selection import GridSearchCV

###We can create a sklearn er model of lgb, using the 

# model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=50,

#                               learning_rate=0.1, n_estimators=200, max_depth=7,

#                               metric='rmse', bagging_fraction = 0.6,feature_fraction = 0.8,

#                               min_child_samples=20, min_child_weight=0.001,

#                              reg_alpha=0.5,reg_lambda=0.001, learning_rate=0.01)



# params_test1={    

# }

# gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test1, scoring='neg_mean_squared_error', cv=2, verbose=20, n_jobs=1,pre_dispatch=2)

# gsearch1.fit(X_cv_train, y_cv_train)

# gsearch1.best_estimator_, gsearch1.best_params_, gsearch1.best_score_ 

%%time

gc.collect()

evals_result = {}

#model = lgb.train(lgb_params, lgb.Dataset(X_train, label=y_train.clip(0,40)), 150)

model_lgb =lgb.train(lgb_params, lgb_train, num_boost_round=500,

                       early_stopping_rounds=200, categorical_feature=cat_features,

                       evals_result=evals_result,verbose_eval=50,valid_sets=[lgb_train, lgb_valid])
pred_lgb = model_lgb.predict(X_test.values).clip(*target_range)

import pickle

pickle.dump(model_lgb,open("firstlevel_lgb_model" , "wb"))
# ### read from disk 

# import pickle

# MODELS_FOLDER = '/kaggle/input/week5work2/'

# model_lgb = pd.read_pickle(os.path.join(MODELS_FOLDER, 'firstlevel_lgb_model') )

# pred_lgb = model_lgb.predict(X_test.values).clip(*target_range)
print(' pred_lgb max %f , pred_lgb min %f' , (pred_lgb.min(), pred_lgb.max()))
pred_lgb_val = model_lgb.predict(X_val)

from sklearn.metrics import mean_squared_error

from math import sqrt

print('Val RMSE for lgb is %f' % sqrt(mean_squared_error(y_val, pred_lgb_val.clip(*target_range))))

print('Val RMSE for lgb is %f' % sqrt(mean_squared_error(y_val, pred_lgb_val)))
lgb.plot_importance(model_lgb, max_num_features=25, figsize=(6,6), title='Feature importance (LightGBM)')

plt.show()


print('Plotting metrics recorded during training...')

ax = lgb.plot_metric(evals_result, metric='rmse')

plt.show()



# print('Plotting split value histogram...')

# ax = lgb.plot_split_value_histogram(model, feature='f26', bins='auto')

# plt.show()



# print('Plotting 54th tree...')  # one tree use categorical feature to split

# ax = lgb.plot_tree(model, tree_index=53, figsize=(15, 15), show_info=['split_gain'])

# plt.show()
# import shap



# shap.initjs()

# explainer = shap.TreeExplainer(model, X_train.head(100), feature_dependence="independent")

# X_test_array = X_test.head(100) # we need to pass a dense version for the plotting functions

# shap_values = explainer.shap_values(X_test_array)

# shap.summary_plot(shap_values, X_test_array, feature_names=X_train.columns)
plt.figure(figsize=(7,5))

plt.scatter(pred_lr, pred_lgb, marker='x', alpha=0.45)

plt.xlabel('Linear regression test set predictions')

plt.ylabel('LightGBM test set predictions')

plt.show()
X_test_level2 = np.c_[pred_lr, pred_lgb]
#dates_train_level2 = dates_train[dates_train.isin([28, 32, 33])]

dates_train_level2 = dates_train[dates_train.isin([27,28, 29, 30, 31, 32])]



# That is how we get target for the 2nd level dataset

y_train_level2 = y_train[dates_train.isin(dates_train_level2)]
%%time

gc.collect()

X_train_level2 = np.zeros([y_train_level2.shape[0], 2])

from sklearn.preprocessing import StandardScaler





# Now fill `X_train_level2` with metafeatures

pbar = tqdm_notebook(dates_train_level2.unique())

for cur_block_num in pbar:

    

    '''

        1. Split `X_train` into parts

           Remember, that corresponding dates are stored in `dates_train` 

        2. Fit linear regression 

        3. Fit LightGBM and put predictions          

        4. Store predictions from 2. and 3. in the right place of `X_train_level2`. 

           We can use `dates_train_level2` for it

           Make sure the order of the meta-features is the same as in `X_test_level2`

    '''      

    

    pbar.set_description('{}: Split'.format(cur_block_num))

    _X_train = all_data.loc[dates <  cur_block_num].drop(to_drop_cols, axis=1)

    _X_test =  all_data.loc[dates == cur_block_num].drop(to_drop_cols, axis=1)



    _y_train = all_data.loc[dates <  cur_block_num, 'target'].values.clip(0,40)

    _y_test = all_data.loc[dates ==  cur_block_num, 'target'].values.clip(0,40)

    

    pbar.set_description('{}: LR'.format(cur_block_num))

    evals_result = {}  # to record eval results for plotting

    lr = LinearRegression(fit_intercept=True, normalize=True)

    scaler = StandardScaler()

    lr.fit(scaler.fit_transform(_X_train.values), _y_train)

    X_train_level2[dates_train_level2 == cur_block_num, 0] = lr.predict(scaler.fit_transform(_X_test.values)).clip(*target_range)

    

    pbar.set_description('{}: LGB'.format(cur_block_num))

    #model = lgb.train(lgb_params, lgb.Dataset(_X_train, label=_y_train), 100)

    lgb_train = lgb.Dataset(_X_train, _y_train)

    lgb_valid = lgb.Dataset(_X_test, _y_test)

    

    model = lgb.train(lgb_params, lgb_train, num_boost_round=200,evals_result = evals_result,

                           valid_sets=[lgb_train,lgb_valid], early_stopping_rounds=50,

                           verbose_eval=50)

    X_train_level2[dates_train_level2 == cur_block_num, 1] = model.predict(_X_test).clip(*target_range)

    

    print('Plotting metrics recorded during 2nd level training for %f' % (cur_block_num))

    ax = lgb.plot_metric(evals_result, metric='rmse')

    plt.show()
print(X_train_level2.mean(axis=0))
plt.figure(figsize=(7,5))

plt.scatter(X_train_level2[:,0], X_train_level2[:,1], marker='o', alpha=0.45)

plt.xlabel('Linear regression test set predictions')

plt.ylabel('LightGBM test set predictions')

plt.show()
alphas_to_try = np.linspace(0, 1, 1001)



r2_scores = np.array([r2_score(y_train_level2, np.dot(X_train_level2, [alpha, 1 - alpha])) for alpha in alphas_to_try])

best_alpha = alphas_to_try[r2_scores.argmax()]

r2_train_simple_mix = r2_scores.max()



print('Best alpha: %f; Corresponding r2 score on train: %f' % (best_alpha, r2_train_simple_mix))
test_preds = best_alpha * pred_lr + (1 - best_alpha) * pred_lgb

test['item_cnt_month'] = test_preds.clip(*target_range)

test[['ID', 'item_cnt_month']].to_csv('submission_averaging.csv', index=False)
test_preds_lr_scaled = best_alpha * pred_lr_scaled + (1 - best_alpha) * pred_lgb

test['item_cnt_month'] = test_preds_lr_scaled.clip(*target_range)

test[['ID', 'item_cnt_month']].to_csv('submission_averaging_lr_scaled.csv', index=False)
metamodel = LinearRegression()

metamodel.fit(X_train_level2, y_train_level2)

import pickle

pickle.dump(metamodel, open("metamodel",  'wb'))
train_preds = metamodel.predict(X_train_level2).clip(*target_range)

r2_train_stacking = r2_score(y_train_level2.clip(*target_range), train_preds)



test_preds = metamodel.predict(np.vstack((pred_lr, pred_lgb)).T)

test['item_cnt_month'] = test_preds.clip(*target_range)

test[['ID', 'item_cnt_month']].to_csv('submission_stacking.csv', index=False)



print('Train R-squared for stacking is %f' % r2_train_stacking)
rmse_train = np.sqrt(mean_squared_error(y_train_level2, train_preds))

print('RMSE Train: %f' % rmse_train)

print( ' mean train preds : %f ' , (train_preds.mean()))
print( ' mean test preds : %f ' , (test_preds.mean()) )