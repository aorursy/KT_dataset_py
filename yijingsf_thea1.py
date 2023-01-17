# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import featuretools as ft
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import pickle
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
new_transactions = pd.read_csv('../input/new_merchant_transactions.csv',parse_dates =["purchase_date"])
his_transactions = pd.read_csv('../input/historical_transactions.csv',parse_dates =["purchase_date"])
train = pd.read_csv( '../input/train.csv',parse_dates =["first_active_month"])
test = pd.read_csv( '../input/test.csv',parse_dates =["first_active_month"])
merchants = pd.read_csv( '../input/merchants.csv')
merchants = pd.read_csv( '../input/merchants.csv')
merchants[merchants.duplicated(subset="merchant_id")].head()

merchants = merchants.drop_duplicates(subset="merchant_id",keep="first")
print(train.shape[0]==train["card_id"].nunique())
print(merchants.shape[0]==merchants["merchant_id"].nunique())
#print(merchants[merchants["merchant_id"].duplicated()])
train_his_transactions=train[["card_id"]].merge(his_transactions, how='left', on="card_id")
print(train_his_transactions.shape)
train_his_transactions.head()
with open('train_his_transactions.pickle', 'wb') as f:
    pickle.dump(train_his_transactions, f)
with open('train_his_transactions.pickle', 'rb') as f:
    train_his_transactions = pickle.load(f)
train_his_transactions=train_his_transactions.loc[np.random.choice(train_his_transactions.index, 1000000, replace=False)]
print(train_his_transactions.shape)
train_his_transactions.head()
authorized_flag = pd.get_dummies(train_his_transactions['authorized_flag'])
authorized_flag.columns = ['authorized_flag_N', 'authorized_flag_Y']
train_his_transactions=train_his_transactions.drop(['authorized_flag'], axis=1)
train_his_transactions=pd.concat([train_his_transactions, authorized_flag], axis=1)
train_his_transactions.head()
category_1 = pd.get_dummies(train_his_transactions['category_1'])
#category_1.head()
category_1.columns = ['category_1_N', 'category_1_Y']
train_his_transactions=train_his_transactions.drop(['category_1'], axis=1)
train_his_transactions=pd.concat([train_his_transactions, category_1], axis=1)
train_his_transactions.head()
category_2 = pd.get_dummies(train_his_transactions['category_2'])
#category_2.head()
category_2.columns = ['category_2_1', 'category_2_2',"category_2_3","category_2_4","category_2_5"]
train_his_transactions=train_his_transactions.drop(['category_2'], axis=1)
train_his_transactions=pd.concat([train_his_transactions, category_2], axis=1)
train_his_transactions.head()
category_3 = pd.get_dummies(train_his_transactions['category_3'])
#category_3.head()
category_3.columns = ['category_3_A', 'category_3_B',"category_3_C"]
train_his_transactions=train_his_transactions.drop(['category_3'], axis=1)
train_his_transactions=pd.concat([train_his_transactions, category_3], axis=1)
train_his_transactions.head()
with open('train_his_transactions.pickle', 'wb') as f:
    pickle.dump(train_his_transactions, f)
with open('train_his_transactions.pickle', 'rb') as f:
    train_his_transactions = pickle.load(f)
train_his_transactions.head()
train_his_transactions.shape
train_his_transactions_mer = train_his_transactions.merge(merchants,how="left",on="merchant_id")
train_his_transactions_mer.shape
train_his_transactions_mer.head()
category_1 = pd.get_dummies(train_his_transactions_mer['category_1'])
#category_1.head()
category_1.columns = ['category_1_N_mer', 'category_1_Y_mer']
train_his_transactions_mer=train_his_transactions_mer.drop(['category_1'], axis=1)
train_his_transactions_mer=pd.concat([train_his_transactions_mer, category_1], axis=1)
train_his_transactions_mer.head()
category_2 = pd.get_dummies(train_his_transactions_mer['category_2'])
#category_2.head()
category_2.columns = ['category_2_1_mer', 'category_2_2_mer',"category_2_3_mer","category_2_4_mer","category_2_5_mer"]
train_his_transactions_mer=train_his_transactions_mer.drop(['category_2'], axis=1)
train_his_transactions_mer=pd.concat([train_his_transactions_mer, category_2], axis=1)
train_his_transactions_mer.head()
category_4 = pd.get_dummies(train_his_transactions_mer['category_4'])
#category_1.head()
category_4.columns = ['category_4_N_mer', 'category_4_Y_mer']
train_his_transactions_mer=train_his_transactions_mer.drop(['category_4'], axis=1)
train_his_transactions_mer=pd.concat([train_his_transactions_mer, category_4], axis=1)
train_his_transactions_mer.head()
with open('train_his_transactions_mer.pickle', 'wb') as f:
    pickle.dump(train_his_transactions_mer, f)
with open('train_his_transactions_mer.pickle', 'rb') as f:
    train_his_transactions_mer = pickle.load(f)
train_his_transactions_mer.head()
#inner join (some training card id is not in new_transactions,delete these)
train_new_transactions=train[["card_id"]].merge(new_transactions, how='inner', on="card_id")
print(train_new_transactions.shape)
train_new_transactions.head()
train_new_transactions.columns=["new_trans_" + str(col) for col in train_new_transactions.columns]
merchants.columns=["merchants_" + str(col) for col in merchants.columns] 
#drop merchants with same merchant id,return the first one
#merchants.drop_duplicates(subset="merchants_merchant_id",keep="first",inplace=True)
merchants.head()
merge_new_trans_merchants=pd.merge(train_new_transactions,merchants,how="left",left_on="new_trans_merchant_id",right_on="merchants_merchant_id")
merge_new_trans_merchants['new_trans_authorized_flag'] = merge_new_trans_merchants['new_trans_authorized_flag'].apply(lambda x: 1 if x == 'Y' else 0)
merge_new_trans_merchants['new_trans_category_1'] = merge_new_trans_merchants['new_trans_category_1'].apply(lambda x: 1 if x == 'Y' else 0)
autorized_card_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['new_trans_authorized_flag'].mean()
new_trans_cate_1_rate_Y = merge_new_trans_merchants.groupby(['new_trans_card_id'])['new_trans_category_1'].mean()
#create dummy variable for new_trans_category_2
merge_new_trans_merchants['new_trans_category_2_1'] = merge_new_trans_merchants['new_trans_category_2'].apply(lambda x: 1 if x == 1 else 0)
merge_new_trans_merchants['new_trans_category_2_2'] = merge_new_trans_merchants['new_trans_category_2'].apply(lambda x: 1 if x == 2 else 0)
merge_new_trans_merchants['new_trans_category_2_3'] = merge_new_trans_merchants['new_trans_category_2'].apply(lambda x: 1 if x == 3 else 0)
merge_new_trans_merchants['new_trans_category_2_4'] = merge_new_trans_merchants['new_trans_category_2'].apply(lambda x: 1 if x == 4 else 0)
merge_new_trans_merchants['new_trans_category_2_5'] = merge_new_trans_merchants['new_trans_category_2'].apply(lambda x: 1 if x == 5 else 0)
#calculate mean of each category in cate_2 group by card_id
cate_2_1_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['new_trans_category_2_1'].mean()
cate_2_2_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['new_trans_category_2_2'].mean()
cate_2_3_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['new_trans_category_2_3'].mean()
cate_2_4_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['new_trans_category_2_4'].mean()
cate_2_5_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['new_trans_category_2_5'].mean()
#create dummy variable for new_trans_category_3
merge_new_trans_merchants['new_trans_category_3_A'] = merge_new_trans_merchants['new_trans_category_3'].apply(lambda x: 1 if x == "A" else 0)
merge_new_trans_merchants['new_trans_category_3_B'] = merge_new_trans_merchants['new_trans_category_3'].apply(lambda x: 1 if x == "B" else 0)
merge_new_trans_merchants['new_trans_category_3_C'] = merge_new_trans_merchants['new_trans_category_3'].apply(lambda x: 1 if x == "C" else 0)

#calculate mean of each category in cate_3 group by card_id
cate_3_A_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['new_trans_category_3_A'].mean()
cate_3_B_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['new_trans_category_3_B'].mean()
cate_3_C_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['new_trans_category_3_C'].mean()

#create dummy variable for merchant_category_1
merge_new_trans_merchants['merchants_category_1'] = merge_new_trans_merchants['merchants_category_1'].apply(lambda x: 1 if x == "Y" else 0)

#calculate mean of each category in merchant_cate_1 group by card_id
merchant_cate_1_Y_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['merchants_category_1'].mean()

#create dummy variable for merchants_category_2
merge_new_trans_merchants['merchants_category_2_1'] = merge_new_trans_merchants['merchants_category_2'].apply(lambda x: 1 if x == 1 else 0)
merge_new_trans_merchants['merchants_category_2_2'] = merge_new_trans_merchants['merchants_category_2'].apply(lambda x: 1 if x == 2 else 0)
merge_new_trans_merchants['merchants_category_2_3'] = merge_new_trans_merchants['merchants_category_2'].apply(lambda x: 1 if x == 3 else 0)
merge_new_trans_merchants['merchants_category_2_4'] = merge_new_trans_merchants['merchants_category_2'].apply(lambda x: 1 if x == 4 else 0)
merge_new_trans_merchants['merchants_category_2_5'] = merge_new_trans_merchants['merchants_category_2'].apply(lambda x: 1 if x == 5 else 0)
#calculate mean of each category in merchant cate_2 group by card_id
merchants_cate_2_1_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['merchants_category_2_1'].mean()
merchants_cate_2_2_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['merchants_category_2_2'].mean()
merchants_cate_2_3_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['merchants_category_2_3'].mean()
merchants_cate_2_4_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['merchants_category_2_4'].mean()
merchants_cate_2_5_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['merchants_category_2_5'].mean()
merge_new_trans_merchants['merchants_category_4'] = merge_new_trans_merchants['merchants_category_4'].apply(lambda x: 1 if x == "Y" else 0)
merchant_cate_4_Y_rate = merge_new_trans_merchants.groupby(['new_trans_card_id'])['merchants_category_4'].mean()

new_trans_merchants_by_card_id={
    "authorized_card_rate":autorized_card_rate,
    "new_trans_category_1_rate_Y":new_trans_cate_1_rate_Y,
    "new_trans_category_2_1_rate":cate_2_1_rate,
    "new_trans_category_2_2_rate":cate_2_2_rate,
    "new_trans_category_2_3_rate":cate_2_3_rate,
    "new_trans_category_2_4_rate":cate_2_4_rate,
    "new_trans_category_2_5_rate":cate_2_5_rate,
    "new_trans_category_3_A_rate":cate_3_A_rate,
    "new_trans_category_3_B_rate":cate_3_B_rate,
    "new_trans_category_3_C_rate":cate_3_C_rate,
    "merchants_category_1_Y_rate":merchant_cate_1_Y_rate,
    "merchants_category_2_1_rate":merchants_cate_2_1_rate,
    "merchants_category_2_2_rate":merchants_cate_2_2_rate,
    "merchants_category_2_3_rate":merchants_cate_2_3_rate,
    "merchants_category_2_4_rate":merchants_cate_2_4_rate,
    "merchants_category_2_5_rate":merchants_cate_2_5_rate,
    "merchants_category_4_Y_rate":merchant_cate_4_Y_rate
}
new_trans_merchants_by_card_id_df=pd.DataFrame(new_trans_merchants_by_card_id)
new_trans_merchants_by_card_id_df=new_trans_merchants_by_card_id_df.reset_index()
new_trans_merchants_by_card_id_df.head()
with open('new_trans_merchants_by_card_id_df.pickle', 'wb') as f:
    pickle.dump(new_trans_merchants_by_card_id_df, f)
with open('new_trans_merchants_by_card_id_df.pickle', 'rb') as f:
    new_trans_merchants_by_card_id_df = pickle.load(f)
es = ft.EntitySet(id = 'card')
es = es.entity_from_dataframe(entity_id = 'train', dataframe = train,index = 'card_id')

es = es.entity_from_dataframe(entity_id = 'train_his_transactions_mer', 
                              dataframe = train_his_transactions_mer,
                              make_index = True,
                              index = "train_his_transactions_mer_id")
es = es.entity_from_dataframe(entity_id = 'new_trans_merchants_by_card_id_df', 
                              dataframe = new_trans_merchants_by_card_id_df,
                              make_index = True,
                              index = "new_trans_merchants_by_card_id_df_id")
new_trans_merchants_by_card_id_df.head()
train_his_transactions_mer.head()
train.head()
r_train_his = ft.Relationship(es['train']['card_id'],
                                   es['train_his_transactions_mer']['card_id'])
r_train_new = ft.Relationship(es['train']['card_id'],
                                    es['new_trans_merchants_by_card_id_df']['new_trans_card_id'])

# Add the relationship to the entity set
es = es.add_relationship(r_train_his)
es = es.add_relationship(r_train_new)
es
# features, feature_names = ft.dfs(entityset=es, target_entity='train', 
#                                  max_depth = 2,
#                                  agg_primitives = ['mean', 'max', 'percent_true', 'last'],
#                                  trans_primitives = ['years', 'month', 'subtract', 'divide'])

# with open('features.pickle', 'wb') as f:
#     pickle.dump([features, feature_names], f)

# features.head()
dic = pd.read_excel('../input/Data_Dictionary.xlsx', sheet_name='train')
dic
# with open('features.pickle', 'wb') as f:
#     pickle.dump([features, feature_names], f)
e = pd.read_excel('../input/Data_Dictionary.xlsx', sheet_name='history')
e
