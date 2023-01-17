# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import SimpleImputer, IterativeImputer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



        

df_train = pd.read_csv('/kaggle/input/amexpert2019/train.csv')

df_demo = pd.read_csv('/kaggle/input/amexpert2019/customer_demographics.csv')

df_coup = pd.read_csv('/kaggle/input/amexpert2019/coupon_item_mapping.csv')

df_cust = pd.read_csv('/kaggle/input/amexpert2019/customer_transaction_data.csv')

df_camp = pd.read_csv('/kaggle/input/amexpert2019/campaign_data.csv')

df_item_data = pd.read_csv('/kaggle/input/amexpert2019/item_data.csv')

df_test = pd.read_csv('/kaggle/input/amex-hack-2019/test_QyjYwdj.csv')

df_train



# Any results you write to the current directory are saved as output.
df_train['dtype'] = 'train'

df_test['dtype'] = 'test'

df_test['redemption_status'] = 0
def impute_family_size(df_family):

    df = df_family.copy() 

    df.loc[(df.family_size=="1") & (df.marital_status.isnull()), "marital_status"] = "Single"

    df.loc[(df.no_of_children.isnull()==False) & (df.marital_status.isnull()), "marital_status"] = "Married"

    df.loc[(df.age_range=="18-25") & (df.marital_status.isnull()), "marital_status"] = "Single"

    df.loc[(df.age_range!="18-25") & (df.marital_status.isnull()), "marital_status"] = "Married"

    df.loc[(df.family_size=="2") & (df.marital_status=="Married") & (df.no_of_children.isnull()), "no_of_children"] = "0"

    df.loc[(df.marital_status=="Single") & (df.no_of_children.isnull()), "no_of_children"] = "-1"

    return df
df_demo = impute_family_size(df_demo)
df_train_test = pd.concat([df_train, df_test])

df_train_test = df_train_test.sample(frac=0.1)
df_train_test.isnull().sum()
df_train_test = df_train_test.set_index("campaign_id").join(df_camp.set_index('campaign_id')).reset_index()

df_train_test = df_train_test.set_index("customer_id").join(df_demo.set_index('customer_id')).reset_index()

df_train_test = df_train_test.set_index("customer_id").join(df_cust.set_index('customer_id')).reset_index()
df_train_test.dropna(inplace=True)
df_train_test = df_train_test.set_index("item_id").join(df_item_data.set_index('item_id')).reset_index()
df_train_test.size
df_train_test.isnull().sum()
features = ['campaign_type', 'age_range', 'marital_status', 'rented', 'family_size',

       'no_of_children', 'income_bracket','quantity', 'selling_price', 'other_discount','coupon_discount', 'brand', 'brand_type','category' ]

label = "redemption_status"
df_train_test.head
df_train_test.isnull().sum()
df_train_test.loc[df_train_test.no_of_children.isnull(),"no_of_children"] = 0
impute_feature_cat= ['marital_status', 'age_range','brand_type','family_size', 'category','income_bracket', 'no_of_children']
df_train_test
simple_imputer_cat = SimpleImputer(strategy='most_frequent')
def simple_impute(df, imputer, cols, test=False):

    df = df.copy()

    df['family_size'] = df['family_size'].apply(lambda x: int(x.strip('+')) if isinstance(x, str) else x)    

    df['no_of_children'] = df['no_of_children'].apply(lambda x: int(x.strip('+')) if isinstance(x, str) else x)

    if not test:

        imputer.fit(df[cols])

    df[cols] = imputer.transform(df[cols])

    return df
df_train_test = impute_family_size(df_train_test)
# df_train_test = simple_impute(df_train_test, simple_imputer_cat, impute_feature_cat)#
df_train_test
orde = OrdinalEncoder()
def encode(df, encoder, cols, test=False):

    df = df.copy()

    df['family_size'] = df['family_size'].apply(lambda x: int(x.strip('+')) if isinstance(x, str) else x)    

    df['no_of_children'] = df['no_of_children'].apply(lambda x: int(x.strip('+')) if isinstance(x, str) else x)

    if not test:

        encoder.fit(df[cols])

    df[cols] = encoder.transform(df[cols])

    return df
df_train_test = encode(df_train_test, orde, ['marital_status', 'age_range', 'campaign_type', 'brand_type', 'category'])
df_split_test  = df_train_test[df_train_test.dtype == "test"].reset_index()

df_split_train  = df_train_test[df_train_test.dtype == "train"].reset_index()



df_split_test = df_split_test.drop(columns=['redemption_status']).reset_index()
df_split_train
x_train, x_valid,y_train,y_valid = train_test_split(df_split_train[features],df_split_train[label], test_size=0.10, random_state=42)
model = DecisionTreeClassifier()

clf = model.fit(x_train,y_train)
print("train score:", clf.score(x_train,y_train))

y_predict = clf.predict(x_valid)

clf.score(x_valid,y_valid)
df_split_test[label] = clf.predict(df_split_test[features])
df_split_test[['id',label]].to_csv("submission.csv", index=False)