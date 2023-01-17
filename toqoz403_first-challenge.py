import math

import gc

import numpy as np

import pandas as pd



submit = True
# NOTE: https://qiita.com/hadacchi/items/ff6528364fed2404eef0

# NOTE: specify dtype for memory...

def read(ic, file_name):

    print('read ' + file_name)



    base_dir = '../input/'

    dtype = {ic: 'uint32', 'shipping': 'uint8', 'price': 'float16', 'item_condition_id': 'uint8'}

    return pd.read_table(base_dir + file_name, index_col=[ic], dtype=dtype)



df_train = read('train_id', 'train.tsv')

df_test = read('test_id', 'test.tsv')
def process_texts(df):

    print('process_text')



    #df['has_rm'] = df['name'].apply(lambda x: 1 if '[rm]' in x else 0).astype('int8')



    #df['has_description'] = df['item_description'].apply(lambda x: 1 if x != 'No description yet' else 0).astype('int8')

    df.drop(['name', 'item_description'], axis=1, inplace=True)

    return df



df_train = process_texts(df_train)

df_test = process_texts(df_test)

gc.collect()
brand_name_master = {}

brand_names = df_train['brand_name'].value_counts().where(lambda x : x > 10).dropna().index

for _, b in enumerate(brand_names):

    brand_name_master[b] = b

del brand_names



def process_brand_name(df):

    print('process_brand_name')



    # NOTE: NaN == NaN -> false

    df['has_brand_name'] = df['brand_name'].apply(lambda x: 1 if x == x else 0).astype('int8')

    df['brand_name_m'] = df['brand_name'].apply(lambda x: brand_name_master.get(x, np.nan))

    df.drop(['brand_name'], axis=1, inplace=True)

    return df



df_train = process_brand_name(df_train)

df_test = process_brand_name(df_test)



del brand_name_master

gc.collect()
def process_category_name(df):

    print('process_category_name')



    return df



df_train = process_category_name(df_train)

df_test = process_category_name(df_test)
def process_categorical(df):

    print('process_categorical')



    df = pd.get_dummies(df, columns=['item_condition_id', 'category_name'], drop_first=True)

    gc.collect()

    

    # DON'T drop_first

    df = pd.get_dummies(df, columns=['brand_name_m'])

    gc.collect()



    return df



df_train = process_categorical(df_train)

gc.collect()

print(df_train.shape)

df_test = process_categorical(df_test)

gc.collect()

print(df_test.shape)
y = df_train['price'].as_matrix()

df_train.drop(['price'], axis=1, inplace=True)

gc.collect()
print('align train.tsv & test.tsv')

# https://www.kaggle.com/dansbecker/using-categorical-data-with-one-hot-encoding

df_train, df_test = df_train.align(df_test, join='left', axis=1, fill_value=0)

gc.collect()
train_columns = df_train.columns

print('generate sparse matrix X')

X = df_train.to_sparse(fill_value=0).to_coo()

print('generate sparse matrix T')

T = df_test.to_sparse(fill_value=0).to_coo()
from sklearn import linear_model

from sklearn.model_selection import train_test_split



def rmsle(a, p):

    n = len(a)

    assert len(p) == n

    terms_to_sum = [(math.log(max(p[i-1], 0) + 1) - math.log(a[i-1] + 1)) ** 2.0 for i in range (1, n)]

    return (sum(terms_to_sum) * (1.0/n)) ** 0.5



clf = linear_model.LinearRegression()

if submit:

    print('fit')

    %timeit clf.fit(X, np.log1p(y))

else:

    print('train_test_split')

    X_train, X_test, y_train, a = train_test_split(X, y, test_size=0.2)

    print('fit')

    %timeit clf.fit(X_train, np.log1p(y_train))

    print('predict')

    p = clf.predict(X_test)

    print('calculate score')

    score = rmsle(a, np.expm1(p))

    print('v-score: ' + str(score))



print("=== Coefficients ===")

print(pd.DataFrame({

  "Name": train_columns,

  "Coefficients": clf.coef_

}).sort_values(by='Coefficients'))



print("=== Intercept ===")

print(clf.intercept_)
file_name = 'submission.csv'



if submit:

    result = clf.predict(T)

    result_df = pd.DataFrame({'test_id': df_test.index, 'price': np.expm1(result)}, columns=['test_id', 'price'])

    result_df['price'] = result_df['price'].where(result_df['price'] >= 0, 0)

    result_df.to_csv(file_name, index=False)

    print('see ' + file_name)