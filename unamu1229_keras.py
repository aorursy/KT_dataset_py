# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/mercari-price-suggestion-challenge"))



# Any results you write to the current directory are saved as output.
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from IPython.display import display

from sklearn import metrics

from sklearn.model_selection import train_test_split

pd.set_option('display.float_format', lambda x:'%.5f' % x)

import numpy as np

from keras.models import Sequential

from keras.layers.core import Dense, Activation, Dropout

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping
# データタイプを指定

types_dict_train = {'train_id':'int64', 'item_condition_id':'int8', 'price':'float64', 'shipping':'int8'}

types_dict_test = {'test_id':'int64', 'item_condition_id':'int8', 'shipping':'int8'}

 

# tsvファイルからPandas DataFrameへ読み込み

train = pd.read_csv('../input/mercari-price-suggestion-challenge/train.tsv', delimiter='\t', low_memory=True, dtype=types_dict_train)

test = pd.read_csv('../input/mercari-price-suggestion-challenge/test.tsv', delimiter='\t', low_memory=True, dtype=types_dict_test)

stg = pd.read_csv('../input/mercari-price-suggestion-challenge/test_stg2.tsv', delimiter='\t', low_memory=True, dtype=types_dict_test)
# trainのカテゴリ名、商品説明、投稿タイトル、ブランド名のデータタイプを「category」へ変換する

train.category_name = train.category_name.astype('category')

train.item_description = train.item_description.astype('category')

train.name = train.name.astype('category')

train.brand_name = train.brand_name.astype('category')

 

# testのカテゴリ名、商品説明、投稿タイトル、ブランド名のデータタイプを「category」へ変換する

test.category_name = test.category_name.astype('category')

test.item_description = test.item_description.astype('category')

test.name = test.name.astype('category')

test.brand_name = test.brand_name.astype('category')



# stgのカテゴリ名、商品説明、投稿タイトル、ブランド名のデータタイプを「category」へ変換する

stg.category_name = stg.category_name.astype('category')

stg.item_description = stg.item_description.astype('category')

stg.name = stg.name.astype('category')

stg.brand_name = stg.brand_name.astype('category')

 

# dtypesで念のためデータ形式を確認しましょう

train.dtypes, test.dtypes, stg.dtypes
train = train.rename(columns={'train_id':'id'})

test = test.rename(columns={'test_id':'id'})



# 両方のセットへ「is_train」のカラムを追加

# 1 = trainのデータ、0 = testデータ

train['is_train'] = 1

test['is_train'] = 0



# trainのprice(価格）以外のデータをtestと連結

train_test_combine = pd.concat([train.drop(['price'], axis=1),test],axis=0)

 

# 念のためデータの中身を表示させましょう

train_test_combine.head()
stg.name = stg.name.cat.codes

stg.category_name = stg.category_name.cat.codes

stg.brand_name = stg.brand_name.cat.codes

stg.item_description = stg.item_description.cat.codes

stg.head()
# train_test_combineの文字列のデータタイプを「category」へ変換

train_test_combine.category_name = train_test_combine.category_name.astype('category')

train_test_combine.item_description = train_test_combine.item_description.astype('category')

train_test_combine.name = train_test_combine.name.astype('category')

train_test_combine.brand_name = train_test_combine.brand_name.astype('category')

# combinedDataの文字列を「.cat.codes」で数値へ変換する

train_test_combine.name = train_test_combine.name.cat.codes

train_test_combine.category_name = train_test_combine.category_name.cat.codes

train_test_combine.brand_name = train_test_combine.brand_name.cat.codes

train_test_combine.item_description = train_test_combine.item_description.cat.codes

# データの中身とデータ形式を表示して確認しましょう

train_test_combine.head(), train_test_combine.dtypes
# 「is_train」のフラグでcombineからtestとtrainへ切り分ける

df_test = train_test_combine.loc[train_test_combine['is_train'] == 0]

df_train = train_test_combine.loc[train_test_combine['is_train'] == 1]

# 「is_train」をtrainとtestのデータフレームから落とす

df_test = df_test.drop(['is_train'], axis=1)

df_train = df_train.drop(['is_train'], axis=1)

# サイズの確認をしておきましょう

df_test.shape, df_train.shape
# df_trainへprice（価格）を戻す

df_train['price'] = train.price

 

# df_trainを表示して確認

df_train.head()
# price（価格）をlog関数で処理

df_train['price'] = df_train['price'].apply(lambda x: np.log(x) if x>0 else x)

# df_trainを表示して確認

df_train.head()

# x ＝ price以外の全ての値、y = price（ターゲット）で切り分ける

x_train, y_train = df_train.drop(['price'], axis=1), df_train.price
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=(int(x_train.shape[0] * 0.1)))
'''

モデル設定

'''

def weight_variable(shape, name=None):

    return np.random.normal(scale=.01, size=shape)



model = Sequential()



model.add(Dense(500, input_dim=7, kernel_initializer=weight_variable))

model.add(Activation('relu'))

model.add(Dropout(0.5))



model.add(Dense(500, input_dim=500, kernel_initializer=weight_variable))

model.add(Activation('relu'))

model.add(Dropout(0.5))



model.add(Dense(500, input_dim=500, kernel_initializer=weight_variable))

model.add(Activation('relu'))

model.add(Dropout(0.5))



model.add(Dense(1, kernel_initializer=weight_variable))

model.add(Activation('linear'))



optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999))
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)



'''

モデル学習

'''

epochs = 50

batch_size = 10000



model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          validation_data=(x_validation, y_validation),

          callbacks=[early_stopping])
preds = model.predict(stg)

preds = pd.Series(np.exp(preds.reshape([1, preds.shape[0]])[0]))
preds.head()
submit = pd.concat([stg.test_id, preds], axis=1)

submit.columns = ['test_id', 'price']

submit.to_csv('submit_stg.csv', index=False)
submit.head()