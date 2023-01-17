# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

sample_submission_stg2 = pd.read_csv("../input/mercari-price-suggestion-challenge/sample_submission_stg2.csv")

train = pd.read_table("../input/mercari-price-suggestion-challenge/train.tsv")

test = pd.read_table("../input/mercari-price-suggestion-challenge/test.tsv")

test_stg2 = pd.read_table("../input/mercari-price-suggestion-challenge/test_stg2.tsv")
# trainとtestのidカラム名を変更する

train = train.rename(columns = {'train_id':'id'})

test = test_stg2.rename(columns = {'test_id':'id'})

 

# 両方のセットへ「is_train」のカラムを追加

# 1 = trainのデータ、0 = testデータ

train['is_train'] = 1

test['is_train'] = 0

 

# trainのprice(価格）以外のデータをtestと連結

train_test_combine = pd.concat([train.drop(['price'], axis=1),test],axis=0)

 

# 念のためデータの中身を表示させましょう

train_test_combine.head()
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

#train_test_combine.head()

train_test_combine = train_test_combine.astype('i4')

train_test_combine.dtypes
# 「is_train」のフラグでcombineからtestとtrainへ切り分ける

df_test = train_test_combine.loc[train_test_combine['is_train'] == 0]

df_train = train_test_combine.loc[train_test_combine['is_train'] == 1]

 

# 「is_train」をtrainとtestのデータフレームから落とす

df_test = df_test.drop(['is_train'], axis=1)

df_train = df_train.drop(['is_train'], axis=1)

 

# サイズの確認をしておきましょう

df_test.shape, df_train.shape
# x ＝ price以外の全ての値、y = price（ターゲット）で切り分ける

x_train, y_train = df_train, train.price

# モデルの作成

#m = RandomForestRegressor(n_jobs=-1, min_samples_leaf=5, n_estimators=200)

m = RandomForestRegressor()

m.fit(x_train, y_train)

# スコアを表示

m.score(x_train, y_train)
preds = m.predict(df_test)

preds = pd.Series(preds)

submit = pd.concat([sample_submission_stg2.test_id, preds], axis=1)

submit.columns = ['test_id', 'price']

# 提出ファイルとしてCSVへ書き出し

submit.to_csv('submission.csv', index=False)