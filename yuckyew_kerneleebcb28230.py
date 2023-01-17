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
# データセットの都合上、encodingを"ISO-8859-1"にしている

log1_df = pd.read_csv('/kaggle/input/nasa-access-log-dataset/log_1.tsv', delimiter='\t', encoding="ISO-8859-1")

log2_df = pd.read_csv('/kaggle/input/nasa-access-log-dataset/log_2.tsv', delimiter='\t', encoding="ISO-8859-1")
print('log_1.tsv shape: {}'.format(log1_df.shape))

print('log_2.tsv shape: {}'.format(log2_df.shape))
# 10000件をランダムサンプリング

nb_samples = 10000

sampled_df = log1_df.sample(nb_samples).reset_index()



# ユーザー列がないので、APIをコールしたユーザーが100名いたと仮定する

sampled_df['userid'] = pd.Series(np.random.randint(0,100, size=nb_samples))
# カラム名一覧とそれぞれのカラムに存在するUniqueな値の数

print('Column Names: ', sampled_df.columns)

print('\n---')

for col in sampled_df.columns:

    nb_uni_vals = len(sampled_df[col].unique())

    nb_nan_vals = sampled_df[col].isnull().sum()

    print('Column {} に含まれるUniqueな値の数： {}'.format(col, nb_uni_vals))

    print('Column {} の欠損値の数: {}'.format(col, nb_nan_vals))

    print('---')
# valuecountsメソッドからplotを行うことで、コールされているAPIの種別を多い順に可視化

# 多い順から50種可視化

sampled_df.url.value_counts()[:50].plot.bar(figsize=(15,5))
user_df = None

for name, group_df in sampled_df.groupby('userid'):

    uri_counts = group_df['url'].value_counts()

    agg_row = uri_counts.T

    agg_row = agg_row.rename(index=name)

    agg_row['count'] = len(group_df)

    

    if user_df is None:

        user_df = pd.DataFrame([agg_row])

    else:

        user_df = user_df.append(agg_row, ignore_index=True)

        

    user_df = user_df.fillna(0)
user_df