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
from surprise import Reader, Dataset, SVD

from surprise.model_selection import cross_validate
#movies

mt = pd.read_csv('../input/netflix-prize-data/movie_titles.csv',encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'])

mt.set_index('Movie_Id', inplace = True)

mt.shape
#combined_data_1.txt

#ps = pd.read_csv('../input/netflix-prize-data/probe.txt')

df = pd.read_csv('../input/netflix-prize-data/combined_data_1.txt',header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])

df.shape
# 处理评分为NAN的数据

df_nan = pd.DataFrame(pd.isnull(df.Rating))

df_nan = df_nan[df_nan['Rating'] == True]

df_nan = df_nan.reset_index()
movie_np = []

movie_id = 1



for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):

    # numpy approach

    temp = np.full((1,i-j-1), movie_id)

    movie_np = np.append(movie_np, temp)

    movie_id += 1
last_record = np.full((1,len(df) - df_nan.iloc[-1, 0] - 1),movie_id)

movie_np = np.append(movie_np, last_record)

movie_np
len(movie_np)
# remove those Movie ID rows

df = df[pd.notnull(df['Rating'])]

df['Movie_Id'] = movie_np.astype(int)

df['Cust_Id'] = df['Cust_Id'].astype(int)



df
# 数据切分

# 依据别人的kernel思路，

#删除评论过少的电影（它们相对不受欢迎）

#删除评论量过少的客户（他们相对不太活跃）



f = ['count','mean']



df_movie_summary = df.groupby('Movie_Id')['Rating'].agg(f)

df_movie_summary.index = df_movie_summary.index.map(int)

movie_benchmark = round(df_movie_summary['count'].quantile(0.7),0)

drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index



print('Movie minimum times of review: {}'.format(movie_benchmark))



df_cust_summary = df.groupby('Cust_Id')['Rating'].agg(f)

df_cust_summary.index = df_cust_summary.index.map(int)

cust_benchmark = round(df_cust_summary['count'].quantile(0.7),0)

drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index



print('Customer minimum times of review: {}'.format(cust_benchmark))
print('Original Shape: {}'.format(df.shape))

df = df[~df['Movie_Id'].isin(drop_movie_list)]

df = df[~df['Cust_Id'].isin(drop_cust_list)]

print('After Trim Shape: {}'.format(df.shape))
reader = Reader()



# 取300k的数据

data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']][:300000], reader)



svd = SVD()

perf = cross_validate(svd, data, measures=['RMSE'], cv=3)

print(perf)