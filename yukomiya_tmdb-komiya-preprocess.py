# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from matplotlib import pyplot as plt
train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv', index_col=0)

train.info()
test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv', index_col=0)

test.info()
df = pd.concat([train, test])
df["log_revenue"] = np.log10(df["revenue"])
train.head().loc[:, :"poster_path"]
train.head().loc[:, "production_companies":]
# 使わない列を消す

df = df.drop(["imdb_id", "poster_path", "status", "overview", "original_title"], axis=1)
# homepage は、有るか無いかにする

df["homepage"] = ~df["homepage"].isnull()
import re

import ast
dfdic_feature = {}
# 文字列から、"name" 情報を抽出しリストに

def to_name_list(text):

    txt_list = re.sub("[\[\]]", "", text).replace("}, {", "}|{").split("|")

    return [ ast.literal_eval(txt)["name"] for txt in txt_list ]



def to_id_list(text):

    txt_list = re.sub("[\[\]]", "", text).replace("}, {", "}|{").split("|")

    return [ ast.literal_eval(txt)["id"] for txt in txt_list ]
df["genre_names"] = df["genres"].fillna("[{'name': 'nashi'}]").map(to_name_list)
# 各ワードの有無を表す 01 のデータフレームを作成

def count_word_list(series):

    len_max = series.apply(len).max() # ジャンル数の最大値

    tmp = series.map(lambda x: x+["nashi"]*(len_max-len(x))) # listの長さをそろえる

    

    word_set = set(sum(list(series.values), [])) # 全ジャンル名のset

    for n in range(len_max):

        word_dfn = pd.get_dummies(tmp.apply(lambda x: x[n]))

        word_dfn = word_dfn.reindex(word_set, axis=1).fillna(0).astype(int)

        if n==0:

            word_df = word_dfn

        else:

            word_df = word_df + word_dfn

    

    return word_df.drop("nashi", axis=1)
dfdic_feature["genre"] = count_word_list(df["genre_names"])
# 各ジャンルの作品数

dfdic_feature["genre"].sum().sort_values(ascending=False)
# 各言語の作品数

df["original_language"].value_counts()
# train内の作品数が10件未満の言語は "small" に集約

n_language = df.loc[:train.index[-1], "original_language"].value_counts()

small_language = n_language[n_language<10].index

df.loc[df["original_language"].isin(small_language), "original_language"] = "small"
# one_hot_encoding

dfdic_feature["original_language"] = pd.get_dummies(df["original_language"])

dfdic_feature["original_language"].head()
df["production_companies"]
# 複数社で製作のケースもある

df["production_companies"][1]
df["production_names"] = df["production_companies"].fillna("[{'name': 'nashi'}]").map(to_name_list)
tmp = count_word_list(df["production_names"])
# 順位:作品数

tmp.sum().sort_values(ascending=False).reset_index(drop=True).plot(loglog=True)

plt.xlabel("rank")

plt.ylabel("number of movies")
tmp.loc[:3000].sum().sort_values(ascending=False).head(20)
# train内の件数が多い物のみ選ぶ

def select_top_n(df, topn=9999, nmin=2):  # topn:上位topn件, nmin:作品数nmin以上

    if "small" in df.columns:

        df = df.drop("small", axis=1)

    n_word = (df.loc[train.index]>0).sum().sort_values(ascending=False)

    # 作品数がnmin件未満

    smallmin = n_word[n_word<nmin].index

    # 上位topn件に入っていない

    smalln = n_word.iloc[topn+1:].index

    small = set(smallmin) | set(smalln)

    # 件数の少ないタグのみの作品

    df["small"] = (df[small].sum(axis=1)>0)*1

    

    return df.drop(small, axis=1)
# trainに50本以上作品のある会社

dfdic_feature["production_companies"] = select_top_n(tmp, nmin=50)

dfdic_feature["production_companies"].head()
# revenue との相関

dfdic_feature["production_companies"].corrwith(df["log_revenue"]).sort_values()

# 全ての会社が正の相関。 ->　欠測だと低い。多数の会社が参加すると高い?
df["production_countries"]
# 複数国のケース

df["production_countries"][7394]
# 欠損

df["production_countries"].isnull().sum()
# 国名のリストに

df["country_names"] = df["production_countries"].str.replace("United States of America", "USA").fillna("[{'name': 'nashi'}]").map(to_name_list)
df_country = count_word_list(df["country_names"])
# 国別製作本数ランキング

df_country.sum().sort_values(ascending=False).head(10)
df_country.sum().sort_values(ascending=False).reset_index(drop=True).plot(loglog=True)

plt.xlabel("rank")

plt.ylabel("number of movies")
# 2か国だったら、0.5ずつに

df_country = (df_country.T/df_country.sum(axis=1)).T.fillna(0)
df_country = select_top_n(df_country, nmin=30)
dfdic_feature["production_countries"] = df_country
dfdic_feature["production_countries"].head()
# revenue との相関

dfdic_feature["production_countries"].corrwith(df["log_revenue"]).sort_values()

df["Keywords"]
df["keyword_list"] = df["Keywords"].fillna("[{'name': 'nashi'}]").map(to_name_list)
# キーワードが149個ある作品も

df["keyword_list"].apply(len).max()
# 全キーワードの種類

keyword_set = set(sum(list(df["keyword_list"].values), []))

len(keyword_set)
import collections

# 多いキーワードtop20

keyword_count = pd.Series(collections.Counter(sum(list(df["keyword_list"].values), [])))

keyword_count = keyword_count.sort_values(ascending=False)

keyword_count.head(20)
keyword_count.sort_values(ascending=False).reset_index(drop=True).plot(loglog=True)

plt.xlabel("rank")

plt.ylabel("number of movies")
# keyword上位100

keyword_count.iloc[:101]

# nashi は欠損を置き換えたもの
df_keyword = df[[]].copy()

# 上位１００件のキーワードのみ

for word in keyword_count.drop("nashi").iloc[:100].index:

    df_keyword[word] = df["keyword_list"].apply(lambda x: word in x)*1

dfdic_feature["Keywords"] = df_keyword
df_keyword.head()
# revenue との相関

dfdic_feature["Keywords"].corrwith(df["log_revenue"]).sort_values()
df[["original_language","spoken_languages"]].head(30)
def to_tag_list(text, tag):

    txt_list = re.sub("[\[\]]", "", text).replace("}, {", "}|{").split("|")

    return [ ast.literal_eval(txt)[tag] for txt in txt_list ]



# df["language_names"] = df["spoken_languages"].fillna("[{'iso_639_1': 'nashi'}]").apply(to_tag_list, tag = 'iso_639_1')

# df_spklanguage = count_word_list(df["language_names"])
df["language_names"] = df["spoken_languages"].fillna("[{'iso_639_1': 'nashi'}]").apply(to_tag_list, tag = 'iso_639_1')

# 欠損値は１になる

df["n_language"] = df["language_names"].apply(len)
# revenue との相関

df["n_language"].corr(df["log_revenue"])
# 複数のシリーズに属していることはない

assert df["belongs_to_collection"].fillna("[{'name': ''}]").map(to_name_list).map(len).max()==1
df["collection_name"] = df["belongs_to_collection"].fillna("[{'name': 'nashi'}]").map(to_name_list).map(lambda x: x[0])
df["collection_name"].value_counts().head(10)
dfdic_feature["belongs_to_collection"] = pd.get_dummies(df["collection_name"])
# revenue との相関

dfdic_feature["belongs_to_collection"].corrwith(df["log_revenue"]).sort_values().dropna()
collection_av = df.groupby("collection_name").mean()[["log_revenue"]].dropna()

collection_av.columns = ["collection_av_logrevenue"]
collection_av.sort_values("collection_av_logrevenue", ascending=False)
df = df.merge(collection_av, on="collection_name", how="left")

nashi_mean = df.loc[df["collection_name"]=="nashi","collection_av_logrevenue"].mean()

df["collection_av_logrevenue"] = df["collection_av_logrevenue"].fillna(nashi_mean)  # train に無いシリーズの場合、シリーズ無しと同じにする

#df = df.rename(columns={"log_revenue":"collection_mean_log_revenue"})
df_features = pd.concat(dfdic_feature, axis=1)
df_features.shape
df_features.head()
df_features.isnull().sum().sum()
df.isnull().sum()
# 平均で埋める

df["runtime"] = df["runtime"].fillna(df["runtime"].mean())
df[df["release_date"].isnull()]["title"]
# May,2000 (https://www.imdb.com/title/tt0210130/) 

# 日は不明。1日を入れておく

df.loc[3828, "release_date"] = "5/1/00"
df["release_date"]
df["release_year"] = pd.to_datetime(df["release_date"]).dt.year

df.loc[df["release_year"]>2020, "release_year"] = df.loc[df["release_year"]>2020, "release_year"]-100



df["release_month"] = pd.to_datetime(df["release_date"]).dt.month

df["release_day"] = pd.to_datetime(df["release_date"]).dt.day



df["release_dayofyear"] = pd.to_datetime(df["release_date"]).dt.dayofyear

df["release_dayofweek"] = pd.to_datetime(df["release_date"]).dt.dayofweek
df.groupby("release_year").mean()["revenue"].plot()

plt.ylabel("mean revenie")
df.groupby("release_year").mean()["log_revenue"].plot()

plt.ylabel("mean log(revenie)")
df.groupby("release_month").mean()["log_revenue"].plot()

plt.ylabel("mean log(revenie)")
df.groupby("release_dayofyear").mean()["log_revenue"].plot()

plt.ylabel("mean log(revenie)")
df.groupby("release_dayofweek").mean()["log_revenue"].plot()

plt.ylabel("mean log(revenie)")
df.groupby("release_day").mean()["log_revenue"].plot()

plt.ylabel("mean log(revenie)")
df.isnull().sum()
df["budget"]
df["release_month"] = df["release_month"].astype('category')

df["release_dayofweek"] = df["release_dayofweek"].astype('category')
plt.scatter(df["budget"]+1, df["log_revenue"], s=1)

plt.xscale("log")

#plt.xrange([])
df.loc[df["budget"]==0, "log_revenue"].hist()

plt.xlabel("log(revenue)")

plt.ylabel("number")
df.isnull().sum()
df[["original_language", "collection_name"]] = df[["original_language", "collection_name"]].astype("category")
df.columns
df_use = df[['budget', 'homepage', 'popularity','runtime','n_language', 'collection_av_logrevenue',

             'release_year', 'release_month','release_dayofweek']]

df_use
df_use = pd.get_dummies(df_use)
df_use.index = df_features.index
df_input = pd.concat([df_use, df_features.drop("belongs_to_collection", axis=1)], axis=1)
df_input.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error 

df["ln_revenue"] = np.log(df["revenue"]+1)
train_X, val_X, train_y, val_y, train_rev, val_rev = train_test_split(df_input[:3000], 

                                                  df.loc[:3000, "ln_revenue"], 

                                                  df.loc[:3000, "revenue"], 

                                                  test_size=0.25)
from sklearn.linear_model import Lasso, Ridge
clf = Lasso(alpha=0.1)  # default alpha=1

clf.fit(train_X, train_y)
coef = pd.Series(clf.coef_, index=train_X.columns)

coef[coef!=0]
val_pred = clf.predict(val_X)

np.sqrt(mean_squared_error(val_pred, val_y))