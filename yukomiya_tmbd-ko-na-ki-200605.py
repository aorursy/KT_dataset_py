import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import ast

import json

import collections

from collections import Counter



import string

#from janome.tokenizer import Tokenizer

import re

from nltk.corpus import stopwords



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error 



%precision 3

pd.set_option('precision', 3)



import warnings

warnings.filterwarnings('ignore')
#データを読み取る

#

train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv', index_col=0)

#

test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv', index_col=0)
train.info()
test.info()
df = pd.concat([train, test])
df["log_revenue"] = np.log10(df["revenue"])
print(train.shape,test.shape)

train.columns
#columnsを確認し、除外する変数をdrop

print(df.columns)

# 使わない列を消す

df = df.drop(["imdb_id", "poster_path", "status", "overview", "original_title"], axis=1)
df["homepage"] = ~df["homepage"].isnull()
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
dfdic_feature["genre"] = dfdic_feature["genre"].drop("TV Movie", axis=1)
df.loc[2997]
# 各言語の作品数

df["original_language"].value_counts()
# train内の作品数が10件未満の言語は "small" に集約

n_language = df.loc[:train.index[-1], "original_language"].value_counts()

large_language = n_language[n_language>=10].index

df.loc[~df["original_language"].isin(large_language), "original_language"] = "small"
# one_hot_encoding

dfdic_feature["original_language"] = pd.get_dummies(df["original_language"])

dfdic_feature["original_language"] = dfdic_feature["original_language"].loc[:, dfdic_feature["original_language"].sum()>0]

dfdic_feature["original_language"].head()
df["original_language"]
# 複数社で製作のケースもある

df["production_companies"][1]
df["production_names"] = df["production_companies"].fillna("[{'name': 'nashi'}]").map(to_name_list)
tmp = count_word_list(df["production_names"])
# 順位:作品数

tmp.sum().sort_values(ascending=False).reset_index(drop=True).plot(loglog=True)

plt.xlabel("rank")

plt.ylabel("number of movies")
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
# revenue との相関

dfdic_feature["production_countries"].corrwith(df["log_revenue"]).sort_values()
df["keyword_list"] = df["Keywords"].fillna("[{'name': 'nashi'}]").map(to_name_list)
# キーワードが149個ある作品も

df["keyword_list"].apply(len).max()
# 全キーワードの種類

keyword_set = set(sum(list(df["keyword_list"].values), []))

len(keyword_set)
# 多いキーワードtop20

keyword_count = pd.Series(collections.Counter(sum(list(df["keyword_list"].values), [])))

keyword_count = keyword_count.sort_values(ascending=False)

keyword_count.head(20)
keyword_count.sort_values(ascending=False).reset_index(drop=True).plot(loglog=True)

plt.xlabel("rank")

plt.ylabel("number of movies")
df_keyword = df[[]].copy()

# 上位１００件のキーワードのみ

for word in keyword_count.drop("nashi").iloc[:100].index:

    df_keyword[word] = df["keyword_list"].apply(lambda x: word in x)*1



dfdic_feature["Keywords"] = df_keyword
# revenue との相関

dfdic_feature["Keywords"].corrwith(df["log_revenue"]).sort_values()
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
dfdic_feature["belongs_to_collection"] = pd.get_dummies(df["collection_name"])

dfdic_feature["belongs_to_collection"] = dfdic_feature["belongs_to_collection"].loc[:, dfdic_feature["belongs_to_collection"][:3000].sum()>0]
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
# 平均で埋める

df["runtime"] = df["runtime"].fillna(df["runtime"].mean())
# May,2000 (https://www.imdb.com/title/tt0210130/) 

# 日は不明。1日を入れておく

df.loc[3828, "release_date"] = "5/1/00"
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
df["release_month"] = df["release_month"].astype('category')

df["release_dayofweek"] = df["release_dayofweek"].astype('category')
plt.scatter(df["budget"]+1, df["log_revenue"], s=1)

plt.xscale("log")

#plt.xrange([])
df.loc[df["budget"]==0, "log_revenue"].hist()

plt.xlabel("log(revenue)")

plt.ylabel("number")
df[["original_language", "collection_name"]] = df[["original_language", "collection_name"]].astype("category")
df_use = df[['budget', 'homepage', 'popularity','runtime','n_language', 'collection_av_logrevenue',

             'release_year', 'release_month','release_dayofweek']]

df_use
df_use = pd.get_dummies(df_use)
#単語数

df['tagline_word_count'] = df['tagline'].apply(lambda x: len(str(x).split()))

#文字数

df['tagline_char_count'] = df['tagline'].apply(lambda x: len(str(x)))

# 記号の個数

df['tagline_punctuation_count'] = df['tagline'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
df['tagline']=df['tagline'].apply(lambda x : str(x))
#全て小文字に変換

def lower_text(text):

    return text.lower()

df['tagline']=df['tagline'].apply(lambda x : lower_text(x))

#記号の排除

def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)

df['tagline']=df['tagline'].apply(lambda x : remove_punct(x))

def most_common(docs, n=100):#(文章、上位n個の単語)#上位n個の単語を抽出

    fdist = Counter()

    for word in docs:      

            fdist[word] += 1

    common_words = {word for word, freq in fdist.most_common(n)}

    #print('{}/{}'.format(n, len(fdist)))

    return common_words

df['tagline'].apply(lambda x : most_common(x))

def get_stop_words(docs, n=100, min_freq=1):#上位n個の単語、頻度がmin_freq以下の単語を列挙（あまり特徴のない単語等）

    fdist = Counter()

    for word in docs:

            fdist[word] += 1

    common_words = {word for word, freq in fdist.most_common(n)}

    rare_words = {word for word, freq in fdist.items() if freq <= min_freq}

    stopwords = common_words.union(rare_words)

    #print('{}/{}'.format(len(stopwords), len(fdist)))

    return stopwords

df_sw = df['tagline'].apply(lambda x : get_stop_words(x))
def remove_stopwords(words, stopwords):#不要な単語を削除

    words = [word for word in words if word not in stopwords]

    return words
#ベクトル化

from sklearn.feature_extraction.text import TfidfVectorizer

vec_tfidf = TfidfVectorizer()

X = vec_tfidf.fit_transform(df['tagline'])

Tfid_tagline = pd.DataFrame(X.toarray(), columns=vec_tfidf.get_feature_names())
#単語数

df['title_word_count'] = df['title'].apply(lambda x: len(str(x).split()))





#文字数

df['title_char_count'] = df['title'].apply(lambda x: len(str(x)))





# 記号の個数

df['title_punctuation_count'] = df['title'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

df_use2 = df[["tagline_char_count","tagline_word_count","tagline_punctuation_count","title_punctuation_count","title_word_count","title_char_count"]]
#変数の内容を全て確認

#for i, e in enumerate(train['cast'][:1]):

#    print(i, e)
#columnsの辞書化のために変数を定義

dict_columns = ['belongs_to_collection', 'genres', 'production_companies','spoken_languages',

                'production_countries', 'Keywords', 'cast', 'crew']
#JSONの形になっている変数を辞書化させ、対応できるような定義を作る

def text_to_dict(df):

    for column in dict_columns:

        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )

    return df



df = text_to_dict(df)

#映画の中にどれだけの人がキャストされたか表示

print('Number of casted persons in films')

df['cast'].apply(lambda x: len(x) if x != {} else 0).value_counts().head()
#castの中にある俳優の名前をリスト化させる

list_of_cast_names = list(df['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

df['num_cast'] = df['cast'].apply(lambda x: len(x) if x != {} else 0)

df['all_cast'] = df['cast'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')





top_cast_names = [m[0] for m in Counter([i for j in list_of_cast_names for i in j]).most_common(15)]

for g in top_cast_names:

    df['cast_name_' + g] = df['all_cast'].apply(lambda x: 1 if g in x else 0)



    

Counter([i for j in list_of_cast_names for i in j]).most_common(15)
"""

#実際の比較



cast_name_Samuel_L_Jackson=df.loc[df['cast_name_Samuel L. Jackson']==1,]

cast_name_Robert_De_Niro=df.loc[df['cast_name_Robert De Niro']==1,]

cast_name_Morgan_Freeman=df.loc[df['cast_name_Morgan Freeman']==1,]

cast_name_J_K_Simmons=df.loc[df['cast_name_J.K. Simmons']==1,]

cast_name_Bruce_Willis=df.loc[df['cast_name_Bruce Willis']==1,]

cast_name_Liam_Neeson=df.loc[df['cast_name_Liam Neeson']==1,]

cast_name_Susan_Sarandon=df.loc[df['cast_name_Susan Sarandon']==1,]

cast_name_Bruce_McGill=df.loc[df['cast_name_Bruce McGill']==1,]

cast_name_John_Turturro=df.loc[df['cast_name_John Turturro']==1,]

cast_name_Forest_Whitaker=df.loc[df['cast_name_Forest Whitaker']==1,]





cast_name_Samuel_L_Jackson_revenue=cast_name_Samuel_L_Jackson.mean()['revenue']

cast_name_Robert_De_Niro_revenue=cast_name_Robert_De_Niro.mean()['revenue']

cast_name_Morgan_Freeman_revenue=cast_name_Morgan_Freeman.mean()['revenue']

cast_name_J_K_Simmons_revenue=cast_name_J_K_Simmons.mean()['revenue']

cast_name_Bruce_Willis_revenue=cast_name_Bruce_Willis.mean()['revenue']

cast_name_Liam_Neeson_revenue=cast_name_Liam_Neeson.mean()['revenue']

cast_name_Susan_Sarandon_revenue=cast_name_Susan_Sarandon.mean()['revenue']

cast_name_Bruce_McGill_revenue=cast_name_Bruce_McGill.mean()['revenue']

cast_name_John_Turturro_revenue=cast_name_John_Turturro.mean()['revenue']

cast_name_Forest_Whitaker_revenue=cast_name_Forest_Whitaker.mean()['revenue']





cast_revenue_concat = pd.Series([cast_name_Samuel_L_Jackson_revenue,cast_name_Robert_De_Niro_revenue,cast_name_Morgan_Freeman_revenue,cast_name_J_K_Simmons_revenue,

                                cast_name_Bruce_Willis_revenue,cast_name_Liam_Neeson_revenue,cast_name_Susan_Sarandon_revenue,cast_name_Bruce_McGill_revenue,

                                cast_name_John_Turturro_revenue,cast_name_Forest_Whitaker_revenue])

cast_revenue_concat.index=['Samuel L. Jackson','Robert De Niro','Morgan Freeman','J.K. Simmons','Bruce Willis','Liam Neeson','Susan Sarandon','Bruce McGill',

                            'John Turturro','Forest Whitaker']



fig = plt.figure(figsize=(13, 7))

cast_revenue_concat.sort_values(ascending=True).plot(kind='barh',title='mean Revenue (100 million dollars) by Top 10 Most Common Cast')

plt.xlabel('Revenue (100 million dollars)')

"""
list_of_cast_genders = list(df['cast'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)



df['genders_0_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

df['genders_1_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

df['genders_2_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))    











df = df.drop(['cast'], axis=1)

#for i, e in enumerate(train['crew'][:1]):

#    print(i, e)
print('Number of casted persons in films')

df['crew'].apply(lambda x: len(x) if x != {} else 0).value_counts().head(20)
#crewのname

list_of_crew_names = list(df['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

df['num_crew'] = df['crew'].apply(lambda x: len(x) if x != {} else 0)

df['all_crew'] = df['crew'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_crew_names = [m[0] for m in Counter([i for j in list_of_crew_names for i in j]).most_common(15)]

for g in top_crew_names:

    df['crew_name_' + g] = df['all_crew'].apply(lambda x: 1 if g in x else 0)
list_of_crew_names
"""

crew_name_Avy_Kaufman=df.loc[df['crew_name_Avy Kaufman'] == 1].mean()['revenue']

crew_name_Robert_Rodriguez=df.loc[df['crew_name_Robert Rodriguez'] == 1].mean()['revenue']

crew_name_Deborah_Aquila=df.loc[df['crew_name_Deborah Aquila'] == 1].mean()['revenue']

crew_name_James_Newton_Howard=df.loc[df['crew_name_James Newton Howard'] == 1].mean()['revenue']

crew_name_Mary_Vernieu=df.loc[df['crew_name_Mary Vernieu'] == 1].mean()['revenue']

crew_name_Steven_Spielberg=df.loc[df['crew_name_Steven Spielberg'] == 1].mean()['revenue']

crew_name_Luc_Besson=df.loc[df['crew_name_Luc Besson'] == 1].mean()['revenue']

crew_name_Jerry_Goldsmith=df.loc[df['crew_name_Jerry Goldsmith'] == 1].mean()['revenue']

crew_name_Francine_Maisler=df.loc[df['crew_name_Francine Maisler'] == 1].mean()['revenue']

crew_name_Tricia_Wood=df.loc[df['crew_name_Tricia Wood'] == 1].mean()['revenue']

crew_name_James_Horner=df.loc[df['crew_name_James Horner'] == 1].mean()['revenue']

crew_name_Kerry_Barden=df.loc[df['crew_name_Kerry Barden'] == 1].mean()['revenue']

crew_name_Bob_Weinstein=df.loc[df['crew_name_Bob Weinstein'] == 1].mean()['revenue']

crew_name_Harvey_Weinstein=df.loc[df['crew_name_Harvey Weinstein'] == 1].mean()['revenue']

crew_name_Janet_Hirshenson=df.loc[df['crew_name_Janet Hirshenson'] == 1].mean()['revenue']



concat_crew_name = pd.Series([crew_name_Avy_Kaufman,crew_name_Robert_Rodriguez,crew_name_Deborah_Aquila,crew_name_James_Newton_Howard,

                             crew_name_Mary_Vernieu,crew_name_Steven_Spielberg,crew_name_Luc_Besson,crew_name_Jerry_Goldsmith,

                             crew_name_Francine_Maisler,crew_name_Tricia_Wood,crew_name_James_Horner,crew_name_Kerry_Barden,

                             crew_name_Bob_Weinstein,crew_name_Harvey_Weinstein,crew_name_Janet_Hirshenson])

concat_crew_name.index = ['Avy Kaufman','Robert Rodriguez','Deborah Aquila','James Newton Howard',

                             'Mary Vernieu','Steven Spielberg','Luc Besson','Jerry Goldsmith',

                             'Francine Maisler','Tricia Wood','James Horner','Kerry Barden',

                             'Bob Weinstein','Harvey Weinstein','Janet Hirshenson']



fig = plt.figure(figsize=(13,7))

concat_crew_name.sort_values(ascending=True).plot(kind='barh',

                                                       title='mean Revenue (100 million dollars) by Top 15 Most Common crew name')

plt.xlabel('Revenue (100 million dollars)')

"""
Counter([i for j in list_of_crew_names for i in j]).most_common(15)
list_of_crew_department = list(df['crew'].apply(lambda x: [i['department'] for i in x] if x != {} else []).values)

top_crew_department = [m[0] for m in Counter(i for j in list_of_crew_department for i in j).most_common(12)]

for g in top_crew_department:

    df['crew_department_' + g] = df['crew'].apply(lambda x: sum([1 for i in x if i['department'] == g]))
Counter([i for j in list_of_crew_department for i in j]).most_common(15)
list_of_crew_job = list(df['crew'].apply(lambda x: [i['job'] for i in x] if x != {} else []).values)

top_crew_job = [m[0] for m in Counter(i for j in list_of_crew_job for i in j).most_common(20)]

for g in top_crew_job:

    df['crew_job_' + g] = df['crew'].apply(lambda x: sum([1 for i in x if i['job'] == g]))
top_crew_job
df['genders_0_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

df['genders_1_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

df['genders_2_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))
df_use3=df[['num_cast',

       'all_cast', 'cast_name_Samuel L. Jackson', 'cast_name_Robert De Niro',

       'cast_name_Bruce Willis', 'cast_name_Morgan Freeman',

       'cast_name_Liam Neeson', 'cast_name_Willem Dafoe',

       'cast_name_Steve Buscemi', 'cast_name_Sylvester Stallone',

       'cast_name_Nicolas Cage', 'cast_name_Matt Damon',

       'cast_name_J.K. Simmons', 'cast_name_John Goodman',

       'cast_name_Julianne Moore', 'cast_name_Christopher Walken',

       'cast_name_Robin Williams', 'genders_0_cast', 'genders_1_cast',

       'genders_2_cast', 'num_crew', 'all_crew', 'crew_name_Avy Kaufman',

       'crew_name_Steven Spielberg', 'crew_name_Robert Rodriguez',

       'crew_name_Mary Vernieu', 'crew_name_Deborah Aquila',

       'crew_name_Bob Weinstein', 'crew_name_Harvey Weinstein',

       'crew_name_Hans Zimmer', 'crew_name_Tricia Wood',

       'crew_name_James Newton Howard', 'crew_name_James Horner',

       'crew_name_Luc Besson', 'crew_name_Francine Maisler',

       'crew_name_Kerry Barden', 'crew_name_Jerry Goldsmith',

       'crew_department_Production', 'crew_department_Sound',

       'crew_department_Art', 'crew_department_Crew',

       'crew_department_Writing', 'crew_department_Costume & Make-Up',

       'crew_department_Camera', 'crew_department_Directing',

       'crew_department_Editing', 'crew_department_Visual Effects',

       'crew_department_Lighting', 'crew_department_Actors',

       'crew_job_Producer', 'crew_job_Executive Producer', 'crew_job_Director',

       'crew_job_Screenplay', 'crew_job_Editor', 'crew_job_Casting',

       'crew_job_Director of Photography', 'crew_job_Original Music Composer',

       'crew_job_Art Direction', 'crew_job_Production Design',

       'crew_job_Costume Design', 'crew_job_Writer', 'crew_job_Set Decoration',

       'crew_job_Makeup Artist', 'crew_job_Sound Re-Recording Mixer',

       'crew_job_Script Supervisor', 'crew_job_Camera Operator',

       'crew_job_Animation', 'crew_job_Visual Effects Supervisor',

       'crew_job_Hairstylist', 'genders_0_crew', 'genders_1_crew',

       'genders_2_crew']]
df_use.index = df_features.index

df_use2.index = df_use.index

df_use3.index = df_use2.index
df_input = pd.concat([df_use, df_features], axis=1) # .drop("belongs_to_collection", axis=1)

df_input = pd.concat([df_input, df_use2], axis=1)

df_input = pd.concat([df_input, df_use3], axis=1)
Tfid_tagline.index = df_use.index

df_use_Tfid = Tfid_tagline.loc[:, Tfid_tagline[:3000].nunique()>1]

df_use_Tfid.shape
df_input = pd.concat([df_input, df_use_Tfid], axis=1)
df["ln_revenue"] = np.log(df["revenue"]+1)
no_numeric = df_input.apply(lambda s:pd.to_numeric(s, errors='coerce')).isnull().all()

no_numeric[no_numeric]
X_all = df_input.drop(["collection_av_logrevenue", "all_cast", "all_crew"], axis=1)

y_all = df["ln_revenue"]

y_all.index = X_all.index
# 標準化

X_train_all_mean = X_all[:3000].mean()

X_train_all_std  = X_all[:3000].std()

X_all = (X_all-X_train_all_mean)/X_train_all_std
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error 

from sklearn.preprocessing import StandardScaler
# 欠損確認

X_all.isnull().sum().sum()
train_X, val_X, train_y, val_y = train_test_split(X_all[:3000], 

                                                  y_all[:3000], 

                                                  test_size=0.25, random_state=1)
from sklearn.linear_model import Lasso, Ridge
clf = Lasso(alpha=0.1, max_iter=3000, random_state=1)  # default alpha=1, max_iter=1000

clf.fit(train_X, train_y)
val_pred = clf.predict(val_X)

print("RMSLE score for validation data")

np.sqrt(mean_squared_error(val_pred, val_y))
plt.scatter(np.exp(val_pred)+1, np.exp(val_y)+1, s=3)

plt.xlabel("prediction")

plt.ylabel("true revenue")

plt.xscale("log")

plt.yscale("log")
coef = pd.Series(clf.coef_, index=train_X.columns)

df_coef = pd.DataFrame(coef[coef!=0], columns=["coef"])

df_coef[abs(df_coef["coef"])>0.1].sort_values("coef", ascending=False)
clf = Lasso(alpha=0.1, max_iter=3000, random_state=1)  # default alpha=1, max_iter=1000

clf.fit(X_all[:3000], y_all[:3000])
coef = pd.Series(clf.coef_, index=train_X.columns)

df_coef = pd.DataFrame(coef[coef!=0], columns=["coef"])

df_coef[abs(df_coef["coef"])>0.1].sort_values("coef", ascending=False)
test_pred = clf.predict(X_all[3000:])
test_revenue = np.exp(test_pred)-1
sample_submission = pd.read_csv('../input/tmdb-box-office-prediction/sample_submission.csv')

sample_submission.head()
sample_submission["revenue"] = test_revenue
sample_submission
sample_submission.to_csv('submission.csv', index=False)
from sklearn.ensemble import RandomForestRegressor
clf2 = RandomForestRegressor(max_depth=15, min_samples_split=5, n_jobs=3, random_state=1)  # default alpha=1, max_iter=1000

clf2.fit(train_X, train_y)
val_pred = clf2.predict(val_X)

print("RMSLE score for validation data")

np.sqrt(mean_squared_error(val_pred, val_y))
plt.scatter(np.exp(val_pred)+1, np.exp(val_y)+1, s=3)

plt.xlabel("prediction")

plt.ylabel("true revenue")

plt.xscale("log")

plt.yscale("log")
clf2 = RandomForestRegressor(max_depth=15, min_samples_split=5, n_jobs=3, random_state=1)  # default alpha=1, max_iter=1000

clf2.fit(X_all[:3000], y_all[:3000])
df_importance = pd.DataFrame([clf2.feature_importances_], columns=train_X.columns, index=["importance"]).T

df_importance.sort_values("importance", ascending=False).head(20)
test_pred = clf2.predict(X_all[3000:])
test_revenue = np.exp(test_pred)-1
submission_RF = sample_submission.copy()

submission_RF["revenue"] = test_revenue
submission_RF
submission_RF.to_csv('submission_RF.csv', index=False)