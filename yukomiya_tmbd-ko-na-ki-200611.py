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

train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')

#

test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')
print(train.shape,test.shape)

train.columns
train.loc[train['id'] == 391,'runtime'] = 96 #The Worst Christmas of My Lifeの上映時間を調べて入力

train.loc[train['id'] == 592,'runtime'] = 90 #А поутру они проснулисьの上映時間を調べて入力

train.loc[train['id'] == 925,'runtime'] = 86 #¿Quién mató a Bambi?の上映時間を調べて入力

train.loc[train['id'] == 978,'runtime'] = 93 #La peggior settimana della mia vitaの上映時間を調べて入力

train.loc[train['id'] == 1256,'runtime'] = 92 #Cry, Onion!の上映時間を調べて入力

train.loc[train['id'] == 1542,'runtime'] = 93 #All at Onceの上映時間を調べて入力

train.loc[train['id'] == 1875,'runtime'] = 93 #Vermistの上映時間を調べて入力

train.loc[train['id'] == 2151,'runtime'] = 108 #Mechenosetsの上映時間を調べて入力

train.loc[train['id'] == 2499,'runtime'] = 86 #Na Igre 2. Novyy Urovenの上映時間を調べて入力

train.loc[train['id'] == 2646,'runtime'] = 98 #My Old Classmateの上映時間を調べて入力

train.loc[train['id'] == 2786,'runtime'] = 111 #Revelationの上映時間を調べて入力

train.loc[train['id'] == 2866,'runtime'] = 96 #Tutto tutto niente nienteの上映時間を調べて入力
test.loc[test['id'] == 3244,'runtime'] = 93 #La caliente niña Julietta	の上映時間を調べて入力

test.loc[test['id'] == 4490,'runtime'] = 90 #Pancho, el perro millonarioの上映時間を調べて入力

test.loc[test['id'] == 4633,'runtime'] = 108 #Nunca en horas de claseの上映時間を調べて入力

test.loc[test['id'] == 6818,'runtime'] = 90 #Miesten välisiä keskustelujaの上映時間を調べて入力



test.loc[test['id'] == 4074,'runtime'] = 103 #Shikshanachya Aaicha Ghoの上映時間を調べて入力

test.loc[test['id'] == 4222,'runtime'] = 91 #Street Knightの上映時間を調べて入力

test.loc[test['id'] == 4431,'runtime'] = 96 #Plus oneの上映時間を調べて入力

test.loc[test['id'] == 5520,'runtime'] = 86 #Glukhar v kinoの上映時間を調べて入力

test.loc[test['id'] == 5845,'runtime'] = 83 #Frau Müller muss weg!の上映時間を調べて入力

test.loc[test['id'] == 5849,'runtime'] = 140 #Shabdの上映時間を調べて入力

test.loc[test['id'] == 6210,'runtime'] = 104 #The Last Breathの上映時間を調べて入力

test.loc[test['id'] == 6804,'runtime'] = 140 #Chaahat Ek Nasha...の上映時間を調べて入力

test.loc[test['id'] == 7321,'runtime'] = 87 #El truco del mancoの上映時間を調べて入力
df = pd.concat([train, test]).set_index("id")
#columnsを確認し、除外する変数をdrop

print(df.columns)

# 使わない列を消す

df = df.drop(["poster_path", "status", "original_title"], axis=1) # "overview",  "imdb_id", 
# logを取っておく

df["log_revenue"] = np.log10(df["revenue"])

# homepage: 有無に

df["homepage"] = ~df["homepage"].isnull()
dfdic_feature = {}
%%time

# JSON text を辞書型のリストに変換

import ast

dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']



for col in dict_columns:

       df[col]=df[col].apply(lambda x: [] if pd.isna(x) else ast.literal_eval(x) )
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

    

    return word_df#.drop("nashi", axis=1)
df["genre_names"] = df["genres"].apply(lambda x : [ i["name"] for i in x])

dfdic_feature["genre"] = count_word_list(df["genre_names"])

# TV movie は1件しかないので削除

dfdic_feature["genre"] = dfdic_feature["genre"].drop("TV Movie", axis=1)

dfdic_feature["genre"].head()
# train内の作品数が10件未満の言語は "small" に集約

n_language = df.loc[:train.index[-1], "original_language"].value_counts()

large_language = n_language[n_language>=10].index

df.loc[~df["original_language"].isin(large_language), "original_language"] = "small"
df["original_language"] = df["original_language"].astype("category")
# one_hot_encoding

dfdic_feature["original_language"] = pd.get_dummies(df["original_language"])

#dfdic_feature["original_language"] = dfdic_feature["original_language"].loc[:, dfdic_feature["original_language"].sum()>0]

dfdic_feature["original_language"].head()
df["production_names"] = df["production_companies"].apply(lambda x : [ i["name"] for i in x])

#.fillna("[{'name': 'nashi'}]").map(to_name_list)
%time tmp = count_word_list(df["production_names"])
# train内の件数が多い物のみ選ぶ

def select_top_n(df, topn=9999, nmin=2):  # topn:上位topn件, nmin:作品数nmin以上

#    if "small" in df.columns:

#        df = df.drop("small", axis=1)

    n_word = (df.loc[train["id"]]>0).sum().sort_values(ascending=False)

    # 作品数がnmin件未満

    smallmin = n_word[n_word<nmin].index

    # 上位topn件に入っていない

    smalln = n_word.iloc[topn+1:].index

    small = set(smallmin) | set(smalln)

    # 件数の少ないタグのみの作品

    df["small"] = df[small].sum(axis=1) #>0

    

    return df.drop(small, axis=1)
# trainに50本以上作品のある会社

dfdic_feature["production_companies"] = select_top_n(tmp, nmin=50)

dfdic_feature["production_companies"].head()
# 国名のリストに

df["country_names"] = df["production_countries"].apply(lambda x : [ i["name"] for i in x])

df_country = count_word_list(df["country_names"])
# 2か国だったら、0.5ずつに

df_country = (df_country.T/df_country.sum(axis=1)).T.fillna(0)
# 30作品以上の国のみ

dfdic_feature["production_countries"] = select_top_n(df_country, nmin=30)

dfdic_feature["production_countries"].head()
df["keyword_list"] = df["Keywords"].apply(lambda x : [ i["name"] for i in x])
def encode_topn_onehot(series, topn):

    # 多いワード順に

    word_count = pd.Series(collections.Counter(sum(list(series.values), [])))

    word_count = word_count.sort_values(ascending=False)

    

    df_topn = df[[]].copy()  # index のみのDF

    # 上位topn件のキーワードのみ

    for word in word_count.iloc[:topn].index:  # .drop("nashi")

        df_topn[word] = series.apply(lambda x: word in x)*1

    

    return df_topn

    
dfdic_feature["Keywords"] = encode_topn_onehot(df["keyword_list"], 100)
df["num_Keywords"] = df["keyword_list"].apply(len)
df["language_names"] = df["spoken_languages"].apply(lambda x : [ i["name"] for i in x])

df["n_language"] = df["language_names"].apply(len)

# 欠損値は１にする(データを見ると無声映画ではない)

df.loc[df["n_language"]==0, "n_language"] = 1
# 英語が含まれるか否か

df["speak_English"] = df["language_names"].apply(lambda x : "English" in x)
import datetime
# 公開日の欠損1件 id=3829

# May,2000 (https://www.imdb.com/title/tt0210130/) 

# 日は不明。1日を入れておく

df.loc[3829, "release_date"] = "5/1/00"
df["release_year"] = pd.to_datetime(df["release_date"]).dt.year.astype(int)

# 年の20以降を、2020年より後の未来と判定してしまうので、補正。

df.loc[df["release_year"]>2020, "release_year"] = df.loc[df["release_year"]>2020, "release_year"]-100



df["release_month"] = pd.to_datetime(df["release_date"]).dt.month.astype(int)

df["release_day"] = pd.to_datetime(df["release_date"]).dt.day.astype(int)
# datetime型に

df["release_date"] = df.apply(lambda s: datetime.datetime(

    year=s["release_year"],month=s["release_month"],day=s["release_day"]), axis=1)
df["release_dayofyear"] = df["release_date"].dt.dayofyear

df["release_dayofweek"] = df["release_date"].dt.dayofweek
# 月、曜日は カテゴリ型に

df["release_month"] = df["release_month"].astype('category')

df["release_dayofweek"] = df["release_dayofweek"].astype('category')
# collection 名を抽出

df["collection_name"] = df["belongs_to_collection"].apply(lambda x : x[0]["name"] if len(x)>0 else "nashi")

# 無い場合、"nashi"に
# シリーズの作品数

#df = pd.merge( df, df.groupby("collection_name").count()[["budget"]].rename(columns={"budget":"count_collection"}), 

#         on="collection_name", how="left")

# indexがずれるので、戻す

#df.index = df.index+1



df["count_collection"] = df["collection_name"].apply(lambda x : (df["collection_name"]==x).sum())

# シリーズ以外の場合0

df.loc[df["collection_name"]=="nashi", "count_collection"] = 0



# シリーズ何作目か

df["number_in_collection"] = df.sort_values("release_date").groupby("collection_name").cumcount()+1

# シリーズ以外の場合0

df.loc[df["collection_name"]=="nashi", "number_in_collection"] = 0



%%time

# 同シリーズの自分より前の作品の平均log(revenue)

df["collection_av_logrevenue"] = [ df.loc[(df["collection_name"]==row["collection_name"]) & 

                                          (df["number_in_collection"]<row["number_in_collection"]),

                                          "log_revenue"].mean() 

     for key,row in df.iterrows() ]

# 欠損(nashi) の場合、nashi での平均

df.loc[df["collection_name"]=="nashi", "collection_av_logrevenue"] = df.loc[df["collection_name"]=="nashi", "log_revenue"].mean()
# train に無くtestだけにあるシリーズの場合、シリーズもの全部の平均

collection_mean = df.loc[df["collection_name"]!="nashi", "log_revenue"].mean()  # シリーズもの全部の平均

df["collection_av_logrevenue"] = df["collection_av_logrevenue"].fillna(collection_mean)  

df_features = pd.concat(dfdic_feature, axis=1)
# 欠測と0は、0ではないものの平均で埋める

df["runtime"] = df["runtime"].fillna(df.loc[df["runtime"]>0, "runtime"].mean())

df.loc[df["runtime"]==0, "runtime"] = df.loc[df["runtime"]>0, "runtime"].mean()
#plt.scatter(df["budget"]+1, df["log_revenue"], s=1)

#plt.xscale("log")

#plt.xrange([])
df.columns
df[["original_language", "collection_name"]] = df[["original_language", "collection_name"]].astype("category")
df_use = df[['budget', 'homepage', 'popularity','runtime','n_language', 

             "num_Keywords", "speak_English",

             'release_year', 'release_month','release_dayofweek', 

             'collection_av_logrevenue' ,"count_collection","number_in_collection"

            ]]

df_use.head()
df_use = pd.get_dummies(df_use)
train_add = pd.read_csv('../input/tmdb-competition-additional-features/TrainAdditionalFeatures.csv')

test_add = pd.read_csv('../input/tmdb-competition-additional-features/TestAdditionalFeatures.csv')

train_add.head()
df = pd.merge(df, pd.concat([train_add, test_add]), on="imdb_id", how="left")
add_cols = ["popularity2", "rating", "totalVotes"]

df[add_cols] = df[add_cols].fillna(df[add_cols].mean())
train2 = pd.read_csv('../input/tmdb-box-office-prediction-more-training-data/additionalTrainData.csv')

train3 = pd.read_csv('../input/tmdb-box-office-prediction-more-training-data/trainV3.csv')

train3.head()
#全て小文字に変換

def lower_text(text):

    return text.lower()



#記号の排除

def remove_punct(text):

    text = text.replace('-', ' ')  # - は単語の区切りとみなす

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



def remove_stopwords(words, stopwords):#不要な単語を削除

    words = [word for word in words if word not in stopwords]

    return words
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
# 英語でよく使う単語が入っていない文章を確認

df.loc[df["overview"].apply(lambda x : str(x)).apply(lambda x : lower_text(x)

                                ).str.contains("nan|the|where|with|from|and|for|his|her|over")==False, "overview"]

#train3.loc[train3["overview"].apply(lambda x : str(x)).apply(lambda x : lower_text(x)).str.contains("nan|the|where|with|from|and|for|his|her|over")==False, "overview"]
no_english_overview_id = [157, 2863, 4616]   # 上のデータを目で確認

no_english_tagline_id = [3255, 3777, 4937]   # Tfidf で非英語の単語があったもの
from gensim.models import word2vec
col_text = ["overview", "tagline"] # "title", 

all_text = pd.concat([df[col_text], train2[col_text], train3[col_text]])
# 英語以外と"nan"は除外

all_text.loc[no_english_overview_id, "overview"] = np.nan

all_text.loc[no_english_tagline_id, "tagline"] = np.nan

all_text.loc[all_text["tagline"]=="nan", "tagline"] = np.nan
all_texts = all_text.stack()

all_texts=all_texts.apply(lambda x : str(x))

all_texts=all_texts.apply(lambda x : lower_text(x))

all_texts=all_texts.apply(lambda x : remove_punct(x))
all_texts.to_csv("./alltexts_for_w2v.txt", index=False, header=False)

docs = word2vec.LineSentence("alltexts_for_w2v.txt")

%%time



model = word2vec.Word2Vec(docs, sg=1, size=100, min_count=5, window=5, iter=100)

model.save("./alltexts_w2v1_sg2.model")
# model = word2vec.Word2Vec.load("./alltexts_w2v1_cbow.model")

# model = word2vec.Word2Vec.load("./alltexts_w2v1_sg2.model")
model.most_similar(positive=['father'])
model.most_similar(positive=['human'])
# 単語ベクトルの mean, max を文章ベクトルにする

def get_doc_vector(doc, method="mean", weight=None):

    split_doc = doc.split(" ")

    if weight==None:

        weight = dict(zip(model.wv.vocab.keys(), np.ones(len(model.wv.vocab))))

        

    word_vecs = [ model[word]*weight[word] for word in split_doc if word in model.wv.vocab.keys() ]

    

    if len(word_vecs)==0:

        doc_vec = []

    elif method=="mean":

        doc_vec =  np.mean(word_vecs, axis=0)

    elif method=="max":

        doc_vec =  np.max(word_vecs, axis=0)

    elif method=="meanmax":

        doc_vec =  np.mean(word_vecs, axis=0)+np.max(word_vecs, axis=0)

    return doc_vec
#単語数

df['overview_word_count'] = df['overview'].apply(lambda x: len(str(x).split()))

#文字数

df['overview_char_count'] = df['overview'].apply(lambda x: len(str(x)))

# 記号の個数

df['overview_punctuation_count'] = df['overview'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
# 前処理

df['_overview']=df['overview'].apply(lambda x : str(x)

                            ).apply(lambda x : lower_text(x)).apply(lambda x : remove_punct(x))

%time df_overview =  df["_overview"].apply(get_doc_vector, method="meanmax").apply(pd.Series)
df_overview = df_overview.fillna(0).add_prefix("overview_")
#単語数

df['tagline_word_count'] = df['tagline'].apply(lambda x: len(str(x).split()))

#文字数

df['tagline_char_count'] = df['tagline'].apply(lambda x: len(str(x)))

# 記号の個数

df['tagline_punctuation_count'] = df['tagline'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
df['_tagline']=df['tagline'].apply(lambda x : str(x)

                                 ).apply(lambda x : lower_text(x)).apply(lambda x : remove_punct(x))

#ベクトル化

# from sklearn.feature_extraction.text import TfidfVectorizer

# vec_tfidf = TfidfVectorizer()

# X = vec_tfidf.fit_transform(df['tagline'])

# Tfidf_tagline = pd.DataFrame(X.toarray(), columns=vec_tfidf.get_feature_names())

# X = vec_tfidf.fit_transform(df['overview'].dropna())

# Tfidf_overview = pd.DataFrame(X.toarray(), columns=vec_tfidf.get_feature_names())
%time df_tagline =  df["_tagline"].apply(get_doc_vector, method="meanmax").apply(pd.Series)
df_tagline = df_tagline.fillna(0).add_prefix("tagline_")
#単語数

df['title_word_count'] = df['title'].apply(lambda x: len(str(x).split()))

#文字数

df['title_char_count'] = df['title'].apply(lambda x: len(str(x)))

# 記号の個数

df['title_punctuation_count'] = df['title'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

df_use2 = df[["tagline_char_count","tagline_word_count","tagline_punctuation_count",

              "overview_char_count","overview_word_count","overview_punctuation_count",

              "title_char_count","title_word_count","title_punctuation_count"]]
# Keywords を全部並べたものを、文とみなしてベクトル化

%time df_keyword_w2v = df["keyword_list"].apply(" ".join).apply(get_doc_vector, method="mean").apply(pd.Series)

df_keyword_w2v = df_keyword_w2v.fillna(0).add_prefix("keyword_")
#映画の中にどれだけの人がキャストされたか表示

print('Number of casted persons in films')

df['cast'].apply(len).value_counts().head()
df['num_cast'] = df['cast'].apply(len)  # 人数

# df['all_cast'] = df['cast'].apply(lambda x: [i['name'] for i in x])  # 



# 出演作品数上位50人について、one-hot-encoding

df_castname = pd.DataFrame([], index=df.index)

list_of_cast_names = list(df['cast'].apply(lambda x: [i['name'] for i in x]).values)  # 俳優名のリストのリスト

top_cast_names = [m[0] for m in Counter([i for j in list_of_cast_names for i in j]).most_common(50)]

for g in top_cast_names:

    df_castname[g] = df['cast'].apply(lambda x: g in [i['name'] for i in x])



    

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
# 性別比

list_of_cast_genders = list(df['cast'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)



df['genders_0_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))/df["num_cast"]

df['genders_1_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))/df["num_cast"]



# 欠損は平均で埋める

df[['genders_0_cast', 'genders_1_cast']] = df[['genders_0_cast', 'genders_1_cast']].fillna(df[['genders_0_cast', 'genders_1_cast']].mean())







# df = df.drop(['cast'], axis=1)

df["crew"][1][0]
# 部署別　のべ人数

department_count = pd.Series(Counter([job for lst in df["crew"].apply(lambda x : [ i["department"] for i in x]).values for job in lst]))

department_count.sort_values(ascending=False)
# job別　のべ人数(top30)

job_count = pd.Series(Counter([job for lst in df["crew"].apply(lambda x : [ i["job"] for i in x]).values for job in lst]))

job_count.sort_values(ascending=False).head(30)
job_count["Visual Effects"]
%%time

df_crew = { idx : pd.DataFrame([ [crew["department"], crew["job"], crew["name"]] 

                        for crew in x], columns=["department", "job", "name"]) 

    for idx, x in df["crew"].iteritems() }
df_crew = pd.concat(df_crew)

df_crew.head()
#crewのname

df['num_crew'] = df['crew'].apply(len)



# crew gender

df['genders_0_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))/df["num_crew"]

df['genders_1_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))/df["num_crew"]



# 欠損は平均で埋める

df[['genders_0_crew', 'genders_1_crew']] = df[['genders_0_crew', 'genders_1_crew']].fillna(df[['genders_0_crew', 'genders_1_crew']].mean())

def select_job(list_dict, key, value):

    return [ dic["name"] for dic in list_dict if dic[key]==value]
# 各部署の人数

for department in department_count.index:

    df['dep_{}_num'.format(department)] = df["crew"].apply(select_job, key="department", value=department).apply(len)  

    
# Animationの人数（アニメ映画で重要そうなので入れてみる）

df['job_Animation_num'] = df["crew"].apply(select_job, key="job", value="Animation").apply(len)
%%time

# 重要と思われるjobについて、参加作品数上位15人で one-hot-encoding

# 製作、監督、脚本、キャスティング、作曲

df_crewname = pd.DataFrame([], index=df.index)

for job in ["Producer", "Director", "Screenplay", "Casting", "Original Music Composer"]:

    col = 'job_{}_list'.format(job)

    df[col] = df["crew"].apply(select_job, key="job", value=job)



    top_list = [m[0] for m in Counter([i for j in df[col] for i in j]).most_common(15)]

    for i in top_list:

        df_crewname['{}_{}'.format(job,i)] = df[col].apply(lambda x: i in x)
# 技術部門はdepartment毎に、参加作品数上位15人で one-hot-encoding

for job in ["Sound", "Art", "Costume & Make-Up", "Camera", "Visual Effects"]:

    col = 'department_{}_list'.format(job)

    df[col] = df["crew"].apply(select_job, key="department", value=job)



    top_list = [m[0] for m in Counter([i for j in df[col] for i in j]).most_common(15)]

    for i in top_list:

        df_crewname['{}_{}'.format(job,i)] = df[col].apply(lambda x: i in x)
# 監督が複数の作品数

(df["job_Director_list"].apply(len)>1).sum()
df.columns
df_use3=df[['num_cast', 'genders_0_cast',

       'genders_1_cast', 'num_crew', 'genders_0_crew', 'genders_1_crew',

       'dep_Directing_num', 'dep_Writing_num', 'dep_Production_num',

       'dep_Sound_num', 'dep_Camera_num', 'dep_Editing_num', 'dep_Art_num',

       'dep_Costume & Make-Up_num', 'dep_Crew_num', 'dep_Lighting_num',

       'dep_Visual Effects_num', 'dep_Actors_num', 'job_Animation_num']]
df
df_features.index = df.index



df_use.index = df.index

# df_use2.index = df.index

df_use4 = df[add_cols]
df_input = pd.concat([df_use, df_use2, df_use3, df_use4, df_features], axis=1) # .drop("belongs_to_collection", axis=1)
#Tfid_tagline.index = df_use.index

#df_use_Tfid = Tfid_tagline.loc[:, Tfid_tagline[:3000].nunique()>1]

#df_use_Tfid.shape
# 全て繋げた特徴量

#df_input = pd.concat([df_input, df_tagline, df_overview, df_keyword_w2v, df_castname, df_crewname], axis=1)

# df_tagline, df_overview, df_keyword_w2v, 
# 欠測ナシを確認

df_input.isnull().sum().sum()
#cols = df_input.loc[:, df_input.isnull().sum()>0].columns

#df_input.loc[:, cols] = df_input[cols].fillna(df_input[cols].mean())
# 保存

import pickle

with open('df_input.pkl', 'wb') as f:

      pickle.dump(df_input , f)
df["ln_revenue"] = np.log(df["revenue"]+1)
# 数値化できい列を確認

no_numeric = df_input.apply(lambda s:pd.to_numeric(s, errors='coerce')).isnull().all()

no_numeric[no_numeric]
X_all = df_input  # .drop(["collection_av_logrevenue"], axis=1)

y_all = df["ln_revenue"]

y_all.index = X_all.index
[ c for c in X_all.columns if "revenue" in str(c)]
# 標準化

# X_train_all_mean = X_all[:3000].mean()

# X_train_all_std  = X_all[:3000].std()

# X_all = (X_all-X_train_all_mean)/X_train_all_std
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error 

from sklearn.preprocessing import StandardScaler
train_X, val_X, train_y, val_y = train_test_split(X_all[:train.index[-1]], 

                                                  y_all[:train.index[-1]], 

                                                  test_size=0.25, random_state=1)
from sklearn.ensemble import RandomForestRegressor
clf2 = RandomForestRegressor(n_jobs=3, random_state=1)  # max_depth=, min_samples_split=, 

clf2.fit(train_X, train_y)
val_pred = clf2.predict(val_X)

print("RMSLE score for validation data")

np.sqrt(mean_squared_error(val_pred, val_y))
plt.scatter(np.exp(val_pred)+1, np.exp(val_y)+1, s=3)

plt.xlabel("prediction")

plt.ylabel("true revenue")

plt.xscale("log")

plt.yscale("log")
clf2 = RandomForestRegressor(n_jobs=3, random_state=1, n_estimators=500)  # 

clf2.fit(X_all[:train.index[-1]], y_all[:train.index[-1]])
df_importance = pd.DataFrame([clf2.feature_importances_], columns=train_X.columns, index=["importance"]).T

df_importance.sort_values("importance", ascending=False).head(20)
test_pred = clf2.predict(X_all[3000:])
test_revenue = np.exp(test_pred)-1
sample_submission = pd.read_csv('../input/tmdb-box-office-prediction/sample_submission.csv')
submission_RF = sample_submission.copy()

submission_RF["revenue"] = test_revenue
submission_RF
submission_RF.to_csv('submission_RF.csv', index=False)