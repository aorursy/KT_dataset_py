import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



import os

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
rating=pd.read_csv('/kaggle/input/anime-recommendations-database/rating.csv')

anime=pd.read_csv('/kaggle/input/anime-recommendations-database/anime.csv')
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
anime=reduce_mem_usage(anime)

rating=reduce_mem_usage(rating)
anime.head()
anime.shape
print(np.median(anime['members']))

anime=anime[anime['members']>(np.percentile(anime['members'], 50))]

anime.dropna(axis=0, how='any', subset = ['rating'] ,inplace=True)
rating.head()
rating.nunique()
rating['rating'] = rating['rating'].replace(-1,np.nan)

rating["user_id"].unique()

user=rating.loc[:,'user_id'].value_counts()

user=user.to_frame()

user = user.drop(user[user.user_id < 150].index)#drop users who rated less than 150 times to decrease the dataset size

user=user.rename(columns={"user_id": "count"})

user['user_id']=user.index
user.head()
rating=pd.merge(user,rating,on='user_id',how='left')

users=rating[['user_id','anime_id','rating']]

users=users.reset_index()

anime=anime.reset_index()

df = pd.merge(anime,users,on='anime_id',how='inner')

df=df.drop(['index_x','index_y'], axis=1)

df = df.rename(columns={'rating_x': 'anime_rating','rating_y':'user_rating'})

df.head(10)
rating_counts=df.loc[:,'anime_id'].value_counts()#每个动漫打分人数 number of raters each anime

rating_counts=rating_counts.to_frame()

rating_counts=rating_counts.rename(columns={'anime_id': 'count_ratings'})

rating_counts['anime_id']=rating_counts.index

rating_counts=rating_counts[rating_counts['count_ratings']>300]#只保留300个人以上评分的动漫only keep anime with more than 300 raters.

rating_counts.head()
df1=pd.merge(rating_counts,df,on='anime_id',how='left')

df_p = df1.pivot_table(index='user_id', columns='anime_id', values='user_rating')

print('Shape User-Movie-Matrix:\t{}'.format(df_p.shape))

df_p.sample(3)
df2=df1.dropna(subset=['user_rating','anime_rating','members'])

df2=df2.drop_duplicates(subset='name')

df2.head()
def weighted_rating(x): #x is the dataframe's name

    m=300

    C=df2.anime_rating.mean()

    v = x['count_ratings']

    R = x['anime_rating']

    return (v/(v+m) * R) + (m/(m+v) * C)
df2['wr'] = df2.apply(weighted_rating, axis=1)

df2=df2.sort_values(by='wr',ascending=False)

df2.head(10)
from sklearn.metrics.pairwise import cosine_similarity

df_p=df_p.fillna(0)

user_similarity = cosine_similarity(df_p) #ger similarity matrix for users

user_similarity.shape
item_similarity = cosine_similarity(df_p.T)#get similarity matrix for animes

item_similarity.shape
item_sim_df = pd.DataFrame(item_similarity, index = df_p.columns, columns = df_p.columns)

item_sim_df.head(3) #show similarity matrix for animes
user_sim_df = pd.DataFrame(user_similarity, index = df_p.index, columns = df_p.index)

user_sim_df.head(3) #show similarity matrix forusers
def similar_users(user):

    

    if user not in df_p.index:

        return('No data available on user {}'.format(user))

    

    print('Most Similar Users:\n')

    sim_values = user_sim_df.sort_values(by=user, ascending=False).loc[:,user].tolist()[1:6] # sort the similar score and get top5

    sim_users = user_sim_df.sort_values(by=user, ascending=False).index[1:6]  # get the user_id of those top 5.  

    zipped = zip(sim_users, sim_values,)

    for user, sim in zipped:

        print('User #{0}, Similarity value: {1:.2f}'.format(user, sim)) 
similar_users(3) 
similar_users(73)
def similar_animes(anime):

    

    if anime not in df_p.columns:

        return('No anime called {}'.format(anime))

    

    print('Most Similar Animes:\n')

    sim_values = item_sim_df.sort_values(by=anime, ascending=False).loc[:,anime].tolist()[1:6]

    sim_animes = item_sim_df.sort_values(by=anime, ascending=False).index[1:6]

    zipped = zip(sim_animes, sim_values,)

    for anime, sim in zipped:

        print('Anime #{0}, Similarity value: {1:.2f}'.format(anime, sim)) 
similar_animes(19)
anime_id_name_match=df1[['anime_id','name']].drop_duplicates()

anime_id_name_match=anime_id_name_match.sort_values(by='anime_id')

item_sim_df_name=item_sim_df.copy()

item_sim_df_name.index = anime_id_name_match['name']

item_sim_df_name.columns = anime_id_name_match['name']

item_sim_df_name.head(3)
def similar_animes_name(anime_name):

    count = 1

    print('Similar shows to {} include:\n'.format(anime_name))

    for item in item_sim_df_name.sort_values(by = anime_name, ascending = False).index[1:11]:

        print('No. {}: {}'.format(count, item))

        count +=1 
import re

#This function is to find exact anime name by inputing in keywords

def find_real_name(x):

    df1_anime=df1.drop_duplicates(subset='name')

    find_name=df1_anime[df1_anime['name'].str.contains(x, flags=re.IGNORECASE)] #case non-sensitive

    return find_name
find_real_name('ping')
similar_animes_name('Ping Pong The Animation')
def user_like_me(user):

    # get the user's row

    s1 = df_p.loc[user,:]



    # get the index of max values in s1, might be more than 1

    s1_argmax = s1[s1 == s1.max()].index.tolist()



    # randomly choose 1 index

    #s1_argmax = np.random.choice(s1_argmax) 

    s1_argmax

    animes=[]

    for i in s1_argmax:

        name_list=anime_id_name_match[anime_id_name_match.anime_id==i]['name'].tolist()

        animes.append(name_list)  

    print('The user like you the most is also watching:')

    print(*animes, sep='\n')
user_like_me(3324)
def predicted_rating(anime_name, user):

    sim_users = user_sim_df.sort_values(by=user, ascending=False).index[1:500] 

    user_values = user_sim_df.sort_values(by=user, ascending=False).loc[:,user].tolist()[1:500]

    rating_list = []

    weight_list = []

    for j, i in enumerate(sim_users):

        item_sim_df_name_2=df_p.copy()

        item_sim_df_name_2.columns = anime_id_name_match['name']

        rating = item_sim_df_name_2.loc[i, anime_name]

        similarity = user_values[j]

        if np.isnan(rating):

            continue

        elif not np.isnan(rating):

            rating_list.append(rating*similarity)

            weight_list.append(similarity)

    return sum(rating_list)/sum(weight_list)   
predicted_rating('Cowboy Bebop', 5)
df['genre_and_type']=df['genre']+','+df['type']

df_anime_name_match=df[['anime_id','name','genre_and_type']].drop_duplicates()

df_anime_name_match.head()
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import seaborn as sns

from collections import Counter

import re

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet

import string

import nltk

import matplotlib.pyplot as plt

pd.set_option("display.max_colwidth", 200)

import spacy

import gensim

from gensim import corpora

!pip install pyLDAvis

import pyLDAvis

import pyLDAvis.gensim

%matplotlib inline



import itertools

nltk.download('stopwords')

nltk.download('punkt')

nltk.download('wordnet')

nltk.download('averaged_perceptron_tagger')
# words to be removed from vocabulary

blockwords = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at','since','paid','don','doesn','close',

 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'doing', "don't", 'down', 'during',

 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's",

 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself',

 "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours' 'ourselves', 'out', 'over', 'own',

 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'like',

 'than', 'that',"that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", 'also','can','could','should',

 "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'not','bit','much',

 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where','within','quite','really','just','together',

 "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's",'will', 'with', "won't", 'would', "wouldn't", 'hole','furniture','put',

 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', "s'yourself'", 'yourselves', 'drawer','sure',

 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'hundred', 'thousand', '1st', '2nd', '3rd','nightstand','nightstands','night',

 '4th', '5th', '6th', '7th', '8th', '9th', '10th']
df_anime_name_match['genre_and_type']=df_anime_name_match['genre_and_type'].apply(str)
stop_words = set(stopwords.words('english'))                      # set of all stop words

lem=WordNetLemmatizer()

#p=inflect.engine()



def process(comment):

  sent = comment.lower()                                          # lower case all words 

  words = nltk.word_tokenize(sent)

  words =  [word for word in words if not word in blockwords]     # remove words present in blockwords

  words = [word for word in words if not word.isdigit()]          # remove digit characters

  #words = [word for word in words if len(word) > 3]               # remove words with length less than 3

  #words = [word for word in words if word.isalpha()]              # remove non alphabetic words

  words = [lem.lemmatize(word) for word in words]                 # lemmatize words to root word

  sent = ' '.join(words)

  sent = re.sub(r'\(', '', sent)

  sent = re.sub(r'\)', '', sent)

  sent = re.sub(r"'", '', sent)

  return sent



def num_words(sent):                                              # returns number of words in the sentence

  word_tok=nltk.word_tokenize(sent)

  return len(word_tok)



df_anime_name_match['Cleaned_g_t']=df_anime_name_match['genre_and_type'].apply(process)

df_anime_name_match['Unclean_len']=df_anime_name_match['genre_and_type'].apply(num_words)                     # word length of uncleaned comments

df_anime_name_match['Clean_len']=df_anime_name_match.Cleaned_g_t.apply(num_words)               # word length of cleaned comments

df_anime_name_match['percentage reduction']=(df_anime_name_match['Unclean_len']-df_anime_name_match['Clean_len'])/df_anime_name_match['Unclean_len']*100 # percentage of reduction
text= " ".join(df_anime_name_match['Cleaned_g_t'])

# Display the generated image:

wordcloud = WordCloud(max_font_size=35, max_words=40, background_color="white",collocations=False).generate(text)

plt.figure(figsize=(8,6))

plt.imshow(wordcloud, interpolation="gaussian")

plt.title('Top words in anime descriptions',size=19)

plt.axis("off")

plt.show()
tf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tf.fit_transform(df_anime_name_match['Cleaned_g_t'])
from sklearn.metrics.pairwise import linear_kernel

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

cosine_sim.shape[0]
tf_sim=pd.DataFrame(data=cosine_sim) 

tf_sim.index = df_anime_name_match['name']

tf_sim.columns = df_anime_name_match['name']

tf_sim.head()
def similar_animes_content_based(anime):

    

    if anime not in tf_sim.columns:

        return('No anime called {}'.format(anime))

    

    print('Most Similar Animes:\n')

    sim_values = tf_sim.sort_values(by=anime, ascending=False).loc[:,anime].tolist()[1:11]

    sim_animes = tf_sim.sort_values(by=anime, ascending=False).index[1:11]

    zipped = zip(sim_animes, sim_values,)

    for anime, sim in zipped:

        print('{0}, {1:.2f}'.format(anime, sim)) 
find_real_name('doro')
similar_animes_content_based('Dororon Enma-kun Meeramera')
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
pca = PCA(n_components=3)

pca.fit(df_p)

print(pca.explained_variance_ratio_)

print(pca.explained_variance_)
pca_df_p = pca.transform(df_p)

pca_df_p = pd.DataFrame(pca_df_p)

pca_df_p.head(2)
cluster_3 = pd.DataFrame(pca_df_p[[0,1,2]])
plt.rcParams['figure.figsize'] = (10, 6)





fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(cluster_3[0],cluster_3[1],cluster_3[2])



plt.title('Data Distribution PCA in 3D', fontsize=20)

plt.show()
from sklearn.cluster import KMeans

 

'利用SSE选择k'

SSE = []  # 存放每次结果的误差平方和 

for k in range(1,9):

    estimator = KMeans(n_clusters=k)  # 构造聚类器

    estimator.fit(pca_df_p[[0,1,2]])

    SSE.append(estimator.inertia_)

X = range(1,9)

plt.xlabel('k')

plt.ylabel('SSE')

plt.plot(X,SSE,'o-')

from sklearn.metrics import silhouette_score

Scores = []  # 存放轮廓系数 put silhouette scores here

for k in range(2, 14):

    estimator = KMeans(n_clusters=k)  # 构造聚类器 build k-means model

    estimator.fit(np.array(pca_df_p[[0,1,2]]))

    Scores.append(silhouette_score(np.array(pca_df_p[[0,1,2]]), estimator.labels_, metric='euclidean'))

X = range(2, 14)

plt.xlabel('k')

plt.ylabel('Silhouette Coefficient')

plt.plot(X, Scores, 'o-')

plt.show()
clusterer = KMeans(n_clusters=2,random_state=30).fit(cluster_3)

centers = clusterer.cluster_centers_

c_preds = clusterer.predict(cluster_3)
fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(cluster_3[0],cluster_3[1],cluster_3[2], c = c_preds)

plt.title('Data points in 3D PCA axis', fontsize=20)

plt.show()
fig = plt.figure(figsize=(10,8))

plt.scatter(cluster_3[0],cluster_3[1],cluster_3[2],c = c_preds)

for ci,c in enumerate(centers):

    plt.plot(c[1], c[0], c[2],'o', markersize=8, color='red', alpha=1)



plt.xlabel('x_values')

plt.ylabel('y_values')



plt.title('Data points in 2D PCA axis', fontsize=20)

plt.show()
df_p_anime=df_p.columns.tolist()

df_p_anime = pd.DataFrame (df_p_anime,columns=['anime_id'])

df_p_anime=pd.merge(df_p_anime,df1,on='anime_id',how='left')#2826 animes name match

df_p_anime=df_p_anime[['anime_id','name','anime_rating']].drop_duplicates()

df_p_name=df_p.copy()

df_p_name.columns = df_p_anime['name']#2826 animes

df_p_name['cluster'] = c_preds

df_p_name.head()
group1 = df_p_name[df_p_name['cluster']==0]

group2 = df_p_name[df_p_name['cluster']==1]
group1_mean=group1.mean().to_frame()

group1_mean.head(10)
group2_mean=group2.mean().to_frame()

group2_mean.head(10)
c=df_p_name.reset_index()

c=c[['user_id','cluster']]

df1_c=pd.merge(df1,c,on='user_id',how='left')

df1_c=df1_c.dropna(subset=['cluster'])

df1_c['cluster']=df1_c['cluster'].apply(int)

df1_c.head()
df1_c0 = df1_c[df1_c['cluster']==0]

df1_c1 = df1_c[df1_c['cluster']==1]
df1_c0['members'].mean()
df1_c1['members'].mean()
df1_c0['user_rating'].mean()
df1_c1['user_rating'].mean()