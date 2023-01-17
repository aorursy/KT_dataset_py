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
import pandas as pd

import numpy as np

import os

pd.set_option("display.max_rows", None, "display.max_columns", None)



import warnings

warnings.filterwarnings('ignore')
posts = pd.read_csv('../input/posts.csv')

posts.rename(columns={'_id':'post_id', ' post_type':'post_type'}, inplace=True)

posts.category.fillna("General", inplace = True) 

posts.head()
print(posts.shape)

posts.info()
users = pd.read_csv('../input/users.csv')

users.rename(columns={'_id':'user_id'}, inplace=True)

users.head()
print(users.shape)

users.info()
views = pd.read_csv('../input/views.csv')

views.head()
print(views.shape)

views.info()
df = views.merge(posts, on='post_id', how='left')

df.head()
df = df.merge(users, on='user_id', how='left')

df.head()
df.shape
df.info()
df1 = df[df.isna().any(axis=1)]

df.shape
df = df.dropna(thresh=8)

df.shape
# df.head()
df.category.value_counts()
# df = df.drop(columns=['name', 'timestamp'])

# df = df.reset_index(drop=True)

# df.head()
df.shape
import matplotlib.pyplot as plt

import seaborn as sns



# plot stylings

plt.style.use('fivethirtyeight')

%matplotlib inline
pd.crosstab(df.title, df.gender, margins=True).style.background_gradient(cmap='summer_r')
fig, ax = plt.subplots(1,2,figsize = (10,6))



sns.countplot('gender',data=df,ax=ax[0])

ax[0].set_title('Posts viewed by Gender')



df['gender'].value_counts().plot.pie(explode=[0,0.1,0],autopct='%1.1f%%',shadow=True,ax=ax[1])

ax[1].set_title('Posts viewed by Gender')

ax[1].axis('equal') 



fig.tight_layout()

plt.show()
pd.crosstab(df.title, [df.academics, df.gender], margins=True).style.background_gradient(cmap='summer_r')
fig, ax = plt.subplots(figsize = (10,8))

sns.countplot('academics',hue='gender',data=df)

ax.set_title('Gender and Academics')



fig.tight_layout()

plt.show()
# Displaying top categories

rest, keys, values = 0, [], []

x = df.category.value_counts()

for i,j in x.items():

    if j>=10:

        keys.append(i)

        values.append(j)

    else:

        rest += j

keys.append('Remaining')

values.append(rest)

# print(len(keys))

# print(len(values))
fig, ax = plt.subplots(figsize = (10, 15))

y_pos = np.arange(len(values))

ax.barh(y_pos, values[::-1], color = ['b'], alpha=0.99)

ax.set_yticks(y_pos)

ax.set_yticklabels(keys[::-1])

ax.set_xlabel('Number of posts')

ax.set_title('Top Categories')



plt.show()
fig, ax = plt.subplots(figsize = (10,8))



df['post_type'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True)

ax.set_title('Popular post types')

ax.axis('equal') 



fig.tight_layout()

plt.show()
# Importing required libraries



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import linear_kernel 
tf = TfidfVectorizer(analyzer='word', sublinear_tf=True, ngram_range=(1, 3), min_df=0, stop_words='english')

tfidf_matrix = tf.fit_transform(posts['category'])

tfidf_matrix.shape
tf.get_feature_names()[:-30:-1]
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

cosine_similarities[0]
postid = posts['post_id']

indices = pd.Series(posts.index, index=posts['post_id'])

indices.head()
def item(id):

    return posts.loc[posts['post_id'] == id]['title'].tolist()[0]
def get_recommendations(postid, num, indices):

    idx = indices[postid]

    sim_scores = list(enumerate(cosine_similarities[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:num+1]

    # print(sim_scores)

    indices = [i[0] for i in sim_scores]

    print("Recommending " + str(num) + " posts similar to \"" + item(postid) + "\" ...")

    return posts.iloc[indices]
# Attempt 1

get_recommendations('5eac305f10426255a7aa9dd3', 10, indices)
# Attempt 2

get_recommendations('5d6d39567fa40e1417a4931c', 10, indices)
# posts.head()
posts['soup'] = posts['title'] + " " +posts['category'] + " " +posts['post_type']

posts.head()
tf = TfidfVectorizer(analyzer='word', sublinear_tf=True, ngram_range=(1, 8), min_df=0, stop_words='english')

tfidf_matrix = tf.fit_transform(posts['soup'])

tfidf_matrix.shape
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

cosine_similarities[0]
# posts.head()
postid = posts['post_id']

indices = pd.Series(posts.index, index=posts['post_id'])

# indices.head()
# Attempt 3

get_recommendations('5eac305f10426255a7aa9dd3', 10, indices)
# Attempt 4

get_recommendations('5d6d39567fa40e1417a4931c', 10, indices)
df.head()
df[df['user_id'] == '5df49b32cc709107827fb3c7']
df[df['post_id']=='5ec821ddec493f4a2655889e']
val = df.post_id.value_counts().tolist()

r_dict = dict(df.post_id.value_counts())

# r_dict
def ratings_norm(ratings_dict):

    (a,b) = (1,5)

    for key, value in ratings_dict.items():

        ratings_dict[key] = (b-a)*((value-min(val))/(max(val)-min(val))) + a



    return ratings_dict



ratings = ratings_norm(r_dict)

# print(ratings)
df['rating'] = 0
for key, value in ratings.items():

    df.loc[df.post_id==key, 'rating'] = value

df.head()
!pip install scikit-surprise

from surprise import Reader, Dataset, SVD, accuracy

from surprise.model_selection import cross_validate, KFold, GridSearchCV
reader = Reader()

data = Dataset.load_from_df(df[['user_id', 'post_id', 'rating']], reader)

kf = KFold(n_splits = 5)
algo = SVD()

cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
for trainset, testset in kf.split(data):



    # train and test algorithm.

    algo.fit(trainset)

    predictions = algo.test(testset)



    # Compute and print Root Mean Squared Error

    accuracy.rmse(predictions, verbose=True)
trainset = data.build_full_trainset()

algo.fit(trainset)
df[df['user_id']=='5d60098a653a331687083238'].head()
algo.predict('5d60098a653a331687083238', '5ed1ff0276027d35905cc60d', 3.588235)