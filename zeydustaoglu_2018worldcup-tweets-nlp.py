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
df = pd.read_csv('/kaggle/input/world-cup-2018-tweets/FIFA.csv', encoding = 'latin1')
df.head(10)
df.columns
df.describe().T
#like ortalamalari
likes = df.Likes.unique()
likes_mean = likes.mean()

#followers ortalamalari
followers = df.Followers.unique()
followers_mean = followers.mean()
df = df[~(df == 0).any(axis=1)]
df.head(10)
data = pd.concat([df.Likes,df.Tweet,df.Followers],axis=1)
data.dropna(axis = 0,inplace = True)
#changed Likes column 
data.Likes = [1 if each > likes_mean else 0 for each in data.Likes]

#changed Followers column 
data.Followers = ['Famous' if each > followers_mean else 'Not Famous' for each in data.Followers]
data.head(10)
# regular expression RE mesela "[^a-zA-Z]"
import re

# import nltk # natural language tool kit
# nltk.download("wordnet")      # corpus diye bir kalsore indiriliyor

from nltk.corpus import stopwords
import nltk as nlp

tweet_list = []
for tweet in data.Tweet:
    tweet = re.sub("[^a-zA-Z]"," ",tweet)
    tweet = tweet.lower()   # buyuk harftan kucuk harfe cevirme
    
    tweet = nlp.word_tokenize(tweet)
#     description = [ word for word in description if not word in set(stopwords.words("english"))]
    
    lemma = nlp.WordNetLemmatizer()
    tweet = [ lemma.lemmatize(word) for word in tweet]
    
    tweet = " ".join(tweet)
    tweet_list.append(tweet)

# %% bag of words

from sklearn.feature_extraction.text import CountVectorizer # bag of words yaratmak icin kullandigim metot
max_features =500

count_vectorizer = CountVectorizer(max_features=max_features,stop_words = "english")

sparce_matrix = count_vectorizer.fit_transform(tweet_list).toarray()  # x

print("en sik kullanilan {} kelimeler: {}".format(max_features,count_vectorizer.get_feature_names()))
# %%
y = data.iloc[:,0].values   # upper then 800 or not classes
x = sparce_matrix
# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 42)
# %% naive bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
#%% prediction
y_pred = nb.predict(x_test)

print("accuracy: ",nb.score(y_pred.reshape(-1,1),y_test))
