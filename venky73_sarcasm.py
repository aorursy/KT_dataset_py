# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_json("../input/Sarcasm_Headlines_Dataset.json",lines=True)
df.sample(2)
print("size : ",df.shape)

print("checking null value:\n",df.isna().sum())
#dropping article link

df.drop(columns=["article_link"], inplace=True)
words = " "

for item in df.headline:

    for inner_item in item.lower().split():

        words+=inner_item+" "

        

from wordcloud import WordCloud, STOPWORDS 

stopwords = set(STOPWORDS) 

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(words) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(df.headline,df.is_sarcastic,test_size = 0.1)
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer()

train_X = vect.fit_transform(train_x)
train_X.shape
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

scores = []

for i in range(6,14):

    model = XGBClassifier(max_depth = i)

    model.fit(train_X,train_y)

    target = model.predict(vect.transform(test_x))

    score = accuracy_score(target,test_y)

    scores.append(accuracy_score(target,test_y))

    print ("accuracy score: ",score," Depth: ",i)

print (scores)

print("best score: ",max(scores))
#first 5 features

vect.get_feature_names()[:5]
import nltk

import nltk.corpus

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

wordnet_lemmatizer.lemmatize("beautifully")
import string

from nltk.corpus import stopwords

stop_words = stopwords.words('english')

print ("punctuations that gonna be removed, : ",string.punctuation)

"""

Punctuation,

removing stop words

words with minimum length 3

lemmatizing words



"""

headlines = []

for item in df.headline:

    word_data = " "

    for item2 in item.split():

        if item2.lower() not in stop_words:

            word_data +=wordnet_lemmatizer.lemmatize(item2.lower())+" "

    headlines.append(word_data)

df['refined'] = headlines

def process(text):

    nopunc = ''.join([char for char in text if char not in string.punctuation])

    clean_words = [word for word in nopunc.split() ]

    clean_words = " ".join(clean_words)

    return clean_words

df['refined'] = df['refined'].apply(process)
df.sample()
words = " "

is_sarcasm = df[df.is_sarcastic==1].refined

for item in is_sarcasm:

    for inner_item in item.lower().split():

        words+=inner_item+" "

        

from wordcloud import WordCloud, STOPWORDS 

import matplotlib.pyplot as plt

stopwords = set(STOPWORDS) 

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(words) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.title("Sarcasm")

plt.show() 

words = " "

is_not_sarcasm = df[df.is_sarcastic==0].refined

for item in is_not_sarcasm:

    for inner_item in item.lower().split():

        words+=inner_item+" "

        

from wordcloud import WordCloud, STOPWORDS 

import matplotlib.pyplot as plt

stopwords = set(STOPWORDS) 

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(words) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.title("NOT Sarcasm")

plt.show() 

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(df.refined,df.is_sarcastic,test_size = 0.1)



from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(token_pattern = "[a-zA-Z]{2,}",max_features=2000,ngram_range=(1,2))

train_X = vect.fit_transform(train_x)

from sklearn.metrics import accuracy_score

scores = []

for i in range(6,14):

    model = XGBClassifier(max_depth = i)

    model.fit(train_X,train_y)

    target = model.predict(vect.transform(test_x))

    score = accuracy_score(target,test_y)

    scores.append(score)

    print ("accuracy score: ",score," Depth: ",i)

print (scores)

print("best score: ",max(scores))
vect.get_feature_names()[:5]