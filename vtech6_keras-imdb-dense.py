# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv(r'/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
encoded_reviews = data.replace(('positive', 'negative'),(1, 0))

import re

def cleaner(data):

    html = re.compile(r'<.*?>')

    return html.sub(r'', data)

encoded_reviews['review'] = encoded_reviews['review'].apply(lambda x: cleaner(x))

encoded_reviews.head()
import nltk

from nltk.corpus import stopwords

from nltk import regexp_tokenize

from nltk.stem import WordNetLemmatizer

lemmatize = WordNetLemmatizer()

patn= '\w+'

sw = stopwords.words('english')

reviews = encoded_reviews['review']

def no_sw(sentence):

    sentence_split=regexp_tokenize(sentence.lower(), patn)

    no_sw_sentence=[]

    for word in sentence_split:

        if word not in sw:

            no_sw_sentence.append(word)

    #return ' '.join(np.asarray(no_sw_sentence))

    return no_sw_sentence

no_sw_reviews = reviews.apply(lambda x: no_sw(x))
c = []

for arr in no_sw_reviews:

    c+=arr
total_freq = nltk.FreqDist(word for word in c)

unwanted_words=['movie', 'film']

for i in unwanted_words:

    total_freq.pop(i)

total_freq
total_words = [list(wds) for wds in zip(*total_freq)][0]

most_common = [list(wds) for wds in zip(*total_freq.most_common(2000))][0]

print(len(most_common), len(total_words))
from sklearn.feature_extraction.text import TfidfVectorizer

def join_review(review):

    return ' '.join(review)

joined_wo_sw = no_sw_reviews.apply(lambda x: join_review(x))



#countvec = CountVectorizer(vocabulary=most_common)

tfidf = TfidfVectorizer(vocabulary=most_common)
reviews_df = pd.DataFrame([i for i in joined_wo_sw], columns=['reviews'])

final_df = pd.DataFrame(tfidf.fit_transform(reviews_df.reviews).toarray(), index=reviews_df.index, columns=tfidf.get_feature_names())

final_df.insert(0, 'sentiment', [i for i in encoded_reviews['sentiment']])

final_df.head()
print(max(final_df.max()))
train, test = train_test_split(final_df, test_size=0.2, shuffle=False)
X_train = train.drop('sentiment', axis=1).to_numpy()

y_train = train['sentiment'].to_numpy()

X_test = test.drop('sentiment', axis=1).to_numpy()

y_test = test['sentiment'].to_numpy()
model = keras.models.Sequential()



model.add(keras.layers.Dense(512, activation='relu'))

model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(256, activation='relu'))

model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',

             loss='binary_crossentropy',

             metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=200, epochs=5)
preds = model.predict_classes(X_test)

preds[0]



from sklearn.metrics import accuracy_score

acc = accuracy_score(preds, y_test)

acc