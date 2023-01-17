# all required imports

import json

from urllib.request import urlopen

import re

import time

!pip install praw

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

import pandas as pd

import matplotlib.pyplot as plt

import praw

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.pipeline import Pipeline

import numpy as np





%matplotlib inline
baseurl = 'https://api.pushshift.io/reddit/search/submission/?subreddit=india&sort=desc&sort_type=created_utc&'



period = 1 # in years



week_seconds = 7 * 24 * 60 * 60

week_count = 1

start_seconds = int(time.time()) - ((week_count - 1) * week_seconds) # start time of collection



all_data = []



while week_count <= int(period * 52):

    before, after = start_seconds, start_seconds - week_seconds

    url = baseurl + 'after={}&before={}&size=1000'.format(after, before)

    start_seconds = after

    r = urlopen(url).read()

    data = json.loads(r.decode('utf-8'))

    all_data += data['data']

    print(f'Week {week_count}/{period*52} done')

    week_count += 1

    

    

all_data_DF = pd.DataFrame(all_data)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
all_data_DF = pd.read_csv("../input/redditindiadata/reddit_india_latest.csv")

all_data_DF
keep_features = ['title','url', 'selftext', 'link_flair_text','comments']
all_data_DF.rename(columns={'body': 'selftext', 'flair': 'link_flair_text'}, inplace=True)
collected_data = all_data_DF[keep_features] 
with_flair = collected_data[collected_data.link_flair_text.notnull()]

with_flair.reset_index(drop=True, inplace=True)

with_flair
from collections import Counter

counts = Counter(with_flair['link_flair_text'])

counts
plt.figure(figsize=(10,4))

with_flair.link_flair_text.value_counts().plot(kind='bar');
threshold = 1000

main_flairs = [flair for flair in counts if counts[flair] > threshold]

#main_flairs = ['Coronavirus','Politics', 'Non-Political','Science/Technology','AskIndia', 'Policy/Economy']
with_flair.link_flair_text.value_counts().plot(kind='bar');
df_col_len = int(with_flair['comments'].str.encode(encoding='utf-8').str.len().max())

df_col_len
STOPWORDS = set(stopwords.words('english'))
def cleaning(text):

    text = str(text).lower() 

    tokens  = word_tokenize(text)

    text = ' '.join(word for word in tokens if word not in STOPWORDS and word.isalpha()) 

    return text



def clean_url(url):

    non_uselful = ['http', 'https', 'www', 'com', 'reddit']

    add_ons = ['cms', 'comments', 'r', 'redd', 'google', 'amp', 'co', 'youtu', 'india', 'jpg', 'article', 'youtube', 'png', 'twitter']

    non_uselful += add_ons

    non_uselful = set(non_uselful)

    delimiters = [':', '/', '_', '-', '.']

    pattern = '|'.join(map(re.escape, delimiters))

    try:

        url = re.split(pattern,url)

    except:

        return ''

    else:

        url = ' '.join(word for word in url if word not in STOPWORDS and word.isalpha() and word not in non_uselful) 

        #print(url)

        return url

with_flair['title'] = with_flair['title'].apply(cleaning)

with_flair['selftext'] = with_flair['selftext'].apply(cleaning)

with_flair['url'] = with_flair['url'].apply(clean_url)

with_flair['comments'] = with_flair['comments'].apply(cleaning)



with_flair['content'] = with_flair['title'].str.cat(with_flair['selftext'], sep =" ") 

with_flair['content'] = with_flair['content'].str.cat(with_flair['url'], sep =" ") 

with_flair['content'] = with_flair['content'].str.cat(with_flair['comments'], sep =" ") 

with_flair['title_url']=with_flair['title'].str.cat(with_flair['url'],sep=" ")

with_flair['title_url_self']=with_flair['title_url'].str.cat(with_flair['selftext'],sep=" ")
df_col_len = int(with_flair['comments'].str.encode(encoding='utf-8').str.len().max())

df_col_len
with_flair = with_flair[with_flair.content!='']

with_flair.reset_index(drop=True, inplace=True)

with_flair
with_flair['title'].apply(lambda x: len(x.split(' '))).sum()
with_flair = with_flair.sample(frac=0.5)
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics import accuracy_score, confusion_matrix

X = with_flair.title

y = with_flair.link_flair_text

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline

import numpy as np

cv=CountVectorizer(stop_words=stopwords.words("english"), analyzer='word',max_features=1000,

                token_pattern=r'\b[^\d\W]+\b')

x=cv.fit_transform(with_flair.iloc[:,0])

#len(cv.get_feature_names()),cv.get_feature_names()

#cv.vocabulary_

tf=TfidfTransformer()

tfx=tf.fit_transform(x)
idf=TfidfVectorizer(stop_words=stopwords.words("english"), analyzer='word',max_features=1000,

                    token_pattern=r'\b[^\d\W]+\b')

xdf=idf.fit_transform(X_train)

xdt=idf.transform(X_test)
def naivebayes(X_train, X_test, y_train, y_test):

    from sklearn.naive_bayes import MultinomialNB

    from sklearn.metrics import classification_report

    from sklearn.pipeline import Pipeline

    from sklearn.feature_extraction.text import TfidfTransformer



    nb = Pipeline([('vect', CountVectorizer()),

               ('tfidf', TfidfTransformer()),

               ('clf', MultinomialNB()),

              ])

    nb.fit(X_train, y_train)





    from sklearn.metrics import classification_report

    y_pred = nb.predict(X_test)

    my_tags = list(set(with_flair.link_flair_text))

    print('accuracy %s' % accuracy_score(y_pred, y_test))

    print(classification_report(y_test, y_pred,target_names=my_tags))
def linearsvm(X_train, X_test, y_train, y_test):

    from sklearn.linear_model import SGDClassifier

    from sklearn.metrics import classification_report

    sgd = Pipeline([('vect', CountVectorizer()),

                ('tfidf', TfidfTransformer()),

                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-4, random_state=42,max_iter=100, tol=None)),

               ])

    sgd.fit(X_train, y_train)





    y_pred = sgd.predict(X_test)

    my_tags = list(set(with_flair.link_flair_text))

    print('accuracy %s' % accuracy_score(y_pred, y_test))

    print(classification_report(y_test, y_pred,target_names=my_tags))
def logisticreg(X_train, X_test, y_train, y_test):

    from sklearn.linear_model import LogisticRegression

    from sklearn.metrics import classification_report



    logreg = Pipeline([('vect', CountVectorizer()),

                ('tfidf', TfidfTransformer()),

                ('clf', LogisticRegression(n_jobs=1, C=1e5)),

               ])

    logreg.fit(X_train, y_train)



    y_pred = logreg.predict(X_test)

    my_tags = list(set(with_flair.link_flair_text))

    print('accuracy %s' % accuracy_score(y_pred, y_test))

    print(classification_report(y_test, y_pred,target_names=my_tags))


def randomforest(X_train, X_test, y_train, y_test):

    from sklearn.ensemble import RandomForestClassifier

    from sklearn.metrics import classification_report

    import seaborn as sns



    ranfor = Pipeline([('vect', CountVectorizer()),

                  ('tfidf', TfidfTransformer()),

                  ('clf', RandomForestClassifier(n_estimators = 400, random_state = 42, max_depth=30)),

                 ])



    ranfor.fit(X_train, y_train)

    y_pred = ranfor.predict(X_test)

    from sklearn.externals import joblib 

#     joblib.dump(ranfor, 'joblib_model.pkl') 

#     f_joblib = joblib.load('filename.pkl')  

#     f_joblib.predict(X_test) 

#     import pickle



#     pkl_filename = "final_model.pkl"

#     with open(pkl_filename, 'wb') as file:

#         pickle.dump(ranfor, file)

    my_tags = list(set(with_flair.link_flair_text))

    



    print('accuracy %s' % accuracy_score(y_pred, y_test))

    print(classification_report(y_test, y_pred,target_names=my_tags))

    

    
def mlpclassifier(X_train, X_test, y_train, y_test):

    from sklearn.neural_network import MLPClassifier

    from sklearn.metrics import classification_report

    mlp = Pipeline([('vect', CountVectorizer()),

                  ('tfidf', TfidfTransformer()),

                  ('clf', MLPClassifier(hidden_layer_sizes=(30,30,30))),

                 ])

    mlp.fit(X_train, y_train)



    y_pred = mlp.predict(X_test)

    my_tags = list(set(with_flair.link_flair_text))

    print('accuracy %s' % accuracy_score(y_pred, y_test))

    print(classification_report(y_test, y_pred,target_names=my_tags))
def train_test(X,y):

    from sklearn.metrics import confusion_matrix

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

    print("Results of Naive Bayes Classifier")

    naivebayes(X_train, X_test, y_train, y_test)

    print("Results of Linear Support Vector Machine")

    linearsvm(X_train, X_test, y_train, y_test)

    print("Results of Logistic Regression")

    logisticreg(X_train, X_test, y_train, y_test)

    print("Results of Random Forest")

    randomforest(X_train, X_test, y_train, y_test)

    print("Results of MLP Classifier")

    mlpclassifier(X_train, X_test, y_train, y_test)

#     print(len(X_train))
target= with_flair.link_flair_text

c=with_flair.content

t= with_flair.title

tu=with_flair.title_url

tus=with_flair.title_url_self

print("title as feature")

train_test(t,target)

print("tile+url as feature")

train_test(tu,target)

print("title+url+selftext as feature")

train_test(tus,target)

print("content as feature")

train_test(c,target)
import itertools

import os

from sklearn.utils import shuffle



%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import tensorflow as tf



from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from sklearn.metrics import confusion_matrix



from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.preprocessing import text, sequence

from keras import utils



with_flair = shuffle(with_flair)

train_size = int(len(with_flair) * 0.7)

train_posts = with_flair['title_url_self'][:train_size]

train_tags = with_flair['link_flair_text'][:train_size]





test_posts = with_flair['title_url_self'][train_size:]

test_tags = with_flair['link_flair_text'][train_size:]



max_words = 5000

tokenize = text.Tokenizer(num_words=max_words, char_level=False)

tokenize.fit_on_texts(train_posts) # only fit on train



x_train = tokenize.texts_to_matrix(train_posts)

x_test = tokenize.texts_to_matrix(test_posts)



encoder = LabelEncoder()

encoder.fit(train_tags)

y_train = encoder.transform(train_tags)

y_test = encoder.transform(test_tags)



num_classes = np.max(y_train) + 1

y_train = utils.to_categorical(y_train, num_classes)

y_test = utils.to_categorical(y_test, num_classes)



batch_size = 32

epochs = 30



# Build the model

model = Sequential()

model.add(Dense(512, input_shape=(max_words,)))

model.add(Activation('relu'))

model.add(Dense(128))

model.add(Dropout(0.5))

model.add(Dense(64))

model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

              

history = model.fit(x_train, y_train,

                    batch_size=batch_size,

                    epochs=epochs,

                    verbose=1,

                    validation_split=0.3)

score = model.evaluate(x_test, y_test,

                       batch_size=batch_size, verbose=1)

print('Test accuracy:', score[1])
submission = pd.DataFrame({

        "title_url_self": with_flair["title_url_self"],

        "link_flair_text": with_flair["link_flair_text"]

    })

submission.to_csv("submission.csv", index=False)

submission.head()