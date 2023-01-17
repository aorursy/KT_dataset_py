# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline

import math

import numpy as np

import pandas as pd

import scipy.stats as ss

import cufflinks as cf

from collections import Counter



import re

import nltk

from nltk.corpus import stopwords

from textblob import TextBlob

from nltk.tokenize import word_tokenize

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator





from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn import svm



import keras

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D, Bidirectional

from keras.callbacks import ModelCheckpoint

from sklearn.metrics import roc_auc_score

from keras.preprocessing.text import Tokenizer

data = pd.read_csv('../input/Tweets.csv')

data.head(100)

data.info()

column_names = []

column_counts = []

for column in data:

    column_names.append(column)

    column_counts.append(data[column].count())
#visualize columns count

plt.subplots(figsize=(25,20))

sns.barplot(x=column_names, y=column_counts)
#visualize columns names

list(data.columns)

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)



data['airline_sentiment'].iplot(

    kind='hist',

    bins=50,

    xTitle='polarity',

    linecolor='black',

    yTitle='count',

    title='airline_sentiment Distribution')
data['text'].iplot(

    kind='hist',

    bins=50,

    xTitle='polarity',

    linecolor='black',

    yTitle='count',

    title='text Distribution')
data = data.drop_duplicates()
data['text_length'] = data['text'].apply(len)

hist = data.hist(bins=4)
dp=data[ data['airline_sentiment'] == 'positive']

dg=data[ data['airline_sentiment'] == 'negative']

dn=data[ data['airline_sentiment'] == 'neutral']

positive_retweet_mean=dp['retweet_count'].mean()

negative_retweet_mean=dg['retweet_count'].mean()

neutral_retweet_mean=dn['retweet_count'].mean()

print("positive_retweet_mean", positive_retweet_mean)

print("negative_retweet_mean", negative_retweet_mean)

print("neutral_retweet_mean",neutral_retweet_mean)
data.corr()

def correlation_ratio(categories, measurements):

    fcat, _ = pd.factorize(categories)

    cat_num = np.max(fcat)+1

    y_avg_array = np.zeros(cat_num)

    n_array = np.zeros(cat_num)

    for i in range(0,cat_num):

        cat_measures = measurements[np.argwhere(fcat == i).flatten()]

        n_array[i] = len(cat_measures)

        y_avg_array[i] = np.average(cat_measures)

    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)

    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))

    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))

    if numerator == 0:

        eta = 0.0

    else:

        eta = np.sqrt(numerator/denominator)

    return eta
#rank for the bad airline

negative_tweets = data[data['airline_sentiment'].str.contains("negative")]

bad_airline = negative_tweets[['airline','airline_sentiment_confidence','negativereason']]

bad_airline_count = bad_airline.groupby('airline', as_index=False).count()

bad_airline_count.sort_values('negativereason', ascending=False)
#rank for the good airline

positive_tweets = data[data['airline_sentiment'].str.contains("positive")]

good_airline = positive_tweets[['airline','airline_sentiment_confidence']]

good_airline_count = good_airline.groupby('airline', as_index=False).count()

good_airline_count.sort_values('airline_sentiment_confidence', ascending=False)

reason = negative_tweets[['airline','negativereason']]

bad_flight_reason_count = reason.groupby('negativereason', as_index=False).count()

bad_flight_reason_count.sort_values('negativereason', ascending=False)
data_new = pd.read_csv('../input/Tweets.csv')

df = pd.DataFrame([])

df['airline_sentiment'] = data_new['airline_sentiment']

df['text'] = data_new['text']

df['negativereason'] = data_new['negativereason']
df.head()
class Normalizer:

    def __init__(self):

        self.stop_words = stopwords.words('english')

    

    def lower(self, text):

        return text.lower()

    

    def remove_punctuations(self, text):

        return re.sub('[^\w\s]','', text)

    

    def remove_mentions(self, text):

        return re.sub('@[a-zA-Z0-9-._]+', '', text)

    

    def remove_url(self, text):

        url_ptrn = r'''(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|

            www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|

            \(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|

            (\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»""'']))'''

        return re.sub(url_ptrn, '', text)

    

    def remove_email(self, text):

        return re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',

                             '', text)

    def remove_stop_words(self, text):

        return " ".join(x for x in text.split() if x not in self.stop_words)

    

    def correct_spelling(self, text):

        return str(TextBlob(text).correct())

    

    def tokenize(self, text):

        return word_tokenize(text)

    

    def normalize(self, text):

        text = self.lower(text)

        text = self.remove_url(text)

        text = self.remove_email(text)

        text = self.remove_mentions(text)

# #         text = self.correct_spelling(text)

#         text = self.remove_stop_words(text)

        text = self.remove_punctuations(text)

        text = text.strip()

        return text
normalizer = Normalizer()

df.text = df.text.apply(normalizer.normalize)

def word_cloud(text, Stop_words=None,path=None,height=3000, width=3000):

    wc = WordCloud(background_color="white", max_words=2000,

                contour_width=3, contour_color='steelblue', stopwords=Stop_words,height=3000,width=3000).generate(text)

    wc.to_file(path)

    plt.imshow(wc, interpolation='bilinear', shape=(width, height ))

    plt.axis("off")

    plt.figure()



all_text = ''

for row in df.text:

    all_text += ' ' + str(row)

word_cloud(all_text, path='Cloud.png')

vectorizer = TfidfVectorizer(ngram_range=(1,1))

X = vectorizer.fit_transform(df.text)

y = df.airline_sentiment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

clf_mnb = MultinomialNB().fit(X_train, y_train)



clf_svm = svm.LinearSVC(penalty='l2',C=1).fit(X_train, y_train)

print ("MNB score for train",clf_mnb.score(X_train, y_train))

print ("MNB score for test",clf_mnb.score(X_test, y_test))

print ("SVM score for train",clf_svm.score(X_train, y_train))

print ("SVM score for test",clf_svm.score(X_test, y_test))
max_fatures = 2000

tokenizer = Tokenizer(num_words=max_fatures, split=' ')

tokenizer.fit_on_texts(df.text.values)

X = tokenizer.texts_to_sequences(df.text.values)

X = pad_sequences(X)
Y = pd.get_dummies(df['airline_sentiment']).values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42, stratify=Y)

X_train.shape
X_vald = X_train[:500]

Y_vald = Y_train[:500]

x_train = X_train[500:]

y_train = Y_train[500:]
embed_dim = 128

lstm_out = 196



model = Sequential()

model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))

model.add(Dropout(0.5))

model.add(Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2)))

model.add(Dense(3,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

print(model.summary())
batch_size = 512

history = model.fit(x_train, 

                    y_train, 

                    epochs = 10, 

                    batch_size=batch_size, 

                    validation_data=(X_vald, Y_vald))
import matplotlib.pyplot as plt

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
plt.clf()

acc = history.history['acc']

val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
scores = model.evaluate(X_test, Y_test)

scores