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
import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

from termcolor import colored

from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

import re

from wordcloud import WordCloud



import warnings

warnings.filterwarnings('ignore', category = DeprecationWarning)
BATCH_SIZE = 64

EPOCHS = 3

VOCAB_SIZE = 25000

MAX_LEN =90

EMBEDDING_DIM = 128
train=pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

train.head()
train=train[['id','text', 'target']]

train.shape
train.target.value_counts()
def expand_tweet(tweet):

    expanded_tweet = []

    for word in tweet:

        if re.search("n't", word):

            expanded_tweet.append(word.split("n't")[0])

            expanded_tweet.append("not")

        else:

            expanded_tweet.append(word)

    return expanded_tweet


def clean_tweet(data, wordNetLemmatizer, porterStemmer):

    data['text'] = data['text']

    print(colored("Removing user handles starting with @", "yellow"))

    data['text'] = data['text'].str.replace("@[\w]*","")

    print(colored("Removing numbers and special characters", "yellow"))

    data['text'] = data['text'].str.replace("[^a-zA-Z' ]","")

    print(colored("Removing urls", "yellow"))

    data['text'] = data['text'].replace(re.compile(r"((www\.[^\s]+)|(https?://[^\s]+))"), "")

    print(colored("Removing single characters", "yellow"))

    data['text'] = data['text'].replace(re.compile(r"(^| ).( |$)"), " ")

    print(colored("Tokenizing", "yellow"))

    data['text'] = data['text'].str.split()

    print(colored("Removing stopwords", "yellow"))

    data['text'] = data['text'].apply(lambda text: [word for word in text if word not in STOPWORDS])

    print(colored("Expanding not words", "yellow"))

    data['text'] = data['text'].apply(lambda text: expand_tweet(text))

    print(colored("Lemmatizing the words", "yellow"))

    data['text'] = data['text'].apply(lambda text: [wordNetLemmatizer.lemmatize(word) for word in text])

    print(colored("Stemming the words", "yellow"))

    data['text'] = data['text'].apply(lambda text: [porterStemmer.stem(word) for word in text])

    print(colored("Combining words back to tweets", "yellow"))

    data['text'] = data['text'].apply(lambda text: ' '.join(text))

    return data
STOPWORDS= stopwords.words("english")

wordNetLemmatizer = WordNetLemmatizer()

porterStemmer = PorterStemmer()
train = clean_tweet(train, wordNetLemmatizer, porterStemmer)

#real_tweets = ' '.join(train[train['target'] == 0]['text'].str.lower())

#fake_tweets = ' '.join(train[train['target'] == 1]['text'].str.lower())
#wordcloud = WordCloud(stopwords = STOPWORDS, background_color = "white", max_words = 1000).generate(real_tweets)

#plt.figure(figsize = (12, 8))

#plt.imshow(wordcloud)

#plt.axis("off")

#plt.title("Real tweets Wordcloud")
#wordcloud = WordCloud(stopwords = STOPWORDS, background_color = "white", max_words = 1000).generate(fake_tweets)

#plt.figure(figsize = (12, 8))

#plt.imshow(wordcloud)

#plt.axis("off")

#plt.title("Fake tweets Wordcloud")
from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense, Embedding, GlobalMaxPooling1D, Flatten, Conv1D, Dropout, Activation

from keras.preprocessing.text import Tokenizer

from nltk.tokenize import RegexpTokenizer
#tokenizer = RegexpTokenizer(r'\w+') 
tokenizer = Tokenizer(num_words=VOCAB_SIZE)

tokenizer.fit_on_texts(train['text'].values)

#train['text'] = train['text'].map(tokenizer.tokenize)
#from sklearn.model_selection import train_test_split

#X_train, X_val, y_train, y_val = train_test_split(train['text'], train['target'], test_size=0.25, random_state=42)
X_train=train['text']

y_train=train['target']
y_train.shape
X_train.head()
x_train_seq = tokenizer.texts_to_sequences(X_train.values)

#x_val_seq = tokenizer.texts_to_sequences(X_val.values)



x_train = sequence.pad_sequences(x_train_seq, maxlen=MAX_LEN, padding="post", value=0)

#x_val = sequence.pad_sequences(x_val_seq, maxlen=MAX_LEN, padding="post", value=0)
y_train.head()
print('First sample before preprocessing: \n', train['text'].values[2], '\n')

print('First sample after preprocessing: \n', x_train[2])
# Model Parameters 



NUM_FILTERS = 250

KERNEL_SIZE = 4

HIDDEN_DIMS = 250
"""

print('Build model...')

model = Sequential()



# we start off with an efficient embedding layer which maps

# our vocab indices into EMBEDDING_DIM dimensions

model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN))

model.add(Dropout(0.25))



# we add a Convolution1D, which will learn NUM_FILTERS filters

model.add(Conv1D(NUM_FILTERS,

                 KERNEL_SIZE,

                 padding='valid',

                 activation='relu',

                 strides=1))



# we use max pooling:

model.add(GlobalMaxPooling1D())



# We add a vanilla hidden layer:

model.add(Dense(HIDDEN_DIMS))

model.add(Dropout(0.2))

model.add(Activation('relu'))



# We project onto a single unit output layer, and squash it with a sigmoid:

model.add(Dense(1))

model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

"""
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, LSTM, Embedding,Bidirectional,Dropout

from tensorflow.keras.preprocessing.sequence import pad_sequences


model = Sequential()



model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM,  input_length=x_train.shape[1]))

model.add(Bidirectional(LSTM(64,return_sequences=True)))

model.add(Bidirectional(LSTM(128)))

model.add(Dense(80, activation='elu'))

model.add(Dropout(0.25))

model.add(Dense(100, activation='elu'))

model.add(Dropout(0.25))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# fit a model

model.fit(x_train, y_train,

          batch_size=BATCH_SIZE,

          epochs=1,

          validation_split=0.15,

          verbose=2)



# Evaluate the model

#score, acc = model.evaluate(x_val, y_val, batch_size=BATCH_SIZE)

#print('\nAccuracy: ', acc*100)



#pred = model.predict_classes(x_val)
test=pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

test=test[['text']]
test.head()
test.shape
test=clean_tweet(test, wordNetLemmatizer, porterStemmer)

tokenizer.fit_on_texts(test['text'].values)

X_test=test['text']

x_test_seq = tokenizer.texts_to_sequences(X_test.values)

x_test = sequence.pad_sequences(x_test_seq, maxlen=MAX_LEN, padding="post", value=0)
prediction = model.predict(x_test)
prediction
submission_sample = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

submission_sample.head()
submission_sample.shape
submit = submission_sample.copy()

submit.target = np.where(prediction > 0.5,1,0)
submit.to_csv('submit4.csv',index=False)