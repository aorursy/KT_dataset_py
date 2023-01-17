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
df = pd.read_csv("/kaggle/input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv",index_col = 0)

df.head()
from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D,SimpleRNN,Flatten

from keras.layers import Dropout

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

import re

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.sentiment.util import *

import scikitplot as skplt

from wordcloud import WordCloud

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer 

from nltk.stem import PorterStemmer, LancasterStemmer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

import matplotlib.pyplot as plt 

from textblob import TextBlob

import nltk

import warnings

warnings.filterwarnings('ignore') 
# only take the review column as predictor and recommended IND as target in the dataset 

# rename the predictor and target

data = df[["Title","Review Text","Recommended IND"]]

data = data.rename(columns = {"Review Text":"text","Recommended IND":"sentiment"})

data.head()
# calculate the null values in the dataset

data.text.isna().sum()
# the null values only occupies a very small proportion thus we can directly delete them

data = data[~data.text.isna()]
print(data.sentiment.isna().sum(),data.text.isna().sum())

# now no na values
# let's check whether the sentiment columns contain other values or not

data.sentiment.unique()

# Great, no other values, just binary result
def count_exclamation_mark(string_text):

    count = 0

    for char in string_text:

        if char == '!':

            count += 1

    return count
# calculate the ! number in text

data['count_exc'] = data['text'].apply(count_exclamation_mark)

data.head(5)


# transfer all the text into lower case

data['text'] = data['text'].str.lower()

# clear all the non-related notation

data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

data.head(6)
# generate new feature, the length of text

data['text_length'] = data['text'].apply(len)

data.head()
# view the distribution of the target variable

print(len(data[data.sentiment == 1]))

print(len(data[data.sentiment == 0 ]))
# generate new feature, the polarity of the one review text

data['Polarity'] = data['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

data.head(5)
# manully set the stop words in this situation

stop_words = list(set(stopwords.words('english')))

clothes_list =['dress','sweater','shirt',

               'skirt','material', 'white', 'black',

              'jeans', 'fabric', 'color','order', 'wear']



for i in clothes_list:

    stop_words.append(i)

def stopwords_removal(messy_str):

    messy_str = word_tokenize(messy_str)

    return [word.lower() for word in messy_str 

            if word.lower() not in stop_words ]
# remove all the words which are in the stop word list

data['text'] = data['text'].apply(stopwords_removal)

data.head()
# stemming transformation of text

porter = PorterStemmer()

def stem_update(text_list):

    text_list_new = []

    for word in text_list:

        word = porter.stem(word)

        text_list_new.append(word) 

    return text_list_new
data['text'] = data['text'].apply(stem_update)

data['text'].head()
data['text'] = data['text'].apply(lambda x:' '.join(x))

data['text'].head()
# create word cloud

pos_df = data[data.sentiment== 1]

neg_df = data[data.sentiment== 0]

pos_df.head(3)
pos_words =[]

neg_words = []



for review in pos_df.text:

    pos_words.append(review) 

pos_words = ' '.join(pos_words)

pos_words[:60]



for review in neg_df.text:

    neg_words.append(review)

neg_words = ' '.join(neg_words)

neg_words[:200]
# word cloud for positive word

wordcloud = WordCloud().generate(pos_words)



wordcloud = WordCloud(background_color="white",max_words=len(pos_words),\

                      max_font_size=40, relative_scaling=.5, colormap='summer').generate(pos_words)

plt.figure(figsize=(13,13))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
# word cloud for negative word

wordcloud = WordCloud().generate(neg_words)



wordcloud = WordCloud(background_color="white",max_words=len(neg_words),\

                      max_font_size=40, relative_scaling=.5, colormap='gist_heat').generate(neg_words)

plt.figure(figsize=(13,13))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
# tokenlizing(vectorizing) the text, which transforms the data into tensor format

samples = data["text"].tolist()

maxlen = 100

max_words = 10000

tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(samples)

sequences = tokenizer.texts_to_sequences(samples)#transfer string into number

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))

X = pad_sequences(sequences, maxlen=maxlen)
# generate the target label

labels =  pd.get_dummies(data['sentiment']).values

print('Shape of data tensor:', X.shape)

print('Shape of label tensor:', labels.shape)
# generate the random dataset

indices = np.arange(data.shape[0])

np.random.shuffle(indices)

data2 = X[indices]

labels = labels[indices]
data2.shape# contains all the text information
a = data[['Polarity','text_length']].values #contains the created features information

a.shape
# the train,val,test is the data about text feature and train2,val2,test2 is the data about two creating new features,finailly we merge them together 

training_samples = 11320

validation_samples = 15848

x_train = data2[:training_samples]

x_train2 = a[:training_samples]

y_train = labels[:training_samples]

x_val = data2[training_samples: validation_samples] 

x_val2 = a[training_samples: validation_samples] 

y_val = labels[training_samples: validation_samples]

x_test = data2[validation_samples:]

x_test2 = a[validation_samples:]

y_test = labels[validation_samples:]

# for text feature, we still need following preprocessing step

x_train = pad_sequences(x_train, maxlen=maxlen)

x_val = pad_sequences(x_val, maxlen=maxlen)



# concat all the features

x_train = np.hstack((x_train2,x_train))

x_val = np.hstack((x_val2,x_val))

x_test = np.hstack((x_test2,x_test))
# This is the baseline, cause in the dataset, lable 1 occupies 82% percent

(np.sum(data['sentiment'] == 1)/data.shape[0]) * 100
x_train.shape


# First, let's build the simple embedding model

def build_model():

    model = Sequential()

    model.add(Embedding(max_words, 102, input_length=maxlen+2))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dense(64, activation='relu'))

    model.add(Dense(32, activation='relu'))

    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=['acc'])

    return model



model = build_model()

model.summary()

history = model.fit(x_train, y_train,

                    epochs=7,

                    batch_size=64,

                    validation_data=(x_val, y_val))


# First, let's build the simple NN model. Considering that the word vector will be a sparse matrix thus we add one embedding layer

def build_model():

    

    model = Sequential()

    model.add(Embedding(max_words, 102,input_length=maxlen+2))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.3))

    model.add(Dense(64, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=['acc'])

    return model

model = build_model()

model.summary()

history = model.fit(x_train, y_train,

                    epochs=7,

                    batch_size=64,

                    validation_data=(x_val, y_val))
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()

#  training loss keep decreasing while val loss keep increasing , overfitting
model.evaluate(x_test, y_test)
# recursion NN is a classic method to process text problem

def build_RNN():

    model = Sequential() 

    model.add(Embedding(max_words, 102, input_length=maxlen+2)) 

    model.add(Dropout(0.3))

    model.add(SimpleRNN(32)) 

    model.add(Dropout(0.3))

    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) 

    return model
RNN_model = build_RNN()

RNN_model.summary()

history_RNN = RNN_model.fit(x_train, y_train,

                    epochs=7,

                    batch_size=64,

                    validation_data=(x_val, y_val))
acc = history_RNN.history['acc']

val_acc = history_RNN.history['val_acc']

loss = history_RNN.history['loss']

val_loss = history_RNN.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()

# training loss keeps decreasing while val loss keeps decreasing,overfitting
RNN_model.evaluate(x_test, y_test)
# RNN and embedding model both exist some problems, let's try LSTM, another advanced version of RNN

def build_LSTM():

    embed_dim = 128

    lstm_out = 196

    max_features = 2000

    model = Sequential()

    model.add(Embedding(max_features, embed_dim,input_length = x_train.shape[1]))

    model.add(SpatialDropout1D(0.4))

    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))

    model.add(Dense(2,activation='softmax'))

    model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])

    return model

LSTM_model = build_LSTM()

LSTM_model.summary()

history_LSTM = LSTM_model.fit(x_train, y_train,

                    epochs=7,

                    batch_size=64,

                    validation_data=(x_val, y_val))
acc = history_LSTM.history['accuracy']

val_acc = history_LSTM.history['val_accuracy']

loss = history_LSTM.history['loss']

val_loss = history_LSTM.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()

# training loss decreases and val loss flucates,better than previous two 
LSTM_model.evaluate(x_test, y_test)
# Because our dataset is not balanced, thus we use another metrics---tpr and tnr to evalute our model
pos_count, neg_count, pos_correct, neg_correct = 0, 0, 0, 0

for x in range(len(x_test)):

    

    result = model.predict(x_test[x].reshape(1,x_test.shape[1]),batch_size=1,verbose = 2)[0]

   

    if np.argmax(result) == np.argmax(y_test[x]):

        if np.argmax(y_test[x]) == 0:

            neg_correct += 1

        else:

            pos_correct += 1

       

    if np.argmax(y_test[x]) == 0:

        neg_count += 1

    else:

        pos_count += 1





print("Embedding model's ablity to identify the positive samples and negative samples")

print("pos_acc", pos_correct/pos_count*100, "%")

print("neg_acc", neg_correct/neg_count*100, "%")
pos_count, neg_count, pos_correct, neg_correct = 0, 0, 0, 0

for x in range(len(x_test)):

    

    result = RNN_model.predict(x_test[x].reshape(1,x_test.shape[1]),batch_size=1,verbose = 2)[0]

   

    if np.argmax(result) == np.argmax(y_test[x]):

        if np.argmax(y_test[x]) == 0:

            neg_correct += 1

        else:

            pos_correct += 1

       

    if np.argmax(y_test[x]) == 0:

        neg_count += 1

    else:

        pos_count += 1





print("RNN's ablity to identify the positive samples and negative samples")

print("pos_acc", pos_correct/pos_count*100, "%")

print("neg_acc", neg_correct/neg_count*100, "%")
pos_count, neg_count, pos_correct, neg_correct = 0, 0, 0, 0

for x in range(len(x_test)):

    

    result = LSTM_model.predict(x_test[x].reshape(1,x_test.shape[1]),batch_size=1,verbose = 2)[0]

   

    if np.argmax(result) == np.argmax(y_test[x]):

        if np.argmax(y_test[x]) == 0:

            neg_correct += 1

        else:

            pos_correct += 1

       

    if np.argmax(y_test[x]) == 0:

        neg_count += 1

    else:

        pos_count += 1





print("LSTM ablity to identify positive samples and negative samples")

print("pos_acc", pos_correct/pos_count*100, "%")

print("neg_acc", neg_correct/neg_count*100, "%")

# let's try some interest samples

review_sample_1 = 'the poor quality and size is not suitable! '

review_sample_2 = 'Oh! nice experience'

review_sample_3 = 'ehh...OK OK, price is cheap! quality is also"cheap"'

review_sample_4 = 'good! very good! everything is good! Only one thing is not very great:what I buy is a shirt but get a pant'

def get_result(review):

    print(review)

    #vectorizing the review by the pre-fitted tokenizer instance

    length = np.array(len(review))

    polarity = np.array(TextBlob(review).sentiment.polarity)

    length = length.reshape(1,-1)

    polarity = polarity.reshape(1,-1)

    rw = tokenizer.texts_to_sequences([review])

    #padding the review to have exactly the same shape as `embedding_2` input

    rw = pad_sequences(rw, maxlen=100, dtype='int32', value=0)

    rw = np.hstack((rw,length,polarity))

    sentiment = LSTM_model.predict(rw,batch_size=1,verbose = 2)[0]

    if(np.argmax(sentiment) == 0):

        print("negative")

    elif (np.argmax(sentiment) == 1):

        print("positive")

for i in [review_sample_1,review_sample_2,review_sample_3,review_sample_4]:

    get_result(i)