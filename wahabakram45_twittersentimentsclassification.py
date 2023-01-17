import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

import pandas as pd

import numpy as np
df = pd.read_csv('../input/Tweets.csv')

df.head()
sentiment_counts = df.airline_sentiment.value_counts()

number_of_tweets = df.tweet_id.count()

print(sentiment_counts)

dff = df.groupby(["airline", "airline_sentiment" ]).count()["name"]

dff['American']
airlines=df.airline.unique()

positive_percentage = []

negative_percentage = []

neutral_percentage = []

for i in airlines:

    positive_percentage.append((dff[i].positive/dff[i].sum())*100)

    negative_percentage.append((dff[i].negative/dff[i].sum())*100)

    neutral_percentage.append((dff[i].neutral/dff[i].sum())*100)

percentage_data = [positive_percentage,negative_percentage,neutral_percentage]

percentage_data = np.array(percentage_data)

percentage_data=percentage_data.reshape(6,3)
my_series = pd.DataFrame(data=percentage_data, index =airlines)

my_series[0] = positive_percentage

my_series[1] = negative_percentage

my_series[2] = neutral_percentage

my_series
import matplotlib.pyplot as plt

import matplotlib.style

%matplotlib inline

import matplotlib.style

from matplotlib.pyplot import subplots



fig, ax = subplots()

my_colors =['blue','red','green']

my_series.plot(kind='bar', stacked=False, ax=ax, color=my_colors, figsize=(14, 7), width=0.8)

ax.legend(["Postive Percentage","Negative Percentage","Neutral Percentage"])

plt.title("Percentages of Sentiments, Tweets Sentiments Analysis Airlines, 2017")

plt.show()
data = df[['text','airline_sentiment']]

data.head()
data.loc[:,('airline_sentiment')] = data.airline_sentiment.map({'neutral':0, 'positive':1,'negative':2})

data.head()
positive_sentiment_words = ''

negative_sentiment_words = ''

neutral_sentiment_words = ''

neutral = data[data.airline_sentiment == 0]

positive = data[data.airline_sentiment ==1]

negative = data[data.airline_sentiment ==2]
import nltk,re

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer



stop_words = set(stopwords.words('english'))

wordnet_lemmatizer = WordNetLemmatizer()

for val in neutral.text:

    text = val.lower()

    only_letters = re.sub("[^a-zA-Z]", " ",text) 

    tokens = nltk.word_tokenize(only_letters )[2:]

    for word in tokens:

        if word not in stop_words:

            word = wordnet_lemmatizer.lemmatize(word)

            neutral_sentiment_words =  neutral_sentiment_words + word + ' '

            

for val in positive.text:

    text = val.lower()

    only_letters = re.sub("[^a-zA-Z]", " ",text) 

    tokens = nltk.word_tokenize(only_letters )[2:]

    for word in tokens:

        if word not in stop_words:

            word = wordnet_lemmatizer.lemmatize(word)

            positive_sentiment_words =  positive_sentiment_words + word + ' '

            

for val in negative.text:

    text = val.lower()

    only_letters = re.sub("[^a-zA-Z]", " ",text) 

    tokens = nltk.word_tokenize(only_letters )[2:]

    for word in tokens:

        if word not in stop_words:

            word = wordnet_lemmatizer.lemmatize(word)

            negative_sentiment_words =  negative_sentiment_words + word + ' '

            

            

from wordcloud import WordCloud

neutral_wordcloud = WordCloud(width=600, height=400).generate(neutral_sentiment_words)

positive_wordcloud = WordCloud(width=600, height=400).generate(positive_sentiment_words)

negative_wordcloud = WordCloud(width=600, height=400).generate(negative_sentiment_words)

plt.figure( figsize=(10,8), facecolor='k')

plt.imshow(neutral_wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
plt.figure( figsize=(10,8), facecolor='k')

plt.imshow(positive_wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()

plt.figure( figsize=(10,8), facecolor='k')

plt.imshow(negative_wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

data['text'] = data['text'].apply(lambda x: x.lower())

data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

X_train,X_test,y_train,y_test = train_test_split(data["text"],data["airline_sentiment"], test_size = 0.2, random_state = 10)

print("train tuples",X_train.shape)

print("test tuples",X_test.shape)

print("train labels",y_train.shape)

print("test labels",y_test.shape)

vect = CountVectorizer()

vect.fit(X_train)

X_train_df = vect.transform(X_train)

X_test_df = vect.transform(X_test)
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X_train_df,y_train)

result=model.predict(X_test_df)
print("Accuracy Score:",accuracy_score(y_test,result))
from sklearn import svm

clf = svm.SVC(kernel='rbf')

clf.fit(X_train_df,y_train)

result=clf.predict(X_test_df)
print("Accuracy Score:",accuracy_score(y_test,result))

import keras

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D

from keras.callbacks import ModelCheckpoint

import os

from sklearn.metrics import roc_auc_score

from keras.preprocessing.text import Tokenizer

max_fatures = 2000

tokenizer = Tokenizer(num_words=max_fatures, split=' ')

tokenizer.fit_on_texts(data['text'].values)

X = tokenizer.texts_to_sequences(data['text'].values)

X = pad_sequences(X)



embed_dim = 128

lstm_out = 196



model = Sequential()

model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))

model.add(Dropout(0.5))

model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(3,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

print(model.summary())



Y = pd.get_dummies(data['airline_sentiment']).values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)
batch_size = 256

history = model.fit(X_train, 

                    Y_train, 

                    epochs = 10, 

                    batch_size=batch_size, 

                    validation_data=(X_test, Y_test))
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper right')

plt.show()
models = ['Naive Bayes','SVM','LSTM']

accuracy = [76.09,60.82,81.11]

result_frame = pd.DataFrame(data = accuracy,index = models)



fig, ax = subplots()

my_colors =['blue','red','green']

result_frame.plot(kind='bar', stacked=False, ax=ax, color=my_colors, figsize=(12, 4), width=0.4)

ax.legend(["Percentage"])

plt.title("Comparison of different models on Twitter Sentiments")

plt.show()