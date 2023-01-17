import numpy as np

import pandas as pd
train = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', sep="\t")

test = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', sep="\t")

sub = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv', sep=",")
train.head()
train.shape
test.head()
test.shape
sub.head()
sub.shape
print('Number of phrases in train: {}. Number of sentences in train: {}.'.format(train.shape[0], len(train.SentenceId.unique())))

print('Number of phrases in test: {}. Number of sentences in test: {}.'.format(test.shape[0], len(test.SentenceId.unique())))
print('Average count of phrases per sentence in train is {0:.0f}.'.format(train.groupby('SentenceId')['Phrase'].count().mean()))

print('Average count of phrases per sentence in test is {0:.0f}.'.format(test.groupby('SentenceId')['Phrase'].count().mean()))
overlapped = pd.merge(train[["Phrase", "Sentiment"]], test, on="Phrase", how="inner")

print(overlapped.shape)
overlapped.head()
overlap_boolean_mask_test = test['Phrase'].isin(overlapped['Phrase'])
# Check sentiment distribution

import seaborn as sns

sns.countplot(x='Sentiment', data = train)
# ramdomly select 40,000 records from neutral sentiment to fit the model

neutral = len(train[train['Sentiment'] == 2])

neutral_indices = train[train.Sentiment == 2].index

random_indices = np.random.choice(neutral_indices,40000, replace=True)

no_neutral_indices = train[train.Sentiment != 2].index

under_sample_indices = np.concatenate([no_neutral_indices,random_indices])

train = train.loc[under_sample_indices]

train.reset_index(inplace = True)

train.head()
del train['index']
train.head()
sns.countplot(x='Sentiment', data = train)
train.shape
import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

import re

from nltk.tokenize import word_tokenize

from nltk.tokenize import RegexpTokenizer
# missing value check

train = train.replace('',np.NaN)

train = train.replace(' ',np.NaN)

train.isnull().any()
# delete missing value

train.dropna(inplace = True)
train.isnull().any().sum()
# add phrase lenghth colunm 

# and the phrase length is for separating long phrase and short Phrase

train['phrase_length'] = train['Phrase'].apply(lambda x: len(x.split()))

# check value counts

phrase_length = train.phrase_length.value_counts()

phrase_length.plot.bar(figsize=(25,10))
# step 1: Normalization

train['Phrase'] = train['Phrase'].apply(lambda x: x.lower())

# separate the dataset 

train1 = train[train['phrase_length'] >5]

train2 = train[train['phrase_length'] <=5]

# step 2:  stopwords removal 

stop = stopwords.words('english')

train1['Phrase'] = train1['Phrase'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# step 3: Tokenization and punctuation removal

tokenizer = RegexpTokenizer(r'\w+')

train1['Phrase'] = train1['Phrase'].apply(lambda x: tokenizer.tokenize(x))

train2['Phrase'] = train2['Phrase'].apply(lambda x: word_tokenize(x))

# train1['Phrase'] = train1['Phrase'].apply(lambda x: remove_punctuations(x))

# merge data

frames = [train1, train2]

train = pd.concat(frames)

train.sort_index(inplace=True)

# step 4: Lemmatization

wl=WordNetLemmatizer()

train['Phrase'] = train['Phrase'].apply(lambda x: [wl.lemmatize(w,pos = 'v') for w in x])

train['Phrase'] = train['Phrase'].apply(lambda x: [wl.lemmatize(w) for w in x])

# re-calculate the phrase length

train['phrase_length'] = train['Phrase'].apply(lambda x: len(x))

train.head()
train.phrase_length.max()
# same data cleaning process on test data set

# add phrase length column

test['phrase_length'] = test['Phrase'].apply(lambda x: len(x.split()))

# cleaning process:

# step 1: lowercase

test['Phrase'] = test['Phrase'].apply(lambda x: x.lower())

# separate dataset

test1 = test[test['phrase_length'] >5]

test2 = test[test['phrase_length'] <=5]

# step 2: stopwords removal 

test1['Phrase'] = test1['Phrase'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# step 3: Tokenization and punctuation removal

test1['Phrase'] = test1['Phrase'].apply(lambda x: tokenizer.tokenize(x))

test2['Phrase'] = test2['Phrase'].apply(lambda x: word_tokenize(x))

# merge the test dataset

frame = [test1, test2]

test = pd.concat(frame)

test.sort_index(inplace=True)

# step 4: Lemmatization

test['Phrase'] = test['Phrase'].apply(lambda x: [wl.lemmatize(w,pos = 'v') for w in x])

test['Phrase'] = test['Phrase'].apply(lambda x: [wl.lemmatize(w) for w in x])

test['phrase_length'] = test['Phrase'].apply(lambda x: len(x))
test.phrase_length.max()
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization

from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D

from keras.models import Model, load_model

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras import backend as K

from keras.engine import InputSpec, Layer

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
import keras

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels
y = train['Sentiment']

y = keras.utils.to_categorical(y,num_classes=5)
phrase_train, phrase_valid, y_train, y_valid = train_test_split(train['Phrase'], y, test_size=0.2, random_state=1000)
tokenizer = Tokenizer(num_words=10000)

tokenizer.fit_on_texts(phrase_train)

tokenizer.fit_on_texts(phrase_valid)

x_train = tokenizer.texts_to_sequences(phrase_train)

x_valid = tokenizer.texts_to_sequences(phrase_valid)

max_len = 32

x_train = pad_sequences(x_train, maxlen = max_len)

x_valid = pad_sequences(x_valid, maxlen = max_len)
x_train
cnn_model1 = Sequential()

filters = 100

cnn_model1.add(Embedding(input_dim = 15000, output_dim=1000, input_length = 32))

# cnn_model1.add(layers.Flatten())

cnn_model1.add(Dropout(0.1))

cnn_model1.add(Conv1D(filters, 3, strides=1, padding='valid', activation='relu'))



cnn_model1.add(GlobalMaxPool1D())

cnn_model1.add(layers.Dense(10, activation='relu'))

cnn_model1.add(layers.Dense(5, activation='softmax'))

cnn_model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

cnn_model1.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience = 10, verbose=2)

history1 = cnn_model1.fit(x_train, y_train,

                         validation_data=(x_valid, y_valid),

                          epochs=20,

                         batch_size=1000,

                         callbacks=[early_stopping])
tokenizer = Tokenizer(num_words=7000)

tokenizer.fit_on_texts(test.Phrase)

x_test = tokenizer.texts_to_sequences(test.Phrase)

max_len = 32

x_test = pad_sequences(x_test, maxlen = max_len)

pred1 = cnn_model1.predict_classes(x_test,verbose=1)
sub.Sentiment = pred1

sub.to_csv('sub1.csv',index=False)

sub.head()
lstm_model = Sequential()

lstm_model.add(Embedding(input_dim = 10000, output_dim=1000, input_length = 32))

lstm_model.add(LSTM(64,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))

lstm_model.add(LSTM(32,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))

lstm_model.add(Dense(5, activation='softmax'))

lstm_model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])

lstm_model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience = 10, verbose=2)

history2 = lstm_model.fit(x_train, y_train,

                          validation_data=(x_valid, y_valid),

                         epochs=20,

                         batch_size=1000

                    )
#tokenizer = Tokenizer(num_words=7000)

#tokenizer.fit_on_texts(test.Phrase)

#x_test = tokenizer.texts_to_sequences(test.Phrase)

#max_len = 32

#x_test = pad_sequences(x_test, maxlen = max_len)

pred2 = lstm_model.predict_classes(x_test,verbose=1)
sub.Sentiment = pred2

sub.to_csv('sub2.csv',index=False)

sub.head()
model3= Sequential()

model3.add(Embedding(10000,1000,input_length = 32))

model3.add(Conv1D(64,kernel_size=3,padding='same',activation='relu'))

model3.add(MaxPooling1D(pool_size=2))

model3.add(Dropout(0.25))

model3.add(GRU(128,return_sequences=True))

model3.add(Dropout(0.3))

model3.add(Flatten())

model3.add(Dense(128,activation='relu'))

model3.add(Dropout(0.5))

model3.add(Dense(5,activation='softmax'))

model3.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

model3.summary()
%%time

early_stopping = EarlyStopping(monitor='val_loss', patience = 10, verbose=2)

history3 = model3.fit(x_train, y_train, validation_data=(x_valid, y_valid),epochs = 15, batch_size = 1000, verbose=1)
pred3 = model3.predict_classes(x_test,verbose=1)
sub.Sentiment = pred3

sub.to_csv('sub3.csv',index=False)

sub.head()
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import model_selection, naive_bayes, svm

from sklearn.metrics import accuracy_score
train['Phrase'] = train['Phrase'].apply(lambda x: ' '.join(x))

train.head()
train.shape
test['Phrase'] = test['Phrase'].apply(lambda x: ' '.join(x))

test.head()
Tfidf_vect = TfidfVectorizer(max_features=10000)

x_train_tf = Tfidf_vect.fit_transform(train.Phrase)

x_test_tf = Tfidf_vect.transform(test.Phrase)

y_train_tf = train['Sentiment']
# print(Tfidf_vect.vocabulary_)
# print(x_train_tf)
# fit the training dataset on the NB classifier

Naive = naive_bayes.MultinomialNB()

Naive.fit(x_train_tf,y_train_tf)
# predict the labels on train dataset

predictions_NB = Naive.predict(x_train_tf)

# Use accuracy_score function to get the accuracy

print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, y_train_tf))
pred4 = Naive.predict(x_test_tf)
sub.Sentiment = pred4

sub.to_csv('sub4.csv',index=False)

sub.head()
sub_all=pd.DataFrame({'model1':pred1,'model2':pred2,'model3':pred3,'model4':pred4})

pred_mode=sub_all.agg('mode',axis=1)[0].values

sub_all.head()
finalpred=(pred1+pred2+pred3+pred4)//4

sub.Sentiment = finalpred

sub.to_csv('sub_all.csv',index=False)

sub.head()
overlapped.head()
overlapped.shape
lapped_id = overlapped.PhraseId
sub_al = sub[~sub['PhraseId'].isin(lapped_id)]
sub_al.shape
overlap_records = overlapped[['PhraseId','Sentiment']]

overlap_records.head()
overlap_records.shape
sub_final= pd.concat([sub_al,overlap_records])

sub_final.shape
sub_final.shape
sub_final.tail(100)
sub_final['PhraseId'].duplicated().sum()
sub_final.reset_index(inplace=True)

type(sub_final.PhraseId[0])
sub = sub_final.reindex(sub_final['PhraseId'].abs().sort_values(ascending=True).index)

sub.head(20)
sub.tail()
sub = sub[['PhraseId','Sentiment']]

sub.tail()
sub.reset_index(inplace=True)

sub.tail()
sub = sub[['PhraseId','Sentiment']]

sub.tail()
sub.to_csv('submission.csv',index=False)

sub.head()