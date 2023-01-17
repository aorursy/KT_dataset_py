import numpy as np 
import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from  keras.preprocessing.sequence import pad_sequences
from keras.models import *
from keras.layers import *
from keras.losses import *
from keras.callbacks import *
from keras.optimizers import *
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re

df = pd.read_csv("../input/train.csv")
df.head(10)
#Number of sentences
sentence_split = re.compile("[.!?'\";]")
df['num_sentences'] = df['text'].apply(lambda x: len(sentence_split.split(x)))
df['num_sentences'].describe()
# Number of words
df['num_words'] = df['text'].apply(lambda x: len(x.split(' ')))
df['num_words'].describe()
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.countplot(x="author", data=df);
plt.subplot(1,3,2)
sns.stripplot(x="author", hue='author', y='num_sentences', data=df, jitter=True);
plt.subplot(1,3,3)
sns.stripplot(x="author", hue='author', y='num_words', data=df, jitter=True);
labelEncoder = LabelEncoder().fit(df['author'])
df['author_id'] = labelEncoder.transform(df['author'])
def gen_ngram(tokens, n):
    length = len(tokens)
    for i in range(2, n+1):
        for j in range(0, length+1-i):
            tokens.append('+'.join(tokens[j:j+i]))
    return tokens
def preprocess_text(text, stem_func=None, stop_words=set()):
    text = nltk.tokenize.word_tokenize(text)
    text = [w for w in text if not w in stop_words]
    if stem_func!=None:
        text = [stem_func(w) for w in text]
    return ' '.join(text)
wnl = nltk.stem.wordnet.WordNetLemmatizer()

df['text_proc'] = df['text'].apply(lambda x: preprocess_text(x, wnl.lemmatize, set()))
df['pos_tags'] = df['text_proc'].apply(lambda x: ' '.join([y[1] for y in nltk.pos_tag(x.split(' '))]))
df['text_proc'].head().apply(lambda x: len(x.split(' ')))
df['pos_tags'].head().apply(lambda x: len(x.split(' ')))
df.head()
bigrams = [' '.join(gen_ngram(sentence.split(' '), 2)) for sentence in df['text_proc']]
tokenizer = Tokenizer(filters='', lower=False, split=' ')
tokenizer.fit_on_texts(bigrams)
y = df['author_id'].values
X = tokenizer.texts_to_sequences(bigrams)
X = pad_sequences(X)
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.1)
model1 = Sequential()
model1.add(InputLayer((X.shape[1],)))
model1.add(Embedding(input_dim=np.max(X)+1, output_dim=40, input_length=X.shape[1]))
model1.add(GlobalAveragePooling1D())
model1.add(Dropout(0.5))
model1.add(Dense(3, activation='softmax'))
model1.summary()
model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
epochs=100
history = model1.fit(X_train, y_train, 
          epochs=epochs, 
          validation_data=(X_validation, y_validation),
          callbacks=[EarlyStopping(patience=3, monitor='val_loss')],
          verbose=2)
plt.figure(figsize=(10,2))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], 'r--')
plt.plot(history.history['val_loss'], 'b-')
plt.subplot(1,2,2)
plt.plot(history.history['acc'], 'r--')
plt.plot(history.history['val_acc'], 'b-')
tokenizer = Tokenizer(filters='', lower=False, split=' ')
tokenizer.fit_on_texts(df['pos_tags'])
y = df['author_id'].values
X = tokenizer.texts_to_sequences(df['pos_tags'])
X = pad_sequences(X, 50)

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2)

model2 = Sequential()
model2.add(InputLayer((X.shape[1],)))
model2.add(Embedding(input_dim=np.max(X)+1, output_dim=30, input_length=X.shape[1]))
model2.add(Conv1D(100, kernel_size=10, activation='relu'))
model2.add(GlobalAveragePooling1D())
model2.add(Dropout(0.3))
model2.add(Dense(3, activation='softmax'))
model2.summary()

model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
epochs=300
history = model2.fit(X_train, y_train, 
          epochs=epochs, 
          validation_data=(X_validation, y_validation),
          callbacks=[EarlyStopping(patience=5, monitor='val_loss')],
          verbose=2)
plt.figure(figsize=(10,2))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], 'r--')
plt.plot(history.history['val_loss'], 'b-')
plt.subplot(1,2,2)
plt.plot(history.history['acc'], 'r--')
plt.plot(history.history['val_acc'], 'b-')
tokenizer_text = Tokenizer(filters='', lower=False, split=' ')
tokenizer_text.fit_on_texts(df['text_proc'])
tokenizer_pos = Tokenizer(filters='', lower=False, split=' ')
tokenizer_pos.fit_on_texts(df['pos_tags'])
y = df['author_id'].values
X_text = tokenizer_text.texts_to_sequences(df['text_proc'])
X_text = pad_sequences(X_text)
X_pos = tokenizer_pos.texts_to_sequences(df['pos_tags'])
X_pos = pad_sequences(X_pos)

X_text_train, X_text_validation, X_pos_train, X_pos_validation, y_train, y_validation = train_test_split(X_text, X_pos, y, test_size=0.1)

pos_input = Input(shape=(X_pos.shape[1],), name='pos_input')
pos_embd = Embedding(input_dim=np.max(X_pos)+1, output_dim=10, input_length=X_pos.shape[1])(pos_input)
text_input = Input(shape=(X_text.shape[1],), name='text_input')
text_embd = Embedding(input_dim=np.max(X_text)+1, output_dim=10, input_length=X_text.shape[1])(text_input)
x = concatenate([pos_embd, text_embd], axis=-1)
x1 = Conv1D(20, kernel_size=2, padding='same', activation='relu')(x)
x1 = GlobalMaxPooling1D()(x1)
x2 = Conv1D(20, kernel_size=3, padding='same', activation='relu')(x)
x2 = GlobalMaxPooling1D()(x2)
x3 = Conv1D(20, kernel_size=4, padding='same', activation='relu')(x)
x3 = GlobalMaxPooling1D()(x3)
x4 = Conv1D(20, kernel_size=5, padding='same', activation='relu')(x)
x4 = GlobalMaxPooling1D()(x4)
x = concatenate([x1,x2,x3,x4], axis=-1)
x = Dropout(0.3)(x)
x = Dense(3, activation='softmax')(x)
model_combined = Model(inputs = [text_input, pos_input], outputs=x)
model_combined.summary()

model_combined.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
epochs=100
history = model_combined.fit({'pos_input':X_pos_train, 'text_input':X_text_train}, y_train, 
          epochs=epochs, 
          validation_data=({'pos_input':X_pos_validation, 'text_input':X_text_validation}, y_validation),
          callbacks=[EarlyStopping(patience=2, monitor='val_loss')],
          verbose=2)
plt.figure(figsize=(10,2))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], 'r--')
plt.plot(history.history['val_loss'], 'b-')
plt.subplot(1,2,2)
plt.plot(history.history['acc'], 'r--')
plt.plot(history.history['val_acc'], 'b-')