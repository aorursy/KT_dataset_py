%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import ModelCheckpoint
train_f = pd.read_csv("../input/train.tsv",sep="\t")
train_f.head()
test_f = pd.read_csv("../input/test.tsv",sep="\t")
test_f.head()
corpus_sentences = list(map(str,train_f["Phrase"] + test_f["Phrase"]))
corpus_sentences[0]
type(corpus_sentences[0])
max_words = 40000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(list(corpus_sentences))

list_tokenized_train = tokenizer.texts_to_sequences(train_f["Phrase"])
list_tokenized_test = tokenizer.texts_to_sequences(test_f["Phrase"])
len(list_tokenized_train)
num_words = [len(i) for i in corpus_sentences]
plt.hist(num_words)
max_len = 80
X_train = pad_sequences(list_tokenized_train,maxlen=max_len)
X_test = pad_sequences(list_tokenized_test,maxlen=max_len)
t = train_f["Sentiment"].values
print(len(t))
y_train = []  
for i in t:
    l = [0]*5
    l[i]=1
    y_train.append(np.array(l))
np.shape(y_train)
X_train_t, X_train_dev, y_train_t, y_train_dev = train_test_split(np.array(X_train),np.array(y_train),test_size=0.2)
type(y_train_t)
embed_size = 128
model = Sequential()

model.add(Embedding(max_words,embed_size,input_length=max_len))

model.add(Bidirectional(LSTM(64,return_sequences=True,use_bias=True,bias_initializer=keras.initializers.Ones())))

model.add(LSTM(64,use_bias=True,bias_initializer=keras.initializers.Ones()))

model.add(Dense(64, activation='relu'))

model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
batch_size=64
epochs = 4
text_model = model.fit(X_train_t,y_train_t,batch_size=batch_size,epochs=epochs,verbose=1, validation_data=(X_train_dev,y_train_dev)) #,callbacks=callbacks_list
