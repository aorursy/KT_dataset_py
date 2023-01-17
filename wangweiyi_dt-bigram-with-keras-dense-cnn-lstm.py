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
train=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

train.head()
test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

test.head()
train_x=train['text']

train_x.head()
train_x.isnull().sum()
train_y=train['target']
test_x=test['text']

test_x.head()
test_x.isnull().sum()
from sklearn.feature_extraction.text import CountVectorizer

vector=CountVectorizer(min_df=1,ngram_range=(2,2))

X=vector.fit_transform(train_x)
from keras.models import Sequential

from keras.layers import Dense,Flatten

from keras.layers.embeddings import Embedding
# 批量处理文本

from keras.preprocessing.text import Tokenizer

tokenizer=Tokenizer()

tokenizer.fit_on_texts(train_x)

word_sequences=tokenizer.texts_to_sequences(train_x)

wordsize=np.max([len(word_sequences[i]) for i in range(len(word_sequences))])

from keras.preprocessing.sequence import pad_sequences

padded_word_sequences=pad_sequences(word_sequences,maxlen=wordsize)
testnizer=Tokenizer()

testnizer.fit_on_texts(test_x)

word_sequences_test=testnizer.texts_to_sequences(test_x)

padded_word_sequences_test=pad_sequences(word_sequences_test,maxlen=wordsize)
padded_word_sequences.shape
vacabsize=np.max([np.max(word_sequences[i]) for i in range(len(word_sequences))])+1

vacabsize
vacabsize_test=np.max([np.max(word_sequences_test[i]) for i in range(len(word_sequences_test))])+1

vacabsize_test
# 全连接网络

model=Sequential()

model.add(Embedding(vacabsize,64,input_length=wordsize))

model.add(Flatten())

model.add(Dense(2000,activation='relu'))

model.add(Dense(500,activation='relu'))

model.add(Dense(200,activation='relu'))

model.add(Dense(50,activation='relu'))

model.add(Dense(1,activation='relu'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())
model.fit(padded_word_sequences,train_y,epochs=2,batch_size=128)
test_y=model.predict_classes(padded_word_sequences_test)
test_y.shape
test_y[0]
submission=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

submission.head()
submission['target']=test_y
submission.head()
# 全连接神经网络的预测输出

submission.to_csv("submissiondense.csv", index=False, header=True)
# 卷积

from keras.layers import Dropout,Conv1D,MaxPooling1D

modelcnn=Sequential()

modelcnn.add(Embedding(vacabsize,64,input_length=wordsize)) #将向量从高维降到低维

modelcnn.add(Conv1D(filters=64,kernel_size=3,padding='same',activation='relu'))

modelcnn.add(MaxPooling1D(pool_size=2))

modelcnn.add(Dropout(0.25))

modelcnn.add(Conv1D(filters=128,kernel_size=3,padding='same',activation='relu'))

modelcnn.add(MaxPooling1D(pool_size=2))

modelcnn.add(Dropout(0.25))

modelcnn.add(Flatten())

modelcnn.add(Dense(64,activation='relu'))

modelcnn.add(Dense(32,activation='relu'))

modelcnn.add(Dense(1,activation='sigmoid'))

modelcnn.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

print(modelcnn.summary())
modelcnn.fit(padded_word_sequences,train_y,epochs=2,batch_size=128)
test_yccn=modelcnn.predict(padded_word_sequences_test)
submission['target']=test_yccn
submission.head()
submission.to_csv("submissioncnn.csv", index=False, header=True)
# LSTM

from keras.layers import LSTM

modellstm=Sequential()

modellstm.add(Embedding(vacabsize,64,input_length=wordsize))

modellstm.add(LSTM(128,return_sequences=True))

modellstm.add(Dropout(0.2))

modellstm.add(LSTM(64,return_sequences=True))

modellstm.add(Dropout(0.2))

modellstm.add(LSTM(32))

modellstm.add(Dropout(0.2))

modellstm.add(Dense(1,activation='sigmoid'))
modellstm.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

print(modellstm.summary())
modellstm.fit(padded_word_sequences,train_y,epochs=2,batch_size=128)
test_ylstm=modellstm.predict(padded_word_sequences_test)
submission['target']=test_ylstm
submission.head()
submission.to_csv("submissionlstm.csv", index=False, header=True)