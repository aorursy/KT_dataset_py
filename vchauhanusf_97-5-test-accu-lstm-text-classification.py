

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


data=pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv',encoding='Latin-1')
data.sample(5)
data.v1.value_counts().plot.bar()
data.v1.value_counts()
747/(4825+747)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(data.v2,data.v1,test_size=.25,shuffle=True,stratify=data.v1)
y_test.value_counts()
187/(1206+187)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# Defining Hyper Parameters

num_words=1000
max_len=150
oov_token='<oov>'
tokenizer=Tokenizer(num_words=1000,oov_token='<oov>')
tokenizer.fit_on_texts(x_train)

word_index=tokenizer.word_index

#word_index
sequences=tokenizer.texts_to_sequences(x_train)

len(sequences)
padded_sequences=pad_sequences(sequences,maxlen=max_len)
len(padded_sequences)
text_sequences=tokenizer.texts_to_sequences(x_test)
padded_text_test=pad_sequences(text_sequences,maxlen=max_len)

from sklearn.preprocessing import LabelEncoder

en=LabelEncoder()

en.fit(y_train)

y_train=en.transform(y_train)
y_test=en.transform(y_test)
from keras.layers import Embedding,Dense,LSTM,Dropout,Flatten

from keras.models import Sequential

model=Sequential([Embedding(num_words,64,input_length=max_len),
                  LSTM(32),
                  Dense(64,activation='relu'),
                  Dense(1,activation='sigmoid')])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(padded_sequences,y_train)
text_hist=model.evaluate(padded_text_test,y_test)