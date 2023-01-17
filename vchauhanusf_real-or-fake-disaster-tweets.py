



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv',index_col=False)

test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv',index_col=False)

train.sample(5)
test.sample(5)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
train.info()
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(train.target.value_counts())
#x_train,x_test,y_train,y_test=train_test_split(train.text,train.target,test_size=.25,shuffle=True)
# defining hyper paramters



num_words=3000

max_len=100

tokenizer=Tokenizer(num_words,oov_token='<oov>')



tokenizer.fit_on_texts(train.text)



word_index=tokenizer.word_index

len(word_index)
# creating train sequences and then padding them



train_sequences=tokenizer.texts_to_sequences(train.text)



padded_train=pad_sequences(train_sequences,maxlen=max_len)



from keras.layers import Dropout,Embedding,Dense,Flatten,Bidirectional,LSTM

from keras.models import Sequential



model=Sequential([Embedding(num_words,8,input_length=max_len),

                  Bidirectional(LSTM(8)),

                 

                              

                               Dense(16,activation='relu'),

                 Dropout(.3), 

                  

                               Dense(1,activation='sigmoid')])

                  

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])                  
history=model.fit(padded_train,train.target,epochs=10)
import matplotlib.pyplot as plt



plt.plot(history.history['accuracy'],'r')

plt.plot(history.history['loss'],'b')
test.sample(5)


test_data_sequences=tokenizer.texts_to_sequences(test.text)
padded_sub_sequences=pad_sequences(test_data_sequences,maxlen=max_len)
test['target']=model.predict(padded_sub_sequences)
test
test=test.drop(['keyword','location','text'],axis=1)
test.sample(5)
test.loc[test['target']<.5,:].count()
test['target']=test.target.apply(lambda x:1 if x>0.5 else 0)
test.to_csv('real_nlp_try2.csv',index=False)