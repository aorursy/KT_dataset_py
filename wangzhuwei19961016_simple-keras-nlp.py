# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_train=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
data_train.columns
data_train.isnull().sum()
data_train['keyword'].value_counts()
data_train['location'].value_counts()
data_train['target'].value_counts()
data_train['text'].describe()
X=data_train.drop('target',axis=1)

y=data_train['target']
print(X.shape)
print(y.shape)
import re 

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer 

lemmatizer=WordNetLemmatizer()

corpus=[]

for i in range(0,len(data_train)):

    review=re.sub('[^a-zA-Z]', ' ', data_train['text'][i])

    review=review.lower()

    review=review.split()

    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]

    review=' '.join(review)

    corpus.append(review)

corpus[0] 
from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.text import one_hot

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Dense

from keras.layers import Dropout
voc_size=5000

onehot_repr=[one_hot(words,voc_size)for words in corpus] 

onehot_repr[0]
sent_length=20

embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)

print(embedded_docs[0])
embedding_vector_features=40

from tensorflow.keras.layers import Dropout

embedding_vector_features=40

model=Sequential()

model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))

model.add(Dropout(0.3))

model.add(LSTM(100))

model.add(Dropout(0.3))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
from sklearn.model_selection import train_test_split
X_final=np.array(embedded_docs)

y_final=np.array(y)
X_train,X_test,y_train,y_test=train_test_split(X_final,y_final,test_size=0.3)
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)
pred=model.predict_classes(X_test)

pred
from sklearn.metrics import accuracy_score

accuracy_score(y_test,pred)