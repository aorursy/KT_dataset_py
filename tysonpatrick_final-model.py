# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

from nltk.tokenize import TreebankWordTokenizer

from tensorflow import keras

from keras.preprocessing.sequence import pad_sequences

from keras import Sequential

from keras.layers import Embedding, Dense, LSTM, Dropout

from keras.preprocessing.text import one_hot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
training_data=pd.read_csv("/kaggle/input/quora-insincere-questions-classification/train.csv")

testing_data=pd.read_csv("/kaggle/input/quora-insincere-questions-classification/test.csv")
punctuations = ['?',"'",'$', '&', '/', '[', ']', '>', '%', '=',',','.','"', ':', ')', '(', '-', '!', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  

 '·', '_', '{', '}']



def clear_text(x):

    x = str(x)

    for m in punctuations:

        x = x.replace(m,'')

    return x
training_data["question_text"]=training_data["question_text"].apply(lambda x: clear_text(x))

testing_data["question_text"]=testing_data["question_text"].apply(lambda x: clear_text(x))
X_Train=training_data["question_text"].str.lower()

X_Test=testing_data["question_text"].str.lower()
X_Train=X_Train.tolist()

X_Test=X_Test.tolist()
encoded=[]

for n in X_Train:

    encoded.append(one_hot(n,20000))

encoded_test=[]

for m in X_Test:

    encoded_test.append(one_hot(m,20000))
number_of_most_frequent_words=1000

max_len=70

X_encoded = pad_sequences(encoded,maxlen=max_len,padding='post' )

X_test_encoded = pad_sequences(encoded_test,maxlen=max_len,padding='post' )
Y_Train=training_data["target"]

print(Y_Train)
model = Sequential()

model.add(Embedding(input_dim=20000, output_dim=128, input_length=70))

model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
result= model.fit(X_encoded, Y_Train,epochs=2)

print(result)
pred=model.predict_proba(X_test_encoded)

print(pred)
testing_data['prediction']=pred

threshold=0.50

for index, row in testing_data.iterrows():

    if row['prediction']>0.5:

        testing_data.prediction[index]=int(1)

    else:

        testing_data.prediction[index]=int(0)
testing_data=testing_data.drop(['question_text'], axis = 1)
testing_data['prediction']=testing_data['prediction'].astype(np.int64)
testing_data.to_csv('submission.csv',index=False) 