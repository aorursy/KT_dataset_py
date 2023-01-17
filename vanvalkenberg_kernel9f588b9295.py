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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
trainFrame = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

testFrame = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
Data = trainFrame['text']

label = trainFrame['target']
testData = testFrame['text']
## coverting to numpy arrays

Data = np.array(Data)

label = np.array(label)

testData= np.array(testData)
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Data, label, test_size = 0.1, random_state = 42)
tokenizer = Tokenizer(num_words = 25000)

tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)

X_test = tokenizer.texts_to_sequences(X_test)

testData= tokenizer.texts_to_sequences(testData)
X_train = pad_sequences(X_train)

X_train.shape
X_test = pad_sequences(X_test, maxlen = 33)

testData = pad_sequences(testData, maxlen = 33)

len(tokenizer.word_index)
Mymodel = tf.keras.models.Sequential([

    

    tf.keras.layers.Input(shape = (33,)),

    tf.keras.layers.Embedding(21116, 20),

    tf.keras.layers.LSTM(10, return_sequences = True,kernel_regularizer='l2'),

    tf.keras.layers.GlobalMaxPool1D(),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(10, activation = 'relu', kernel_regularizer='l2'),

    #tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(20, activation = 'relu',kernel_regularizer='l2'),

    #tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(1, activation = 'sigmoid',kernel_regularizer='l2')

])



Mymodel.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'binary_crossentropy', metrics = ['accuracy'] )



retVAl = Mymodel.fit(X_train, y_train, validation_data = (X_test,y_test), epochs = 20 , batch_size = 19)
from sklearn.ensemble import RandomForestClassifier

Rm = RandomForestClassifier(max_depth=15, random_state=0, criterion='entropy')

Rm.fit(X_train,y_train)
from sklearn.naive_bayes import MultinomialNB

multNb = MultinomialNB()

multNb.fit(X_test, y_test)
from sklearn.metrics import accuracy_score
pred = Rm.predict(X_test)

pred.shape

accuracy_score(y_test, pred.reshape((762,1)))
pred = multNb.predict(X_test)

pred.shape

accuracy_score(y_test, pred.reshape((762,1)))
testLabels = testFrame['id']

testLabels = np.array(testLabels)
predVect = Mymodel.predict(testData)
## first mapp

pred1 = predVect.flatten()



for mem in pred1:

    if mem >= 0.6:

        mem = 1

    else:

        mem = 0

pred1 = np.round_(pred1)
pred1
df = {'id' : testLabels,

       'target': pred1  }
df = pd.DataFrame(df)

df.to_csv('naivePred10.csv')
testData = testFrame['text']

testData = np.array(testData)

testData = testData.flatten()
spam  = []

ham = []



i = 0 

for ii in pred1:

    if  ii == 1:

        spam.append(testData[i])

    else:

        ham.append(testData[i])

    

    i = i + 1

    
from wordcloud import WordCloud

spam = np.array(spam, dtype = 'str')



sp = ""

for mem in spam :

    sp = sp + " " + mem





wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white',  

                min_font_size = 10).generate(sp) 
plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show()