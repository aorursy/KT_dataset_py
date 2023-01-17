# Importing important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 
import keras
from keras.models import Sequential
from keras.preprocessing import text
from keras.layers import Dense,Dropout
# Reading datasets from csv file
train_set = pd.read_csv('../input/train.csv')
train_set.head()
# Separating Comments and Labels
comments,labels= train_set['comment_text'],train_set[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]
comments.head()
# Use of tokenizer to convert texts into word array
num_words=3000
tokenizer = text.Tokenizer(lower=True,num_words=num_words)
tokenizer.fit_on_texts(comments)
encoded_text = tokenizer.texts_to_matrix(comments,mode='tfidf')
encoded_text.shape
def Model():
    model = Sequential()
    model.add(Dense(1024,input_shape=(num_words,),activation='relu'))
    model.add(Dropout(0.4))
    
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(64,activation='relu'))
    model.add(Dense(6,activation='sigmoid'))
    
    return model
model = Model()
model.compile(loss=keras.losses.binary_crossentropy,optimizer='adam',metrics=['accuracy'])
history = model.fit(encoded_text,labels,verbose=1,epochs=2,validation_split=0.3)
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()