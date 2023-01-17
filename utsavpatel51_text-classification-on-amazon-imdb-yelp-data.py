import numpy as np

import pandas as pd

#a = pd.read_csv('../input/amazon_cells_labelled.txt',names=['sentence','label'],sep='\t')[:10]
files={'yelp':'yelp_labelled.txt',

       'amazon':'amazon_cells_labelled.txt',

       'imdb':'imdb_labelled.txt'}

df_list=[]

for k,v in files.items():

    path='../input/'+v

    df = pd.read_csv(path,names=['sentence','label'],sep='\t')

    df['source']=k

    df_list.append(df)

df = pd.concat(df_list)

print(df.iloc[0])

print(type(df))
from sklearn.model_selection import train_test_split

#Split the data

df_yelp = df[df['source']=='yelp']

sentence = df_yelp['sentence']

label = df_yelp['label']

sen_train,sen_test,y_train,y_test = train_test_split(sentence,label,test_size=0.25,random_state=1000)

len(sen_train),len(sen_test)
#Best way to understand feature vector

'''

sentences = ['John John likes icecream','utsav','John is utsav']

from sklearn.feature_extraction.text import CountVectorizer



vec = CountVectorizer(min_df=0,lowercase=False)

vec.fit(sentences)

vec.vocabulary_

vec.transform(sentences).toarray()

'''

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

vectorizer.fit(sen_train)

X_train = vectorizer.transform(sen_train)

X_test = vectorizer.transform(sen_test)

X_train,X_test
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='lbfgs')

lr.fit(X_train,y_train)

predict_lr = lr.predict(X_test)

score = lr.score(X_test,y_test)

result_lr = pd.DataFrame({'Predict':predict_lr,'actual':y_test})

print(f'Accuracy: {score}')

result_lr[0:10]
#Acuuracy

from sklearn.metrics import classification_report,confusion_matrix

clr = classification_report(y_test,predict_lr)

com = confusion_matrix(y_test,predict_lr)

print(clr)

print("Confusion_matrix")

print(com)
#Test for unseen data

for source in df['source'].unique():

    df_yelp = df[df['source']==source]

    sentence = df_yelp['sentence']

    label = df_yelp['label']

    sen_train,sen_test,y_train,y_test = train_test_split(sentence,label,test_size=0.25,random_state=1000)

    

    vectorizer = CountVectorizer()

    vectorizer.fit(sen_train)

    X_train = vectorizer.transform(sen_train)

    X_test = vectorizer.transform(sen_test)



    lr = LogisticRegression(solver='lbfgs')

    lr.fit(X_train,y_train)

    score = lr.score(X_test,y_test)

    print('Source: {} Accuracy: {:.3f}'.format(source,score))

    
from sklearn.model_selection import train_test_split

#Split the data

df_yelp = df[df['source']=='yelp']

sentence = df_yelp['sentence']

label = df_yelp['label']

sen_train,sen_test,y_train,y_test = train_test_split(sentence,label,test_size=0.25,random_state=1000)

len(sen_train),len(sen_test)



from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

vectorizer.fit(sen_train)

X_train = vectorizer.transform(sen_train)

X_test = vectorizer.transform(sen_test)

X_train,X_test
from keras.models import Sequential

from keras import layers

input_dim = X_train.shape[1]

model = Sequential()

model.add(layers.Dense(10,input_dim=input_dim,activation='relu'))

model.add(layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])

model.summary()
history = model.fit(X_train,y_train,

                   epochs=100,verbose=True,

                   validation_data=(X_test,y_test),

                   batch_size=10)
loss,accuracy = model.evaluate(X_train,y_train,verbose=False)

print('Training accuracy {}'.format(accuracy))

loss,accuracy = model.evaluate(X_test,y_test,verbose=False)

print('Testing accuracy {}'.format(accuracy))
def hist(history):

    from matplotlib import pyplot as plt

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    x = range(1,len(loss)+1)

    plt.subplot(1,2,1)

    plt.plot(x,loss,'r',label='training loss')

    plt.plot(x,val_loss,'b',label='validation loss')

    plt.legend()

    plt.subplot(1,2,2)

    plt.plot(x,acc,'r',label='training accuracy')

    plt.plot(x,val_acc,'b',label='validation accuracy')

    plt.legend()

hist(history)
#Two possible ways to represent a word as a vector are one-hot encoding and word embeddings

#1)label encoding

from sklearn.preprocessing import LabelEncoder

cities = ['London', 'Berlin', 'Berlin', 'New York', 'London']

le =LabelEncoder()

city_label = le.fit_transform(cities)

print("Label encoding -> " ,city_label)



#2)one-hot encoding

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False,categories='auto')

city_label = city_label.reshape((5,1))

ohe.fit_transform(city_label)
#Word embeddings

#This method represents words as dense word vectors (also called word embeddings) 

#which are trained unlike the one-hot encoding which are hardcoded.

#This means that the word embeddings collect more information into fewer dimensions.

#Now you need to tokenize the data into a format that can be used by the word embeddings

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(sen_train)



vocab_size = len(tokenizer.word_index) + 1



X_train = tokenizer.texts_to_sequences(sen_train)

X_test = tokenizer.texts_to_sequences(sen_test)



from keras.preprocessing.sequence import pad_sequences

maxlen=100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
#input_dim: the size of the vocabulary

#output_dim: the size of the dense vector

#input_length: the length of the sequence

from keras.models import Sequential

from keras import layers



out_dim=50

model=Sequential()

model.add(layers.Embedding(input_dim=vocab_size,

                          output_dim=out_dim,

                          input_length=maxlen))

model.add(layers.Flatten())

model.add(layers.Dense(10, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

model.summary()
history = model.fit(X_train,y_train,

                   epochs=20,

                   verbose=True,

                   validation_data=(X_test,y_test),

                   batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))

hist(history)
from keras.models import Sequential

from keras import layers



out_dim=50

model=Sequential()

model.add(layers.Embedding(input_dim=vocab_size,

                          output_dim=out_dim,

                          input_length=maxlen))

model.add(layers.GlobalMaxPool1D())

model.add(layers.Dense(10, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train,

                    epochs=20,

                    verbose=False,

                    validation_data=(X_test, y_test),

                    batch_size=10)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))

hist(history)