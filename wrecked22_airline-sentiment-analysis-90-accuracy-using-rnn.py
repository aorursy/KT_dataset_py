import re

import warnings

import itertools

import numpy as np 

import pandas as pd 

import seaborn as sns

import tensorflow as tf

import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore", category=FutureWarning)
data = pd.read_csv("../input/Tweets.csv")

df = data[["text","airline_sentiment"]]

df['text'] = df['text'].map(lambda x: x.lstrip('@VirginAmerica@UnitedAir@Southwestairline@DeltaAir@USAirways@American').rstrip('@'))
sns.countplot(data.airline);
df = df[df.airline_sentiment!="neutral"] # To remove neutral responses

df['text'] = df['text'].apply(lambda x: x.lower()) # To lower 

df['text'] = df['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x))) # To keep numbers and strings only
df.head(5) #Quick Look 
sns.countplot(df.airline_sentiment);#Mostly Negative Reviews(Class Imbalance found) 
df = df.drop(df[df.airline_sentiment == "negative"].iloc[:5000].index)
sns.countplot(df.airline_sentiment);
max_fatures = 4000

tokenizer = Tokenizer(num_words = max_fatures, split=' ')

tokenizer.fit_on_texts(df['text'].values)

X = tokenizer.texts_to_sequences(df['text'].values)

X = pad_sequences(X)

Y = df['airline_sentiment']

L = Y.values
k = []

for i in range(6541):

    if L[i]=="negative":

                        k.append(0)

    elif L[i]=="positive":

                        k.append(1)
X_train, X_test, Y_train, Y_test = train_test_split(X, k, 

                                                    test_size=0.3, 

                                                    shuffle = True,

                                                    stratify = k,

                                                    random_state = 1 )
embed_dim = 128

lstm_out = 196

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Embedding(max_fatures, 128, input_length=X_train.shape[1]))

model.add(tf.keras.layers.SpatialDropout1D(0.5))

model.add(tf.keras.layers.LSTM(196, dropout = 0.3, recurrent_dropout = 0.3 ))

model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(100, activation = tf.nn.relu))

model.add(tf.keras.layers.Dropout(0.4))

model.add(tf.keras.layers.Dense(2, activation = tf.nn.softmax))
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
Model = model.fit(X_train,

                         Y_train,

                         epochs=8,

                         batch_size=32,

                         validation_split = 0.2,

                         verbose = 2)
score = model.evaluate(X_test, Y_test, verbose=False)

print("loss = ",score[0])

print("accuracy = ",score[1])
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred,axis = 1) 

confusion_mtx = confusion_matrix(Y_test, y_pred_classes) 

plot_confusion_matrix(confusion_mtx, classes = range(2)) 
sample = ['Meetings: Air crew is so dumb.']

sample = tokenizer.texts_to_sequences(sample)

sample = pad_sequences(sample, maxlen=31, dtype='int32', value=0)

print(sample)

sentiment = model.predict(sample,batch_size=1,verbose = 2)[0]

if(np.argmax(sentiment) == 0):

    print("negative")

elif (np.argmax(sentiment) == 1):

    print("positive")