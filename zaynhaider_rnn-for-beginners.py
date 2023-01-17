import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Reading csv file

ds = pd.read_csv('/kaggle/input/question-classification/Question_Classification_Dataset.csv')
ds.head()
# Droping the other Columns because we deel with only Category0

ds.drop(['Unnamed: 0','Category1','Category2'], axis=1, inplace=True)
ds
# Counting the target values of Category0

ds['Category0'].value_counts()
ds.info
# Importing Necessary Libraries

import seaborn as sns
import matplotlib.pyplot as plt

# Printing the Count plot of the target values of Category0

sns.countplot(ds['Category0'])
plt.xlabel('Classes')
# Encoding the Target Labels

from sklearn.preprocessing import LabelEncoder

Category0_n = LabelEncoder()
ds['Category0_n'] = Category0_n.fit_transform(ds['Category0'])
ds
# Habilitate the data for spliting into train and test data

x = ds['Questions']
y = ds['Category0_n']
y = y.to_numpy()
x
y
# Spliting the data into train and test data

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2)
x_train.shape , y_train.shape
# Converting the textual data into the form of sequence arrays

max_words = 1100
max_len = 200

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

tok = Tokenizer(num_words=max_words)  # Initializing the Tokenizer
tok.fit_on_texts(x_train)             # fit the train model 
seq = tok.texts_to_sequences(x_train) # Converting Text to Sequence array
seq_matrix = sequence.pad_sequences(sequences=seq, maxlen=max_len)   

seq_matrix
# Converting test data set into text sequence array for testing...!

test_seq = tok.texts_to_sequences(x_test)
test_seq_matrix = sequence.pad_sequences(test_seq, maxlen=max_len)

test_seq_matrix
# Importing the necessary libraries

import keras
import tensorflow as tf
# RNN Model 

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=max_words, output_dim=6, input_length=max_len))
model.add(tf.keras.layers.LSTM(64, activation='tanh'))

model.add(tf.keras.layers.Dense(220, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(6, activation='softmax'))


model.compile(optimizer=tf.keras.optimizers.Adam(), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(seq_matrix,y_train, batch_size=60, epochs=10, validation_split=0.2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuacy')
plt.legend(['Acc','Val'], loc = 'upper left')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Loss','Val'], loc = 'upper left')
from sklearn.metrics import accuracy_score,confusion_matrix

y_pred = model.predict_classes(test_seq_matrix)
print('Accuracy Score : ',accuracy_score(y_test,y_pred))
cm = confusion_matrix(y_test,y_pred)
print('Confusion Matrix :\n',cm)
plt.figure(figsize=(7,5))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('truth')