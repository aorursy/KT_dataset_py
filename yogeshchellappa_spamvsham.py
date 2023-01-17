import os
import pandas as pd
import numpy as np
df = pd.read_csv("../input/spam.csv", encoding='latin-1')
df.shape
df.head()
df.columns.values[2:]
df.drop(df.columns.values[2:], axis=1, inplace=True)
df.head()
df.shape
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
%matplotlib inline
# Let's consider only the top 10000 words appearing in the messages
maxWords = 10000
# Get the size of each word
sizes = df['v2'].map(lambda x: len(x.split(" ")))
plt.hist(sizes, normed=True, bins=50);
maxMessageSize = 100
labelText = df['v1'].tolist()
texts = df['v2'].tolist()
tokenizer = Tokenizer(num_words=maxWords)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=maxMessageSize)
labels = []
for i in labelText:
    if i == "ham":
        labels.append(0)
    elif i == "spam":
        labels.append(1)
labels = np.asarray(labels)
data.shape
labels.shape
indices = np.arange(data.shape[0])
np.random.shuffle(indices)

data = data[indices]
labels = labels[indices]
trainingSetSize = 3500
validationSetSize = 1000
trainingSet = data[:trainingSetSize]
trainingLabels = labels[:trainingSetSize]

validationSet = data[trainingSetSize: trainingSetSize + validationSetSize]
validationLabels = labels[trainingSetSize: trainingSetSize + validationSetSize]

testSet = data[trainingSetSize + validationSetSize:]
testLabels = labels[trainingSetSize + validationSetSize:]
from keras import models
from keras import layers
from keras import activations
from keras import optimizers
from keras import losses
from keras import metrics
embeddingDimension = 100
model = models.Sequential()
model.add(layers.Embedding(maxWords, embeddingDimension, input_length=maxMessageSize))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation=activations.relu))
model.add(layers.Dense(1, activation=activations.sigmoid))

model.compile(optimizer=optimizers.Adam(lr=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])

history = model.fit(trainingSet, trainingLabels, epochs=10, batch_size=64, validation_data=(validationSet, validationLabels))
history.history.keys()
acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
model.save("spamVsHam.h5")
testSet.shape
testLabels.shape
testLoss, testAccuracy = model.evaluate(testSet, testLabels)
testAccuracy
