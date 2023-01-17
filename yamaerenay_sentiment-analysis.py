import numpy as np 

import pandas as pd

import os

from tqdm.notebook import tqdm

from tensorflow.keras import layers, models

import spacy as sp

nlp = sp.load("en_core_web_lg")
dfs = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        dfs.append(pd.read_csv(os.path.join(dirname, filename)))

df = pd.concat(dfs, 0).reset_index(drop = True)



labellist = ["sad", "happy", "angry"]



parts = []

for label in labellist:

    part = df[df["sentiment"] == label].copy()

    sentiment = []

    for index in range(len(part.index)):

        if(part["sentiment"].iloc[index] == label):

            sentiment.append(1)

        else:

            sentiment.append(0)

    part[label] = sentiment

    parts.append(part)

df = pd.concat(parts, 0).drop("sentiment", 1).fillna(0)
df
def vectorize(texts):

    with nlp.disable_pipes():

        vectors = np.array(

            [

                nlp(texts[text_index]).vector

                for text_index in tqdm(range(len(texts)))

            ]

        )

    return vectors
def train(X_data, y_data, epochs = 50, batch_size = 8):

    model = models.Sequential()

    model.add(layers.Dense(300, activation = "relu", input_shape = (300,)))

    model.add(layers.Dense(1200, activation = "relu"))

    model.add(layers.Dense(3, activation = "softmax"))

    model.compile(optimizer = "adam", metrics = ["accuracy"], loss = "categorical_crossentropy")

    model.fit(X_data, y_data, epochs = epochs, batch_size = batch_size, shuffle = True)

    return model
def predict(model, text, labellist = labellist):

    if(type(text) == list):

        pred = model.predict(nlp(text).vector)

    else:

        pred = model.predict(np.expand_dims(nlp(text).vector, 0))

    print("Predictions:\n" + "\n".join([labellist[x] + ": " + str(pred[0][x]) for x in range(len(pred[0]))]))
def evaluate(model, text, label, labellist = labellist):

    label = np.array(list(map(lambda x: (x==label)*1, labellist)))

    if(type(text) == list):

        loss, accuracy = model.evaluate(nlp(text).vector, label)

    else:

        loss, accuracy = model.evaluate(np.expand_dims(nlp(text).vector, 0), np.expand_dims(label, 0))

    print("Loss: %d" % loss)

    print("Accuracy: %d" % accuracy)
X_data = vectorize(df["content"].str.lower().values)

y_data = df[labellist]
model = train(X_data, y_data)
predict(model, "It hurts me so much...")
predict(model, "Cheers to health")
predict(model, "He is a sick bastard")
evaluate(model, "You are wonderful and gorgeous! I love you", "happy")