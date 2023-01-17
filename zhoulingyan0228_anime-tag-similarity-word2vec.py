import numpy as np 
import pandas as pd 
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import keras
import matplotlib.pyplot as plt
df = pd.read_csv("../input/all_data.csv")
n_samples = 10000
recent_tags = df['tags'].tail(n_samples)
count_vectorizer = CountVectorizer(min_df=0.01,
                                   analyzer=lambda x: x.split(' ')
                                  )
count_vectorizer.fit(recent_tags)

N_CLASSES = len(count_vectorizer.vocabulary_)

def fit_generator(all_tags):
    while True:
        for tags in all_tags:
            words = count_vectorizer.transform(tags.split(' ')).toarray()
            contexts = count_vectorizer.transform([tags]).toarray().repeat(len(words), axis=0)
            yield words, contexts

model = keras.models.Sequential()
model.add(keras.layers.InputLayer((N_CLASSES,)))
model.add(keras.layers.Dense(2))
model.add(keras.layers.Activation('linear',name='embedding'))
model.add(keras.layers.Dense(N_CLASSES))
model.add(keras.layers.Activation('sigmoid'))
model.summary()

embedding = keras.models.Model(inputs=model.input, outputs=model.get_layer('embedding').output)
model.compile(loss='binary_crossentropy',
             optimizer=keras.optimizers.Adam(0.001),
             metrics=[])
model.fit_generator(fit_generator(recent_tags), 
                   epochs=10,
                   steps_per_epoch=n_samples,
                   callbacks=[keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=0, mode='auto')],
                   verbose=2)
keys = count_vectorizer.vocabulary_.keys()
input_vecs = count_vectorizer.transform(keys).toarray()
latents = embedding.predict_on_batch(input_vecs)
plt.figure(figsize=(30,30))
colors = ['b','g','r','c','m','y','k']
for k, l in zip(keys, latents):
    plt.scatter(latents.T[0], latents.T[1])
for k, l in zip(keys, latents):
    plt.text(l[0], l[1], k,
             alpha=0.5, 
             color=colors[np.random.randint(0, len(colors))],
             rotation=np.random.randint(-30, 30))