# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras import layers
!ls ../input
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
train_df.info()
train_df.loc[1:3]["question_text"]
X_train = train_df["question_text"].fillna("kh").values

X_test = test_df["question_text"].fillna("kh").values

y = train_df["target"]
import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('ggplot')



def plot_history(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    x = range(1, len(acc) + 1)



    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(x, acc, 'b', label='Training acc')

    plt.plot(x, val_acc, 'r', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.plot(x, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()
from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer()

vectorizer.fit(list(X_train))

X_train_vec = vectorizer.transform(list(X_train))

X_test_vec  = vectorizer.transform(list(X_test))



#feature selection

from sklearn.feature_selection import SelectKBest, chi2

max_features = 50000

ch2 = SelectKBest(chi2, max_features)

x_train = ch2.fit_transform(X_train_vec, y)

x_test = ch2.transform(X_test_vec)



from sklearn.model_selection import train_test_split

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y, test_size = 0.1, random_state=42)



from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(x_train, y)

pre = classifier.predict_proba(x_test)



# For submission 

#y_pre= [ np.argmax(i) for i in pre]

#submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_pre})

#submit_df.to_csv("submission.csv", index=False)
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer()

X_train_vec= vectorizer.fit_transform(list(X_train))

X_test_vec = vectorizer.transform(list(X_test))



feature_names = vectorizer.get_feature_names()

len(feature_names)
#feature selection

max_features = 50000

ch2 = SelectKBest(chi2, max_features)

x_train = ch2.fit_transform(X_train_vec, y)

x_test = ch2.transform(X_test_vec)
from keras.models import Sequential

from keras import layers



model = Sequential()

model.add(layers.Dense(10, input_dim=max_features, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])



model.summary()
from sklearn.model_selection import train_test_split

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y, test_size = 0.1, random_state=42)



#vars

batch_size = 32

epochs = 4



hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),

                  verbose=True)



y_pred = model.predict(x_test, batch_size=1024)



plot_history(hist)



y_te = (y_pred[:,0] > 0.5).astype(np.int)



#for submission

#submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})

#submit_df.to_csv("submission.csv", index=False)
# Performance 

loss, accuracy = model.evaluate(X_tra,y_tra, verbose=False)

print("Training split Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_val,y_val, verbose=False)

print("Validation Accuracy:  {:.4f}".format(accuracy))
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(list(X_train))



x_train = tokenizer.texts_to_sequences(list(X_train))

x_test = tokenizer.texts_to_sequences(list(X_test))

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
print(list(X_train)[2])

x_train[2]
for word in [ 'why', 'does', 'velocity', 'affect', 'time']:

    print('{}: {}'.format(word, tokenizer.word_index[word]))
# make all tokenized sentences in same size

from keras.preprocessing.sequence import pad_sequences

maxlen = 100

x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)

x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)
print(list(X_train)[2])

print(x_train[2])
print(x_train.shape)
from keras.models import Sequential

from keras import layers



embedding_dim = 50



model = Sequential()

model.add(layers.Embedding(input_dim=vocab_size, 

                           output_dim=embedding_dim, 

                           input_length=maxlen))

model.add(layers.Flatten())

model.add(layers.Dense(10, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

model.summary()



batch_size = 32

epochs = 4



from sklearn.model_selection import train_test_split

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y, test_size = 0.1, random_state=42)



hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),

                  verbose=True)



y_pred = model.predict(x_test, batch_size=1024)

plot_history(hist)



y_te = (y_pred[:,0] > 0.5).astype(np.int)

#for submission 

#submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})

#submit_df.to_csv("submission.csv", index=False)



#performance

loss, accuracy = model.evaluate(X_tra,y_tra, verbose=False)

print("Training split Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_val,y_val, verbose=False)

print("Validation Accuracy:  {:.4f}".format(accuracy))
from keras.models import Sequential

from keras import layers



embedding_dim = 50



model = Sequential()

model.add(layers.Embedding(input_dim=vocab_size, 

                           output_dim=embedding_dim, 

                           input_length=maxlen))

model.add(layers.GlobalMaxPool1D())

model.add(layers.Dense(10, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

model.summary()



batch_size = 32

epochs = 5



from sklearn.model_selection import train_test_split

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y, test_size = 0.1, random_state=42)



hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),

                  verbose=True)



y_pred = model.predict(x_test, batch_size=1024)

plot_history(hist)



y_te = (y_pred[:,0] > 0.5).astype(np.int)

#submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})

#submit_df.to_csv("submission.csv", index=False)



loss, accuracy = model.evaluate(X_tra,y_tra, verbose=False)

print("Training split Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_val,y_val, verbose=False)

print("Validation Accuracy:  {:.4f}".format(accuracy))
! ls ../input/embeddings/*/
def create_embedding_matrix(filepath, word_index, embedding_dim):

    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index

    embedding_matrix = np.zeros((vocab_size, embedding_dim))



    with open(filepath) as f:

        for line in f:

            word, *vector = line.split(' ')

            if word in word_index:

                idx = word_index[word] 

                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]



    return embedding_matrix
embedding_dim = 50

embedding_matrix_glove = create_embedding_matrix('../input/embeddings/glove.840B.300d/glove.840B.300d.txt',

        tokenizer.word_index, embedding_dim)



#embedding_matrix.shape



nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix_glove, axis=1))

nonzero_elements / vocab_size
embedding_matrix_wiki = create_embedding_matrix('../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec',

        tokenizer.word_index, embedding_dim)



nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix_wiki, axis=1))

nonzero_elements / vocab_size
from gensim.models import KeyedVectors

def create_embedding_matrix_google(filepath, word_index, embedding_dim):

    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index

    embedding_matrix = np.zeros((vocab_size, embedding_dim))



    wv_from_bin = KeyedVectors.load_word2vec_format(filepath, binary=True) 

    for word, vector in zip(wv_from_bin.vocab, wv_from_bin.vectors):

        if word in word_index:

                idx = word_index[word] 

                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]



    return embedding_matrix
filepath = "../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"



embedding_matrix_google = create_embedding_matrix_google(filepath,tokenizer.word_index, embedding_dim)

#embedding_matrix_google.shape

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix_google, axis=1))

nonzero_elements / vocab_size
model = Sequential()

model.add(layers.Embedding(vocab_size, embedding_dim, 

                           weights=[embedding_matrix_glove], 

                           input_length=maxlen, 

                           trainable=False))

model.add(layers.GlobalMaxPool1D())

model.add(layers.Dense(10, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

model.summary()



batch_size = 32

epochs = 5



from sklearn.model_selection import train_test_split

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y, test_size = 0.1, random_state=42)



hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),

                  verbose=True)



y_pred = model.predict(x_test, batch_size=1024)

plot_history(hist)
model = Sequential()

model.add(layers.Embedding(vocab_size, embedding_dim, 

                           weights=[embedding_matrix_glove], 

                           input_length=maxlen, 

                           trainable=True))

model.add(layers.GlobalMaxPool1D())

model.add(layers.Dense(10, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

model.summary()



batch_size = 32

epochs = 5



from sklearn.model_selection import train_test_split

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y, test_size = 0.1, random_state=42)



hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),

                  verbose=True)



y_pred = model.predict(x_test, batch_size=1024)

plot_history(hist)



y_te = (y_pred[:,0] > 0.5).astype(np.int)

#submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})

#submit_df.to_csv("submission.csv", index=False)



loss, accuracy = model.evaluate(X_tra,y_tra, verbose=False)

print("Training split Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_val,y_val, verbose=False)

print("Validation Accuracy:  {:.4f}".format(accuracy))
embedding_dim = 100



model = Sequential()

model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))

model.add(layers.Conv1D(128, 5, activation='relu'))

model.add(layers.GlobalMaxPooling1D())

model.add(layers.Dense(10, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

model.summary()



batch_size = 32

epochs = 5



from sklearn.model_selection import train_test_split

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y, test_size = 0.1, random_state=42)



hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),

                  verbose=True)



y_pred = model.predict(x_test, batch_size=1024)

plot_history(hist)



y_te = (y_pred[:,0] > 0.5).astype(np.int)

#submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})

#submit_df.to_csv("submission.csv", index=False)



loss, accuracy = model.evaluate(X_tra,y_tra, verbose=False)

print("Training split Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_val,y_val, verbose=False)

print("Validation Accuracy:  {:.4f}".format(accuracy))
from keras.layers import LSTM, Dense, Bidirectional, Input,Dropout,BatchNormalization, CuDNNGRU, CuDNNLSTM



embedding_dim = 100

batch_size = 32

epochs = 1



model = Sequential()

model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))

model.add(CuDNNLSTM(128,return_sequences=True))

model.add(layers.GlobalMaxPooling1D())

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])



model.summary()



from sklearn.model_selection import train_test_split

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y, test_size = 0.1, random_state=42)



hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),

                  verbose=True)



y_pred = model.predict(x_test, batch_size=1024)



y_te = (y_pred[:,0] > 0.5).astype(np.int)

#submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})

#submit_df.to_csv("submission.csv", index=False)



loss, accuracy = model.evaluate(X_tra,y_tra, verbose=False)

print("Training split Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_val,y_val, verbose=False)

print("Validation Accuracy:  {:.4f}".format(accuracy))
from keras.layers import LSTM, Dense, Bidirectional, Input,Dropout,BatchNormalization, CuDNNGRU, CuDNNLSTM



embedding_dim = 100

batch_size = 32

epochs = 1



model = Sequential()

model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))

model.add(CuDNNGRU(64, return_sequences=True))

model.add(layers.GlobalMaxPooling1D())

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

model.summary()



from sklearn.model_selection import train_test_split

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y, test_size = 0.1, random_state=42)



hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),

                  verbose=True)



y_pred = model.predict(x_test, batch_size=1024)



y_te = (y_pred[:,0] > 0.5).astype(np.int)

#submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})

#submit_df.to_csv("submission.csv", index=False)



loss, accuracy = model.evaluate(X_tra,y_tra, verbose=False)

print("Training split Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_val,y_val, verbose=False)

print("Validation Accuracy:  {:.4f}".format(accuracy))

from keras.models import Model

from keras.layers import Input, Dense, Embedding, concatenate

from keras.layers import CuDNNGRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D



maxlen = 100

max_features = 50000



def get_model():

    inp = Input(shape=(maxlen, ))

    x = Embedding(max_features, 100)(inp)

    x = CuDNNGRU(64, return_sequences=True)(x)

    avg_pool = GlobalAveragePooling1D()(x)

    max_pool = GlobalMaxPooling1D()(x)

    conc = concatenate([avg_pool, max_pool])

    outp = Dense(1, activation="sigmoid")(conc)

    

    model = Model(inputs=inp, outputs=outp)

    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])



    return model



model = get_model()
batch_size = 32

epochs = 1



from sklearn.model_selection import train_test_split

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y, test_size = 0.1, random_state=42)



hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),

                  verbose=True)



y_pred = model.predict(x_test, batch_size=1024)



y_te = (y_pred[:,0] > 0.5).astype(np.int)

submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})

submit_df.to_csv("submission.csv", index=False)