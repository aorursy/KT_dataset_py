import numpy as np

import pandas as pd

from keras.layers import (Input, Embedding, Conv1D, Activation, GlobalMaxPooling1D, BatchNormalization,

                          Concatenate, CuDNNLSTM, Flatten, Dropout, Dense)

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

from keras.models import Model, Sequential

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.initializers import Constant

from keras.metrics import top_k_categorical_accuracy

from keras.utils import to_categorical, Sequence, plot_model

import matplotlib.pyplot as plt
news = pd.read_pickle('../input/stopwords-removal-and-lemmatization/data.pkl')

news = shuffle(news, random_state=29)

news.head()
news['category'] = news['category'].apply(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)

labels = list(set(news['category']))

news['category'] = news['category'].apply(labels.index)

X_train, X_test, y_train, y_test = train_test_split(news['text'], news['category'], test_size=0.3, random_state=4)

X_holdout, X_test, y_holdout, y_test = train_test_split(X_test, y_test, test_size=0.66, random_state=98)

y_train, y_test, y_holdout = list(y_train), list(y_test), list(y_holdout)
tokenizer = Tokenizer(lower=True)

tokenizer.fit_on_texts(news['text'])

X_train = tokenizer.texts_to_sequences(X_train)

X_holdout = tokenizer.texts_to_sequences(X_holdout)

X_test = tokenizer.texts_to_sequences(X_test)
def get_embed_mat(EMBEDDING_FILE):

    embeddings_index = {}

    with open(EMBEDDING_FILE, 'r') as f:

        for line in f:

            values = line.split()

            word = values[0]

            coefs = np.asarray(values[1:], dtype='float32')

            embeddings_index[word] = coefs

        f.close()

    

    word_index = tokenizer.word_index

    num_words = len(word_index) + 1

    all_embs = np.stack(embeddings_index.values())

    embedding_matrix = np.random.normal(all_embs.mean(), all_embs.std(), 

                                        (num_words, embed_dim))

    for word, i in word_index.items():

        if i >= num_words:

            break

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector

    return num_words, embedding_matrix



EMBEDDING_FILE = '../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt'

embed_dim = 200

num_words, embedding_matrix = get_embed_mat(EMBEDDING_FILE)
plt.hist([len(x) for x in X_train + X_test+X_holdout], bins=100)

plt.ylabel('Number of sequences')

plt.xlabel('Length')

plt.show()
plt.hist([len(x) for x in X_train + X_holdout + X_test], bins=100)

plt.ylabel('Number of sequences')

plt.xlabel('Length')

plt.show()
max_length = 50

print('Sequence length:', max_length)
num_classes = len(labels)

layers = []

filters = [2, 3, 5]



sequence_input1 = Input(shape=(max_length, ), dtype='int32')

embedding_layer_static1 = Embedding(num_words, embed_dim, embeddings_initializer=Constant(embedding_matrix),

                            input_length=max_length, trainable=False)(sequence_input1)



for sz in filters:

    conv_layer1 = Conv1D(filters=256, kernel_size=sz)(embedding_layer_static1)

    batchnorm_layer1 = BatchNormalization()(conv_layer1)

    act_layer1 = Activation('relu')(batchnorm_layer1)

    pool_layer1 = GlobalMaxPooling1D()(act_layer1)

    layers.append(pool_layer1)



merged1 = Concatenate(axis=1)(layers)



drop1 = Dropout(0.5)(merged1)

dense1 = Dense(512, activation='relu')(drop1)

out1 = Dense(num_classes, activation='softmax')(dense1)



cnn_static = Model(sequence_input1, out1)

cnn_static.summary()
def top_3_acc(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=3)



class dataseq(Sequence):

    def __init__(self, X, y, batch_size, padding='post'):

        self.x, self.y = X, y

        self.batch_size = batch_size

        self.m = len(self.y)

        self.padding = padding



    def __len__(self):

        return int(np.ceil(len(self.x) / float(self.batch_size)))



    def __getitem__(self, idx):

        batch_x = self.x[idx * self.batch_size:min((idx + 1) * self.batch_size, self.m)]

        batch_y = self.y[idx * self.batch_size:min((idx + 1) * self.batch_size, self.m)]



        return pad_sequences(batch_x, maxlen=max_length, truncating='post', padding=self.padding), to_categorical(

            batch_y, num_classes=num_classes)



cnn_static.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', top_3_acc])



batch_size = 128

cnn_static_history = cnn_static.fit_generator(dataseq(X_train, y_train, batch_size), epochs=15, verbose=2,

                              validation_data = dataseq(X_holdout, y_holdout, batch_size), shuffle=True)
plt.plot(cnn_static_history.history['loss'], label='train')

plt.plot(cnn_static_history.history['val_loss'], label='holdout')

plt.title('CNN - Static learning curve')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(loc='upper right')

plt.show()
cnn_static.evaluate_generator(dataseq(X_test, y_test, batch_size))
layers = []



embedding_layer_dynamic1 = Embedding(num_words, embed_dim, embeddings_initializer=Constant(embedding_matrix),

                            input_length=max_length)(sequence_input1)



for sz in filters:

    conv_layer2 = Conv1D(filters=256, kernel_size=sz)(embedding_layer_dynamic1)

    batchnorm_layer2 = BatchNormalization()(conv_layer2)

    act_layer2 = Activation('relu')(batchnorm_layer2)

    pool_layer2 = GlobalMaxPooling1D()(act_layer2)

    layers.append(pool_layer2)



merged2 = Concatenate(axis=1)(layers)



drop2 = Dropout(0.5)(merged2)

dense2 = Dense(512, activation='relu')(drop2)

out2 = Dense(num_classes, activation='softmax')(dense2)



cnn_dynamic = Model(sequence_input1, out2)
cnn_dynamic.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', top_3_acc])

cnn_dynamic_history = cnn_dynamic.fit_generator(dataseq(X_train, y_train, batch_size), epochs=5, verbose=2,

                              validation_data = dataseq(X_holdout, y_holdout, batch_size), shuffle=True)
plt.plot(cnn_dynamic_history.history['loss'], label='train')

plt.plot(cnn_dynamic_history.history['val_loss'], label='holdout')

plt.title('CNN - Dynamic learning curve')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(loc='upper right')

plt.show()
cnn_dynamic.evaluate_generator(dataseq(X_test, y_test, batch_size))
sequence_input2 = Input(shape=(max_length, ), dtype='int32')

embedding_layer_static2 = Embedding(num_words, embed_dim, embeddings_initializer=Constant(embedding_matrix),

                            input_length=max_length, trainable=False)(sequence_input2)

lstm1 = CuDNNLSTM(500, return_sequences=True)(embedding_layer_static2)

drop3 = Dropout(0.5)(lstm1)

lstm2 = CuDNNLSTM(200)(drop3)

drop4 = Dropout(0.5)(lstm2)

out3 = Dense(num_classes, activation='softmax')(drop4)

lstm_static = Model(sequence_input2, out3)

lstm_static.summary()
lstm_static.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', top_3_acc])

lstm_static_history = lstm_static.fit_generator(dataseq(X_train, y_train, batch_size, 'pre'), epochs=5, verbose=2,

                              validation_data = dataseq(X_holdout, y_holdout, batch_size, 'pre'))
plt.plot(lstm_static_history.history['loss'], label='train')

plt.plot(lstm_static_history.history['val_loss'], label='test')

plt.title('LSTM - Static learning curve')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(loc='upper right')

plt.show()
lstm_static.evaluate_generator(dataseq(X_test, y_test, batch_size, 'pre'))
embedding_layer_dynamic2 = Embedding(num_words, embed_dim, embeddings_initializer=Constant(embedding_matrix),

                           input_length=max_length)(sequence_input2)

lstm3 = CuDNNLSTM(500, return_sequences=True)(embedding_layer_dynamic2)

drop5 = Dropout(0.5)(lstm3)

lstm4 = CuDNNLSTM(200)(drop5)

drop6 = Dropout(0.5)(lstm4)

out4 = Dense(num_classes, activation='softmax')(drop6)

lstm_dynamic = Model(sequence_input2, out4)
lstm_dynamic.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', top_3_acc])

lstm_dynamic_history = lstm_dynamic.fit_generator(dataseq(X_train, y_train, batch_size, 'pre'), epochs=4, verbose=2,

                              validation_data = dataseq(X_holdout, y_holdout, batch_size, 'pre'))
plt.plot(lstm_dynamic_history.history['loss'], label='train')

plt.plot(lstm_dynamic_history.history['val_loss'], label='test')

plt.title('LSTM - Dynamic learning curve')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(loc='upper right')

plt.show()
lstm_dynamic.evaluate_generator(dataseq(X_test, y_test, batch_size, 'pre'))
models = [cnn_static, cnn_dynamic, lstm_static, lstm_dynamic]



for i in range(len(models)):

    for layer in models[i].layers:

        layer.trainable = False



input_layers = [sequence_input1, sequence_input2]

output_layers = [model.output for model in models]

ensemble_merge = Concatenate()(output_layers)

ensemble_dense1 = Dense(120, activation='relu')(ensemble_merge)

ensemble_dense2 = Dense(80, activation='relu')(ensemble_dense1)

output = Dense(num_classes, activation='softmax')(ensemble_dense2)

model = Model(inputs=input_layers, outputs=output)
class meta_dataseq(Sequence):

    def __init__(self, X, y, batch_size):

        self.x, self.y = X, y

        self.batch_size = batch_size

        self.m = len(self.y)



    def __len__(self):

        return int(np.ceil(len(self.x) / float(self.batch_size)))



    def __getitem__(self, idx):

        batch_x = self.x[idx * self.batch_size:min((idx + 1) * self.batch_size, self.m)]

        batch_y = to_categorical(self.y[idx * self.batch_size:min((idx + 1) * self.batch_size, self.m)],

                                 num_classes=num_classes)

        

        return [pad_sequences(batch_x, maxlen=max_length, truncating='post', padding='post'),

                 pad_sequences(batch_x, maxlen=max_length, truncating='post', padding='pre')], batch_y



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', top_3_acc])

model_history = model.fit_generator(meta_dataseq(X_holdout, y_holdout, batch_size), epochs=20, verbose=2,

                              validation_data = meta_dataseq(X_test, y_test, batch_size))
plt.plot(model_history.history['loss'], label='train')

plt.plot(model_history.history['val_loss'], label='test')

plt.title('Meta-classifer learning curve')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(loc='upper right')

plt.show()
plot_model(model, to_file='model.png')

model_json = model.to_json()

with open('model.json', 'w') as json_file:

    json_file.write(model_json)

model.save_weights('model.h5')

print('Saved model to disk')