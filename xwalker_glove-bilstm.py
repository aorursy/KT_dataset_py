import keras

from keras.models import Sequential

from keras.initializers import Constant

from keras.layers import (LSTM, 

                          Embedding, 

                          BatchNormalization,

                          Dense, 

                          TimeDistributed, 

                          Dropout, 

                          Bidirectional,

                          Flatten, 

                          GlobalMaxPool1D)



from nltk.tokenize import word_tokenize

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers.embeddings import Embedding

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras.optimizers import Adam





import pandas as pd

import numpy as np



from sklearn.metrics import (

    precision_score, 

    recall_score, 

    f1_score, 

    classification_report,

    accuracy_score

)



import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
def plot(history, arr):



    fig, ax = plt.subplots(1, 2, figsize=(20, 5))

    for idx in range(2):

        ax[idx].plot(history.history[arr[idx][0]])

        ax[idx].plot(history.history[arr[idx][1]])

        ax[idx].legend([arr[idx][0], arr[idx][1]],fontsize=18)

        ax[idx].set_xlabel('A ',fontsize=16)

        ax[idx].set_ylabel('B',fontsize=16)

        ax[idx].set_title(arr[idx][0] + ' X ' + arr[idx][1],fontsize=16)
dataset = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')

submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
train = dataset.text.values

test = test.text.values

sentiments = dataset.target.values
word_tokenizer = Tokenizer()

word_tokenizer.fit_on_texts(train)

vocab_length = len(word_tokenizer.word_index) + 1
def metrics(pred_tag, y_test):



    print("F1-score: ", f1_score(pred_tag, y_test))

    print("Precision: ", precision_score(pred_tag, y_test))

    print("Recall: ", recall_score(pred_tag, y_test))

    print("Acuracy: ", accuracy_score(pred_tag, y_test))



    print("-"*50)

    print(classification_report(pred_tag, y_test))

    

def embed(corpus): 

    return word_tokenizer.texts_to_sequences(corpus)
longest_train = max(train, key=lambda sentence: len(word_tokenize(sentence)))

length_long_sentence = len(word_tokenize(longest_train))

padded_sentences = pad_sequences(embed(train), length_long_sentence, padding='post')



test_sentences = pad_sequences(

    embed(test), 

    length_long_sentence,

    padding='post'

)
# #Twitter Gloves



embeddings_dictionary = dict()

embedding_dim = 200

# glove_file = open('../input/glove-global-vectors-for-word-representation/glove.6B.' + str(embedding_dim) + 'd.txt', encoding="utf8")

glove_file = open('../input/glove-twitter/glove.twitter.27B.' + str(embedding_dim) + 'd.txt', encoding="utf8")



for line in glove_file:

    records = line.split()

    word = records[0]

    vector_dimensions = np.asarray(records[1:], dtype='float32')

    embeddings_dictionary [word] = vector_dimensions



glove_file.close()
embedding_matrix = np.zeros((vocab_length, embedding_dim))

for word, index in word_tokenizer.word_index.items():

    if index >= vocab_length:

        continue

    embedding_vector = embeddings_dictionary.get(word)

    if embedding_vector is not None:

        embedding_matrix[index] = embedding_vector
def BLSTM():

    model = Sequential()

    model.add(Embedding(input_dim=embedding_matrix.shape[0], 

                        output_dim=embedding_matrix.shape[1], 

                        weights = [embedding_matrix], 

                        input_length=length_long_sentence,

                        trainable=False))

    

    model.add(Bidirectional(LSTM(16)))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model
def BLSTM():

    model = Sequential()

    model.add(Embedding(input_dim=embedding_matrix.shape[0], 

                        output_dim=embedding_matrix.shape[1], 

                        weights = [embedding_matrix], 

                        input_length=length_long_sentence,

                        trainable=False))

    

    model.add(Bidirectional(LSTM(length_long_sentence, return_sequences = True, recurrent_dropout=0.2)))

    model.add(GlobalMaxPool1D())

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(length_long_sentence, activation = "relu"))

    model.add(Dropout(0.5))

    model.add(Dense(length_long_sentence, activation = "relu"))

    model.add(Dropout(0.5))

    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model




reduce_lr = ReduceLROnPlateau(

    monitor='val_loss', 

    factor=0.2, 

    verbose =1, 

    patience=5,                        

    min_lr=0.001

)



for idx in range(5):

    

    print("*"*20 + '\nModelo: ' + str(idx) + '\n')

    

    reduce_lr = ReduceLROnPlateau(

        monitor='val_loss', 

        factor=0.2, 

        verbose =1, 

        patience=5,                        

        min_lr=0.001

    )

    checkpoint = ModelCheckpoint(

        'model_' + str(idx)+ '.h5', 

        monitor='val_loss',

        mode='auto',

        verbose=1,

        save_weights_only = True,

        save_best_only=True

    )

    

    X_train, X_test, y_train, y_test = train_test_split(

        padded_sentences, 

        sentiments, 

        test_size=0.5

    )

    

    model = BLSTM()

    model.fit(X_train,

              y_train,

              batch_size=32,

              epochs=15,

              validation_data=[X_test, y_test],

              callbacks = [reduce_lr, checkpoint],

              verbose=1)
%%time

from glob import glob

import scipy



x_models = []

labels = []



# Carregando os Modelos

for idx in glob('*.h5'):

    model = BLSTM()

    model.load_weights(idx)

    x_models.append(model)

    

# Predizendo Classes para o conjunto de Testes

for idx in x_models:

    preds = idx.predict_classes(test_sentences)

    labels.append(preds)



#Votando nas classes, baseando na moda estatística 

labels = scipy.stats.mode(labels)[0]

labels = np.squeeze(labels)   
submission.target = labels

submission.to_csv("submission.csv", index=False)

submission.target.value_counts().plot.bar();
submission