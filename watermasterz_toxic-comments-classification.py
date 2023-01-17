import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

dir = "../input/cleaned-toxic-comments/train_preprocessed.csv"
data = pd.read_csv(dir)

data.head()
Features = data['comment_text']

Labels = np.array([0 if y == 0 else 1 for y in data['toxicity']])
NUM_WORDS = 40000 # Maximum number of unique words which need to be tokenized

MAXLEN = 50 # Maximum length of a sentence/ comment

PADDING = 'post' # The type of padding done for sentences shorter than the Max len
tokenizer = Tokenizer(num_words=NUM_WORDS)



# Fit the tokenizer on the comments 

tokenizer.fit_on_texts(Features)



# Get the word index of the top 20000 words from the dataset

word_idx = tokenizer.word_index



# Convert the string sentence to a sequence of their numerical values

Feature_sequences = tokenizer.texts_to_sequences(Features)



# Pad the sequences to make them of uniform length

padded_sequences = pad_sequences(Feature_sequences, maxlen = MAXLEN, padding = PADDING)
print("The Transformation of sentence::")

print("\n\nThe normal Sentencen:\n")

print(Features[2])

print("\n\nThe tokenized sequence:\n")

print(Feature_sequences[2])

print("\n\nThe padded sequence:\n")

print(padded_sequences[2])



# Convert to array for passing through the model

X = np.array(padded_sequences)
with open("../input/glove6b50dtxt/glove.6B.50d.txt", encoding='utf-8') as f:

    for x in f:

        print(x)

        break
EMBEDDING_DIM = 50 # number of dimensions of the word embeddings
# initialize the word to index dictionary

word_2_vec = {}

with open("../input/glove6b50dtxt/glove.6B.50d.txt", encoding='utf-8') as f:

    for line in f:

        

        # spilt the elements by space

        elements = line.split()

        word = elements[0]

        # convert to np array

        vecs = np.asarray(elements[1:], dtype='float32')

        word_2_vec[word] = vecs

        

print("Done....\n")
print(f"Number of words {len(word_2_vec)}")

print(f"Shape of the vector {len(word_idx)}")

print(f"Number of max words to be saved {NUM_WORDS}")
# get the max number of words that exist in word index and vocabulary both

num = min(NUM_WORDS, len(word_idx)+ 1)



# Matrix containing the word index and the vector of the word

embedding_matrix = np.zeros((num, EMBEDDING_DIM))



for word, idx in word_idx.items():

    if idx < NUM_WORDS:

        word_vec = word_2_vec.get(word)

        if word_vec is not None:

            embedding_matrix[idx] = word_vec

            

print(embedding_matrix.shape)
def train(models, epochs, graph=True, verbose=2):

    n = 1

    plt.figure(figsize=(10, 7))

    

    histories = []

    for model in models:

        print(f"model number : {n} is training")

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])



        history = model.fit(

            X, 

            Labels,

            batch_size=128,

            epochs=epochs,

            validation_split=0.2, # 20 percent data reserved for validation to avoid or monitor overfitting/ underfitting

            verbose=verbose,

        )

        histories.append(history)

        

        if graph:

            plt.plot(history.history['val_acc'], label=f"Model {n}")

        n+=1

            

    plt.xlabel('Epochs')

    plt.ylabel('Validation Accuracy')

    plt.legend()
model = tf.keras.models.Sequential([

    

    # Embedding layers that takes in the embedding matrix. Be sure to set trainable to false or else it will mess up your 

    # nicely pre trained vectors

    tf.keras.layers.Embedding(num, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAXLEN,trainable=False),

    tf.keras.layers.LSTM(5, return_sequences=True),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.summary()
model2= tf.keras.models.Sequential([

    tf.keras.layers.Embedding(num, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAXLEN,trainable=False),

    tf.keras.layers.LSTM(50, return_sequences=True),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(5, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])

model2.summary()
model3= tf.keras.models.Sequential([

    tf.keras.layers.Embedding(num, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAXLEN,trainable=False),

    

    tf.keras.layers.LSTM(50, return_sequences=True),

    tf.keras.layers.Conv1D(10,15, activation='relu'),



    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(5, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



model3.summary()
model4= tf.keras.models.Sequential([

    

    # Embedding layers that takes in the embedding matrix. Be sure to set trainable to false or else it will mess up your 

    # nicely pre trained vectors

    tf.keras.layers.Embedding(num, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAXLEN,trainable=False),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(5, return_sequences=True)),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(1, activation='sigmoid')

])

model4.summary()
models = [model, model2, model3, model4]

train(models, epochs=10)