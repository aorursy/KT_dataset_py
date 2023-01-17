import numpy as np 

import pandas as pd

import json
import tqdm 

# for some fancy loading bars
dir = "../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json"
with open(dir) as f:

    for i in f:

        print(i)

        break
with open(dir) as f:

    

    for i in f:

        json_loaded = json.loads(i)

        

        for key in json_loaded:

            print(key," | ",  json_loaded[key])

            

        break

        
headlines = []

labels = []



with open(dir) as f:

    for i in f:

        json_loaded = json.loads(i)

        headlines.append(json_loaded['headline'])

        labels.append(json_loaded['is_sarcastic'])



print(f"Number of sentences/headlines: {len(headlines)}")

print("A random entry and its label:\n")

print(headlines[42])

print(labels[42])

# is_sarcastic = 1 for a sarcastic comment and 0 for a non sarcastic one

import re

import nltk
# Example

s = "My github page is https://github.com/The-Bread please follow me "

text = re.sub(r'(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*', '', s, flags=re.MULTILINE)

text
# example

s = "Buy buys . Likes. seems . Doesn't, does, easy,"



s = s.lower()

token = nltk.word_tokenize(s)

sentence = [nltk.stem.SnowballStemmer('english').stem(word) for word in token]



print(" ".join(sentence))
def preprocess(sentence):

    sentence = re.sub(r'(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*', '', sentence, flags=re.MULTILINE)

    sentence = sentence.lower()

    token = nltk.word_tokenize(sentence)

    sentence = [nltk.stem.SnowballStemmer('english').stem(word) for word in token]

    return " ".join(sentence)
example = headlines[52]

print("Original:")

print(example)

print("\nPreprocessed: ")

print(preprocess(example))
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
S = ["Hello", "How are you today", "Where are you"]

tokenizer = Tokenizer()

tokenizer.fit_on_texts(S)

print(f"Actual Sentence: {S[2]}\n\n")



S = tokenizer.texts_to_sequences(S)

print(f"Tokenized Sequence: {S[2]}\n\n")



print(tokenizer.word_index)

padded = pad_sequences(S, maxlen=3, padding='post', truncating='post')

print(f"Padded sequences:\n{padded}")
def tokenize_data(X):

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(X)

    X = tokenizer.texts_to_sequences(X)

    X = pad_sequences(X, maxlen=15, padding='post', truncating='post')

    X = np.array(X) # convert to numpy array 

    return X, tokenizer
stemmed = [preprocess(s) for s in tqdm.tqdm(headlines, desc='Stemming the Headlines')]

print("Stemming complete")



print("\nTokenizing the headlines")

X, tokenizer = tokenize_data(stemmed)

print("Headlines Tokenized")

print(f"\nInput Shape: {X.shape}")



Y = np.array(labels)

print(f"Ouput Shape: {Y.shape}")



# Reshape Y to make it 2D like the inputs

Y = np.reshape(Y, (-1, 1))

print(f"Final Output Shape: {Y.shape}")
print("Number of words tokenized: ", len(tokenizer.word_index))

num_words = len(tokenizer.word_index)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, LSTM, Bidirectional, GlobalAveragePooling1D, Conv1D, Embedding
model = Sequential([

    Embedding(num_words+1, output_dim=10, input_length=15),

    LSTM(15, dropout=0.1, return_sequences=True),

    Conv1D(15, 10, activation='relu'),

    GlobalAveragePooling1D(),

    

    Dense(32,activation='relu'),

    Dense(1, activation='sigmoid')

])

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics = ['acc'])

model.summary()
model.fit(X, Y, epochs=3)