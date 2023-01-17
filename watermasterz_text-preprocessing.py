import pandas as pd

import numpy as np

import plotly.express as px



from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

import re

from nltk.stem.snowball import SnowballStemmer

import tqdm

dir = "../input/sarcastic-comments-on-reddit/train-balanced-sarcasm.csv"

data = pd.read_csv(dir)

data.head(5)
comments = data['comment'].values

labels = data['label'].values
text_cleaning = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

stemmer = SnowballStemmer('english', ignore_stopwords=False)
def preprocess_data(text):

    text = re.sub(text_cleaning, ' ', str(text).lower()).strip()

    text = stemmer.stem(str(text))

    return text



X = []

for i in tqdm.tqdm(range(len(comments))):

    X.append(preprocess_data(comments[i]))
tokenizer = Tokenizer(oov_token='<OOV>')

tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)

padded = pad_sequences(sequences, padding='post', maxlen=20)
xtrain, xtest, ytrain, ytest = train_test_split(np.array(padded), np.array(labels))
print(f"Actual Sentence: {comments[860]}\nStemmed Sentence: {X[860]}\nTokenized: {sequences[860]}\nPadded: {padded[860]}")