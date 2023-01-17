# Some useful imports



from wordcloud import WordCloud, STOPWORDS

import matplotlib.pylab as plt

import pandas as pd

import numpy as np

from PIL import Image

import requests

from io import BytesIO

%matplotlib inline
# Loading train and test datasets

train_df = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")

test_df = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
# Checking some of the stopwords

count = 0

for sw in STOPWORDS:

    print(sw)

    count += 1

    if count == 10:

        break
# Here is an "appropriate" mask.

url = "https://static01.nyt.com/images/2014/08/10/magazine/10wmt/10wmt-articleLarge-v4.jpg"

response = requests.get(url)

img = Image.open(BytesIO(response.content))



mask = np.array(img)

img
# And finally, generating the train wordcloud

text = " ".join(train_df["text"].dropna().str.lower().values)

stopwords = set(STOPWORDS)



wc = WordCloud(max_words=3000, mask=mask, stopwords=stopwords, margin=10,

               random_state=1, contour_color='white', contour_width=1).generate(text)



fig, ax = plt.subplots(1, 1, figsize=(15, 15))



ax.imshow(wc, interpolation="bilinear")

ax.set_title("Tweeter Sentiment Extraction Train")
# Let's do the same thing but this time for the test dataset

text = " ".join(test_df["text"].dropna().str.lower().values)



wc = WordCloud(max_words=3000, mask=mask, stopwords=stopwords, margin=10,

               random_state=1, contour_color='white', contour_width=1).generate(text)



fig, ax = plt.subplots(1, 1, figsize=(15, 15))



ax.imshow(wc, interpolation="bilinear")

ax.set_title("Tweeter Sentiment Extraction Test")
# First, we need to install the library.

!pip install shifterator
# We also need to get the frequency (i.e. occurence) of each word, thus this short utility function.

from collections import Counter

from itertools import chain



def get_word_freq(s):



    return Counter(v for v in chain(*s.dropna().str.lower().str.split().values) if v not in STOPWORDS)
from shifterator import relative_shift as rs





train_freq = dict(get_word_freq(train_df["text"]))

test_freq = dict(get_word_freq(test_df["text"]))



# TODO: These doesn't look right, fix! If you have any idea in the comments, pleas share!

# TODO: How to make the sentiment dict?

train_pos =  get_word_freq(train_df.loc[lambda df: df["sentiment"] == "positive", "text"].copy())

train_neg = get_word_freq(train_df.loc[lambda df: df["sentiment"] == "negative", "text"].copy())

test_pos = get_word_freq(test_df.loc[lambda df: df["sentiment"] == "positive", "text"].copy())

test_neg = get_word_freq(test_df.loc[lambda df: df["sentiment"] == "negative", "text"].copy())

train_sentiment_score = {**train_pos, **train_neg}

test_sentiment_score = {**test_pos, **test_neg}









sentiment_shift = rs.SentimentShift(train_freq, test_freq, train_sentiment_score, test_sentiment_score)



sentiment_shift.get_shift_graph(title="Word Shift for Train (left) vs Test (right) datasets")
from shifterator import relative_shift as rs





train_pos_freq = get_word_freq(train_df.loc[lambda df: df["sentiment"] == "positive", "text"])

train_neg_freq = get_word_freq(train_df.loc[lambda df: df["sentiment"] == "negative", "text"])







sentiment_shift = rs.EntropyShift(reference=train_pos_freq,

                                  comparison=train_neg_freq)



sentiment_shift.get_shift_graph(title="Entropy Shift for Train Positive (left) vs Negative (right) Sentiments")
test_pos_freq = get_word_freq(test_df.loc[lambda df: df["sentiment"] == "positive", "text"])

test_neg_freq = get_word_freq(test_df.loc[lambda df: df["sentiment"] == "negative", "text"])







sentiment_shift = rs.EntropyShift(reference=test_pos_freq,

                                  comparison=test_neg_freq)

sentiment_shift.get_shift_graph(title="Entropy Shift for Test Positive (left) vs Negative (right) Sentiments")
from shifterator import relative_shift as rs





train_freq = get_word_freq(train_df["text"])

test_freq = get_word_freq(test_df["text"])







sentiment_shift = rs.EntropyShift(reference=train_freq,

                                  comparison=test_freq)

sentiment_shift.get_shift_graph(title="Entropy Shift for Train (left) vs Test (right) datasets")