import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from os import listdir

from re import finditer

from functools import partial

from itertools import *



%matplotlib inline
listdir('/kaggle/input/dataset-preprocessing/')
data = pd.read_csv('/kaggle/input/dataset-preprocessing/preprocessed-data.csv')
data.info()
plt.figure(figsize=(13, 6))

plt.title('Number of movies by genre')

data.drop(columns=['plot']).sum(axis=0).sort_values(ascending=False).plot.bar();
data.sum(axis=1).value_counts().plot.pie(explode=[0, 0.1], shadow=True, autopct='%1.0f%%')

plt.title('Number of genres on each movie by mean')

plt.ylabel(None);
def get_word_frequencies(x=None, min_length=0):

    if x is None:

        x = data

    frequencies = {}



    for result in chain.from_iterable(map(partial(finditer, '\w+'), data['plot'])):

        word = result.group(0)

        frequencies[word] = frequencies.get(word, 0)+1

    frequencies = pd.Series(frequencies)

    frequencies = frequencies[np.array(list(map(lambda word: len(word) >= min_length, frequencies.index)))]

    return frequencies
frequencies = get_word_frequencies()

len(frequencies)
frequencies = get_word_frequencies(min_length=6)

frequencies.sort_values(ascending=False).head()
plt.figure(figsize=(13, 6))

frequencies.sort_values(ascending=False).iloc[0:20].plot.bar()