# Import libraries to build Word2Vec model, and load Newsgroups data

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import sys

import re

from gensim.models import Word2Vec

from gensim.models.phrases import Phraser, Phrases

TEXT_DATA_DIR = '../input/20_newsgroups/20_newsgroups'

print(os.listdir("../input/20_newsgroups/20_newsgroups"))

# Newsgroups data is split between many files and folders.

# Directory stucture 20_newsgroup/<newsgroup label>/<post ID>



texts = []         # list of text samples

labels_index = {}  # dictionary mapping label name to numeric id

labels = []        # list of label ids

label_text = []    # list of label texts
# Go through each directory

for name in sorted(os.listdir(TEXT_DATA_DIR)):

    path = os.path.join(TEXT_DATA_DIR, name)

    if os.path.isdir(path):

        label_id = len(labels_index)

        labels_index[name] = label_id

        for fname in sorted(os.listdir(path)):

            # News groups posts are named as numbers, with no extensions.

            if fname.isdigit():

                fpath = os.path.join(path, fname)

                f = open(fpath, encoding='latin-1')

                t = f.read()

                i = t.find('\n\n')  # skip header in file (starts with two newlines.)

                if 0 < i:

                    t = t[i:]

                texts.append(t)

                f.close()

                labels.append(label_id)

                label_text.append(name)



print('Found %s texts.' % len(texts))
# Cleaning data - remove punctuation from every newsgroup text

sentences = []

# Go through each text in turn

for ii in range(len(texts)):

    sentences = [re.sub(pattern=r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]', 

                        repl='', 

                        string=x

                       ).strip().split(' ') for x in texts[ii].split('\n') 

                      if not x.endswith('writes:')]

    sentences = [x for x in sentences if x != ['']]

    texts[ii] = sentences
print(texts[6])
# concatenate all sentences from all texts into a single list of sentences

all_sentences = []

for text in texts:

    all_sentences += text
# Phrase Detection

# Give some common terms that can be ignored in phrase detection

# For example, 'state_of_affairs' will be detected because 'of' is provided here: 

common_terms = ["of", "with", "without", "and", "or", "the", "a"]

# Create the relevant phrases from the list of sentences:

phrases = Phrases(all_sentences, common_terms=common_terms)

# The Phraser object is used from now on to transform sentences

bigram = Phraser(phrases)
len(all_sentences)
print(bigram[all_sentences[5676]])
# Applying the Phraser to transform our sentences is simply

all_sentences = list(bigram[all_sentences])
model = Word2Vec(all_sentences, 

                 min_count=3,   # Ignore words that appear less than this

                 size=200,      # Dimensionality of word embeddings

                 workers=2,     # Number of processors (parallelisation)

                 window=5,      # Context window for words during training

                 iter=30)       # Number of epochs training over corpus
model.vector_size
len(model.wv.vocab)
model.most_similar('New_York')