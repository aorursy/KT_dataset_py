import numpy as np

import pandas as pd



import spacy 

nlp = spacy.load('en_core_web_sm')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
FILE_PATH = '/kaggle/input/customer-support-on-twitter/twcs/twcs.csv'

df = pd.read_csv(FILE_PATH, nrows=500)

df.head()
len(df)
# build a vocbulary - vocab is the dictionary of all the words 

# then embed that vocabulary
def build_vocab(sentences):

    vocab = {}

    for sentence in sentences:

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    

    return vocab





vocab = build_vocab(df.loc[:1000, 'text'].apply(lambda x: x.split()).values)
list(vocab)[:5], len(vocab)
# simple embeddings

%time

count = 0

embedding = nlp

k = {}

for text in df.text:

    doc = embedding(text)

    k[text] = doc

    print(doc)

    count += 1

    if count == 100:

        break
len(k), len(k)
count = 0

for word in vocab:

    print(word)

    count += 1

    if count == 4: break
# function to check coverage of the words being embedded



def check_coverage(vocab, embedding):

    # oov = out of vocabulary

    a = {}

    oov = {}

    embedded_words = 0     # number of embedded words

    oov_words = 0          # number of oov words

    count = 0

    for word in vocab:

        try:

            #a[word] = embedding(word)

            #embedded_words += vocab[word]

            embedding(word)

            count += 1

        except:

            oov[word] = vocab[word]

            #oov_words += vocab[word]

            pass

        

    emb = count / len(vocab)

    print(f"Found embeddings for {emb:.4f}")

    #print(f"Founds embeddings for {(k / (k + i))} text")
%%time

check_coverage(vocab, nlp)
# remove wierd spaces

# tokenization

# spelling correction

# contraction mapping

# stemming

# emoji handling

# cleaning html

# removing links