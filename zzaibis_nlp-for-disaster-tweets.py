# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import re

import nltk

import matplotlib.pyplot as plt

pd.options.display.max_colwidth = 200

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.listdir('/kaggle/input/nlp-getting-started')
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train.head()
train.shape
test.shape
def missing_data(df):

    total = df.isnull().sum()

    percentage = round(total / df.shape[0] *100)

    missing = pd.concat([total, percentage], axis=1, keys= ['Total', 'Percent']).sort_values(by='Percent', ascending = False)

    missing = missing[missing['Total'] > 0]

    return missing
missing_data(train)
train.info()
train.keyword.value_counts().sum()
wpt = nltk.WordPunctTokenizer()

stop_words = nltk.corpus.stopwords.words('english')



def normalize_document(doc):

    # lower case and remove special characters\whitespaces

    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)

    doc = doc.lower()

    doc = doc.strip()

    # tokenize document

    tokens = wpt.tokenize(doc)

    # filter stopwords out of document

    filtered_tokens = [token for token in tokens if token not in stop_words]

    # re-create document from filtered tokens

    doc = ' '.join(filtered_tokens)

    return doc



normalize_corpus = np.vectorize(normalize_document)
norm_corpus = normalize_corpus(train['text'])
norm_corpus_test = normalize_corpus(test['text'])
# from sklearn.feature_extraction.text import CountVectorizer

# # CountVectorizer(ngram_range=(2,2))

# cv = CountVectorizer(min_df=0., max_df=1.)

# cv_matrix = cv.fit_transform(norm_corpus)

# cv_matrix = cv_matrix.toarray()

# cv_matrix
# cv_matrix_test = cv.transform(norm_corpus_test)



# cv_matrix_test = cv_matrix_test.toarray()
# # get all unique words in the corpus

# vocab = cv.get_feature_names()

# # show document feature vectors

# pd.DataFrame(cv_matrix, columns=vocab)
# from sklearn.feature_extraction.text import TfidfVectorizer



# tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)

# tv_matrix = tv.fit_transform(norm_corpus)

# tv_matrix = tv_matrix.toarray()



# vocab = tv.get_feature_names()

# pd.DataFrame(np.round(tv_matrix, 2), columns=vocab)
# tv_matrix_test = tv.transform(norm_corpus_test)

# tv_matrix_test = tv_matrix_test.toarray()



# vocab = tv.get_feature_names()

# pd.DataFrame(np.round(tv_matrix_test, 2), columns=vocab)


# from sklearn.feature_extraction.text import TfidfVectorizer



# tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)

# tv_matrix = tv.fit_transform(norm_corpus)

# tv_matrix = tv_matrix.toarray()
# clf = linear_model.RidgeClassifier()


# from sklearn.metrics.pairwise import cosine_similarity



# similarity_matrix = cosine_similarity(tv_matrix)

# similarity_df = pd.DataFrame(similarity_matrix)

# similarity_df
# scores = model_selection.cross_val_score(clf, tv_matrix, train["target"], cv=3, scoring="f1")

# scores
# clf.fit(tv_matrix, train["target"])
# sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
# sample_submission["target"] = clf.predict(tv_matrix_test)
# sample_submission.to_csv("submission4.csv", index=False)
from nltk.corpus import gutenberg

from string import punctuation



# bible = gutenberg.sents('bible-kjv.txt') 

bible = norm_corpus

# remove_terms = punctuation + '0123456789'



# norm_bible = [[word.lower() for word in sent if word not in remove_terms] for sent in bible]

# norm_bible = [' '.join(tok_sent) for tok_sent in norm_bible]

# norm_bible = filter(None, normalize_corpus(norm_bible))

norm_bible = [tok_sent for tok_sent in bible if len(tok_sent.split()) > 2]



print('Total lines:', len(bible))

print('\nSample line:', bible[10])

print('\nProcessed line:', norm_bible[10])
from keras.preprocessing import text

from keras.utils import np_utils

from keras.preprocessing import sequence



tokenizer = text.Tokenizer()

tokenizer.fit_on_texts(norm_bible)

word2id = tokenizer.word_index



# build vocabulary of unique words

word2id['PAD'] = 0

id2word = {v:k for k, v in word2id.items()}

wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in norm_bible]



vocab_size = len(word2id)

embed_size = 100

window_size = 2 # context window size



print('Vocabulary Size:', vocab_size)

print('Vocabulary Sample:', list(word2id.items())[:10])
def generate_context_word_pairs(corpus, window_size, vocab_size):

    context_length = window_size*2

    for words in corpus:

        sentence_length = len(words)

        for index, word in enumerate(words):

            context_words = []

            label_word   = []            

            start = index - window_size

            end = index + window_size + 1

            

            context_words.append([words[i] 

                                 for i in range(start, end) 

                                 if 0 <= i < sentence_length 

                                 and i != index])

            label_word.append(word)



            x = sequence.pad_sequences(context_words, maxlen=context_length)

            y = np_utils.to_categorical(label_word, vocab_size)

            yield (x, y)

            

            

# Test this out for some samples

i = 0

for x, y in generate_context_word_pairs(corpus=wids, window_size=window_size, vocab_size=vocab_size):

    if 0 not in x[0]:

        print('Context (X):', [id2word[w] for w in x[0]], '-> Target (Y):', id2word[np.argwhere(y[0])[0][0]])

    

        if i == 10:

            break

        i += 1


import keras.backend as K

from keras.models import Sequential

from keras.layers import Dense, Embedding, Lambda



# build CBOW architecture

cbow = Sequential()

cbow.add(Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=window_size*2))

cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))

cbow.add(Dense(vocab_size, activation='softmax'))

cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')



# view model summary

print(cbow.summary())



# visualize model structure

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot



SVG(model_to_dot(cbow, show_shapes=True, show_layer_names=False, 

                 rankdir='TB').create(prog='dot', format='svg'))
for epoch in range(1, 6):

    loss = 0.

    i = 0

    for x, y in generate_context_word_pairs(corpus=wids, window_size=window_size, vocab_size=vocab_size):

        i += 1

        loss += cbow.train_on_batch(x, y)

        if i % 100000 == 0:

            print('Processed {} (context, word) pairs'.format(i))



    print('Epoch:', epoch, '\tLoss:', loss)

    print()
weights = cbow.get_weights()[0]

weights = weights[1:]

print(weights.shape)



# pd.DataFrame(weights, index=list(id2word.values())[1:]).head()
train.sample(20)
from sklearn.metrics.pairwise import euclidean_distances



# compute pairwise distance matrix

distance_matrix = euclidean_distances(weights)

print(distance_matrix.shape)



# view contextually similar words

similar_words = {search_term: [id2word[idx] for idx in distance_matrix[word2id[search_term]-1].argsort()[1:6]+1] 

                   for search_term in ['disaster', 'earthquake', 'flood', 'fire', 'hurricane', 'bombing', 'crime','crash']}



similar_words