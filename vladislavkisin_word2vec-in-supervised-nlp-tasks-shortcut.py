import pandas as pd

import nltk

import seaborn as sns

import re

import numpy as np

import warnings

warnings.filterwarnings("ignore")



df = pd.read_csv('../input/bag-of-words-meets-bags-of-popcorn-/labeledTrainData.tsv', sep='\t')

df.head()
%%time

# Here we get transform the documents into sentences for the word2vecmodel

# we made a function such that later on when we make the submission, we don't need to write duplicate code

def preprocess(df):

    df['review'] = df.review.str.lower()

    df['document_sentences'] = df.review.str.split('.') 

    df['tokenized_sentences'] = list(map(lambda sentences: list(map(nltk.word_tokenize, sentences)), df.document_sentences))  

    df['tokenized_sentences'] = list(map(lambda sentences: list(filter(lambda lst: lst, sentences)), df.tokenized_sentences))



preprocess(df)
from sklearn.model_selection import train_test_split

train, test, y_train, y_test = train_test_split(df.drop(columns='sentiment'), df['sentiment'], test_size=.2)
#Collecting a vocabulary

voc = []

for sentence in train.tokenized_sentences:

    voc.extend(sentence)



print("Number of sentences: {}.".format(len(voc)))

print("Number of rows: {}.".format(len(train)))
%%time

from gensim.models import word2vec, Word2Vec



num_features = 300    

min_word_count = 3    

num_workers = 4       

context = 8           

downsampling = 1e-3   



# Initialize and train the model

W2Vmodel = Word2Vec(sentences=voc, sg=1, hs=0, workers=num_workers, size=num_features, min_count=min_word_count, window=context,

                    sample=downsampling, negative=5, iter=6)
%%time

def sentence_vectors(model, sentence):

    #Collecting all words in the text

    words=np.concatenate(sentence)

    #Collecting words that are known to the model

    model_voc = set(model.wv.vocab.keys()) 

    

    sent_vector = np.zeros(model.vector_size, dtype="float32")

    

    # Use a counter variable for number of words in a text

    nwords = 0

    # Sum up all words vectors that are know to the model

    for word in words:

        if word in model_voc: 

            sent_vector += model[word]

            nwords += 1.



    # Now get the average

    if nwords > 0:

        sent_vector /= nwords

    return sent_vector



train['sentence_vectors'] = list(map(lambda sen_group:

                                      sentence_vectors(W2Vmodel, sen_group),

                                      train.tokenized_sentences))
def vectors_to_feats(df, ndim):

    index=[]

    for i in range(ndim):

        df[f'w2v_{i}'] = df['sentence_vectors'].apply(lambda x: x[i])

        index.append(f'w2v_{i}')

    return df[index]

X_train = vectors_to_feats(train, 300)

X_train.head()
%%time

test['sentence_vectors'] = list(map(lambda sen_group:sentence_vectors(W2Vmodel, sen_group), test.tokenized_sentences))

X_test=vectors_to_feats(test, 300)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train, y_train)
from sklearn.metrics import roc_auc_score, confusion_matrix

roc_auc_score(y_test,lr.predict_proba(X_test)[:,1])

import seaborn as sns

import matplotlib.pyplot as plt

df_cm = pd.DataFrame(confusion_matrix(y_test,lr.predict(X_test)), index = ['predicted positive', 'predicted negative'],

                  columns = ['actual positive', 'actual negative'])

plt.figure(figsize = (10,7))

sns.heatmap(df_cm, annot=True)

plt.show()
voc_df = []

for sentence_group in df.tokenized_sentences:

    voc_df.extend(sentence_group)



print("Number of sentences: {}.".format(len(voc_df)))

print("Number of texts: {}.".format(len(df)))
%%time

from gensim.models import word2vec, Word2Vec



num_features = 300    

min_word_count = 3    

num_workers = 4       

context = 8           

downsampling = 1e-3   



# Initialize and train the model

W2Vmodel = Word2Vec(sentences=voc_df, sg=1, hs=0, workers=num_workers, size=num_features, min_count=min_word_count, window=context,

                    sample=downsampling, negative=5, iter=6)
%%time

df['sentence_vectors'] = list(map(lambda sen_group: sentence_vectors(W2Vmodel, sen_group), df.tokenized_sentences))

df = vectors_to_feats(df, 300)

y = pd.read_csv('../input/bag-of-words-meets-bags-of-popcorn-/labeledTrainData.tsv', sep='\t')['sentiment'].values
from sklearn.model_selection import ShuffleSplit, cross_val_score

cv = ShuffleSplit(n_splits=5, random_state=1)



cv_score = cross_val_score(lr, df, y ,cv=cv, scoring='roc_auc')

print(cv_score, cv_score.mean())