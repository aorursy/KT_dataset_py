
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer 
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import ngrams, FreqDist
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from sklearn import feature_extraction, linear_model, model_selection, preprocessing


train_df  = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
train_df.head()
test_df.head()
train_df.info()
train_df.target.value_counts()
train_df.isnull().sum()
test_df.isnull().sum()
first_sent_example = train_df.text[0].lower()
first_sent_example
stop_words=set(stopwords.words('english'))
tokens_first= word_tokenize(first_sent_example)
print(tokens_first)
filtered_words = [w for w in tokens_first if w not in stop_words]
filtered_words
wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in filtered_words]
lemmatized_word
print(nltk.pos_tag(lemmatized_word))
only_words = [word for word in lemmatized_word if word.isalpha()]
only_words
anthoer_sent_example = train_df.text[15].lower()
anthoer_sent_example
tokens_not_disaster= word_tokenize(anthoer_sent_example)
print(tokens_not_disaster)
filtered_words_not_disaster = [w for w in tokens_not_disaster if w not in stop_words]
filtered_words_not_disaster
lemmatized_word_not_disaster = [wordnet_lemmatizer.lemmatize(word) for word in filtered_words_not_disaster]
lemmatized_word_not_disaster
only_words_not_disaster = [word for word in lemmatized_word_not_disaster if word.isalpha()]
only_words_not_disaster
anthoer_sent_example_2 = train_df.text[32].lower()
anthoer_sent_example_2
tokens_not_disaster_2= word_tokenize(anthoer_sent_example_2)
print(tokens_not_disaster_2)
filtered_words_not_disaster_2 = [w for w in tokens_not_disaster_2 if w not in stop_words]
print(filtered_words_not_disaster_2)
lemmatized_word_not_disaster_2 = [wordnet_lemmatizer.lemmatize(word) for word in filtered_words_not_disaster_2]
print(lemmatized_word_not_disaster_2)
only_words_not_disaster_2 = [word for word in lemmatized_word_not_disaster_2 if word.isalpha()]
only_words_not_disaster_2
print(nltk.pos_tag(only_words_not_disaster_2))
vectorizer = CountVectorizer()
train_vectors = vectorizer.fit_transform(train_df.text)
test_vectors = vectorizer.transform(test_df.text)
print(train_vectors)
vectorizer_tfid = TfidfVectorizer()
vectorizer_tfid.fit(train_df.text)
print(vectorizer_tfid.vocabulary_)
print(vectorizer_tfid.idf_)
# encode document
vector = vectorizer_tfid.transform([train_df.text[1]])
# summarize encoded vector
print(vector.shape)
print(vector.toarray())
clean_tex = []
PorterS = PorterStemmer()
for i in range(train_df.text.shape[0]):
    without_stop_words = [PorterS.stem(word.lower()) for word in train_df.text.str.split()[i] if not word  in stop_words  if word.isalpha()]
    ff = ' '.join(without_stop_words)
    clean_tex.append(ff)
print(pd.DataFrame(clean_tex)[0].head(5))
vectorizer_tfid_after_cleaning = TfidfVectorizer()
vectorizer_tfid_after_cleaning.fit(clean_tex)
print(vectorizer_tfid_after_cleaning)
print(vectorizer_tfid_after_cleaning.vocabulary_)
dict_count = vectorizer_tfid_after_cleaning.vocabulary_

d_xtrain = dict(list(dict_count.items())[len(dict_count)//2:])
d_ytrain = dict(list(dict_count.items())[:len(dict_count)//2])




