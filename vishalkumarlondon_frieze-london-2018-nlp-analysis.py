import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 300) # specifies number of rows to show
pd.options.display.float_format = '{:40,.2f}'.format # specifies default number format to 4 decimal places
pd.options.display.max_colwidth
pd.options.display.max_colwidth = 1000
# This line tells the notebook to show plots inside of the notebook
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sb
import string
import nltk

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sklearn
# # Install pip package in the current Jupyter kernel
# import sys
# !{sys.executable} -m pip install --upgrade pip
# # Install Gensim Library pip package in the current Jupyter kernel
# import sys
# !{sys.executable} -m pip install --upgrade gensim
# #load in twitter data csv file

twittertext = pd.read_csv('../input/frieze-nlp.csv')
twittertext.head(2)
#import nltk and download all stopwords and punctuations
nltk.download('stopwords')
nltk.download('punkt')

import re
#remove 'http' links and remove all '\n' line-brakes

twittertext.Text = twittertext.Text.str.replace(r'http\S+','')
twittertext.Text = twittertext.Text.str.replace(r'\n','')
#remove 'non-ascii' characters

twittertext.Text = twittertext.Text.str.replace(r'[^\x00-\x7F]','')
#make all the text lowercase

twittertext.Text = twittertext.Text.str.lower()
twittertext.Text = twittertext.Text.astype(str)
twittertext.Text = twittertext.Text.str.replace(r'#',' #')
twittertext.head(2)
#create a column for all #hasthtags that appears in the text
twittertext['hashtags'] = twittertext.Text.str.findall(r'#\S+')
#clean the text by removing all punctuation

# https://stackoverflow.com/questions/23175809/str-translate-gives-typeerror-translate-takes-one-argument-2-given-worked-i

twittertext['cleanText'] = twittertext.Text.apply(lambda x: x.translate(
                                                str.maketrans('','', string.punctuation)))

# twittertext.head(2)
twittertext.cleanText = twittertext.cleanText.str.replace(r'art fair','art-fair')
twittertext.cleanText = twittertext.cleanText.str.replace(r'art work','artwork')
twittertext.cleanText = twittertext.cleanText.str.replace(r'frieze week','frieze-week')
#tokenise all of the clean text

twittertext['tokenText'] = twittertext.cleanText.apply(word_tokenize)
twittertext.head(2)
#count the number of words in the post

twittertext['tokenCount'] = twittertext.tokenText.apply(len)
# twittertext.head(2)
# Import stopwords with nltk.
stop = stopwords.words('english')

# Exclude stopwords with Python's list comprehension and pandas.DataFrame.apply.
twittertext['noStopWords'] = twittertext['cleanText'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#tokenise all of the text with Stop words
twittertext['tokenTextnoStop'] = twittertext.noStopWords.apply(word_tokenize)
twittertext.head(2)
#remove 'http' links and remove all '\n' line-brakes

twittertext['Text_nohashtags'] = twittertext.Text.str.replace('#\S+', '')
#clean the text by removing all punctuation
# https://stackoverflow.com/questions/23175809/str-translate-gives-typeerror-translate-takes-one-argument-2-given-worked-i
twittertext['Text_nohashtags_cleanText'] = twittertext.Text_nohashtags.apply(lambda x: x.translate(
                                                str.maketrans('','', string.punctuation)))

#tokenise all of the clean text
twittertext['Text_nohashtags_tokenText'] = twittertext.Text_nohashtags_cleanText.apply(word_tokenize)

# Import stopwords with nltk.
stop = stopwords.words('english')

# Exclude stopwords with Python's list comprehension and pandas.DataFrame.apply.
twittertext['Text_nohashtags_noStopWords'] = twittertext['Text_nohashtags_cleanText'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#tokenise all of the text with Stop words
twittertext['Text_nohashtags_tokenTextnoStop'] = twittertext.Text_nohashtags_noStopWords.apply(word_tokenize)
twittertext_nohashtag = twittertext.sort_values(by='Text_nohashtags_cleanText')
twittertext_nohashtag = twittertext_nohashtag.loc[800:, :]
twittertext_nohashtag.head(2)
#google's langdetect: https://stackoverflow.com/questions/43377265/determine-if-text-is-in-english

#import Google's langdetect library to check where tweets are english or not
from langdetect import detect
lang = []

try:
    for index, row in twittertext['cleanText'].iteritems():
        lang = detect(row)
        twittertext.loc[index, 'Languagereveiw'] = lang

except Exception:
    pass
twittertext.Languagereveiw.value_counts()

twittertext = twittertext.loc[twittertext['Languagereveiw'] == 'en']
twittertext.shape
twittertext_nodup = twittertext.drop_duplicates(subset='cleanText')
twittertext_nodup.shape
twittertext_nodup.head(2)
# Load in the sentiment analysis csv file

sentiment = pd.read_csv('../input/frieze-sentiment-nodup.csv')
sentiment.head(1)
sentiment.DateTime.dtypes
sentiment.DateTime = pd.to_datetime(sentiment.DateTime)
sentiment.DateTime.dtypes
sentiment.shape
score = sentiment.score
date_time = sentiment.DateTime

DF = pd.DataFrame()
DF['score'] = score
DF = DF.set_index(date_time)

fig, ax = plt.subplots()
plt.plot(DF)
x = pd.DataFrame(sentiment.groupby('Day')['score'].mean())
x.columns = ['mean']
x['median'] = sentiment.groupby('Day')['score'].median()
x = x.reset_index()
x.head()
people = pd.DataFrame(sentiment.groupby('UserHandle')['score'].mean())
people.columns = ['mean']
people.head()
people['median'] = sentiment.groupby('UserHandle')['score'].median()
people.head()
people['count'] = sentiment.groupby('UserHandle')['score'].count()
people.head()
people = people.reset_index()
people.dtypes
people = people[people['count'] > 4]
people.sort_values(by = ['mean'],
                   ascending = False).head(10)
twittertext_nodup.head(2)
#flatten list of lists of no stop words (TextnoStop) into one large list

tokens = []

for sublist in twittertext_nodup.tokenTextnoStop:
    for word in sublist:
        tokens.append(word)
tokens_df = pd.DataFrame(tokens)
tokens_df.columns = ['words']
tokens_df['freq'] = tokens_df.groupby('words')['words'].transform('count')

tokens_df.shape
tokens_df.head()
tokens_df.words.value_counts()[:20]
word_count = pd.DataFrame(tokens_df.words.value_counts()[:20])
word_count.reset_index(inplace=True)
word_count.columns = ['word', 'count_words']
twittertext_nodup.head(1)
hashtags = []

for sublist in twittertext_nodup.hashtags:
    for word in sublist:
        hashtags.append(word)
hashtags_df = pd.DataFrame(hashtags)
hashtags_df.columns = ['words']
hashtags_df['freq'] = hashtags_df.groupby('words')['words'].transform('count')

hashtags_df.shape
hashtags_df.words.value_counts()[:20]
hash_count = pd.DataFrame(hashtags_df.words.value_counts()[:20])
hash_count.reset_index(inplace=True)
hash_count.columns = ['hashtag', 'count_hashtag']
#remove all hashtags
hashtags_df['words'] = hashtags_df.words.str.replace(r'#', '')
#create a df by merging the hashtag_df with the token_df ie. words
words_minus_hashtags = pd.merge(hashtags_df, tokens_df, on='words', how='left')
#take the hashtag count away from the word count
words_minus_hashtags['count_minushashtag'] = words_minus_hashtags.freq_y - words_minus_hashtags.freq_x
words_minus_hashtags.sort_values(by='count_minushashtag', ascending=False, inplace=True)
#some cleaning steps
words_minus_hashtags = words_minus_hashtags.loc[:,['words', 'count_minushashtag']]
words_minus_hashtags = words_minus_hashtags.drop_duplicates()
words_minus_hashtags = words_minus_hashtags.dropna()
#create a new dataframe the top 20 word counts minus the hasthtag count
withouthash_count = pd.DataFrame(words_minus_hashtags.head(20))
withouthash_count.reset_index(inplace=True)
#drop old index column
withouthash_count = withouthash_count.iloc[:, 1:3]
#merge all three dataframes together

count = pd.concat([word_count, hash_count, withouthash_count], axis=1)
count
from nltk.stem import WordNetLemmatizer
from collections import Counter
#import word lemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
#group the text by day
day_corpus = twittertext_nodup.groupby('Day').apply(lambda x: x['noStopWords'].str.cat())

#tokenize the text for that day using word lemmatizer and then count
day_corpus = day_corpus.apply(lambda x: Counter([
    wordnet_lemmatizer.lemmatize(w) 
    for w in word_tokenize(x) 
    if w.lower() not in stop and not w.isdigit()]))
#count the five most frequent words per day
day_freq = day_corpus.apply(lambda x: x.most_common(5))

# create a dataframe of the five most frequent words per day
day_freq = pd.DataFrame.from_records(
    day_freq.values.tolist()).set_index(day_freq.index)
day_freq
def normalize_row(x):
    label, repetition = zip(*x)
    t = sum(repetition)
    r = [n/t for n in repetition]
    return list(zip(label,r))

day_freq = day_freq.apply(lambda x: normalize_row(x), axis=1)
twittertext_nodup.head(1)
twittertext_nohashtag.head(2)
# create a corpus of all the text which does not have stop words in it
# this corupus will be a list of lists

corpus_with_hashtags = list(twittertext_nodup['tokenTextnoStop'])

corpus_without_hashtags = list(twittertext_nohashtag['Text_nohashtags_tokenTextnoStop'])
print(len(corpus_with_hashtags))
print(len(corpus_without_hashtags))
from gensim.models import word2vec

model1 = word2vec.Word2Vec(corpus_with_hashtags, 
                          size=100, 
                          window=10, 
                          min_count=30, 
                          workers=10)

model1.wv['frieze']
model2 = word2vec.Word2Vec(corpus_without_hashtags, 
                          size=100, 
                          window=10, 
                          min_count=30, 
                          workers=10)

model2.wv['frieze']
from sklearn.decomposition import PCA
#model1
vocab1 = list(model1.wv.vocab)
X1 = model1[model1.wv.vocab]

pca1 = PCA(n_components = 2)
result1 = pca1.fit_transform(X1)
#model2
vocab2 = list(model2.wv.vocab)
X2 = model2[model2.wv.vocab]

pca2 = PCA(n_components = 2)
result2 = pca2.fit_transform(X2)
import matplotlib.pyplot as pyplot

pyplot.scatter(result1[:, 0], result1[:, 1])
pyplot.scatter(result2[:, 0], result2[:, 1])
#model_1 with hashtags
wrds1 = list(model1.wv.vocab)
len(wrds1)
#model_2 without hashtags
wrds2 = list(model2.wv.vocab)
len(wrds2)
#nearest neighbours for model_1 with hashtags
model1.wv.most_similar('frieze')
#nearest neighbours for model_2 without hashtags
model2.wv.most_similar('frieze')
#model_1 with hashtags

#zip the two lists containing vectors and words
zipped1 = zip(model1.wv.index2word, model1.wv.vectors)

#the resulting list contains `(word, wordvector)` tuples. We can extract the entry for any `word` or `vector` (replace with the word/vector you're looking for) using a list comprehension:
wordresult1 = [i for i in zipped1 if i[0] == word]
vecresult1 = [i for i in zipped1 if i[1] == vector]
#model_2 without hashtags

#zip the two lists containing vectors and words
zipped2 = zip(model2.wv.index2word, model2.wv.vectors)

#the resulting list contains `(word, wordvector)` tuples. We can extract the entry for any `word` or `vector` (replace with the word/vector you're looking for) using a list comprehension:
wordresult2 = [i for i in zipped2 if i[0] == word]
vecresult2 = [i for i in zipped2 if i[1] == vector]
#dataframe of similar words for model_1 with hashtags

# https://stackoverflow.com/questions/43776572/visualise-word2vec-generated-from-gensim

word_frame1 = pd.DataFrame(result1, index=vocab1, columns=['x', 'y'])
word_frame1.reset_index(inplace=True)
word_frame1.columns = ['words', 'x', 'y']
#dataframe of similar words for model_2 without hashtags

# https://stackoverflow.com/questions/43776572/visualise-word2vec-generated-from-gensim

word_frame2 = pd.DataFrame(result2, index=vocab2, columns=['x', 'y'])
word_frame2.reset_index(inplace=True)
word_frame2.columns = ['words', 'x', 'y']
#nearest neighbours dataframe for model_1 with hashtags

nearest1 = word_frame1[word_frame1.words.isin(['frieze','friezeartfair','regentspark', 'friezeweek', 'friezelondon2018', 
                                         'friezemasters', 'london', 'davidshrigley', 'friezelondon',
                                         'stephenfriedmangallery', 'sculpture'])]
nearest1.columns = ['words1', 'x1', 'y1']
nearest1 = nearest1.reset_index()
#nearest neighbours dataframe for model_2 without hashtags

nearest2 = word_frame2[word_frame2.words.isin(['frieze','thanks','friezemasters', 'london', 'fair', 
                                         'masters', 'art', 'fairs', 'highlights',
                                         'best', 'street','instagram'])]

nearest2.columns = ['words2', 'x2', 'y2']
nearest2 = nearest2.reset_index()
nearest = pd.concat([nearest1, nearest2], axis=1)
nearest