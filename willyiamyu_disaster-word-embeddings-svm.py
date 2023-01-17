# import necessary packages

import pandas as pd 

import numpy as np

import re
# import train and test data

train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')

train.head()
# function to remove html tags in text

def htmlremove(text):

    return re.sub('<[^<]+?>', '', text)



train['text'] = train['text'].apply(htmlremove)

test['text'] = test['text'].apply(htmlremove)     
# remove url and @'s

def urlremove(text):

    return re.sub(r"(?:\@|https?\://)\S+", "", text)



train['text'] = train['text'].apply(urlremove)

test['text'] = test['text'].apply(urlremove)
# remove emojis

# https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python

def remove_emoji(text):

    emoji_pattern = re.compile(

        u"(\ud83d[\ude00-\ude4f])|"  # emoticons

        u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)

        u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)

        u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols

        u"(\ud83c[\udde0-\uddff])"  # flags (iOS)

        "+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



train['text'] = train['text'].apply(remove_emoji)

test['text'] = test['text'].apply(remove_emoji)
# remove contractions

# taken from: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python

def decontracted(text):

    if '’' in text:

        text = text.replace("’", "'")

    # specific

    text = re.sub(r"won\'t", "will not", text)

    text = re.sub(r"can\'t", "can not", text)



    # general

    text = re.sub(r"n\'t", " not", text)

    text = re.sub(r"\'re", " are", text)

    text = re.sub(r"\'s", " is", text)

    text = re.sub(r"\'d", " would", text)

    text = re.sub(r"\'ll", " will", text)

    text = re.sub(r"\'t", " not", text)

    text = re.sub(r"\'ve", " have", text)

    text = re.sub(r"\'m", " am", text)

    return text



train['text'] = train['text'].apply(decontracted)

test['text'] = test['text'].apply(decontracted)
# remove special characters

def characterremove(text):

    return re.sub('\W+|_', ' ', text)



train.text = train.text.apply(characterremove)

test.text = test.text.apply(characterremove)
# remove numbers in text

def remove_numbers(text):

    # define the pattern to keep

    pattern = r'[^a-zA-z.,!?/:;\"\'\s]' 

    return re.sub(pattern, '', text)



train.text = train.text.apply(remove_numbers)

# lower-case text and remove unnecessary whitespaces

train.text = train.text.str.lower()

train.text = train.text.str.strip()



test.text = test.text.apply(remove_numbers)

test.text = test.text.str.lower()

test.text = test.text.str.strip()
from sklearn.feature_extraction.text import CountVectorizer

import nltk

from nltk.corpus import stopwords



stopwords = nltk.corpus.stopwords.words('english')



def get_top_20(ngram_range, new_corpus, stopwords, min_df):

    init_vec = CountVectorizer(ngram_range=ngram_range, stop_words = stopwords, min_df=min_df)

    # get sparse matrix from vocabulary

    X = init_vec.fit_transform(new_corpus)

    count_list = X.toarray().sum(axis=0) 

    word_list = init_vec.get_feature_names()

    counts = dict(zip(word_list,count_list))

    top_20 = sorted(counts.items(), key=lambda x:x[1], reverse=True)[:20]

    return top_20, counts



top_20, counts = get_top_20((1,1), train.text, stopwords, 4)

top_20
# wordcloud image of above (with more words) 

from PIL import Image

from wordcloud import WordCloud



def generate_wc_from_counts(counts):

    wc = WordCloud(background_color="white", width=1000, height=600, contour_width=3,

               contour_color='steelblue', max_words=300, 

               relative_scaling=0.5,normalize_plurals=False, random_state=0).generate_from_frequencies(counts)

    return wc



wc = generate_wc_from_counts(counts)

wc.to_image()
# lemmatize words in text

# lemmatize with nltk before spacy seems to give higher acc?

from nltk.stem import WordNetLemmatizer



lemmer = WordNetLemmatizer()

train.text = [' '.join([lemmer.lemmatize(word) for word in text.split(' ')]) for text in train.text]



test.text = [' '.join([lemmer.lemmatize(word) for word in text.split(' ')]) for text in test.text]
# split data into X and y

X_text = train.text

y_text = train.target

X_test_text = test.text
import spacy

nlp = spacy.load("en_core_web_lg")

nlp.vocab["amp"].is_stop = True



docs = [nlp(d).vector for d in X_text]

X_text = np.vstack(docs)



docs_test = [nlp(d).vector for d in X_test_text]

X_test_text = np.vstack(docs_test)
# train test split

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_text, y_text, random_state=0)
# gridsearch svm hyperparameters 

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

from sklearn.model_selection import KFold



param_grid = {'kernel': ['rbf','linear'], 'C': [0.001, 0.01, 0.1, 1, 1.5, 2, 3, 10]}



grid = GridSearchCV(SVC(), param_grid, cv=10)

grid.fit(X_train, y_train)

print(grid.best_params_)

grid.score(X_test,y_test)
# get predictions for test data and export

y_test_text = grid.predict(X_test_text)

test['target'] = y_test_text

test_export = test.loc[:, ['id', 'target']]

test_export.to_csv('disaster_test_word2vec.csv', index=False)