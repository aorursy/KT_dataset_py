# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
import nltk

from nltk.tokenize.toktok import ToktokTokenizer

import re



# nlp = spacy.load('en_core', parse=True, tag=True, entity=True)

tokenizer = ToktokTokenizer()

stopword_list = nltk.corpus.stopwords.words('english')

stopword_list.remove('no')

stopword_list.remove('not')

import unicodedata
remove_accented_chars(train['comment_text'][0])
def remove_special_chars(text, remove_digits=False):

    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'

    text = re.sub(pattern, '', text)

    return text



remove_special_chars(train['comment_text'][0], remove_digits=True)
def simple_stemmer(text):

	ps = nltk.porter.PorterStemmer()

	text = ' '.join([ps.stem(word) for word in text.split()])

	return text

simple_stemmer(train['comment_text'][0])

from nltk.stem import WordNetLemmatizer



def lemmatize_text(text):

    wordnet_lemmatizer = WordNetLemmatizer()

    text = ' '.join(wordnet_lemmatizer.lemmatize(elem) for elem in text.split(' '))    

    return text
lemmatize_text(train['comment_text'][0])

def remove_stopwords(text, is_lower_case=False):

    tokens = tokenizer.tokenize(text)

    tokens = [token.strip() for token in tokens]

    if is_lower_case:

        filtered_tokens = [token for token in tokens if token not in stopword_list]

    else:

        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]

    filtered_text = ' '.join(filtered_tokens)    

    return filtered_text
remove_stopwords(train['comment_text'][0])
def normalize_corpus(corpus, html_stripping=False, contraction_expansion=True,

                     accented_char_removal=False, text_lower_case=True, 

                     text_lemmatization=False, special_char_removal=True, 

                     stopword_removal=True, remove_digits=True):

    

    normalized_corpus = []

    # normalize each document in the corpus

    for doc in corpus:

        # strip HTML

        if html_stripping:

            doc = strip_html_tags(doc)

        # remove accented characters

        if accented_char_removal:

            doc = remove_accented_chars(doc)

        # expand contractions    

        if contraction_expansion:

            doc = expand_contractions(doc)

        # lowercase the text    

        if text_lower_case:

            doc = doc.lower()

        # remove extra newlines

        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)

        # lemmatize text

        if text_lemmatization:

            doc = lemmatize_text(doc)

        # remove special characters and\or digits    

        if special_char_removal:

            # insert spaces between special characters to isolate them    

            special_char_pattern = re.compile(r'([{.(-)!}])')

            doc = special_char_pattern.sub(" \\1 ", doc)

            doc = remove_special_chars(doc, remove_digits=remove_digits)  

        # remove extra whitespace

        doc = re.sub(' +', ' ', doc)

        # remove stopwords

        if stopword_removal:

            doc = remove_stopwords(doc, is_lower_case=text_lower_case)

            

        normalized_corpus.append(doc)

        

    return normalized_corpus
CONTRACTION_MAP = {

"ain't": "is not",

"aren't": "are not",

"can't": "cannot",

"can't've": "cannot have",

"'cause": "because",

"could've": "could have",

"couldn't": "could not",

"couldn't've": "could not have",

"didn't": "did not",

"doesn't": "does not",

"don't": "do not",

"hadn't": "had not",

"hadn't've": "had not have",

"hasn't": "has not",

"haven't": "have not",

"he'd": "he would",

"he'd've": "he would have",

"he'll": "he will",

"he'll've": "he he will have",

"he's": "he is",

"how'd": "how did",

"how'd'y": "how do you",

"how'll": "how will",

"how's": "how is",

"I'd": "I would",

"I'd've": "I would have",

"I'll": "I will",

"I'll've": "I will have",

"I'm": "I am",

"I've": "I have",

"i'd": "i would",

"i'd've": "i would have",

"i'll": "i will",

"i'll've": "i will have",

"i'm": "i am",

"i've": "i have",

"isn't": "is not",

"it'd": "it would",

"it'd've": "it would have",

"it'll": "it will",

"it'll've": "it will have",

"it's": "it is",

"let's": "let us",

"ma'am": "madam",

"mayn't": "may not",

"might've": "might have",

"mightn't": "might not",

"mightn't've": "might not have",

"must've": "must have",

"mustn't": "must not",

"mustn't've": "must not have",

"needn't": "need not",

"needn't've": "need not have",

"o'clock": "of the clock",

"oughtn't": "ought not",

"oughtn't've": "ought not have",

"shan't": "shall not",

"sha'n't": "shall not",

"shan't've": "shall not have",

"she'd": "she would",

"she'd've": "she would have",

"she'll": "she will",

"she'll've": "she will have",

"she's": "she is",

"should've": "should have",

"shouldn't": "should not",

"shouldn't've": "should not have",

"so've": "so have",

"so's": "so as",

"that'd": "that would",

"that'd've": "that would have",

"that's": "that is",

"there'd": "there would",

"there'd've": "there would have",

"there's": "there is",

"they'd": "they would",

"they'd've": "they would have",

"they'll": "they will",

"they'll've": "they will have",

"they're": "they are",

"they've": "they have",

"to've": "to have",

"wasn't": "was not",

"we'd": "we would",

"we'd've": "we would have",

"we'll": "we will",

"we'll've": "we will have",

"we're": "we are",

"we've": "we have",

"weren't": "were not",

"what'll": "what will",

"what'll've": "what will have",

"what're": "what are",

"what's": "what is",

"what've": "what have",

"when's": "when is",

"when've": "when have",

"where'd": "where did",

"where's": "where is",

"where've": "where have",

"who'll": "who will",

"who'll've": "who will have",

"who's": "who is",

"who've": "who have",

"why's": "why is",

"why've": "why have",

"will've": "will have",

"won't": "will not",

"won't've": "will not have",

"would've": "would have",

"wouldn't": "would not",

"wouldn't've": "would not have",

"y'all": "you all",

"y'all'd": "you all would",

"y'all'd've": "you all would have",

"y'all're": "you all are",

"y'all've": "you all have",

"you'd": "you would",

"you'd've": "you would have",

"you'll": "you will",

"you'll've": "you will have",

"you're": "you are",

"you've": "you have"

}
def expand_contractions(text):

    return ' '.join([CONTRACTION_MAP[elem] if elem.lower() in CONTRACTION_MAP else elem for elem in text.lower().split(" ")])
# train_sample = train[0:20]

# new_train = train_sample['comment_text'].apply(lambda x: normalize_corpus(x.split(" ")))



# sample_train_df = pd.DataFrame(new_train)

# sample_train_df['target'] = train['target'][0:20]
# entire train data, processing faster

# vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word', stop_words='english')

# tfidf_matrix = vectorizer.fit_transform(train['comment_text'])

# feature_names = vectorizer.get_feature_names()

# matrix = vectorizer.fit_transform(train['comment_text']).todense()

# train_tfidf = pd.DataFrame(matrix, columns=vectorizer.get_feature_names())
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
# tfidf_vec = TfidfVectorizer(stop_words='english')

# # tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3))

# tfidf_vec.fit_transform(train['comment_text'].values.tolist() + test['comment_text'].values.tolist())

# train_tfidf = tfidf_vec.transform(train['comment_text'].values.tolist())

# test_tfidf = tfidf_vec.transform(test['comment_text'].values.tolist())
# from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import SGDClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2
y = train.target

# tfidf_train = tfidf_vec.fit_transform(train['comment_text'])



# X_train, X_test, y_train, y_test = train_test_split(train['comment_text'], y, test_size=0.33,random_state=53)
all_text = pd.concat([train['comment_text'], test['comment_text']])



word_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='word',

    token_pattern=r'\w{1,}',

    ngram_range=(1, 2),

    max_features=50000)

word_vectorizer.fit(all_text)

print('Word TFIDF 1/3')

train_word_features = word_vectorizer.transform(train['comment_text'])

print('Word TFIDF 2/3')

test_word_features = word_vectorizer.transform(test['comment_text'])

print('Word TFIDF 3/3')
from sklearn.feature_selection import SelectFromModel



from sklearn.linear_model import LogisticRegression

import lightgbm as lgb



train['new_target'] = np.where(train['target'] > 0, 1, 0)



train_target = train['new_target']

model = LogisticRegression(solver='sag')

sfm = SelectFromModel(model, threshold=0.2)

print(train_word_features.shape)

train_sparse_matrix = sfm.fit_transform(train_word_features, train_target)

print(train_sparse_matrix.shape)

train_sparse_matrix, valid_sparse_matrix, y_train, y_valid = train_test_split(train_sparse_matrix, train_target, test_size=0.05, random_state=144)





test_sparse_matrix = sfm.transform(test_word_features)



d_train = lgb.Dataset(train_sparse_matrix, label=y_train)

d_valid = lgb.Dataset(valid_sparse_matrix, label=y_valid)

watchlist = [d_train, d_valid]



params = {'learning_rate': 0.2,

              'application': 'binary',

              'num_leaves': 31,

              'verbosity': -1,

              'metric': 'auc',

              'data_random_seed': 2,

              'bagging_fraction': 0.8,

              'feature_fraction': 0.6,

              'nthread': 4,

              'lambda_l1': 1,

              'lambda_l2': 1}

model = lgb.train(params,

                  train_set=d_train,

                  num_boost_round=140,

                  valid_sets=watchlist,

                  verbose_eval=10)

submission = pd.DataFrame.from_dict({'id': test['id']})

submission['prediction'] = model.predict(test_sparse_matrix)
submission.to_csv('submission.csv', index=False)
