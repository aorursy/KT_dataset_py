# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# NLTK modules
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import re

from gensim.models import Word2Vec # Word2Vec module

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Loading dataset
news_data = pd.read_csv('/kaggle/input/bbc-fulltext-and-category/bbc-text.csv')

# Basic info of the dataset
print(f"Shape : {news_data.shape}, \n\nColumns: {news_data.columns}, \n\nCategories: {news_data.category.unique()}")

# print sample data
news_data.head().append(news_data.tail())
# Plot category data
plt.figure(figsize=(10,6))
sns.countplot(news_data.category)
plt.show()
class DataPreparation:
    def __init__(self, data, column='text'):
        self.df = data
        self.column = column
    
    def preprocess(self):
        self.tokenize()
        self.remove_stopwords()
        self.remove_non_words()
        self.lemmatize_words()
        
        return self.df
    
    def tokenize(self):
        self.df['clean_text'] = self.df[self.column].apply(nltk.word_tokenize)
        print("Tokenization is done.")
    
    def remove_stopwords(self):
        stopword_set = set(nltk.corpus.stopwords.words('english'))
        
        rem_stopword = lambda words: [item for item in words if item not in stopword_set]
        
        self.df['clean_text'] = self.df['clean_text'].apply(rem_stopword)
        print("Remove stopwords done.")
    
    def remove_non_words(self):
        """
            Remove all non alpha characters from the text data
            :numbers: 0-9
            :punctuation: All english punctuations
            :special characters: All english special characters
        """
        regpatrn = '[a-z]+'
        rem_special_chars = lambda x: [item for item in x if re.match(regpatrn, item)]
        self.df['clean_text'] = self.df['clean_text'].apply(rem_special_chars)
        print("Removed non english characters is done.")
        
    def lemmatize_words(self):
        lemma = nltk.stem.wordnet.WordNetLemmatizer()
        
        on_word_lemma = lambda x: [lemma.lemmatize(w, pos='v') for w in x]
        
        self.df['clean_text'] = self.df['clean_text'].apply(on_word_lemma)
        print("Lemmatization on the words.")
# Preprocessing activities on the data
data_prep = DataPreparation(news_data)

cleanse_df = data_prep.preprocess()
def vectorize(vector, X_train, X_test):
    vector_fit = vector.fit(X_train)
    
    X_train_vec = vector_fit.transform(X_train)
    X_test_vec = vector_fit.transform(X_test)
    
    print("Vectorization is completed.")
    return X_train_vec, X_test_vec

def label_encoding(y_train):
    """
        Encode the given list of class labels
        :y_train_enc: returns list of encoded classes
        :labels: actual class labels
    """
    lbl_enc = LabelEncoder()
    
    y_train_enc = lbl_enc.fit_transform(y_train)
    labels = lbl_enc.classes_
    
    return y_train_enc, labels
# Encode the class labels
y_enc_train, labels = label_encoding(news_data['category'])

# Split from the loaded dataset
X_train, X_valid, y_train, y_test = train_test_split(news_data['text'], y_enc_train, test_size=0.2, shuffle=True)

# Bag of words (BOW) matrix
bow_vector = CountVectorizer(ngram_range=(1, 1), analyzer='word', max_features=5000, max_df=2, min_df=1)
bow_vector.fit(X_train) 


pipe = Pipeline([('bow', bow_vector),
                ('tfidf', TfidfTransformer())]).fit(X_train)
train_tfidf = pipe.transform(X_train)
valid_tfidf = pipe.transform(X_valid)
print(train_tfidf.shape, valid_tfidf.shape)
def lsa_reduction(X_train, X_test, n_comp=120):
    svd = TruncatedSVD(n_components=n_comp)
    normalizer = Normalizer()
    
    lsa_pipe = Pipeline([('svd', svd),
                        ('normalize', normalizer)]).fit(X_train)
    
    train_reduced = lsa_pipe.transform(X_train)
    test_reduced = lsa_pipe.transform(X_test)
    return train_reduced, test_reduced
    
def lsa_nmf_reduction(X_train, X_test, n_comp=120):
    nmf = NMF(n_components=n_comp)
    normalizer = Normalizer()
    
    lsa_pipe = Pipeline([('nmf', nmf),
                        ('normalize', normalizer)]).fit(X_train)
    
    train_reduced = lsa_pipe.transform(X_train)
    test_reduced = lsa_pipe.transform(X_test)
    return train_reduced, test_reduced
xtrain_svd, xtest_svd = lsa_reduction(train_tfidf, valid_tfidf, 2000)

# NMF dimensionality function is called only for Multinomial Naive Bayes
# xtrain_svd, xtest_svd = lsa_nmf_reduction(train_tfidf, valid_tfidf)
sgd = SGDClassifier(random_state=0,loss='log',alpha=0.01,penalty='elasticnet')
lr = LogisticRegression(C=1.0)
svc = SVC(kernel='linear')
nb = MultinomialNB()

# One vs Restclassifier
orc_clf = OneVsRestClassifier(estimator=svc).fit(xtrain_svd, y_train)
print(orc_clf.get_params)
print(orc_clf.intercept_)
# Predict the test data
y_pred = orc_clf.predict(xtest_svd)
print("Accuracy: %1.3f \tPrecision: %1.3f \tRecall: %1.3f \tF1-Score: %1.3f\n" % (accuracy_score(y_test, y_pred),
                                                                                     precision_score(y_test, y_pred, average='micro'),
                                                                                     recall_score(y_test, y_pred, average='micro'),
                                                                                     f1_score(y_test, y_pred, average='micro')))
