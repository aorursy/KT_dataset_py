import numpy as np
import pandas as pd
from functools import reduce

from nltk.corpus import stopwords
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
def build_collocation_applier(set_colloc):
    """
    Returns a function which applies the given set of collocations to a sentence
    Parameters:
    - set_colloc: Iterable of collocations, each of which is a tuple
    """
    def apply_collocations(sentence):
        """
        Transforms each collocation in the given sequence into a single word
        """
        res = sentence.lower()
        for b1,b2 in set_colloc:
            res = res.replace("%s %s" % (b1 ,b2), "%s%s" % (b1 ,b2))
        return res
    return apply_collocations

class TwoStepClassifier():
    """
    This classifier will divide the 5-class problem into simpler problems:
    Problem 1: Classify review between negative(0-1), neutral(2) and positive(3-4)
    Problem 2: If review is negative, classify 0 or 1. If review is positive, classify 3 or 4
    """
    def __init__(self, clf, clf_params = {}):
        self.clf = clf
        self.clf_params = clf_params
        
    def fit(self, X, Y):
        Y_1 = Y.apply(lambda x : {0:0,1:0,2:2,3:4,4:4}[x])
        self.clf_1 = self.clf(**self.clf_params).fit(X, Y_1)
        self.clf_2 = self.clf(**self.clf_params).fit(X[Y<2,:], Y[Y<2])
        self.clf_3 = self.clf(**self.clf_params).fit(X[Y>2,:], Y[Y>2])
        
    def predict(self, X):
        Y = self.clf_1.predict(X)
        Y[Y<2] = self.clf_2.predict(X[Y<2])
        Y[Y>2] = self.clf_3.predict(X[Y>2])
        return Y
        
    def score(X, Y):
        return np.mean(self.predict(X) == Y)
df_train = pd.read_csv('../input/train.tsv', sep = '\t')
df_test = pd.read_csv('../input/test.tsv', sep = '\t')
df_train_unique = df_train.groupby('SentenceId').agg({'Phrase' : lambda x : reduce(lambda a, b: a, x)})
tokenizer = CountVectorizer(stop_words = stopwords.words('english')).build_analyzer()
tokens = df_train_unique.Phrase.apply(tokenizer)
tokens = tokens.apply(lambda x : ' '.join(x)).drop_duplicates().drop(3).reset_index(drop = True).apply(lambda x : x.split())
bigram_measures = BigramAssocMeasures()
finder = BigramCollocationFinder.from_documents(tokens)
finder.apply_freq_filter(5)
colloc = finder.nbest(bigram_measures.pmi, 100)
applier = build_collocation_applier(colloc)
tokens_colloc_train = df_train.Phrase.apply(tokenizer)
tokens_colloc_train = tokens_colloc_train.apply(lambda x : applier(' '.join(x)))
tokens_colloc_test = df_test.Phrase.apply(tokenizer)
tokens_colloc_test = tokens_colloc_test.apply(lambda x : applier(' '.join(x)))
bow = TfidfVectorizer(binary = True,
                      sublinear_tf = True,
                      stop_words = stopwords.words('english'),
                      ngram_range=(1,2),
                      min_df = 10, max_df = 0.5, max_features = 5000)
X_train = bow.fit_transform(tokens_colloc_train)
X_test = bow.transform(tokens_colloc_test)
Y_train = df_train.Sentiment
clf = TwoStepClassifier(LogisticRegression)
clf.fit(X_train.toarray(), Y_train)
pred = pd.read_csv('../input/sampleSubmission.csv')
pred.Sentiment = clf.predict(X_test.toarray())
pred.to_csv('output.csv', index = False)