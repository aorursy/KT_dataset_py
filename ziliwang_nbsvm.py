from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import re
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from scipy import sparse
from sklearn.svm import LinearSVC
import json
from sklearn.model_selection import train_test_split
import copy
with open('../input/train.json') as f:
    raw_train = json.load(f)
with open('../input/test.json') as f:
    raw_test = json.load(f)
def ru_token(string):
    """russian tokenize based on nltk.word_tokenize. only russian letter remaind."""
    return [i for i in word_tokenize(string) if re.match(r'[\u0400-\u04ffа́]+$', i)]
tfidf = TfidfVectorizer(ngram_range=(1, 3), tokenizer=ru_token, stop_words=stopwords.words('russian'),
                        min_df=3,
                        use_idf=1, smooth_idf=1, sublinear_tf=1)
tfidf.fit([i['text'] for i in raw_train + raw_test])
class NBFeaturer(BaseEstimator, ClassifierMixin):
    """from https://www.kaggle.com/sermakarevich"""
    def __init__(self, alpha):
        self.alpha = alpha

    def preprocess_x(self, x, r):
        return x.multiply(r)

    def transform(self, x):
        x_nb = self.preprocess_x(x, self._r)
        return x_nb

    def fit(self, x, y):
        self._r = sparse.csr_matrix(np.log(self.pr(x, 1, y) / self.pr(x, 0, y)))
        return self

    def pr(self, x, y_i, y):
        p = x[y == y_i].sum(0)
        return (p + self.alpha) / ((y == y_i).sum() + self.alpha)
nbf = NBFeaturer(alpha=10)
model = LinearSVC(C=4, max_iter=30) # One-vs.-rest balanced category with 1:1:1

p = pipeline = Pipeline([
    ('nbf', nbf),
    ('lr', model)
])
train, val = train_test_split(raw_train, test_size=0.2, random_state=2018)
train_x = tfidf.transform([i['text'] for i in train])
lab = LabelEncoder()
lab.fit([i['sentiment'] for i in raw_train])
train_y = []
for i in range(3):
    train_y.append((lab.transform([d['sentiment'] for d in train]) == i).astype(int))
val_x = tfidf.transform([d['text'] for d in val])
val_y = lab.transform([d['sentiment'] for d in val])
test_x = tfidf.transform([i['text'] for i in raw_test])
a = lab.fit_transform([i['sentiment'] for i in raw_train])
w = [(len(a) - np.sum(a == i))/(2 * np.sum(a == i)) for i in range(max(a)+1)]
pred = []
test_pred = []
for i in range(3):
    p.get_params()['lr'].class_weight = {0: 1, 1:w[i] }
    p.fit(train_x, train_y[i])
    pred.append(p.decision_function(val_x))
    test_pred.append(p.decision_function(test_x))
accuracy_score(val_y, np.argmax(np.array(pred), axis=0))
print(classification_report(val_y, np.argmax(np.array(pred), axis=0), target_names=lab.classes_, digits=5))
sub_df = pd.DataFrame()
sub_df['id'] =  [i['id'] for i in raw_test]
sub_df['sentiment'] = np.argmax(np.array(test_pred), axis=0)
sub_df['sentiment']= sub_df['sentiment'].apply(lambda x: lab.classes_[x])
sub_df.head()
sub_df.to_csv('nb.csv', index=False)
