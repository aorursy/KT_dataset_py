import os

import pandas as pd

import numpy as np

import sklearn as sk

import matplotlib.pyplot as plt

import seaborn as sns

from string import punctuation

from nltk import pos_tag

from nltk.stem import PorterStemmer

from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.corpus import stopwords

from nltk import FreqDist

from wordcloud import WordCloud, STOPWORDS



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()

print("--- Shape ---")

print(train.shape)

print("--- Missing values ---")

train.isnull().sum() * 100 / len(train)
sns.countplot(train.author)
# Takes a column and concatenate strings

def build_corpus(data):

    data = str(data)

    corpus = ""

    for sent in data:

        corpus += str(sent)

    return corpus
# Gather text of authors in different dataframes

eap = train[train.author == "EAP"]

hpl = train[train.author == "HPL"]

mws = train[train.author == "MWS"]
plt.figure(figsize=(15,10))

plt.subplot(331)

eap_wc = WordCloud(background_color="white", max_words=100, stopwords=STOPWORDS)

eap_wc.generate(build_corpus(eap.text))

plt.title("Edgar Allan Poe", fontsize=20)

plt.imshow(eap_wc, interpolation='bilinear')

plt.axis("off")



plt.subplot(332)

hpl_wc = WordCloud(background_color="white", max_words=100, stopwords=STOPWORDS)

hpl_wc.generate(build_corpus(hpl.text))

plt.title("HP Lovecraft", fontsize=20)

plt.imshow(hpl_wc, interpolation='bilinear')

plt.axis("off")



plt.subplot(333)

mws_wc = WordCloud(background_color="white", max_words=100, stopwords=STOPWORDS)

mws_wc.generate(build_corpus(mws.text))

plt.title("Marry Shelley", fontsize=20)

plt.imshow(mws_wc, interpolation='bilinear')

plt.axis("off")
le = LabelEncoder()

author_encoded = le.fit_transform(train.author)
seed = 12

X_train, X_test, y_train, y_test = train_test_split(train.text, author_encoded, 

    test_size=0.3, random_state=seed)

metric = 'accuracy'

kfold = KFold(n_splits=10, random_state=seed)
# Return called columns of a DataFrame

class ColumnExtractor(TransformerMixin):

    def __init__(self, cols):

        self.cols = cols

    def transform(self, X):

        Xcols = X[self.cols]

        return Xcols

    def fit(self, X, y=None):

        return self



# Enables to train an estimator within the pipeline

class ModelTransformer(TransformerMixin):

    def __init__(self, model):

        self.model = model

    def fit(self, *args, **kwargs):

        self.model.fit(*args, **kwargs)

        return self

    def transform(self, X, **transform_params):

        return pd.DataFrame(self.model.predict(X))
# Calculate the length of each text

class LengthTransformer(TransformerMixin):

    def transform(self, X, **transform_params):

        return pd.DataFrame(X.apply(lambda x: len(str(x)))) 

    def fit(self, X, y=None, **fit_params):

        return self

    

# Count the number of words in each text

class WordCountTransformer(TransformerMixin):

    def transform(self, X, **transform_params):

        return pd.DataFrame(X.apply(lambda x: len(str(x).split()))) 

    def fit(self, X, y=None, **fit_params):

        return self

    

# Count the number of unique words in each text

class UniqueWordCountTransformer(TransformerMixin):

    def transform(self, X, **transform_params):

        return pd.DataFrame(X.apply(lambda x: len(set(str(x).split())))) 

    def fit(self, X, y=None, **fit_params):

        return self

    

# Calculate the average length of words in each text

class MeanLengthTransformer(TransformerMixin):

    def transform(self, X, **transform_params):

        return pd.DataFrame(X.apply(lambda x: np.mean([len(w) for w in str(x).split()]))) 

    def fit(self, X, y=None, **fit_params):

        return self



# Count the number of punctuation in each sentence

class PunctuationCountTransformer(TransformerMixin):

    def transform(self, X, **transform_params):

        return pd.DataFrame(X.apply(lambda x: len([p for p in str(x) if p in punctuation]))) 

    def fit(self, X, y=None, **fit_params):

        return self



# Count the number of unique words in each text

class StopWordsCountTransformer(TransformerMixin):

    def transform(self, X, **transform_params):

        return pd.DataFrame(X.apply(lambda x: len([sw for sw in str(x).lower().split() if sw in set(stopwords.words("english"))]))) 

    def fit(self, X, y=None, **fit_params):

        return self
pipeline = Pipeline([

    ('features', FeatureUnion([

        ('text_length', LengthTransformer()),

        ('word_count', WordCountTransformer()),

        ('mean_length', MeanLengthTransformer()),

        ('punctuation_count', PunctuationCountTransformer()),

        ('stop_words_count', StopWordsCountTransformer()),

        ('count_vect', CountVectorizer(lowercase=False)),

        ('tf_idf', TfidfVectorizer())

    ])),

  ('classifier', XGBClassifier(objective='multi:softprob', random_state = 12, eval_metric='mlogloss'))

])
clf_pipe = pipeline.fit(X_train, y_train)

score_pipe = cross_val_score(clf_pipe, X_train, y_train, cv=kfold, scoring=metric)

print("Mean score = %.3f, Std deviation = %.3f"%(np.mean(score_pipe),np.std(score_pipe)))

score_pipe_test = clf_pipe.score(X_test,y_test)

print("Mean score = %.3f, Std deviation = %.3f"%(np.mean(score_pipe_test),np.std(score_pipe_test)))
conf_mat = confusion_matrix(y_test, clf_pipe.predict(X_test))

sns.heatmap(conf_mat, annot=True)

plt.xticks(range(3), ('EAP', 'HPL', 'MWS'), horizontalalignment='left')

plt.yticks(range(3), ('EAP', 'HPL', 'MWS'), rotation=0)
target_names = ['EAP', 'HPL', 'MWS']

y_pred = pd.DataFrame(clf_pipe.predict(test.text), columns=target_names)

submission = pd.concat([test["id"],y_pred], 1)

submission.to_csv("./submission.csv", index=False)