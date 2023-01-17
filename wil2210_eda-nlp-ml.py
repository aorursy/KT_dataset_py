import os
import numpy as np
import pandas as pd
import spacy

import matplotlib.pyplot as plt
import seaborn as sns

sns.set
%matplotlib inline
df_train = pd.read_csv(os.path.join('..', 'input', 'train.csv'))
df_test = pd.read_csv(os.path.join('..', 'input', 'test.csv'))
df_train.head()
df_train.shape, df_test.shape
df_train.info()
df_train['target'].value_counts().plot(kind='bar');
insincere_ratio = (80810 / 1225312) * 100
insincere_ratio
y = df_train['target']
X = df_train['question_text']
X_insincere = X[y == 1]
X_insincere.head()
X_sincere = X[y == 0]
X_sincere.head()
from nltk.tokenize import word_tokenize
corpus = [word_tokenize(token) for token in X]
lowercase_train = [[token.lower() for token in doc] for doc in corpus]
alphas = [[token for token in doc if token.isalpha()] for doc in lowercase_train]
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
train_no_stop = [[token for token in doc if token not in stop_words] for doc in alphas]
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stemmed = [[stemmer.stem(token) for token in doc] for doc in train_no_stop]
train_clean_str = [ ' '.join(doc) for doc in stemmed]
nb_words = [len(tokens) for tokens in alphas]
alphas_unique = [set(doc) for doc in alphas]
nb_words_unique = [len(doc) for doc in alphas_unique]
train_str = [ ' '.join(doc) for doc in lowercase_train]
nb_characters = [len(doc) for doc in train_str]
train_stopwords = [[token for token in doc if token in stop_words] for doc in alphas]
nb_stopwords = [len(doc) for doc in train_stopwords]
non_alphas = [[token for token in doc if token.isalpha() == False] for doc in lowercase_train]
nb_punctuation = [len(doc) for doc in non_alphas]
train_title = [[token for token in doc if token.istitle() == True] for doc in corpus]
nb_title = [len(doc) for doc in train_title]
df_clean = pd.DataFrame(data={'text_clean': train_clean_str})
df_clean.head()
nb_words = pd.Series(nb_words)
nb_words_unique = pd.Series(nb_words_unique)
nb_characters = pd.Series(nb_characters)
nb_stopwords = pd.Series(nb_stopwords)
nb_punctuation = pd.Series(nb_punctuation)
nb_title = pd.Series(nb_title)
df_show = pd.concat([df_clean, nb_words, nb_words_unique, nb_characters, nb_stopwords, nb_punctuation, nb_title], axis=1).rename(columns={
    0: "Number of words", 1: 'Number of unique words', 2: 'Number of characters', 3: 'Number of stopwords', 4: 'Number of punctuations',
    5: 'Number of titlecase words'
})
df_show.head()
df_feat = df_show.drop(['text_clean'], axis=1)
df_feat.head()
df_feat.info()
from nltk.tokenize import word_tokenize
%%time
corpus_insincere = [word_tokenize(t) for t in X_insincere]
lowercase = [[t.lower() for t in doc] for doc in corpus_insincere]
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
no_stops = [[t for t in doc if t not in stop_words] for doc in lowercase]
alphas_insincere = [[token for token in doc if token.isalpha()] for doc in no_stops]
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stemmed_insincere = [[stemmer.stem(token) for token in doc] for doc in alphas_insincere]
nb_words_insincere_nostop = [len(tokens) for tokens in no_stops]
avg_nostop = np.mean(nb_words_insincere_nostop)
avg_nostop
nb_words_insincere_stop = [len(tokens) for tokens in lowercase]
avg_stop = np.mean(nb_words_insincere_stop)
avg_stop
np.median(nb_words_insincere_nostop)
np.median(nb_words_insincere_stop)
nb_words_insincere_stop = pd.Series(nb_words_insincere_stop)
nb_words_insincere_nostop = pd.Series(nb_words_insincere_nostop)
df_insincere =  pd.DataFrame(X_insincere)
df_insincere.info()
df_insincere = pd.concat([X_insincere.reset_index(), nb_words_insincere_nostop, nb_words_insincere_stop], axis=1).set_index('index').rename(columns={
    0: "nb_words_no_stop", 1: 'nb_words_stop'
})
df_insincere.head()
#plt.hist(nb_words_insincere_nostop, bins=30)
#plt.hist(nb_words_insincere_stop, bins=30)
sns.distplot(np.log1p(nb_words_insincere_nostop), kde=False, label="No stop")
sns.distplot(np.log1p(nb_words_insincere_stop), kde=False, label="Stop")
plt.legend();
sns.distplot(nb_words_insincere_stop, hist=False, color='red', label='Stop')
sns.distplot(nb_words_insincere_nostop, hist=False, color='blue', label='No stop')
plt.legend();
from collections import defaultdict

counter = defaultdict(int)
for doc in alphas_insincere:
    for token in doc:
        counter[token] += 1

from collections import Counter

c = Counter(counter)

c.most_common(10)
corpus_sincere = [word_tokenize(t) for t in X_sincere]
lowercase_sincere = [[t.lower() for t in doc] for doc in corpus_sincere]
no_stop_sincere = [[t for t in doc if t not in stop_words] for doc in lowercase_sincere]
alphas_sincere = [[token for token in doc if token.isalpha()] for doc in no_stop_sincere]
nb_words_sincere_nostop = [len(t) for t in no_stop_sincere]
avg_words_sincere_nostop = np.mean(nb_words_sincere_nostop)
avg_words_sincere_nostop
nb_words_sincere_stop = [len(t) for t in lowercase_sincere]
avg_words_sincere = np.mean(nb_words_sincere_stop)
avg_words_sincere
np.median(nb_words_sincere_nostop)
np.median(nb_words_sincere_stop)
nb_words_sincere_stop = pd.Series(nb_words_sincere_stop)
nb_words_sincere_nostop = pd.Series(nb_words_sincere_nostop)
df_sincere =  pd.DataFrame(X_sincere)
df_sincere.info()
df_sincere = pd.concat([X_sincere.reset_index(), nb_words_sincere_nostop, nb_words_sincere_stop], axis=1).set_index('index').rename(columns={
    0: "nb_words_no_stop", 1: 'nb_words_stop'
})
df_sincere.head()
from collections import defaultdict

counter_sincere = defaultdict(int)
for doc in alphas_sincere:
    for token in doc:
        counter[token] += 1

from collections import Counter

c_sincere = Counter(counter)

c_sincere.most_common(10)
from gensim import corpora
dictionary = corpora.Dictionary(alphas_insincere)
corpus_1 = [dictionary.doc2bow(t) for t in alphas_insincere]
from gensim.models.ldamodel import LdaModel
%%time
lda_model = LdaModel(
    corpus=corpus_1, id2word=dictionary, num_topics=4, random_state=42)
from pprint import pprint
pprint(lda_model.print_topics())
%%time
lda_model_1 = LdaModel(
    corpus=corpus_1, id2word=dictionary, num_topics=4, random_state=42, iterations=10)
from pprint import pprint
pprint(lda_model_1.print_topics())
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(lda_model, corpus_1, dictionary)
pyLDAvis.gensim.prepare(lda_model_1, corpus_1, dictionary)
weight_topic = lda_model_1.top_topics(corpus=corpus_1, dictionary=dictionary, topn=30)
politic, religion, sex, america = weight_topic
politic = politic[0]
politic = [tup[1] for tup in politic]
politic
religion = religion[0]
religion = [tup[1] for tup in religion]
religion
sex = sex[0]
sex = [tup[1] for tup in sex]
sex
america = america[0]
america = [tup[1] for tup in america]
america
y_labeled = []
for doc in train_no_stop:
    counter = 0
    for word in doc:
        if word in politic or word in religion or word in sex or word in america:
            counter += 1
    if counter >= 3:
        y_labeled.append(1)
    else:
        y_labeled.append(0)
        
y_labeled[:10]
y_labeled = pd.Series(y_labeled)
y_labeled[:3]
df_feat['y_topic_labeled'] = y_labeled
df_show['y_topic_labeled'] = y_labeled
df_feat.head()
df_feat['y_topic_labeled'].value_counts()
df_show.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
tvec = TfidfVectorizer(stop_words='english')
tf = tvec.fit_transform(X_train)
tf
cvec = CountVectorizer(stop_words='english')
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=100, random_state=42)
from sklearn.pipeline import Pipeline
preprocessing_pipeline = Pipeline([('tvec', tvec), ('svd', svd)])
preprocessing_pipeline.fit_transform(X_train)
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
pipe_mnb = Pipeline([('vectorizer', cvec), ('mnb', mnb)])
pipe_mnb.fit(X_train, y_train)
y_pred_mnb = pipe_mnb.predict(X_test)
y_pred_mnb
cm = confusion_matrix(y_test, y_pred_mnb)
cm
cr = classification_report(y_test, y_pred_mnb)
print(cr)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
#pipe_rf = Pipeline([('vectorizer', tvec), ('rf', rf)])
#pipe_rf.fit(X_train, y_train)
#y_pred = pipe_rf.predict(X_test)
#y_pred
#cm = confusion_matrix(y_test, y_pred)
#cm
#cr = classification_report(y_test, y_pred)
#print(cr)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
pipe_lr = Pipeline([('vectorizer', cvec), ('lr', lr)])
pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict(X_test)
y_pred_lr
cm = confusion_matrix(y_test, y_pred_lr)
cm
cr = classification_report(y_test, y_pred_lr)
print(cr)
