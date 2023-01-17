# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import csv, collections
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import roc_curve, auc
print("Libraries Imported")
df = pd.read_csv('../input/sentiment140/training.1600000.processed.noemoticon.csv', encoding='latin', header=None, names=['target', 'ids', 'date', 'flag', 'user', 'tweet'])
df.head()
df = df.drop(["ids", "date", "flag", "user"], axis=1)
df.head()
df['target'] = df['target'].replace(4, 1)
df.tail()
col ='target'

fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x=col, data=df, alpha=0.5)

plt.show()
df["tweet"] = df["tweet"].map(lambda x: x.lower())
print(df.head())
    
df["tweet"] = df["tweet"].str.replace(r'(\w+\.)*\w+@(\w+\.)+[a-z]+', '')
print(df.head())
df["tweet"] = df["tweet"].str.replace(r'(http|ftp|https)://[-\w.]+(:\d+)?(/([\w/_.]*)?)?|www[\.]\S+', '')
print(df.head())
df["tweet"] = df["tweet"].str.replace(r'[\@\#]\S+', '')
print(df.head())
df["tweet"] = df["tweet"].str.replace(r'<\w+[^>]+>', '')
print(df.head())
emo_info = {
    # positive emoticons
    ":‑)": " good ",
    ":)": " good ",
    ";)": " good ",
    ":-}": " good ",
    "=]": " good ",
    "=)": " good ",
    ";d": " good ",
    ":d": " good ",
    ":dd": " good ",
    "xd": " good ",
    ":p": " good ",
    "xp": " good ",
    "<3": " love ",

    # negativve emoticons
    ":‑(": " sad ",
    ":‑[": " sad ",
    ":(": " sad ",
    "=(": " sad ",
    "=/": " sad ",
    ":{": " sad ",
    ":/": " sad ",
    ":|": " sad ",
    ":-/": " sad ",
    ":o": " shock "

}
emo_info_order = [k for (k_len, k) in reversed(sorted([(len(k), k) for k in emo_info.keys()]))]
def emo_repl(phrase):
    for k in emo_info_order:
        phrase = phrase.replace(k, emo_info[k])
    return phrase
df['tweet'] = df['tweet'].apply(emo_repl)
print(df.head())
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"\bdon't\b", "do not", phrase)
    phrase = re.sub(r"\bdoesn't\b", "does not", phrase)
    phrase = re.sub(r"\bdidn't\b", "did not", phrase)
    phrase = re.sub(r"\bdidnt\b", "did not", phrase)
    phrase = re.sub(r"\bhasn't\b", "has not", phrase)
    phrase = re.sub(r"\bhaven't\b", "have not", phrase)
    phrase = re.sub(r"\bhavent\b", "have not", phrase)
    phrase = re.sub(r"\bhadn't\b", "had not", phrase)
    phrase = re.sub(r"\bwon't\b", "will not", phrase)
    phrase = re.sub(r"\bwouldn't\b", "would not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)

    # using regular expressions to expand the contractions
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    return phrase
df['tweet'] = df['tweet'].apply(decontracted)
print(df.head())
stop = stopwords.words('english')
# 불용어 추가
manual_sw_list = ['retweet', 'retwet', 'rt', 'oh', 'dm', 'mt', 'ht', 'ff', 'shoulda', 'woulda', 'coulda', 'might', 'im', 'tb', 'mysql', 'hah', "a", "an", "the", "and", "but", "if",
                  "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
                  "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "nor", "only", "own", "same", "so", "than", "too", "very", "s",
                  "t", "just", "don", "now", 'tweet', 'x', 'f']

stop.extend(manual_sw_list)
df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
print(df.head())
lem = WordNetLemmatizer()
df['tweet'] = df['tweet'].apply(lambda x: ' '.join([lem.lemmatize(word, 'v') for word in x.split()]))
print(df.head())
df["tweet"] = df["tweet"].str.replace(r'[^\w\s]', '')
print(df.head())
df["tweet"] = df["tweet"].str.replace(r'[0-9]+', '')
print(df.head())

non_alphabet = re.compile(r'[^a-z]+')
df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if non_alphabet.search(word) is None]))
print(df.head())
df['tweet'] = df['tweet'].str.replace(r'([a-z])\1{1,}', r'\1\1')
df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word if len(wordnet.synsets(word)) > 0 else re.sub(r'([a-z])\1{1,}', r'\1', word) for word in x.split()]))
print(df.head())
df['tweet'] = df['tweet'].str.replace(r'(ha)\1{1,}', r'\1')
print(df.head())
df.drop(df[df["tweet"] == ''].index, inplace=True)
df = df.reset_index(drop=True)
col ='target'

fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x=col, data=df, alpha=0.5)

plt.show()

print(len(df[df[col] == 0]))
print(len(df[df[col] == 1]))
text_train, text_tst, y_train, y_tst = train_test_split(df['tweet'], df['target'], random_state=0, test_size=0.50)
text_train, text_test, y_train, y_test = train_test_split(text_train, y_train, random_state=0, test_size=0.10)
print(len(text_train))
print(len(text_test))
def load_sent_word_net():
    sent_scores = collections.defaultdict(list)

    with open("../input/sentiwordnet/sentiwordnet/SentiWordNet_3.0.0.txt","r") as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='"')

        for line in reader:
            if line[0].startswith("#"):
                continue
            if len(line) == 1:
                continue
            POS, ID, PosScore, NegScore, SynsetTerms, Glos = line
            if len(POS) == 0 or len(ID) == 0:
                continue
            for term in SynsetTerms.split(" "):
                term = term.split('#')[0]
                # print(term)
                term = term.replace("-", " ").replace("_", " ")
                key = "%s/%s" % (POS, term)
                # print(key)
                sent_scores[key].append((float(PosScore), float(NegScore)))
                # print(sent_scores)
        for key, value in sent_scores.items():
            sent_scores[key] = np.mean(value, axis=0)

        return sent_scores


sent_word_net = load_sent_word_net()
class LinguisticVectorizer(BaseEstimator):

    def get_feature_names(self):
        return np.array(['sent_pos', 'sent_neg', 'nouns', 'adjectives', 'verbs', 'adverbs'])

    def fit(self, documents, y=None):
        return self

    def _get_sentiments(self, d):
        sent = tuple(d.split())
        tagged = nltk.pos_tag(sent)

        pos_vals = []
        neg_vals = []

        nouns = 0.
        adjectives = 0.
        verbs = 0.
        adverbs = 0.

        i = 0
        for w, t in tagged:

            p, n = 0, 0
            sent_pos_type = None
            if t.startswith("NN"):
                #noun
                sent_pos_type = "n"
                nouns += 1
            elif t.startswith("JJ"):
                #adjective
                sent_pos_type = "a"
                adjectives += 1
            elif t.startswith("VB"):
                #verb
                sent_pos_type = "v"
                verbs += 1
            elif t.startswith("RB"):
                #adverb
                sent_pos_type = "r"
                adverbs += 1
            else:
                sent_pos_type = "Nan"

                i += 1
                l = len(sent) - i

                if l == 0:
                    l = 1
                else:
                    pass

            if sent_pos_type is not None:

                sent_word = "%s/%s" % (sent_pos_type, w)

                if sent_word in sent_word_net:
                    p, n = sent_word_net[sent_word]
                elif sent_word == "Nan":
                    p, n = 0, 0

                pos_vals.append(p)
                neg_vals.append(n)

        if i == 0:
            l = len(sent)
        else:
            pass

        avg_pos_val = np.mean(pos_vals)
        avg_neg_val = np.mean(neg_vals)

        return [avg_pos_val, avg_neg_val, nouns / l, adjectives / l, verbs / l, adverbs / l]

    # print(_get_sentiments('This be fantastic'))

    def transform(self, documents):
        pos_val, neg_val, nouns, adjectives, verbs, adverbs = np.array([self._get_sentiments(d) for d in documents]).T
        result = np.array([pos_val, neg_val, nouns, adjectives, verbs, adverbs]).T

        return result
tfidf_ngrams = TfidfVectorizer(min_df=5, ngram_range=(1, 3))
ling_stats = LinguisticVectorizer()
all_features = FeatureUnion([('ling', ling_stats), ('tfidf', tfidf_ngrams)])
clf = MultinomialNB(alpha=5)

pipeline = Pipeline([('all', all_features), ('clf', clf)])

pipeline.fit(text_train, y_train)
y_train_pred = pipeline.predict_proba(text_train)[:, 1] 
y_test_pred = pipeline.predict_proba(text_test)[:, 1]
train_fpr, train_tpr, train_thresholds = roc_curve(y_train, y_train_pred)
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_test_pred)
plt.plot(train_fpr, train_tpr, label="train AUC =" + str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC =" + str(auc(test_fpr, test_tpr)))
plt.legend()
plt.title("AUC PLOTS")
plt.grid()
plt.show()
trauc=round(auc(train_fpr, train_tpr),3)
teauc=round(auc(test_fpr, test_tpr),3)
print('Train AUC=',trauc)
print('Test AUC=',teauc)
