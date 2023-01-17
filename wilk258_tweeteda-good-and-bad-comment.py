# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt 
import seaborn as sns 
train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
train.head(3)
train.isnull().sum()
train = train.dropna()
print("train shape:", train.shape)
print("train len:", len(train))
print("test shape:", test.shape)
print("test len:", len(test))

train.dtypes
test.dtypes
import spacy
from spacy.symbols import nsubj, VERB
nlp = spacy.load('en_core_web_lg')
train.head(3)
sns.countplot(train['sentiment']);
plt.title('Data: Target distribution');
def text_entities(text):
    doc = nlp(text)
    for ent in doc.ents:
        print(f'Entity: {ent}, Label: {ent.label_}, {spacy.explain(ent.label_)}')
text_entities(train['text'][9])
one_sentence = train['text'][0]
doc = nlp(one_sentence)
spacy.displacy.render(doc, style='ent',jupyter=True)
one_sentence = train['text'][240]
doc = nlp(one_sentence)
spacy.displacy.render(doc, style='ent',jupyter=True)
one_sentence = train['text'][300]
doc = nlp(one_sentence)
spacy.displacy.render(doc, style='ent',jupyter=True)
one_sentence = train['text'][450]
doc = nlp(one_sentence)
spacy.displacy.render(doc, style='ent',jupyter=True)
def redact_names(text):
    doc = nlp(text)
    redacted_sentence = []
    for ent in doc.ents:
        ent.merge()
    for token in doc:
        if token.ent_type_ == "PERSON":
            redacted_sentence.append("[REDACTED]")
        else:
            redacted_sentence.append(token.string)
    return "".join(redacted_sentence)
one_sentence = train['text'][450]
doc = nlp(one_sentence)
spacy.displacy.render(doc, style='ent',jupyter=True)
one_sentence = redact_names(train['text'][500])
doc = nlp(one_sentence)
spacy.displacy.render(doc, style='ent',jupyter=True)

text = train['text'][9]
doc = nlp(text)
spacy.displacy.render(doc, style='ent', jupyter=True)

for idx, sentence in enumerate(doc.sents):
    for noun in sentence.noun_chunks:
        print(f"sentence {idx+1} has noun chunk '{noun}'")
one_sentence = train['text'][300]
doc = nlp(one_sentence)
spacy.displacy.render(doc, style='ent', jupyter=True)

for token in doc:
    print(token, token.pos_)
text = train['text'].str.cat(sep=' ')

max_length = 1000000-1
text = text[:max_length]

# removing URLs and '&amp' substrings using regex
import re
url_reg  = r'[a-z]*[:.]+\S+'
text   = re.sub(url_reg, '', text)
noise_reg = r'\&amp'
text   = re.sub(noise_reg, '', text)
doc = nlp(text)
items_of_interest = list(doc.noun_chunks)
items_of_interest = [str(x) for x in items_of_interest]
# pronoun in corona keyword  
df_nouns = pd.DataFrame(items_of_interest, columns=["Corona"])
plt.figure(figsize=(5,4))
sns.countplot(y="Corona",
             data=df_nouns,
             order=df_nouns["Corona"].value_counts().iloc[:10].index)
plt.show()
corona = []
for token in doc:
    if (not token.is_stop) and (token.pos_ == "NOUN") and (len(str(token))>2):
        corona.append(token)
        
corona = [str(x) for x in corona]
df_nouns = pd.DataFrame(corona, columns=["Corona Topics"])
df_nouns
plt.figure(figsize=(5,4))
sns.countplot(y="Corona Topics",
             data=df_nouns,
             order=df_nouns["Corona Topics"].value_counts().iloc[:10].index)
plt.show()
# I wanna see how about trump noun keyword 
trump_topics = []
for ent in doc.ents:
    if ent.label_ not in ["PERCENT", "CARDINAL", "DATE"]:
        trump_topics.append(ent.text.strip())
df_ttopics = pd.DataFrame(trump_topics, columns=["Trump Nouns"])
plt.figure(figsize=(5,4))
sns.countplot(y="Trump Nouns",
             data=df_ttopics,
             order=df_ttopics["Trump Nouns"].value_counts().iloc[1:11].index)
plt.show()
from spacy.lang.en.stop_words import STOP_WORDS
from wordcloud import WordCloud
plt.figure(figsize=(10,5))
wordcloud = WordCloud(background_color="white",
                      stopwords = STOP_WORDS,
                      max_words=45,
                      max_font_size=30,
                      random_state=42
                     ).generate(str(corona))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
from spacy.lang.en.stop_words import STOP_WORDS
from wordcloud import WordCloud
plt.figure(figsize=(10,5))
wordcloud = WordCloud(background_color="white",
                      stopwords = STOP_WORDS,
                      max_words=45,
                      max_font_size=30,
                      random_state=42
                     ).generate(str(trump_topics))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
text_ = train['text'][200]
doc = nlp(text_)
options = {'compact': True, 'bg': '#09a3d5',
           'color': 'white', 'font': 'Trebuchet MS'}
spacy.displacy.render(doc, jupyter=True, style='dep', options=options)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from matplotlib import pyplot as plt
import seaborn as sns
import eli5
from IPython.display import Image
from sklearn.model_selection import train_test_split
train_, test_ = train_test_split(train, test_size=0.2)
print("Train DF: ",train_.shape)
print("Test DF: ",test_.shape)
train_.head(3)
text_transformer = TfidfVectorizer(stop_words='english', 
                                   ngram_range=(1, 2), lowercase=True, max_features=150000)
X_train_text = text_transformer.fit_transform(train_['text'])
X_test_text = text_transformer.transform(test_['text'])
X_train = X_train_text
X_test = X_test_text
print("X Train DF: ",X_train.shape)
print("X Test DF: ", X_test.shape)
logit = LogisticRegression(C=5e1, solver='lbfgs', multi_class='multinomial',
                           random_state=17, n_jobs=4)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

cv_results = cross_val_score(logit, X_train, train_['sentiment'], cv=skf, scoring='f1_macro')
cv_results, cv_results.mean()
logit.fit(X_train, train_['sentiment'])

eli5.show_weights(estimator=logit, 
                  feature_names= text_transformer.get_feature_names(),top=(50, 5))