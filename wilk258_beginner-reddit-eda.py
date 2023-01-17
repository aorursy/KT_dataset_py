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
import tensorflow as tf 
import keras 
import eli5
import spacy
from spacy.symbols import nsubj, VERB
nlp = spacy.load('en_core_web_lg')
from scipy.sparse import hstack
from matplotlib import pyplot as plt
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
stopword=set(STOPWORDS)
post = pd.read_csv('/kaggle/input/dataisbeautiful/r_dataisbeautiful_posts.csv')
post.head(3)
post.shape
sns.countplot(x='removed_by', data=post)
sns.countplot(x='over_18', data=post)
def text_entities(text):
    doc = nlp(text)
    for ent in doc.ents:
        print(f'Entity: {ent}, Label: {ent.label_}, {spacy.explain(ent.label_)}')
text_entities(post['title'][9])

one_sentence = post['title'][5]
doc = nlp(one_sentence)
spacy.displacy.render(doc, style='ent',jupyter=True)
one_sentence = post['title'][1500]
doc = nlp(one_sentence)
spacy.displacy.render(doc, style='ent',jupyter=True)
one_sentence = post['title'][5500]
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

text = post['title'][10000]
doc = nlp(text)
spacy.displacy.render(doc, style='ent', jupyter=True)

for idx, sentence in enumerate(doc.sents):
    for noun in sentence.noun_chunks:
        print(f"sentence {idx+1} has noun chunk '{noun}'")
text = post['title'].str.cat(sep=' ')

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
df_nouns = pd.DataFrame(items_of_interest, columns=["Trump"])
plt.figure(figsize=(5,4))
sns.countplot(y="Trump",
             data=df_nouns,
             order=df_nouns["Trump"].value_counts().iloc[:10].index)
plt.show()
Coronavirus = []
for token in doc:
    if (not token.is_stop) and (token.pos_ == "NOUN") and (len(str(token))>2):
        Coronavirus.append(token)
        
corona = [str(x) for x in Coronavirus]
df_nouns = pd.DataFrame(corona, columns=["Coronavirus"])
df_nouns
plt.figure(figsize=(5,4))
sns.countplot(y="Coronavirus",
             data=df_nouns,
             order=df_nouns["Coronavirus"].value_counts().iloc[:10].index)
plt.show()
text_ = post['title'][5000]
doc = nlp(text_)
options = {'compact': True, 'bg': '#09a3d5',
           'color': 'white', 'font': 'Trebuchet MS'}
spacy.displacy.render(doc, jupyter=True, style='dep', options=options)
def plot_count(feature, title, df, size=1, show_percents=False):
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    total = float(len(df))
    g = sns.countplot(df[feature], order = df[feature].value_counts().index[0:20], palette='Set3')
    g.set_title("Number of {}".format(title))
    if(size > 2):
        plt.xticks(rotation=90, size=10)
    if(show_percents):
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,
                    height + 3,
                    '{:1.2f}%'.format(100*height/total),
                    ha="center") 
    ax.set_xticklabels(ax.get_xticklabels());
    plt.show()    
plot_count('title','reddit title', post, 3.5)
def nonan(x):
    if type(x) == str:
        return x.replace("\n", "")
    else:
        return ""

text = ' '.join([nonan(abstract) for abstract in post["title"]])
wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,
                      width=1200, height=1000).generate(text)
fig = px.imshow(wordcloud)
fig.update_layout(title_text='reddit title noun')
