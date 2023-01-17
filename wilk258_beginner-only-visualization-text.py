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
import matplotlib.pyplot as plt 
import seaborn as sns 
from wordcloud import WordCloud,STOPWORDS
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re,string,unicodedata
from nltk.stem import WordNetLemmatizer,PorterStemmer
import os
import gc
from nltk.tokenize import word_tokenize
from collections import  Counter
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)
df = pd.read_csv("/kaggle/input/political-advertisements-from-facebook/fbpac-ads-en-US.csv")
df.head(3)
df.columns
fe = ['title','message','paid_for_by']
text_df = df[fe]
text_df.head(3)
text_df.shape
stop=set(stopwords.words('english'))

def build_list(df,col="title"):
    corpus=[]
    lem=WordNetLemmatizer()
    stop=set(stopwords.words('english'))
    new= df[col].dropna().str.split()
    new=new.values.tolist()
    corpus=[lem.lemmatize(word.lower()) for i in new for word in i if(word) not in stop]
    
    return corpus
corpus=build_list(text_df)
counter=Counter(corpus)
most=counter.most_common()
x=[]
y=[]
for word,count in most[:10]:
    if (word not in stop) :
        x.append(word)
        y.append(count)
plt.figure(figsize=(9,7))
sns.barplot(x=y,y=x)
plt.title("most common word in title")
corpus=build_list(text_df,"paid_for_by")
counter=Counter(corpus)
most=counter.most_common()
x=[]
y=[]
for word,count in most[:10]:
    if (word not in stop) :
        x.append(word)
        y.append(count)
        
plt.figure(figsize=(9,7))
sns.barplot(x=y,y=x)
plt.title("most common word in paid_for_by")
corpus=build_list(text_df,"message")
counter=Counter(corpus)
most=counter.most_common()
x=[]
y=[]
for word,count in most[:10]:
    if (word not in stop) :
        x.append(word)
        y.append(count)
        
plt.figure(figsize=(9,7))
sns.barplot(x=y,y=x)
plt.title("most common word in message")
def plot_count(feature, title,df, size=1, show_percents=False):
    f, ax = plt.subplots(1,1, figsize=(4*size, 4))
    total = float(len(df))
    g = sns.countplot(df[feature],order = df[feature].value_counts().index[0:20], palette='Set3')
    g.set_title("Number of {}".format(title))
    if (size > 2):
        plt.xticks(rotation=90, size=10)
    if(show_percents):
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2.,
                   height + 3, '{:1.2f}%'.format(100*height/total),
                   ha="center")
    ax.set_xticklabels(ax.get_xticklabels());
    plt.show()
plot_count('title','Title countplot', text_df, 3.5)
plot_count('message','message countplot', text_df, 3.5)
plot_count('paid_for_by','paid_for_by countplot', text_df, 3.5)
def clean(text):
    text = text.fillna("fillna").str.lower()
    text = text.map(lambda x: re.sub('\\n',' ',str(x)))
    text = text.map(lambda x: re.sub("\[\[User.*",'',str(x)))
    text = text.map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))
    text = text.map(lambda x: re.sub("\(http://.*?\s\(http://.*\)",'',str(x)))
    return text
text_df['title'] = clean(text_df['title'])
text_df['message'] = clean(text_df['message'])
text_df['paid_for_by'] = clean(text_df['paid_for_by'])
stemmer = PorterStemmer()
def stem_text(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            word = stemmer.stem(i.strip())
            final_text.append(word)
    return " ".join(finan_text)
plt.figure(figsize = (20, 20))
wc = WordCloud(max_words=1500, width=1600,height = 800 , stopwords = STOPWORDS).generate(" ".join(text_df.title))
plt.imshow(wc , interpolation = 'bilinear')
plt.figure(figsize = (20, 20))
wc = WordCloud(max_words=1500, width=1600,height = 800 , stopwords = STOPWORDS).generate(" ".join(text_df.message))
plt.imshow(wc , interpolation = 'bilinear')
plt.figure(figsize = (20, 20))
wc = WordCloud(max_words=1500, width=1600,height = 800 , stopwords = STOPWORDS).generate(" ".join(text_df.paid_for_by))
plt.imshow(wc , interpolation = 'bilinear')

import spacy 
nlp = spacy.load('en_core_web_lg')
def text_entity(text):
    doc = nlp(text)
    for ent in doc.ents:
        print(f'Entity: {ent}, Label: {ent.label_}, {spacy.explain(ent.label_)}')
text_entity(text_df['message'][10])

first = text_df['message'][50]
doc = nlp(first)
spacy.displacy.render(doc, style='ent',jupyter=True)
second = text_df['message'][130]
doc = nlp(second)
spacy.displacy.render(doc, style='ent',jupyter=True)
third = text_df['message'][1500]
doc = nlp(third)
spacy.displacy.render(doc, style='ent',jupyter=True)
first = text_df['message'][50]
doc = nlp(first)
spacy.displacy.render(doc, style='ent',jupyter=True)

for idx, sentence in enumerate(doc.sents):
    for noun in sentence.noun_chunks:
        print(f"sentence {idx+1} has noun chunk '{noun}'")
txt = text_df['message'][1500]
doc = nlp(txt)
spacy.displacy.render(doc, style='ent', jupyter=True)

for token in doc:
    print(token, token.pos_)

df_ = text_df['message'].str.cat(sep=' ')

max_length = 1000000-1
df_ =  df_[:max_length]

import re
url_reg  = r'[a-z]*[:.]+\S+'
df_   = re.sub(url_reg, '', df_)
noise_reg = r'\&amp'
df_   = re.sub(noise_reg, '', df_)
doc = nlp(df_)
items_of_interest = list(doc.noun_chunks)
items_of_interest = [str(x) for x in items_of_interest]
df_nouns = pd.DataFrame(items_of_interest, columns=["facebook"])
plt.figure(figsize=(5,4))
sns.countplot(y="facebook",
             data=df_nouns,
             order=df_nouns["facebook"].value_counts().iloc[:10].index)
plt.show()
text_df.head(3)
distri = text_df['message'][150]
doc = nlp(distri)
options = {'compact': True, 'bg': '#09a3d5',
           'color': 'white', 'font': 'Trebuchet MS'}
spacy.displacy.render(doc, jupyter=True, style='dep', options=options)
da = text_df['message'][2500]
doc = nlp(da)
options = {'compact': True, 'bg': '#09a3d5',
           'color': 'white', 'font': 'Trebuchet MS'}
spacy.displacy.render(doc, jupyter=True, style='dep', options=options)
_ = text_df['message'][561]
doc = nlp(_)
options = {'compact': True, 'bg': '#09a3d5',
           'color': 'white', 'font': 'Trebuchet MS'}
spacy.displacy.render(doc, jupyter=True, style='dep', options=options)
for token in doc:
    print(f"token: {token.text},\t dep: {token.dep_},\t head: {token.head.text},\t pos: {token.head.pos_},\
    ,\t children: {[child for child in token.children]}")