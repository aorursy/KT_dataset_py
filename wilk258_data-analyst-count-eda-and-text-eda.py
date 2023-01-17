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
!pip install rake_nltk
import matplotlib.pyplot as plt 
import seaborn as sns
import re
import pyLDAvis
import gensim
from rake_nltk import Rake
from spacy.tokens import Span 
import pyLDAvis.gensim
from wordcloud import WordCloud,STOPWORDS
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re,string,unicodedata
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize
from collections import  Counter
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)
import spacy 
nlp = spacy.load('en_core_web_lg')
df = pd.read_csv('/kaggle/input/data-analyst-jobs/DataAnalyst.csv')
df.head(3)
plt.rcParams["figure.figsize"] = (15,8)
plt.style.use("fivethirtyeight")
df['Salary Estimate'].value_counts().sort_values(ascending=False).head(30).plot(kind='bar')
plt.title('Salary Estimate Frequency')
plt.rcParams["figure.figsize"] = (15,8)
plt.style.use("fivethirtyeight")
sns.barplot(y=df['Rating'].value_counts().index,
            x=df['Rating'].value_counts().sort_values(ascending=False))
plt.rcParams["figure.figsize"] = (15,8)
plt.style.use("fivethirtyeight")
df['Job Title'].value_counts().sort_values(ascending=False).head(30).plot(kind='bar')
plt.title('Job Title Frequency')
plt.show()
plt.rcParams["figure.figsize"] = (15,8)
plt.style.use("fivethirtyeight")
df['Company Name'].value_counts().sort_values(ascending=False).head(30).plot(kind='bar')
plt.title("Company Name Frequency")
plt.show()
plt.rcParams["figure.figsize"] = (15,8)
plt.style.use("fivethirtyeight")
df['Location'].value_counts().sort_values(ascending=False).head(30).plot(kind='bar') 
plt.title("Location Frequency")
plt.show()
plt.rcParams["figure.figsize"] = (15,8)
plt.style.use("fivethirtyeight")
df['Headquarters'].value_counts().sort_values(ascending=False).head(30).plot(kind='bar')
plt.title("Headquarters Frequency")
plt.show()
plt.rcParams["figure.figsize"] = (15,8)
plt.style.use("fivethirtyeight")
df['Size'].value_counts().sort_values(ascending=False).head(30).plot(kind='bar') 
plt.title("Company Size Frequency")
plt.show()
plt.rcParams["figure.figsize"] = (15,8)
plt.style.use("fivethirtyeight")
df['Founded'].value_counts().sort_values(ascending=False).head(30).plot(kind='bar')
plt.title("Founded Frequency")
plt.show()
plt.rcParams["figure.figsize"] = (15,8)
plt.style.use("fivethirtyeight")
df['Type of ownership'].value_counts().sort_values(ascending=False).head(30).plot(kind='bar') 
plt.title("Type Of Ownership Frequency")
plt.show()
plt.rcParams["figure.figsize"] = (15,8)
plt.style.use("fivethirtyeight")
df['Industry'].value_counts().sort_values(ascending=False).head(30).plot(kind='bar')
plt.title("Industry Frequency")
plt.show()
plt.rcParams["figure.figsize"] = (15,8)
plt.style.use("fivethirtyeight")
df['Sector'].value_counts().sort_values(ascending=False).head(30).plot(kind='bar')
plt.title("Sector Frequency")
plt.show()
plt.rcParams["figure.figsize"] = (15,8)
plt.style.use("fivethirtyeight")
df['Revenue'].value_counts().sort_values(ascending=False).head(30).plot(kind='bar') 
text = df['Job Description']
def text_pro(df):
    corpus=[]
    stem = PorterStemmer()
    lem=WordNetLemmatizer()

    for news in text:
        words=[w for w in word_tokenize(news) if (w not in stop)]
        
        words=[lem.lemmatize(w) for w in words if len(w)>2]
        
        corpus.append(words)
    return corpus
corpus=text_pro(text)
dic=gensim.corpora.Dictionary(corpus)
bow_corpus = [dic.doc2bow(doc) for doc in corpus]
lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 4, 
                                   id2word = dic,                                    
                                   passes = 10,
                                   workers = 2)
lda_model.show_topics()
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dic)
vis
df.head(3)
stop=set(stopwords.words('english'))

def build_list(df,col="Job Description"):
    corpus=[]
    lem=WordNetLemmatizer()
    stop=set(stopwords.words('english'))
    new= df[col].dropna().str.split()
    new=new.values.tolist()
    corpus=[lem.lemmatize(word.lower()) for i in new for word in i if(word) not in stop]
    
    return corpus
corpus=build_list(df)
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
plt.title("Most Common Word In Job Description")
def clean(text):
    text = text.fillna("fillna").str.lower()
    text = text.map(lambda x: re.sub('\\n',' ',str(x)))
    text = text.map(lambda x: re.sub("\[\[User.*",'',str(x)))
    text = text.map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))
    text = text.map(lambda x: re.sub("\(http://.*?\s\(http://.*\)",'',str(x)))
    return text
df['Job Description'] = clean(df['Job Description'])
stemmer = PorterStemmer()
def stem_text(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            word = stemmer.stem(i.strip())
            final_text.append(word)
    return " ".join(finan_text)
plt.figure(figsize = (20, 20))
wc = WordCloud(max_words=1500, width=1600,height = 800 , stopwords = STOPWORDS).generate(" ".join(df['Job Description']))
plt.imshow(wc , interpolation = 'bilinear')
def text_entity(text):
    doc = nlp(text)
    for ent in doc.ents:
        print(f'Entity: {ent}, Label: {ent.label_}, {spacy.explain(ent.label_)}')
text_entity(df['Job Description'][10])
first = df['Job Description'][50]
doc = nlp(first)
spacy.displacy.render(doc, style='ent',jupyter=True)
second = df['Job Description'][125]
doc = nlp(second)
spacy.displacy.render(doc, style='ent',jupyter=True)
third = df['Job Description'][500]
doc = nlp(third)
spacy.displacy.render(doc, style='ent',jupyter=True)
first = df['Job Description'][75]
doc = nlp(first)
spacy.displacy.render(doc, style='ent',jupyter=True)

for idx, sentence in enumerate(doc.sents):
    for noun in sentence.noun_chunks:
        print(f"sentence {idx+1} has noun chunk '{noun}'")
snd = df['Job Description'][195]
doc = nlp(snd)
spacy.displacy.render(doc, style='ent',jupyter=True)

for idx, sentence in enumerate(doc.sents):
    for noun in sentence.noun_chunks:
        print(f"sentence {idx+1} has noun chunk '{noun}'")
thrd = df['Job Description'][195]
doc = nlp(thrd)
spacy.displacy.render(doc, style='ent',jupyter=True)

for idx, sentence in enumerate(doc.sents):
    for noun in sentence.noun_chunks:
        print(f"sentence {idx+1} has noun chunk '{noun}'")
df_ = df['Job Description'].str.cat(sep=' ')

max_length = 1000000-1
df_ =  df_[:max_length]
doc = nlp(df_)
items_of_interest = list(doc.noun_chunks)
items_of_interest = [str(x) for x in items_of_interest]
df_nouns = pd.DataFrame(items_of_interest, columns=["data"])
plt.figure(figsize=(5,4))
sns.countplot(y="data",
             data=df_nouns,
             order=df_nouns["data"].value_counts().iloc[:10].index)
plt.show()
distribution = df['Job Description'][155]
doc = nlp(distribution)
options = {'compact': True, 'bg': '#09a3d5',
           'color': 'white', 'font': 'Trebuchet MS'}
spacy.displacy.render(doc, jupyter=True, style='dep', options=options)
distribution1 = df['Job Description'][175]
doc = nlp(distribution1)
options = {'compact': True, 'bg': '#09a3d5',
           'color': 'white', 'font': 'Trebuchet MS'}
spacy.displacy.render(doc, jupyter=True, style='dep', options=options)
distribution2 = df['Job Description'][375]
doc = nlp(distribution2)
options = {'compact': True, 'bg': '#09a3d5',
           'color': 'white', 'font': 'Trebuchet MS'}
spacy.displacy.render(doc, jupyter=True, style='dep', options=options)
for token in doc:
    print(token.text, token.dep_, token.head.text, token.head.pos_,
          [child for child in token.children])
for token in doc:
    print(f"token: {token.text},\t dep: {token.dep_},\t head: {token.head.text},\t pos: {token.head.pos_},\
    ,\t children: {[child for child in token.children]}")