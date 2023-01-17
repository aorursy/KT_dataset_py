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
!python -m spacy download es_core_news_sm
!pip install rake_nltk
import matplotlib.pyplot as plt 
import seaborn as sns 
import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from wordcloud import WordCloud,STOPWORDS
from nltk.corpus import stopwords
import re,string,unicodedata
from collections import  Counter
import pyLDAvis
import gensim
import pyLDAvis.gensim
from tqdm import tqdm
from rake_nltk import Rake
from spacy.tokens import Span 
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer,PorterStemmer
import es_core_news_sm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
import spacy
nlp = es_core_news_sm.load()
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)
df = pd.read_csv('/kaggle/input/the-best-sarcasm-annotated-dataset-in-spanish/sarcasmo.tsv',sep='\t')
df.head(3)
df.shape
feature = ['Locutor', 'Locución','Sarcasmo']
df = df[feature]
print(df.Sarcasmo.value_counts())
sns.countplot(x='Sarcasmo', data=df)
text_length=df['Locución'].str.len()
sns.distplot(text_length)
plt.show()
stop=set(stopwords.words('english'))

def build_list(df,col="Locución"):
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
plt.title("palabra más común en texto")
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
plot_count('Locución', 'Top 20 Locución', df, 3.5)
stemmer = PorterStemmer()
def stem_text(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            word = stemmer.stem(i.strip())
            final_text.append(word)
    return " ".join(finan_text)
plt.figure(figsize = (20, 20))
wc = WordCloud(background_color = 'lightgray', max_words=1500, width=1600,height = 1000 , stopwords = STOPWORDS).generate(" ".join(df.Locución))
plt.imshow(wc , interpolation = 'bilinear')
plt.title("Visualización De Recuento De Palabras De Locución")
df.head(3)
def clean(text):
    text = text.fillna("fillna").str.lower()
    text = text.map(lambda x: re.sub('\\n',' ',str(x)))
    text = text.map(lambda x: re.sub("\[\[User.*",'',str(x)))
    text = text.map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))
    text = text.map(lambda x: re.sub("\(http://.*?\s\(http://.*\)",'',str(x)))
    return text

text = clean(df['Locución'])
def text_pro(df):
    corpus=[]
    stem = PorterStemmer()
    lem=WordNetLemmatizer()

    for news in text:
        #df['Sarcasmo'].dropna()[:900]:
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
def text_entity(text):
    doc = nlp(text)
    for ent in doc.ents:
        print(f'Entity: {ent}, Label: {ent.label_}, {spacy.explain(ent.label_)}')
text_entity(df['Locución'][10])
first = df['Locución'][11]
doc = nlp(first)
spacy.displacy.render(doc, style='ent',jupyter=True)
first = df['Locución'][150]
doc = nlp(first)
spacy.displacy.render(doc, style='ent',jupyter=True)
sen1 = df['Locución'][75]
doc = nlp(sen1)
spacy.displacy.render(doc, style='ent',jupyter=True)

for idx, sentence in enumerate(doc.sents):
    for noun in sentence.noun_chunks:
        print(f"sentence {idx+1} has noun chunk '{noun}'")
distri = df['Locución'][75]
doc = nlp(distri)
options = {'compact': True, 'bg': '#09a3d5',
           'color': 'white', 'font': 'Trebuchet MS'}
spacy.displacy.render(doc, jupyter=True, style='dep', options=options)
distri = df['Locución'][250]
doc = nlp(distri)
options = {'compact': True, 'bg': '#09a3d5',
           'color': 'white', 'font': 'Trebuchet MS'}
spacy.displacy.render(doc, jupyter=True, style='dep', options=options)
distri = df['Locución'][370]
doc = nlp(distri)
options = {'compact': True, 'bg': '#09a3d5',
           'color': 'white', 'font': 'Trebuchet MS'}
spacy.displacy.render(doc, jupyter=True, style='dep', options=options)
for token in doc:
    print(f"token: {token.text},\t dep: {token.dep_},\t head: {token.head.text},\t pos: {token.head.pos_},\
    ,\t children: {[child for child in token.children]}")
df.head(2)
X = df.Locución
Y = df.Sarcasmo
le = LabelEncoder()
Y = le.fit_transform(Y)
#Y = Y.reshape(-1,1)
X = np.array(X)
vocab_size = 500
embedding_dim = 16
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 900 
training_sentences = X[0:training_size]
testing_sentences = X[training_size:]
training_labels = Y[0:training_size]
testing_labels = Y[training_size:]
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
num_epochs = 10
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=1)
import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
model.evaluate(training_padded,training_labels)[1]