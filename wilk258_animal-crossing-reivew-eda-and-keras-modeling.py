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
import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from wordcloud import WordCloud,STOPWORDS
from nltk.corpus import stopwords
import re,string,unicodedata
from collections import  Counter
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer,PorterStemmer
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)
item = pd.read_csv('/kaggle/input/animal-crossing/items.csv')
review = pd.read_csv('/kaggle/input/animal-crossing/user_reviews.csv')
critic = pd.read_csv('/kaggle/input/animal-crossing/critic.csv')
villagers = pd.read_csv('/kaggle/input/animal-crossing/villagers.csv')
plt.rcParams['figure.figsize'] = (15, 8)
sns.countplot(y=item['name'].sort_values().tail(15), palette = 'hsv')
plt.title('Distribution of Item Name Top15', fontsize = 20)
plt.show()
print('category:',item['category'].value_counts())
plt.rcParams['figure.figsize'] = (15, 8)
sns.countplot(y=item['category'], palette = 'hsv')
plt.title('Distribution of Category', fontsize = 20)
plt.show()
sns.countplot(x='grade', data=review)
print('date count:', review.date.value_counts())
sns.countplot(y='date', data=review)
sns.distplot(review['text'].str.len())
plt.title('Review Text Length Distribution')
plt.show()
sns.distplot(critic['text'].str.len())
plt.title('Critic Text Length Distribution')
plt.show()
def clean(text):
    text = text.fillna("fillna").str.lower()
    text = text.map(lambda x: re.sub('\\n',' ',str(x)))
    text = text.map(lambda x: re.sub("\[\[User.*",'',str(x)))
    text = text.map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))
    text = text.map(lambda x: re.sub("\(http://.*?\s\(http://.*\)",'',str(x)))
    return text
text = clean(review['text'])
critic_text= clean(critic['text'])
stemmer = PorterStemmer()
def stem_text(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            word = stemmer.stem(i.strip())
            final_text.append(word)
    return " ".join(final_text) 
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(text))
plt.imshow(wc , interpolation = 'bilinear')
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(critic_text))
plt.imshow(wc , interpolation = 'bilinear')
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
plot_count('text', 'Top 20 Review', review, 3.5)
plot_count('text', 'Top 20 Crtiic', critic, 3.5)
stop=set(stopwords.words('english'))

def build_list(df,col="text"):
    corpus=[]
    lem=WordNetLemmatizer()
    stop=set(stopwords.words('english'))
    new= df[col].dropna().str.split()
    new=new.values.tolist()
    corpus=[lem.lemmatize(word.lower()) for i in new for word in i if(word) not in stop]
    
    return corpus
corpus=build_list(review)
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
plt.title("most common word in text")
corpus=build_list(critic,"text")
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
plt.title("most common word in critic text")
label_grade  = review.grade.mean()
review.loc[review['grade'] <= label_grade, 'label'] = 0
review.loc[review['grade'] > label_grade, 'label'] = 1

review.head(3)
review.shape
sentences = np.array(review['text'])
labels = np.array(review['label'])
vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 2000
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]
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