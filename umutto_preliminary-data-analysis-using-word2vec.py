%matplotlib inline



# Data wrapper libraries

import pandas as pd

import numpy as np



# Visualization Libraries

import matplotlib.pyplot as plt

from matplotlib.patches import Patch

from matplotlib.markers import MarkerStyle

import seaborn as sns



# Text analysis helper libraries

from gensim.summarization import summarize

from gensim.summarization import keywords



# Text analysis helper libraries for word frequency etc..

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from string import punctuation



# Word cloud visualization libraries

from scipy.misc import imresize

from PIL import Image

from wordcloud import WordCloud, ImageColorGenerator

from collections import Counter



# Word2Vec related libraries

from gensim.models import KeyedVectors



# Dimensionaly reduction libraries

from sklearn.decomposition import PCA



# Clustering library

from sklearn.cluster import KMeans



# Set figure size a bit bigger than default so everything is easily red

plt.rcParams["figure.figsize"] = (11, 7)
df_variants = pd.read_csv('../input/training_variants').set_index('ID')

df_variants.head()
df_text = pd.read_csv('../input/training_text', sep='\|\|', engine='python', 

                      skiprows=1, names=['ID', 'Text']).set_index('ID')

df_text.head()
df = pd.concat([df_variants, df_text], axis=1)

df.head()
df['Variation'].describe()
plt.figure()

ax = df['Gene'].value_counts().plot(kind='area')



ax.get_xaxis().set_ticks([])

ax.set_title('Gene Frequency Plot')

ax.set_xlabel('Gene')

ax.set_ylabel('Frequency')



plt.tight_layout()

plt.show()
fig, axes = plt.subplots(nrows=3, ncols=3, sharey=True, figsize=(11,11))



# Normalize value counts for better comparison

def normalize_group(x):

    label, repetition = x.index, x

    t = sum(repetition)

    r = [n/t for n in repetition]

    return label, r



for idx, g in enumerate(df.groupby('Class')):

    label, val = normalize_group(g[1]["Gene"].value_counts())

    ax = axes.flat[idx]

    ax.bar(np.arange(5), val[:5],

           tick_label=label[:5]) 

    ax.set_title("Class {}".format(g[0]))

    

fig.text(0.5, 0.97, '(Top 5) Gene Frequency per Class', ha='center', fontsize=14, fontweight='bold')

fig.text(0.5, 0, 'Gene', ha='center', fontweight='bold')

fig.text(0, 0.5, 'Frequency', va='center', rotation='vertical', fontweight='bold')

fig.tight_layout(rect=[0.03, 0.03, 0.95, 0.95])
plt.figure()

ax = df['Class'].value_counts().plot(kind='bar')



ax.set_title('Class Distribution Over Entries')

ax.set_xlabel('Class')

ax.set_ylabel('Frequency')



plt.tight_layout()

plt.show()
df.drop(['Gene', 'Variation'], axis=1, inplace=True)



# Additionaly we will drop the null labeled texts too

df = df[df['Text'] != 'null']
t_id = 0

text = df.loc[t_id, 'Text']



word_scores = keywords(text, words=5, scores=True, split=True, lemmatize=True)

word_scores = ', '.join(['{}-{:.2f}'.format(k, s[0]) for k, s in word_scores])

summary = summarize(text, word_count=100)



print('ID [{}]\nKeywords: [{}]\nSummary: [{}]'.format(t_id, word_scores, summary))
custom_words = ["fig", "figure", "et", "al", "al.", "also",

                "data", "analyze", "study", "table", "using",

                "method", "result", "conclusion", "author", 

                "find", "found", "show", '"', "’", "“", "”"]



stop_words = set(stopwords.words('english') + list(punctuation) + custom_words)

wordnet_lemmatizer = WordNetLemmatizer()



class_corpus = df.groupby('Class').apply(lambda x: x['Text'].str.cat())

class_corpus = class_corpus.apply(lambda x: Counter(

    [wordnet_lemmatizer.lemmatize(w) 

     for w in word_tokenize(x) 

     if w.lower() not in stop_words and not w.isdigit()]

))
class_freq = class_corpus.apply(lambda x: x.most_common(5))

class_freq = pd.DataFrame.from_records(class_freq.values.tolist()).set_index(class_freq.index)



def normalize_row(x):

    label, repetition = zip(*x)

    t = sum(repetition)

    r = [n/t for n in repetition]

    return list(zip(label,r))



class_freq = class_freq.apply(lambda x: normalize_row(x), axis=1)



# set unique colors for each word so it's easier to read

all_labels = [x for x in class_freq.sum().sum() if isinstance(x,str)]

unique_labels = set(all_labels)

cm = plt.get_cmap('Blues_r', len(all_labels))

colors = {k:cm(all_labels.index(k)/len(all_labels)) for k in all_labels}



fig, ax = plt.subplots()



offset = np.zeros(9)

for r in class_freq.iteritems():

    label, repetition = zip(*r[1])

    ax.barh(range(len(class_freq)), repetition, left=offset, color=[colors[l] for l in label])

    offset += repetition

    

ax.set_yticks(np.arange(len(class_freq)))

ax.set_yticklabels(class_freq.index)

ax.invert_yaxis()



# annotate words

offset_x = np.zeros(9) 

for idx, a in enumerate(ax.patches):

    fc = 'k' if sum(a.get_fc()) > 2.5 else 'w'

    ax.text(offset_x[idx%9] + a.get_width()/2, a.get_y() + a.get_height()/2, 

            '{}\n{:.2%}'.format(all_labels[idx], a.get_width()), 

            ha='center', va='center', color=fc, fontsize=14, family='monospace')

    offset_x[idx%9] += a.get_width()

    

ax.set_title('Most common words in each class')

ax.set_xlabel('Word Frequency')

ax.set_ylabel('Classes')



plt.tight_layout()

plt.show()
whole_text_freq = class_corpus.sum()



fig, ax = plt.subplots()



label, repetition = zip(*whole_text_freq.most_common(25))



ax.barh(range(len(label)), repetition, align='center')

ax.set_yticks(np.arange(len(label)))

ax.set_yticklabels(label)

ax.invert_yaxis()



ax.set_title('Word Distribution Over Whole Text')

ax.set_xlabel('# of repetitions')

ax.set_ylabel('Word')



plt.tight_layout()

plt.show()