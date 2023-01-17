# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/yelp-dataset'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
chunks = pd.read_json('../input/yelp-dataset/yelp_academic_dataset_review.json', lines=True, orient='records', chunksize=500)
reviews = []

for chunk in chunks:

    reviews.append(chunk)



reviews_df = pd.concat(reviews)

reviews_df.head()



del reviews



reviews_df = reviews_df[['business_id', 'stars', 'useful', 'funny', 'cool', 'text', 'date']]



reviews_df.head()
reviews_df.shape
import nltk; nltk.download('stopwords')
!pip install pyLDAvis
import re

import numpy as np

import pandas as pd

from pprint import pprint



# Gensim

import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel



# spacy for lemmatization

import spacy



# Plotting tools

import pyLDAvis

import pyLDAvis.gensim  # don't skip this

import matplotlib.pyplot as plt

%matplotlib inline



# Enable logging for gensim - optional

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)



import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)
# NLTK Stop words

from nltk.corpus import stopwords

stop_words = stopwords.words('english')

stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
data1=reviews_df[reviews_df['stars']==1]
data1.shape
df=data1.sample(100000)
df.shape
x=df['text']
x.head()
def clean_text(df):

    data_words = list()

    lines = df.values.tolist()

    for text in lines:

        text = text.lower()

        pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

        text = pattern.sub('', text)

        text = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", text)

        

        words = ' '.join(text)

        data_words.append(words)

    return data_words
data=clean_text(x)
data[1]
# Tokenize

data = df.text.values.tolist()

def sent_to_words(sentences):

    for sentence in sentences:

        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations



data_words = list(sent_to_words(data))



print(data_words[:1])
# Build the bigram and trigram models

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.

trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  



# Faster way to get a sentence clubbed as a trigram/bigram

bigram_mod = gensim.models.phrases.Phraser(bigram)

trigram_mod = gensim.models.phrases.Phraser(trigram)



# See trigram example

print(trigram_mod[bigram_mod[data_words[0]]])
# Define functions for stopwords, bigrams, trigrams and lemmatization

def remove_stopwords(texts):

    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]



def make_bigrams(texts):

    return [bigram_mod[doc] for doc in texts]



def make_trigrams(texts):

    return [trigram_mod[bigram_mod[doc]] for doc in texts]



def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    """https://spacy.io/api/annotation"""

    texts_out = []

    for sent in texts:

        doc = nlp(" ".join(sent)) 

        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    return texts_out
# Remove Stop Words

data_words_nostops = remove_stopwords(data_words)



# Form Bigrams

data_words_bigrams = make_bigrams(data_words_nostops)



# Initialize spacy 'en' model, keeping only tagger component (for efficiency)

# python3 -m spacy download en

nlp = spacy.load('en', disable=['parser', 'ner'])



# Do lemmatization keeping only noun, adj, vb, adv

data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])



print(data_lemmatized[:1])
# Create Dictionary

id2word = corpora.Dictionary(data_lemmatized)



# Create Corpus

texts = data_lemmatized



# Term Document Frequency

corpus = [id2word.doc2bow(text) for text in texts]



# View

print(corpus[:1])
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,

                                           id2word=id2word,

                                           num_topics=10, 

                                           random_state=100,

                                           update_every=1,

                                           chunksize=100,

                                           passes=10,

                                           alpha='auto',

                                           per_word_topics=True)
# Print the Keyword in the 10 topics

pprint(lda_model.print_topics())

doc_lda = lda_model[corpus]
# Compute Perplexity

print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.



# Compute Coherence Score

coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()

print('\nCoherence Score: ', coherence_lda)
# Dominant Topic

def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data):

    # Init output

    sent_topics_df = pd.DataFrame()



    # Get main topic in each document

    for i, row_list in enumerate(ldamodel[corpus]):

        row = row_list[0] if ldamodel.per_word_topics else row_list            

        # print(row)

        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # Get the Dominant topic, Perc Contribution and Keywords for each document

        for j, (topic_num, prop_topic) in enumerate(row):

            if j == 0:  # => dominant topic

                wp = ldamodel.show_topic(topic_num)

                topic_keywords = ", ".join([word for word, prop in wp])

                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)

            else:

                break

    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']



    # Add original text to the end of the output

    contents = pd.Series(texts)

    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    return(sent_topics_df)





df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_lemmatized)



# Format

df_dominant_topic = df_topic_sents_keywords.reset_index()

df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

df_dominant_topic.head(10)
# Display setting to show more characters in column

pd.options.display.max_colwidth = 100



sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')



for i, grp in sent_topics_outdf_grpd:

    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 

                                             grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 

                                            axis=0)



# Reset Index    

sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)



# Format

sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]



# Show

sent_topics_sorteddf_mallet.head(10)
#  Frequency Distribution of Word Counts in Documents

doc_lens = [len(d) for d in df_dominant_topic.Text]



# Plot

plt.figure(figsize=(16,7), dpi=160)

plt.hist(doc_lens, bins = 100, color='navy')

plt.text(750, 100, "Mean   : " + str(round(np.mean(doc_lens))))

plt.text(750,  90, "Median : " + str(round(np.median(doc_lens))))

plt.text(750,  80, "Stdev   : " + str(round(np.std(doc_lens))))

plt.text(750,  70, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))

plt.text(750,  60, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))



plt.gca().set(xlim=(0, 1000), ylabel='Number of Documents', xlabel='Document Word Count')

plt.tick_params(size=16)

plt.xticks(np.linspace(0,1000,9))

plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))

plt.show()
# Word Clouds of Top N Keywords in Each Topic



# 1. Wordcloud of Top N words in each topic

from matplotlib import pyplot as plt

from wordcloud import WordCloud, STOPWORDS

import matplotlib.colors as mcolors



cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'



cloud = WordCloud(stopwords=stop_words,

                  background_color='white',

                  width=2500,

                  height=1800,

                  max_words=10,

                  colormap='tab10',

                  color_func=lambda *args, **kwargs: cols[i],

                  prefer_horizontal=1.0)



topics = lda_model.show_topics(formatted=False)



fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)



for i, ax in enumerate(axes.flatten()):

    fig.add_subplot(ax)

    topic_words = dict(topics[i][1])

    cloud.generate_from_frequencies(topic_words, max_font_size=300)

    plt.gca().imshow(cloud)

    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))

    plt.gca().axis('off')





plt.subplots_adjust(wspace=0, hspace=0)

plt.axis('off')

plt.margins(x=0, y=0)

plt.tight_layout()

plt.show()
pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

vis