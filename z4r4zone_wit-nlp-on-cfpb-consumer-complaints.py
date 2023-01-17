# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



# Handle table-like data and matrices

import numpy as np

import pandas as pd

from pprint import pprint



# Text processing

import nltk



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

plt.style.use('seaborn')

import seaborn as sns
def plot_categories( df , cat , target , **kwargs ):

    row = kwargs.get( 'row' , None )

    col = kwargs.get( 'col' , None )

    facet = sns.FacetGrid( df , row = row , col = col )

    facet.map( sns.barplot , cat , target  )

    facet.add_legend()

    

# NLTK Stop words

from nltk.corpus import stopwords

stop_words = stopwords.words('english')
# get CFPB complaints csv file as a DataFrame

df = pd.read_csv("/kaggle/input/Consumer_Complaints.csv", 

                 usecols=('Issue', 'Consumer complaint narrative'), 

                 nrows=100000)



print('Training Set Dataframe Shape: ', df.shape)



#lets see the data

df.head(5)
cnt_pro = df['Issue'].value_counts()



# Plot 'Product' by frequency

plt.figure(figsize=(24,6))

sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Issue', fontsize=1)

plt.xticks(rotation=90)

plt.show();
# let's see how many Null values we have:

df.isnull().sum()
# drop row if have Null in any column

df = df.dropna()

print('Dataset after removing Null values: ', df.shape)



df.head(10)
# rename column to 'narrative'

df.rename(columns = {'Consumer complaint narrative':'narrative'}, inplace = True)



# Convert to list

narrative = df.narrative.values.tolist()



# Let's see how our narrative text looks like before cleaning

pprint(narrative[:1])
from wordcloud import WordCloud



def plot_wordcloud(wordcloud):

    plt.figure(figsize=(12, 10))

    plt.imshow(wordcloud, interpolation = 'bilinear')

    plt.axis("off")

    plt.show()
wordcloud = WordCloud(max_font_size=None, max_words=200, background_color="white", 

                      width=5000, height=4000, stopwords=stop_words).generate(str(narrative))



plot_wordcloud(wordcloud)
# We will clean text using regex

import re 



# Remove spacial characters

narrative = [re.sub(r'[^\w\s]','',str(item)) for item in narrative]



# Remove distracting single quotes

narrative = [re.sub("\'", "", str(item)) for item in narrative]



# Remove masked data 'XXXX'

narrative = [re.sub("XXXX", "", str(item)) for item in narrative]
# Let's see how our narrative text looks after cleaning

pprint(narrative[:1])
def sent_to_words(sentences):

    for sentence in sentences:

        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations



narrative_words = list(sent_to_words(narrative))



pprint(narrative_words[:1])
# Build the bigram and trigram models

bigram = gensim.models.Phrases(narrative_words, min_count=5, threshold=100) # higher threshold fewer phrases.

trigram = gensim.models.Phrases(bigram[narrative_words], threshold=100)  



# Faster way to get a sentence clubbed as a trigram/bigram

bigram_mod = gensim.models.phrases.Phraser(bigram)

trigram_mod = gensim.models.phrases.Phraser(trigram)



# See trigram example

print(trigram_mod[bigram_mod[narrative_words[0]]])
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

narrative_words_nostops = remove_stopwords(narrative_words)



# Form Bigrams

narrative_words_bigrams = make_bigrams(narrative_words_nostops)



# Initialize spacy 'en' model, keeping only tagger component (for efficiency)

nlp = spacy.load('en', disable=['parser', 'ner'])



# Do lemmatization keeping only noun, adj, vb, adv

narrative_lemmatized = lemmatization(narrative_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])



print(narrative_lemmatized[:1])
# Create Dictionary

id2word = corpora.Dictionary(narrative_lemmatized)



# Create Corpus

texts = narrative_lemmatized



# Term Document Frequency

corpus = [id2word.doc2bow(text) for text in texts]



# View

print(corpus[:1])
# Human readable format of corpus (term-frequency)

[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
# Build LDA model

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

coherence_model_lda = CoherenceModel(model=lda_model, texts=narrative_lemmatized, dictionary=id2word, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()

print('\nCoherence Score: ', coherence_lda)
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):

    """

    Compute c_v coherence for various number of topics



    Parameters:

    ----------

    dictionary : Gensim dictionary

    corpus : Gensim corpus

    texts : List of input texts

    limit : Max num of topics



    Returns:

    -------

    model_list : List of LDA topic models

    coherence_values : Coherence values corresponding to the LDA model with respective number of topics

    """

    coherence_values = []

    model_list = []

    for num_topics in range(start, limit, step):

        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word)

        model_list.append(model)

        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')

        coherence_values.append(coherencemodel.get_coherence())



    return model_list, coherence_values
# Can take a long time to run...

model_list, coherence_values = compute_coherence_values(dictionary=id2word, 

                                                        corpus=corpus, 

                                                        texts=narrative_lemmatized, 

                                                        start=4, 

                                                        limit=30, 

                                                        step=2)
## Show graph

limit=30; start=4; step=2;

x = range(start, limit, step)

plt.plot(x, coherence_values)

plt.xlabel("Num Topics")

plt.ylabel("Coherence score")

plt.legend(("coherence_values"), loc='best')

plt.show()
# Print the coherence scores

for m, cv in zip(x, coherence_values):

    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
# Select the model and print the topics

optimal_model = model_list[1]

model_topics = optimal_model.show_topics(formatted=False)

pprint(optimal_model.print_topics(num_words=10))
# Visualize the topics

pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(optimal_model, corpus, id2word)

vis
#nltk.download('vader_lexicon')



# load the SentimentIntensityAnalyser object in

from nltk.sentiment.vader import SentimentIntensityAnalyzer



# assign it to another name to make it easier to use

analyzer = SentimentIntensityAnalyzer()
# use the polarity_scores() method to get the sentiment metrics

def print_sentiment_scores(sentence):

    snt = analyzer.polarity_scores(sentence)

    print("{:-<40} {}".format(sentence, str(snt)))
print_sentiment_scores("Women in Tech is a great event.")
print_sentiment_scores("Women in Tech is a GREAT event.")
print_sentiment_scores("Women in Tech is a GREAT event. :)")
print_sentiment_scores("Women in Tech is a GREAT event!!! :)")
# getting only the negative score

def negative_score(text):

    negative_value = analyzer.polarity_scores(text)['neg']

    return negative_value



# getting only the neutral score

def neutral_score(text):

    neutral_value = analyzer.polarity_scores(text)['neu']

    return neutral_value



# getting only the positive score

def positive_score(text):

    positive_value = analyzer.polarity_scores(text)['pos']

    return positive_value



# getting only the compound score

def compound_score(text):

    compound_value = analyzer.polarity_scores(text)['compound']

    return compound_value
df['sentiment_neg'] = df['narrative'].apply(negative_score)

df['sentiment_neu'] = df['narrative'].apply(neutral_score)

df['sentiment_pos'] = df['narrative'].apply(positive_score)

df['sentiment_compound'] = df['narrative'].apply(compound_score)
df.head(10)
# all scores in 4 histograms

fig, axes = plt.subplots(2, 2, figsize=(10,8))



# plot all 4 histograms

df.hist('sentiment_neg', bins=25, ax=axes[0,0], color='lightcoral', alpha=0.6)

axes[0,0].set_title('Negative Sentiment Score')

df.hist('sentiment_neu', bins=25, ax=axes[0,1], color='lightsteelblue', alpha=0.6)

axes[0,1].set_title('Neutral Sentiment Score')

df.hist('sentiment_pos', bins=25, ax=axes[1,0], color='chartreuse', alpha=0.6)

axes[1,0].set_title('Positive Sentiment Score')

df.hist('sentiment_compound', bins=25, ax=axes[1,1], color='navajowhite', alpha=0.6)

axes[1,1].set_title('Compound')



# plot common x- and y-label

fig.text(0.5, 0.04, 'Sentiment Scores',  fontweight='bold', ha='center')

fig.text(0.04, 0.5, 'Number of Complaints', fontweight='bold', va='center', rotation='vertical')



# plot title

plt.suptitle('Sentiment Analysis of CFPB Complaints\n\n', fontsize=12, fontweight='bold');
# full dataframe with POSITIVE comments

df_pos = df.loc[df.sentiment_compound >= 0.9]



# only corpus of POSITIVE comments

pos_comments = df_pos['narrative'].tolist()



# full dataframe with NEGATIVE comments

df_neg = df.loc[df.sentiment_compound < 0.0]



# only corpus of NEGATIVE comments

neg_comments = df_neg['narrative'].tolist()
# read some positive comments

pos_comments[2:3]
# read some positive comments

neg_comments[2:3]
df_pos['text_length'] = df_pos['narrative'].apply(len)

df_neg['text_length'] = df_neg['narrative'].apply(len)
sns.set_style("whitegrid")

plt.figure(figsize=(8,5))



sns.distplot(df_pos['text_length'], kde=True, bins=50, color='chartreuse')

sns.distplot(df_neg['text_length'], kde=True, bins=50, color='lightcoral')



plt.title('\nDistribution Plot for Length of Complaint\n')

plt.legend(['Positive Narratives', 'Negative Narratives'])

plt.xlabel('\nText Length')

plt.ylabel('Percentage of Comments\n');