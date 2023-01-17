import pandas as pd 
import numpy as np 
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

#nltk.download("stopwords")
#nltk.download('wordnet')
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from gensim.models import word2vec
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import json
import os

import warnings
warnings.filterwarnings("ignore")
# load the meta data from the CSV file 
df=pd.read_csv("../input/covid19-json-to-csv-file/df.csv")
print (df.shape)

df["abstract"] = df["abstract"].str.lower()
df['title'] = df['title'].str.lower()
df['full_text'] = df['full_text'].str.lower()
#show 10 lines of the new dataframe
print (df.shape)
metadata=pd.read_csv("../input/CORD-19-research-challenge/metadata.csv", usecols=['title','abstract'])
metadata["abstract"] = metadata["abstract"].str.lower()
metadata['title'] = metadata['title'].str.lower()
print(metadata.shape)
papers = pd.merge(df, metadata, how = 'left')
papers
papers=papers.dropna()
papers
stop = set(stopwords.words('english'))
stop |= set(['title','abstract','preprint','biorxiv','read','author','funder','copyright','holder','https','license','et','al','may',
             'also','medrxiv','granted','reuse','rights','used','reserved','peer','holder','figure','fig','table','doi','within'])
lemmatizer = WordNetLemmatizer()
def data_preprocessing(text):
    text = ' '.join(re.sub('https?://\S+|www\.\S+','',text).split())
    text = text.replace('\n', '')
    text = re.sub("[!@#$+%*:()/<.=,—']", '', text)
    text = ' '.join([word for word in text.split() if word not in stop])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text
papers['title'] = papers['title'].apply(lambda x: data_preprocessing(x))
papers['abstract'] = papers['abstract'].apply(lambda x: data_preprocessing(x))
papers['full_text'] = papers['full_text'].apply(lambda x: data_preprocessing(x))
papers.reset_index()
contentCorpus = papers.full_text.values
plt.figure(figsize = (12, 8))
wordcloud = WordCloud(width = 3000,height = 2000,background_color="white",max_words=1000).generate(str(contentCorpus))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Figure 1. Full_text Corpus Word Cloud')
contentCorpus = papers.abstract.values
plt.figure(figsize = (12, 8))
wordcloud = WordCloud(width = 3000,height = 2000,background_color="white",max_words=1000).generate(str(contentCorpus))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Figure 2. Abstract Corpus Word Cloud')
contentCorpus = papers.title.values
plt.figure(figsize = (12, 8))
wordcloud = WordCloud(width = 3000,height = 2000,background_color="white",max_words=10000).generate(str(contentCorpus))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Figure 3. Title Corpus Word Cloud')
papers['virus'] = np.where(papers.full_text.str.contains('covid-19|covid|wuhan'), 'covid-19',
              np.where(papers.full_text.str.contains('alphacoronavirus|alpha-cov'), 'alphacoronavirus',
              np.where(papers.full_text.str.contains('betacoronavirus|mers|mers-cov|sars|sars-cov|sars-cov2'), 'betacoronavirus',
              np.where(papers.full_text.str.contains('gammacoronavirus|ibv'), 'gammacoronavirus',
              "None"))))
papers['virus'].value_counts()
plt.figure(figsize = (16, 8))
ax = sns.countplot(x="virus", data=papers)
ax.set_title('Figure 4. Distribution of different virus covered in the papers')
plt.xticks(rotation=45)
papers['topic'] = np.where(papers.abstract.str.contains('transmission|transmitting'), 'transmission',
              np.where(papers.abstract.str.contains('incubation'), 'incubation',
              np.where(papers.abstract.str.contains('vaccines|vaccine|vaccination|therapeutics|therapeutic|drug|drugs'), 'vaccines|therapeutics',
              np.where(papers.abstract.str.contains('gene|origin|evolution|genetics|genomes|genomic'), 'genetics|origin|evolution',
              np.where(papers.abstract.str.contains('npi|npis|interventions|distancing|isolating|isolation|isolate|mask'), 'non-pharmaceutical interventions',
              np.where(papers.abstract.str.contains('ards|ecmo|respirators|eua|clia|ventilation|cardiomyopathy|ai'), 'medical care',
              np.where(papers.abstract.str.contains('ethical|social|media|rumor|misinformation|ethics|multidisciplinary'), 'ethical|social',
              "None")))))))
papers['topic'].value_counts()
plt.figure(figsize = (12, 8))
ax = sns.countplot(x="topic", data=papers)
ax.set_title('Figure 5. Distribution of different topics covered in the matadata')
plt.xticks(rotation=30)
tokenized_sentences_title = [sentence.split() for sentence in papers['title'].values]
tokenized_sentences_abstract = [sentence.split() for sentence in papers['abstract'].values]
tokenized_sentences_full_text = [sentence.split() for sentence in papers['full_text'].values]
papers['title_tokenized'] = tokenized_sentences_title
papers['abstract_tokenized'] = tokenized_sentences_abstract
papers['full_text_tokenized'] = tokenized_sentences_full_text
model = word2vec.Word2Vec(tokenized_sentences_abstract, size = 100, min_count=1)
def buildWordVector(word_list, size):
    #function to average all words vectors in a given paragraph
    vec = np.zeros(size)
    count = 0.
    for word in word_list:
        if word in model.wv:
            vec += model.wv[word]
            count += 1.
    if count != 0:
        vec /= count
    return vec
papers['title_embedding'] = papers['title_tokenized'].apply(lambda x: buildWordVector(x, size = 100))
papers['abstract_embedding'] = papers['abstract_tokenized'].apply(lambda x: buildWordVector(x, size = 100))

papers.head(10)
def embedding_query(query):
    query = query.split(' ')
    query_vec = np.zeros(100).reshape((1,100))
    count = 0
    for word in query:
        if word in model.wv:
            query_vec += model.wv[word]
            count += 1.
    if count != 0:
        query_vec /= count
    return query_vec
# reference: https://www.kaggle.com/mathijs02/recommend-a-paper-by-using-word-embeddings
def get_similarity(query,n_top):
    query_vec = embedding_query(query)
    papers["cos_sim"] = papers['abstract_embedding'].apply(
        lambda x: metrics.pairwise.cosine_similarity(
            [x],query_vec.reshape(1,-1))[0][0])
    top_list = (papers.sort_values("cos_sim", ascending=False)
                [["title","abstract","cos_sim"]]
                .drop_duplicates()[:n_top])
    return top_list
get_similarity('transmission incubation in human ',10)
get_similarity('risk covid-19',10)
get_similarity('covid-19 genetics origin evolution',10)
get_similarity('drugs medicine to treat covid-19 patients',10)