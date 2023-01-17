# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from pathlib import Path

import os

import glob



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.io.json import json_normalize

import json



import pprint

import string



import matplotlib.pyplot as plt

import seaborn as sns



import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# for dirname, _, filenames in os.walk('C:/Users/trivikram.cheedella/OneDrive - JD Power/Data Science Data/CORD-19-research-challenge'):

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        # print(os.path.join(dirname, filename))

        pass



# Any results you write to the current directory are saved as output.
input = Path('/kaggle/input/CORD-19-research-challenge')

output = Path('/kaggle/output')

biorxiv_medrxiv = Path('/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv')
df_all_sources_metadata = pd.read_csv(input / 'metadata.csv')
print(df_all_sources_metadata.shape)

df_all_sources_metadata.info()
df_all_sources_metadata.head(3)
pd.pivot_table(df_all_sources_metadata, 

               index='full_text_file', 

               values=['cord_uid','sha', 'source_x', 'has_pdf_parse', 'has_pmc_xml_parse'], 

               aggfunc={'cord_uid': 'count','sha': 'count', 'source_x': 'count', 'has_pdf_parse': np.sum, 'has_pmc_xml_parse': np.sum}, 

               margins=True)
%%time

all_json = glob.glob(f'{biorxiv_medrxiv}/**/*.json', recursive=True)

len(all_json)
class FileReader:

    def __init__(self, file_path):

        with open(file_path) as file:

            content = json.load(file)

            self.paper_id = content['paper_id']

            self.abstract = []

            self.body_text = []

            # Abstract

            for entry in content['abstract']:

                self.abstract.append(entry['text'])

            # Body text

            for entry in content['body_text']:

                self.body_text.append(entry['text'])

            self.abstract = '\n'.join(self.abstract)

            self.body_text = '\n'.join(self.body_text)

            # Extend Here

            #

            #

    def __repr__(self):

        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'

first_row = FileReader(all_json[0])

print(first_row)
%%time

dict_ = {'paper_id': [], 'abstract': [], 'body_text': []}

for idx, entry in enumerate(all_json):

    if idx % (len(all_json) // 10) == 0:

        print(f'Processing index: {idx} of {len(all_json)}')

    content = FileReader(entry)

    dict_['paper_id'].append(content.paper_id)

    dict_['abstract'].append(content.abstract)

    dict_['body_text'].append(content.body_text)

df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text'])

df_covid.head()
dict_ = None
%%time

df_covid['abstract_word_count'] = df_covid['abstract'].apply(lambda x: len(x.strip().split()))

df_covid['body_word_count'] = df_covid['body_text'].apply(lambda x: len(x.strip().split()))

df_covid.head()
df_covid.describe(include='all').T
df_covid.drop_duplicates(['body_text'], inplace=True)

df_covid.describe(include='all').T
import nltk

from nltk.corpus import stopwords

from nltk.probability import FreqDist

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from wordcloud import STOPWORDS
lemmatizer = WordNetLemmatizer()



print("Number of stopwrods from STOPWORDS: ", len(STOPWORDS))

print("Number of stopwrods from stopwords.words('english'): ", len(stopwords.words('english')))

other_stopwords = ['q', 'license', 'preprint', 'copyright', 'http', 'doi', 'preprint', 'copyright', 

                   'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 'rights', 

                   'reserved', 'permission', 'used', 'using', 'biorxiv', 'fig', 'fig.', 'al.', 

                   'di', 'la', 'il', 'del', 'le', 'della', 'dei', 'delle', 'una', 'da',  'dell',  

                   'non', 'si', 'cc']



custom_stopwords = []

custom_stopwords = list(set(stopwords.words('english') + list(STOPWORDS))) + other_stopwords



print("Number of stopwrods from custom_stopwords: ", len(custom_stopwords))

print(custom_stopwords[-25:])
def clean_the_text(text):

        text = re.sub('[^a-zA-Z0-9-]', ' ', text)

        tokens = word_tokenize(text)

        # remove_punc = [word for word in tokens if word not in string.punctuation]

        remove_stopwords = [word.lower() for word in tokens if word.lower() not in custom_stopwords]

        more_than_three = [w for w in remove_stopwords if len(w)>3]

        lem = [lemmatizer.lemmatize(w) for w in more_than_three]

        return ' '.join(lem)

df_covid_for_nlp = df_covid.copy()

df_covid = None
%%time

df_covid_for_nlp['cleaned_text'] = df_covid_for_nlp['body_text'].apply(lambda x: clean_the_text(x))
df_covid_for_nlp['cleaned_text'].head()
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import LatentDirichletAllocation
%%time 



tfidf_vect = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')

doc_term_matrix_Tfidf = tfidf_vect.fit_transform(df_covid_for_nlp['cleaned_text'].values.astype('U'))
doc_term_matrix_Tfidf
%%time

# Define Search Param

search_params = {'n_components': [10, 15, 20, 25, 30, 50], 'learning_decay': [.5, .7, .9]}



# Init the Model

lda = LatentDirichletAllocation()



# Init Grid Search Class

grid_search_model = GridSearchCV(lda, param_grid=search_params, n_jobs=-1)



# Do the Grid Search

grid_search_model.fit(doc_term_matrix_Tfidf)
# Best Model

best_lda_model = grid_search_model.best_estimator_



# Model Parameters

print("Best Model's Params: ", grid_search_model.best_params_)



# Log Likelihood Score

print("Best Log Likelihood Score: ", grid_search_model.best_score_)



# Perplexity

print("Model Perplexity: ", best_lda_model.perplexity(doc_term_matrix_Tfidf))
df_results = pd.DataFrame(grid_search_model.cv_results_)



current_palette = sns.color_palette("Set2", 3)



plt.figure(figsize=(12,8))



sns.lineplot(data=df_results,

             x='param_n_components',

             y='mean_test_score',

             hue='param_learning_decay',

             palette=current_palette,

             marker='o')



plt.show()
import random



for i in range(10):

    random_id = random.randint(0,len(tfidf_vect.get_feature_names()))

    print(tfidf_vect.get_feature_names()[random_id])
first_topic = best_lda_model.components_[0]
top_topic_words = first_topic.argsort()[-10:]

top_topic_words
for i in top_topic_words:

    print(tfidf_vect.get_feature_names()[i])
for i,topic in enumerate(best_lda_model.components_):

    print(f'Top 10 words for topic #{i}:')

    print([tfidf_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])

    print('\n')
topic_values_tfidf = best_lda_model.transform(doc_term_matrix_Tfidf)

topic_values_tfidf.shape
topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]



rows = df_covid_for_nlp['paper_id']
df_topic_values_tfidf = pd.DataFrame(topic_values_tfidf, columns=topicnames, index=rows)
df_topic_values_tfidf['topic_number_tfidf'] = topic_values_tfidf.argmax(axis=1)
# Styling

def color_green(val):

    color = 'green' if val > .1 else 'black'

    return 'color: {col}'.format(col=color)



def make_bold(val):

    weight = 700 if val > .1 else 400

    return 'font-weight: {weight}'.format(weight=weight)
df_topic_values_tfidf.head(10).style.applymap(color_green).applymap(make_bold)
dict_topic = {'topic_number_tfidf': [], 'topic_words_tfidf': []}



for i,topic in enumerate(best_lda_model.components_):

    dict_topic['topic_number_tfidf'].append(i)

    dict_topic['topic_words_tfidf'].append([tfidf_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])



df_covid_topics_tfidf = pd.DataFrame(dict_topic, columns=['topic_number_tfidf', 'topic_words_tfidf'])

df_covid_topics_tfidf.head(10)
import pyLDAvis

import pyLDAvis.sklearn
%%time



pyLDAvis.enable_notebook()

panel = pyLDAvis.sklearn.prepare(best_lda_model, doc_term_matrix_Tfidf, tfidf_vect, mds='tsne', sort_topics=False)

panel
df_covid_for_nlp['topic_number_tfidf'] = topic_values_tfidf.argmax(axis=1)
df_covid_for_nlp.head(3)
df_covid_for_nlp['topic_number_tfidf'].value_counts()
df_covid_for_nlp = df_covid_for_nlp.merge(df_covid_topics_tfidf,

                                          how='left', 

                                          left_on='topic_number_tfidf', 

                                          right_on='topic_number_tfidf')
df_covid_for_nlp.head(3)
df_covid_for_nlp.columns
df_all_sources_metadata_with_topics = df_all_sources_metadata.copy()

df_all_sources_metadata_with_topics.shape
df_all_sources_metadata_with_topics = df_all_sources_metadata.merge(

    df_covid_for_nlp[['paper_id', 'abstract_word_count', 'body_word_count', 'cleaned_text', 'topic_number_tfidf', 'topic_words_tfidf']], 

    how='left', 

    left_on='sha', 

    right_on='paper_id')
print(df_all_sources_metadata_with_topics.columns)

print(df_all_sources_metadata_with_topics.shape)

df_all_sources_metadata_with_topics.head()
# We can export the data for further analysis by executing the following code.

# df_all_sources_metadata_with_topics.to_csv(output / 'df_all_sources_metadata_with_topics_biorxiv.csv', index = False)