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
df_all_sources_metadata = pd.read_csv(input / 'metadata.csv')
print(df_all_sources_metadata.shape)

df_all_sources_metadata.info()
df_all_sources_metadata.head(3)
pd.pivot_table(df_all_sources_metadata, 

               index='full_text_file', 

               values=['cord_uid','sha', 'source_x', 'has_pdf_parse', 'has_pmc_xml_parse', 'abstract'], 

               aggfunc={'cord_uid': 'count','sha': 'count', 'source_x': 'count', 'has_pdf_parse': np.sum, 'has_pmc_xml_parse': np.sum, 'abstract': 'count'}, 

               dropna = False,

               margins=True)
df_all_sources_metadata.describe(include='all').T
df_all_sources_metadata_deduped = df_all_sources_metadata.copy()

print(df_all_sources_metadata_deduped.shape)

df_all_sources_metadata_deduped.dropna(axis=0, subset=['abstract'], inplace=True)

df_all_sources_metadata_deduped.drop_duplicates(['abstract'], inplace=True)

df_all_sources_metadata_deduped.describe(include='all').T
%%time

df_all_sources_metadata_deduped['abstract_word_count'] = df_all_sources_metadata_deduped['abstract'].apply(lambda x: len(x.strip().split()))
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

%%time

df_all_sources_metadata_deduped['abstract_cleaned_text'] = df_all_sources_metadata_deduped['abstract'].apply(lambda x: clean_the_text(x))
df_all_sources_metadata_deduped['abstract_cleaned_text'].head()
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import LatentDirichletAllocation
%%time 



tfidf_vect = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')

doc_term_matrix_Tfidf = tfidf_vect.fit_transform(df_all_sources_metadata_deduped['abstract_cleaned_text'].values.astype('U'))
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



rows = df_all_sources_metadata_deduped['cord_uid']
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
df_all_sources_metadata_deduped['topic_number_tfidf'] = topic_values_tfidf.argmax(axis=1)
df_all_sources_metadata_deduped['topic_number_tfidf'].value_counts()
df_all_sources_metadata_deduped['year'] = df_all_sources_metadata_deduped['publish_time'].str.split('-', n=1, expand=True)[0]
df_all_sources_metadata_deduped['year'].value_counts().head(10)
df_all_sources_metadata_deduped = df_all_sources_metadata_deduped.merge(df_covid_topics_tfidf,

                                                                        how='left', 

                                                                        left_on='topic_number_tfidf', 

                                                                        right_on='topic_number_tfidf')
df_all_sources_metadata_deduped.head(3)
# We can export the data for further analysis by executing the following code.

# df_all_sources_metadata_deduped.to_csv(output / 'df_all_sources_metadata_deduped.csv', index = False)