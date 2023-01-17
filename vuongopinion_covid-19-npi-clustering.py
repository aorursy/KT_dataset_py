import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import glob

import json

import pickle
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 50)

pd.set_option('display.width', 100)

pd.set_option('display.max_colwidth', -1)
# Load metadata from Kaggle

root_path = '/kaggle/input/CORD-19-research-challenge'

metadata_path = f'{root_path}/metadata.csv'

meta_df = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})

meta_df = meta_df.drop_duplicates().reset_index(drop=True)
# Clean numbers and special characters

regex_arr = ['\d','\t','\r','\)','\(','\/', ':', ';', '&', '#', '-', '\.']

def clean_numbers_and_special_characters (df, col):

    meta_df[col] = meta_df[col].replace(regex=regex_arr, value='')

    meta_df[col] = meta_df[col].str.strip()

    meta_df[col] = meta_df[col].str.lower()

    return meta_df    



meta_df = clean_numbers_and_special_characters(meta_df, 'title')

meta_df = clean_numbers_and_special_characters(meta_df, 'abstract')

meta_df = clean_numbers_and_special_characters(meta_df, 'authors')
# Obtain file paths for all .json files

all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

len(all_json)
# Remove .xml.json files because most of the these files have duplicate information compared to the .json files

all_json = [x for x in all_json if 'xml.json' not in x ]

len(all_json)
# Method is needed to group doi into a list to avoid duplicate rows for title, abstract, and body_text

def agg_doi(df):

    no_doi = list(df.copy())

    no_doi.remove('doi')

    df_doi = df.groupby(no_doi)['doi'].apply(list).reset_index(name = "doi")

    return df_doi
class FileReader:

    def __init__(self, all_json):

        self.all_json = all_json

    

    def extract_all_json(self, meta_df):

        meta_df = meta_df.fillna('')

        sha_meta = agg_doi(meta_df[['sha', 'doi']])

        pmc_meta = agg_doi(meta_df[['pmcid', 'doi']])

        json_dict = []

        all_json_div = len(all_json) // 10

        for file_path in all_json:

            with open(file_path) as file:

                if len(json_dict) % all_json_div == 0:

                    print(f'{len(json_dict)} of {len(all_json)} json files processed')

                data = json.load(file)

                paper_id = data['paper_id']

                title = data['metadata']['title'].lower().strip()

                author_list = self._extract_name_list(data)

                body_text = self._extract_json_text(data, 'body_text')

                abstract= self._extract_json_text(data, 'abstract')

                doi = []

                if len(meta_df[meta_df.sha == paper_id]):

                    doi = sha_meta[sha_meta.sha == paper_id]['doi'].tolist()[0]

                    if not title:

                        if len(meta_df[(meta_df.sha == paper_id) & (~meta_df.abstract.isna())]):

                            title = meta_df[(meta_df.sha == paper_id) & (~meta_df.title.isna())]['title'].tolist()[0]

                    if not abstract:

                        if len(meta_df[(meta_df.sha == paper_id) & (~meta_df.abstract.isna())]):

                            abstract = meta_df[(meta_df.sha == paper_id) & (~meta_df.abstract.isna())]['abstract'].tolist()[0]

                elif len(meta_df[meta_df.pmcid == paper_id]):

                    doi = pmc_meta[pmc_meta.pmcid == paper_id]['doi'].tolist()[0]

                    if not title:

                        if len(meta_df[(meta_df.pmcid == paper_id) & (~meta_df.title.isna())]):

                            title = meta_df[(meta_df.pmcid == paper_id) & (~meta_df.title.isna())]['title'].tolist()[0]

                    if not abstract:

                        if len(meta_df[(meta_df.pmcid == paper_id) & (~meta_df.title.isna())]):

                            abstract = meta_df[(meta_df.pmcid == paper_id) & (~meta_df.title.isna())]['abstract'].tolist()[0]

                json_dict.append({'paper_id': paper_id,

                                  'title': title,

                                  'author_list': author_list,

                                  'abstract': abstract,

                                  'body_text': body_text,

                                  'doi': doi})

        return pd.DataFrame(json_dict)



    def _extract_name_list(self, data):

        name_list = []

        for a in data['metadata']['authors']:

            first = a['first']

            middle = ''

            if a['middle']:

                middle = a['middle'][0] + ' '

            last = a['last']

            name_list.append(f'{first} {middle}{last}')

        return name_list

    

    def _extract_json_text(self, data, key):

        return ' '.join(str(item['text']).lower().strip() for item in data[key])



files = FileReader(all_json)

df_covid = files.extract_all_json(meta_df)
keywords = ['incident command system',

            'emergency operations',

            'joint information center',

            'social distancing',

            'childcare closers',

            'travel advisory',

            'travel warning',

            'isolation',

            'quarantine',

            'mass gathering cancellations',

            'school closures',

            'facility closures'

            'evacuation',

            'relocation',

            'restricting travel',

            'travel ban',

            'patient cohort',

            'npi']
all_json_df = df_covid[df_covid['abstract'].str.contains('|'.join(keywords), na=False, regex=True)].reset_index(drop=True)
len(all_json_df)
all_json_df.to_pickle('data.pkl')
# Run this cell to read a previously created pickle file instead of re-running the script

# all_json_df = pd.read_pickle('/kaggle/working/data.pkl')
import nltk

nltk.download('punkt')

from nltk.corpus import stopwords

from nltk.tokenize import sent_tokenize, word_tokenize

from collections import Counter

import re
def remove_punc(df, columns):

    for col in columns:

        df[col] = df[col].str.replace('[^a-zA-Z\s]+','')

    return df
def remove_stopwords(df, columns):

    stop = stopwords.words('english')

    for col in columns:

        df[col] = df[col].astype(str).apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    return df
all_json_df = remove_punc(all_json_df, ['body_text','abstract'])

all_json_df = remove_stopwords(all_json_df, ['body_text','abstract'])
from sklearn.feature_extraction.text import TfidfVectorizer
def to_tfidf(df, columns):

    for col in columns:

        tfidfv = TfidfVectorizer()

        df[col + '_tfidf'] = list(tfidfv.fit_transform(df[col]).toarray())

    return df
all_json_df = to_tfidf(all_json_df, ['body_text','abstract'])
def tokenize(row):

    title_tokens = []

    title = row['title']

    if title == title:

        title = re.sub('(/|\|:|&|#|-|\.)', '', title)

        tokens = word_tokenize(title)

        remove_sw = [word for word in tokens if word not in stopwords.words('english')]

        remove_numbers = [word for word in remove_sw if not word.isnumeric()]

        remove_comas = [word for word in remove_numbers if not word in [',', '(', ')', '"', ':', '``', '.', '?']]

        title_tokens.extend(remove_comas)

    return [value[0] for value in Counter(title_tokens).most_common()[0:30]]
all_json_df['tokens'] = all_json_df.apply(tokenize, axis=1)
model_df = all_json_df.copy()
LABELED_FILE = '/kaggle/input/labeled-data/labeled_npi.csv'

df_labels = pd.read_csv(LABELED_FILE)
model_df = model_df.merge(df_labels, on="title", how="inner")

model_df = model_df.loc[model_df['isNPI'].notna()]
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
def pca_apply(df, columns, n_comp):

    new_df = df.copy()

    for col in columns:

        pca = PCA(n_components=n_comp, random_state=1)

        new_df[col+'_pca'] = list(pca.fit_transform(np.stack(df[col].to_numpy())))

    return new_df.reset_index(drop=True)
def apply_scaler(df, columns):

    new_df = df.copy()

    for col in columns:

        scaler = StandardScaler()

        new_df[col + '_scaled'] = list(scaler.fit_transform(np.stack(df[col].to_numpy())))

    return new_df.reset_index(drop=True)
model_df = pca_apply(model_df, ['abstract_tfidf','body_text_tfidf'], 10)

model_df = apply_scaler(model_df,['abstract_tfidf_pca','body_text_tfidf_pca'])
import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import accuracy_score
# Set X and y for model development

X = np.stack(model_df['body_text_tfidf_pca_scaled'].to_numpy())

y = model_df["isNPI"]



# Set the training and testing set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)
# Set params and XGBoost classifier

clf_xgb = xgb.XGBClassifier(max_depth=6, learning_rate=0.1,silent=False, objective='binary:logistic', \

                  booster='gbtree', n_jobs=8, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, \

                  subsample=0.8, colsample_bytree=0.8, colsample_bylevel=1, reg_alpha=0, reg_lambda=1)



# Fit training sets into the model

clf_xgb.fit(X_train, y_train)
y_pred = clf_xgb.predict(X_test)

precision_recall_fscore_support(y_test, y_pred, average='macro')
score = accuracy_score(y_test, y_pred)

print(f'Accuracy Score: {score}')
core_booster = clf_xgb.get_booster()
xgb.plot_importance(core_booster)

plt.show()
from sklearn.metrics import plot_roc_curve
plot_roc_curve(clf_xgb, X_test, y_test)

plt.show()
final_df = all_json_df.copy()
def npi_slice(df):

    def get_count(row):

        return sum([row['abstract'].count(keyword) for keyword in keywords])

    df = df[df.apply(get_count, axis=1) > 1]

    return df

final_df = npi_slice(final_df)

len(final_df)
# Apply PCE without merging the pre-labeled dataframe. Slicing is done on the full dataset

final_df = pca_apply(final_df, ['abstract_tfidf','body_text_tfidf'], 10)

final_df = apply_scaler(final_df,['abstract_tfidf_pca','body_text_tfidf_pca'])
def npi_col(row):

    x = np.array([row['body_text_tfidf_pca_scaled']])

    y_pred = clf_xgb.predict(x)[0]

    if y_pred > 0:

        return True

    return False
final_df['npi_pred'] = final_df.apply(npi_col, axis=1)

final_df = final_df[final_df['npi_pred']].reset_index(drop=True)

final_df = final_df.drop(columns=['npi_pred'])
len(final_df)
from sklearn.cluster import KMeans
def cluster(df, columns, clust_nums):

    new_df = df.copy()

    for col in columns:

        kmeans = KMeans(n_clusters = clust_nums)

        new_df[col + "_clusterID"] = list(kmeans.fit_predict(np.stack(df[col].to_numpy())))

    return new_df
# Let's try to create 10 clusters

clustered_df = cluster(final_df, ['abstract_tfidf_pca_scaled', 'body_text_tfidf_pca_scaled'], 10)
clustered_df.body_text_tfidf_pca_scaled_clusterID.value_counts()
clustered_df[clustered_df.body_text_tfidf_pca_scaled_clusterID == 3][['title']].head()
# Reduce dimensions for plotting later

def reduce_dimension(row):

    return row[:3]

clustered_df['abstract_tfidf_pca_scaled'] = clustered_df['abstract_tfidf_pca_scaled'].apply(reduce_dimension)

clustered_df['body_text_tfidf_pca_scaled'] = clustered_df['body_text_tfidf_pca_scaled'].apply(reduce_dimension)
!pip install country_list

from country_list import countries_for_language
countries = set([k.lower() for k in dict(countries_for_language('en')).values()])
def set_country_columns(df):

    def get_country(row):

        text_set = set(row['body_text'].split(' '))

        return list(countries.intersection(text_set))

    df['countries'] = df.apply(get_country, axis=1)

    return df

clustered_df = set_country_columns(clustered_df)
def get_country_df(df):

    country = []

    count = []

    for k in dict(countries_for_language('en')).values():

        len_country = len(df[df['countries'].map(set([k.lower()]).issubset)])

        country.append(k.lower())

        count.append(len_country)

    return pd.DataFrame({'country': country, 'count': count})

country_frequency = get_country_df(clustered_df)
country_frequency.sort_values(by='count', ascending=False)
import plotly.express as px
fig = px.scatter_geo(country_frequency,

                     locationmode='country names',

                     locations='country',

                     hover_name='country',

                     size='count',

                     projection='natural earth')

fig.update_layout(title='Research Papers Mentioned by Country',

                  autosize=False,

                  width=500,

                  height=250,

                  paper_bgcolor='rgba(0,0,0,0)'

                 )

fig.show()
clustered_df[['x', 'y', 'z']] = pd.DataFrame(clustered_df['body_text_tfidf_pca_scaled'].values.tolist(),

                                             index = clustered_df.index)
fig = px.scatter(clustered_df, x='x', y='y',

                 color='body_text_tfidf_pca_scaled_clusterID',

                 hover_name='title',

                 hover_data=['paper_id', 'doi'])

fig.update_layout(title = '2D cluster of research papers',

                  xaxis = dict(dtick=1, range=[-5,5], scaleratio = 1),

                  yaxis = dict(dtick=1, range=[-5,5], scaleratio = 1),

                  hoverlabel=dict(

                    bgcolor='white', 

                    font_size=8, 

                    font_family='Rockwell'

                  ),

                  coloraxis=dict(

                    colorbar=dict(title='Cluster ID')

                  ))

fig.show()
fig = px.scatter_3d(clustered_df, x='x', y='y', z='z',

                    color='body_text_tfidf_pca_scaled_clusterID',

                    hover_name='title',

                    hover_data=['paper_id', 'doi'])

fig.update_layout(title = '3D cluster of research papers by body_text',

                  paper_bgcolor='rgba(0,0,0,0)',

                  scene = dict(

                    xaxis = dict(dtick=1, range=[-5,5],),

                    yaxis = dict(dtick=1, range=[-5,5],),

                    zaxis = dict(dtick=1, range=[-5,5],),),

                  hoverlabel=dict(

                    bgcolor='white', 

                    font_size=8, 

                    font_family='Rockwell'

                  ),

                  coloraxis=dict(

                    colorbar=dict(title='Cluster ID')

                  ))

fig.show()