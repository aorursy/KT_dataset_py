import pandas as pd
import numpy as np
import json
import os
import glob
from pathlib import Path, PurePath
from ipywidgets import interact
import ipywidgets as widgets
from IPython.display import FileLink
from collections import defaultdict
from sklearn.decomposition import LatentDirichletAllocation
!pip install transformers -q #==2.2.2

from transformers import *
import logging
import torch
from tqdm.notebook import tqdm
!pip install bert-extractive-summarizer -q
from summarizer import Summarizer

!pip install --upgrade git+https://github.com/zalandoresearch/flair.git -q
from flair.data import Sentence
from flair.embeddings import BertEmbeddings,DocumentPoolEmbeddings
!pip install whoosh -q
import whoosh
from whoosh.qparser import *
from whoosh.fields import Schema, TEXT, KEYWORD, ID, STORED, NUMERIC, NGRAMWORDS
from whoosh.analysis import StemmingAnalyzer,StandardAnalyzer, NgramFilter
from whoosh import index
def setup_local_data():
  input_dir = '/kaggle/input/'
  for item in list(Path(input_dir).glob('*')):
    print(item)
  return input_dir

# this loads the precomupted dataframe that includes the original metadata along with features for bert summaries and scibert embeddings
def read_full_data_json(input_dir):
  path = input_dir + 'covid19-corpus-data/full_metadata_with_scibert_embeddings_v2.json'
  data = pd.read_json(path)
  return data

def read_metadata_csv(input_dir):
    metadata_path = input_dir + '/metadata.csv'
    metadata = pd.read_csv(metadata_path,
                           dtype={'cord_uid':str,
                                  'sha':str,
                                  'publish_time': str, 
                                  'authors':str,
                                  'title': str,
                                  'abstract':str,
                                  'url': str},
                           parse_dates = ['publish_time']
                          )
    #get publish year
    metadata['publish_year'] = pd.DatetimeIndex(metadata['publish_time']).year
    #set the abstract to the paper title if it is null
    metadata['abstract'] = metadata['abstract'].fillna(metadata['title'])
    #remove if abstract is empty or contains only one word
    metadata = metadata.dropna(subset=['abstract'], axis = 0)
    metadata['number_tokens'] = metadata['abstract'].apply(lambda x: len(x.split()))
    metadata = metadata[metadata['number_tokens']>1].reset_index(drop=True)
    return metadata
local_dir = setup_local_data()
# run the below code to load the metadata from the official data set

# initial_metadata = read_metadata_csv(local_dir + '/CORD-19-research-challenge')
# print(initial_metadata.info())
# Load model, model config and tokenizer via Transformers
def create_custom_model_and_tokenizer(pretrain_model):
    custom_config = AutoConfig.from_pretrained(pretrain_model)
    custom_config.output_hidden_states = True
    custom_tokenizer = AutoTokenizer.from_pretrained(pretrain_model)
    custom_model = AutoModel.from_pretrained(pretrain_model, config=custom_config)
    return custom_model, custom_tokenizer

def extract_summary(text, custom_model=None, custom_tokenizer=None):
    model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)
    return model(text)
# instantiate custom model and tokenizer

# sciBert, sciBert_tokenizer = create_custom_model_and_tokenizer('allenai/scibert_scivocab_uncased')
#extract summaries from the abstracts
#chunk processing due to long processing time and possible notebook runtime shutdowns
def get_abstract_bert_summaries_and_save(num_chunks=20):
  chunks = np.array_split(initial_metadata, num_chunks)
  for index in range(len(chunks)):
      chunk = chunks[index].reset_index()
      summary_list = []
      for i in tqdm(chunk.index):
          summary = extract_summary(chunk.iloc[i]['abstract'], 
                                    custom_model=sciBert, 
                                    custom_tokenizer=sciBert_tokenizer)
          summary_list.append(
              {
                  "cord_uid": chunk.iloc[i]['cord_uid'],
                  "sha": chunk.iloc[i]['sha'],
                  "summary": summary
              }
          )
      summary_df = pd.DataFrame(data=summary_list)
      summary_df.to_json(local_dir + 'abstract_summaries_part{}.json'.format(index))
# uncomment below if you need to rerun bert summaries
# we've included the files for you already to save time since it takes >10 hours

# get_abstract_bert_summaries_and_save()
# get scibert embeddings of the abstract summaries
def get_embeddings(text, model):
  sentence = Sentence(text)
  document_embedding = DocumentPoolEmbeddings([model],
                                             pooling= 'mean')
  document_embedding.embed(sentence)
  # now check out the embedded sentence.
  return sentence.get_embedding().data.numpy()

def get_scibert_embeddings_and_save(emb_model, chunks=3):
  files = glob.glob(local_dir+'/covid19-corpus-data/AbstractSummaries/*.json')

  for index in range(len(files)):
      df = pd.read_json(files[index]).dropna(subset=['summary']).reset_index(drop=True)

      chunks = np.array_split(df, CHUNKS_COUNT)
      
      for chunk_idx in range(len(chunks)):
        chunk = chunks[chunk_idx].reset_index()
        emb_list=[]
        for i in tqdm(chunk.index):
          try:
            embedding = get_embeddings(chunk.iloc[i]['summary'], emb_model)
            emb_list.append(
            {
              "cord_uid": chunk.iloc[i]['cord_uid'],
              "sha": chunk.iloc[i]['sha'],
              "scibert_emb": embedding
            })
          except RuntimeError:
            emb_list.append(
            {
              "cord_uid": chunk.iloc[i]['cord_uid'],
              "sha": chunk.iloc[i]['sha'],
              "scibert_emb": np.nan
            })
        emb_df = pd.DataFrame(data=emb_list)
        emb_df.to_json(local_dir+'covid19-corpus-data/AbstractEmbeddings/abstract_embeddings_part{}_{}.json'.format(index, chunk_idx),
                      default_handler=str)
        del emb_df
# uncomment below if you need to rerun scibert embeddings
# we've included the files for you already to save time since it takes >3 hours

# emb_model = BertEmbeddings(bert_model_or_path="allenai/scibert_scivocab_uncased", layers='-2')
# get_scibert_embeddings_and_save(emb_model)
#now that we have the summaries and embeddings, we will combine them into one main df

# read preprocessed SciBERT embeddings
def read_summary_data(input_dir):
  summary_path = input_dir+'covid19-corpus-data/AbstractSummaries' 
  summaries = pd.concat([pd.read_json(f) for f in Path(summary_path).glob('*')]).reset_index(drop=True)
  return summaries

def read_embeddings(input_dir):
  vector_path = input_dir+'covid19-corpus-data/AbstractEmbeddings' 
  embeddings = pd.concat([pd.read_json(f) for f in Path(vector_path).glob('*')]).reset_index(drop=True)
  return embeddings

# uncomment below to load if needed

# summaries = read_summary_data(local_dir)
# embeddings = read_embeddings(local_dir)
# print(summaries.info())
# print(embeddings.info())
'''
this function will attempt to merge metadata with summaries/embeddings data based
on uid or sha. It will also correct for any nulls
'''
def merge_metadata_with_summaries_and_embeddings(metadata, summaries, embeddings):
  merged = metadata.merge(summaries, on=['cord_uid','sha']).merge(embeddings, on=['cord_uid','sha'])
  #for debugging
  # print(merged.info())
  # print(merged[pd.isnull(merged['scibert_emb'])]['summary'])
  # print(merged[pd.isnull(merged['scibert_emb'])]['abstract'])
  for row in merged.loc[merged['scibert_emb'].isnull(), 'scibert_emb'].index:
    try:
      merged.at[row, 'scibert_emb'] = get_embeddings(merged.iloc[row]['abstract'], emb_model)
    except RuntimeError:
      #truncate articles have very long abstracts that exceeds bert's sequence length limit
      merged.at[row, 'scibert_emb'] = get_embeddings(merged.iloc[row]['abstract'][:512], emb_model)
  return merged
# uncomment below to run code that will combine summaries, embeddings, and metadata csv into one json

# merged = merge_metadata_with_summaries_and_embeddings(initial_metadata, summaries, embeddings)
# merged.info()
# # save this combined data so we can reuse it below in main application code
# merged.to_json('/kaggle/working/full_metadata_with_scibert_embeddings.json')
# del merged
#uncomment below to download the file

# os.chdir(r'/kaggle/working')
# merged.to_json(r'full_metadata_with_scibert_embeddings.json')
# FileLink(r'full_metadata_with_scibert_embeddings.json')
local_dir = setup_local_data()
full_data = read_full_data_json(local_dir)
print(full_data.info())
print(full_data.shape)
print(full_data.head(5))
#to view how long each scibert embedding is
print(len(full_data.iloc[0]['scibert_emb'])) 
#now we need to create a new df of just n sample x scibert embedding features
df = pd.DataFrame(data=full_data['scibert_emb'].tolist(), index=full_data['title'])
print(df.shape)
df.head(5)
import pickle

def get_lda_model(num_topics):
    lda_model = LatentDirichletAllocation(n_components=num_topics, max_iter=10, learning_method='online')
    return lda_model

# this loads an lda model that has already been run with 10 topics to save time re-running.
def get_lda_model_saved():
    saved_lda_model = pickle.load(open('/kaggle/input/covid19-corpus-data/saved_lda_model.pk', 'rb'))
    return saved_lda_model

def save_model(model, file_name):
    pickle.dump(model, open(f"/kaggle/working/{file_name}.pk", 'wb'))
# Since LDA requires non-negative values, we perform unity-based normalization, essentially standardizes from 0 to 1, on the dataset
# https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
# https://datascience.stackexchange.com/questions/5885/how-to-scale-an-array-of-signed-integers-to-range-from-0-to-1
normalized_df = (df-df.min())/(df.max()-df.min())

# we use the pretained model to save time from rerunning the lda model's fit operation since that takes awhile in Kaggle.
# lda = get_lda_model(10)
lda = get_lda_model_saved()
lda_output = lda.transform(normalized_df)
print(lda_output.shape)  # (NO_DOCUMENTS, NO_TOPICS)
results = pd.DataFrame(lda_output, index=full_data['title'])
print("Printing top articles for each topic based on % comprised of that topic")
for i in range(0, 10):
  print(f"Printing the top 10 articles for topic {i}")
  sorted_df = results.sort_values(i, ascending=False)
  print(sorted_df[i].head(10))
  print()
#get schema for the index
def get_search_schema():
  schema = Schema(uid = TEXT(stored=True),
                  sha = TEXT(stored=True),
                  year = TEXT(stored=True),
                  author = TEXT(stored=True),
                  #here we set minisize = 1 to preserve numeric values
                  #so we can differentiate between sars-cov and sars-cov-2
                  title = TEXT(analyzer=StandardAnalyzer(minsize=1),stored=True),
                  abstract = TEXT(analyzer=StandardAnalyzer(minsize=1),stored=True),
                  url = TEXT(stored=True))
  return schema

# creates an index in a dictionary (only need to run once)
# noop if it's already created
def create_search_index(drive_path, search_schema):
  if not os.path.exists(drive_path + 'indexdir'):
      os.mkdir(drive_path + 'indexdir')
  ix = index.create_in(drive_path + 'indexdir', search_schema)
  #open an existing index object
  ix = index.open_dir(drive_path + 'indexdir')
  return ix

def add_documents_to_index(ix, metadata):
  # cancel writer in case re-indexing is needed
  # if 'writer' in locals(): #doesn't work on Kaggle
  # writer.cancel()

  #create a writer object to add documents to the index
  writer = ix.writer()

  #now we can add documents to the index
  uid = metadata['cord_uid']
  sha = metadata['sha']
  year = metadata['publish_year']
  author = metadata['authors']
  title = metadata['title']
  abstract = metadata['abstract']
  url = metadata['url']

  for UID, SHA, YEAR, AUTHOR, TITLE, ABSTRACT, URL in zip(uid, sha, year, author, title, abstract, url):
    writer.add_document(uid = str(UID),
                        sha= str(SHA),
                        year= str(YEAR),
                        author=str(AUTHOR),
                        title=str(TITLE),
                        abstract=str(ABSTRACT),
                        url=str(URL))

  #close the writer and save the added documents in the index
  #you should call the commit() function once you finish adding the documents otherwise you will cause an error-
  #when you try to edit the index next time and open another writer. 
  writer.commit()

  # need to cancel writer if error or need to reset
  # writer.cancel()
  return

# get a multifield parser for the list of inptted fields
def get_multifield_parser(fields, search_schema):
  parser = MultifieldParser(fields, schema=search_schema)
  parser.add_plugin(SequencePlugin())
  parser.add_plugin(PhrasePlugin())
  return parser

# this takes in a parser and query string to return the actual query that'll be sent to the searcher
def get_parser_query(parser, query):
  result = parser.parse(query) # use boolean operators in quotation
  print(result)
  return result
    
# this method takes in a search index and query to return a dataframe of results
# ix is the document index we created before
# query is the string found from the parser
def get_search_results(ix, query):
  #you can open the searcher using a with statement so the searcher is automatically closed when you’re done with it
  with ix.searcher() as searcher:
      #The Results object acts like a list of the matched documents
      results = searcher.search(query, limit=None)
      print('Total Hits: {}\n'.format(len(results)))
      output_dict = defaultdict(list)

      for result in results:
        output_dict['cord_uid'].append(result['uid'])
        output_dict['sha'].append(result['sha'])
        output_dict['bm25_score'].append(result.score)
        output_dict['title'].append(result['title'])
        output_dict['abstract'].append(result['abstract'])
        output_dict['publish_year'].append(result['year'])
        output_dict['authors'].append(result['author'])
        output_dict['url'].append(result['url'])
        
  output_df = pd.DataFrame(output_dict)
  return output_df
search_schema = get_search_schema()
#we set this for Kaggle but can be any directory where you're working in. It takes a while to run.
ix = create_search_index('/kaggle/working/', search_schema) 
add_documents_to_index(ix, full_data)
#the code below creates an interactive query builder
from ipywidgets import interact, Layout, HBox, VBox, Box
from IPython.display import HTML, display, clear_output
import ipywidgets as widgets
from IPython.display import update_display

def get_new_text_box():
  textW = widgets.Textarea(
        value='',
        placeholder='Type something like "covid" or incubation',
        description='',
        disabled=False,
        layout=Layout(width='100%', height='50px')
    )
  return textW

def get_new_plus_button():
  button = widgets.Button(description="+")
  return button

def get_new_dropdown():
  dropdown = widgets.Dropdown(
      options=['AND', 'OR', 'NOT'],
      value='AND',
      description='Operator: ',
      disabled=False,
    )
  return dropdown

def dynamic_search_query(parser, ix):
  textW = widgets.Textarea(
        value='',
        placeholder='Type something like "covid" or incubation',
        description='',
        disabled=False,
        layout=Layout(width='100%', height='50px')
    )
  
  button = widgets.Button(description="+")
  search_rows_list = []
  search_rows_list.append( HBox([textW, button], layout=Layout(align_items='center')) )
  display_handle = display(VBox(search_rows_list, layout=Layout(align_items='center')), display_id='disp')

  #search_rows_list is a list of HBox objects
  # the first index will just be a text box and '+' button
  # subsequent rows will have operator, text box, and '+' button
  def on_button_clicked(b):
    global STORED_SEARCH_QUERY
    clear_output(wait=True)
    new_text_box = get_new_text_box()
    dropdown = get_new_dropdown()
    search_rows_list.append( HBox([dropdown, new_text_box, button], layout=Layout(align_items='center')) )
    display_handle.update(VBox(search_rows_list, layout=Layout(align_items='center')))

    combined = ''
    for i in range(0, len(search_rows_list)-1): #we do len - 1 since newet row has no values
      row = search_rows_list[i]
      if i == 0:
        temp = combined + row.children[0].value
        combined = temp
      else:
        temp = combined + ' ' + row.children[0].value + ' ' + row.children[1].value
        combined = temp
    
    print("Current raw search query:\n" + combined)
    print("Current query from parser:")
    query = get_parser_query(parser, combined) #already prints in method
    STORED_SEARCH_QUERY = query

  button.on_click(on_button_clicked)

#read saved index  
ix = index.open_dir('/kaggle/working/indexdir')
fields = ["title", "abstract"] #set search fields
parser = get_multifield_parser(fields, search_schema)
STORED_SEARCH_QUERY = '' #query is stored as a global so the last search query from the parser can be used in later cells
dynamic_search_query(parser, ix)
#see the stored query
print(STORED_SEARCH_QUERY)
#get search engine output
search_results = get_search_results(ix, STORED_SEARCH_QUERY)
print(search_results.shape)
search_results.head(5)
#for each of the output articles, find the closest 20 articles by comparing cosine scores
from scipy.spatial import distance

cord_uid_lda_df = pd.DataFrame(lda_output, index=full_data['cord_uid'])
cord_uids = search_results['cord_uid'].tolist()

list_of_similar_articles = []
duplicates = []
for cord_uid in cord_uids:
  if isinstance(cord_uid_lda_df.loc[cord_uid], pd.DataFrame):
    duplicates.append(cord_uid)
    continue
  topic_score = cord_uid_lda_df.loc[cord_uid].tolist()
  distances = []
  for entry in lda_output:
    distances.append(distance.cosine(topic_score, entry))
  full_data['cosine_distance'] = np.asarray(distances)
  #ascending must be true since scipy does 1 - cosine(theta), so smaller values are closer
  full_data_sorted = full_data.sort_values('cosine_distance', ascending=True)
  list_of_similar_articles.append(full_data_sorted[:20]['cord_uid'].tolist())
  del full_data_sorted

# print(duplicates) #this is helpful for debugging purposes if duplicates are found after doing a search
print(len(list_of_similar_articles))
# ideally we want to see if there are any documents that are included across all lists of top similar articles 
# we'll use set operations for this.
overlapping_articles = list_of_similar_articles[0]
for articles in list_of_similar_articles:
  overlapping_articles = set(overlapping_articles) & set(articles)
print(len(overlapping_articles))
import operator
import collections

article_counts = {}
for articles in list_of_similar_articles:
  for article in articles:
    curValue = article_counts.get(article, 0)
    article_counts[article] = curValue + 1

article_counts = sorted(article_counts.items(), key=lambda kv: kv[1], reverse=True)
sorted_articles = collections.OrderedDict(article_counts)
relevant_articles = []
for k,v in sorted_articles.items():
    if v > 3:
        row = full_data.loc[full_data['cord_uid'] == k]
        title = row['title'].tolist()[0]
        print(f"{title}: {v}")
        relevant_articles.append(k)
full_data = full_data.drop('cosine_distance', axis=1)
full_data.info()
def read_query_dictionary(input_dir):
  path = input_dir+'covid19-corpus-data/'+'all_dz_query_dictionary_v5.json'
  with open(path) as f:
    query_dict = json.load(f)
  return query_dict

# Get a list of queries formatted for the whoosh search engine
whooshified_query_dict = {}
query_dictionary = read_query_dictionary(local_dir) #reads json query file 
for key, value in query_dictionary.items():
  query = get_parser_query(parser, value) #converts dict values to whoosh objects
  whooshified_query_dict[key] = query
# Run search for each query and append results to dataframe
all_search_results_dict = {}
for key, value in whooshified_query_dict.items():
  #Stores results as dictionary with key as `query` and values as dataframes
  search_results_df = get_search_results(ix, value)
  if len(search_results_df) > 0:
    search_results_df['source'] = 'search' #indicate origin of result (search vs. lda)
    search_results_df['query'] = key
    all_search_results_dict[key] = search_results_df
from scipy.spatial import distance

dict_of_similar_articles = {}
cord_uid_lda_df = pd.DataFrame(lda_output, index=full_data['cord_uid'])

for query, titles in all_search_results_dict.items():
  cord_uids = titles['cord_uid'].tolist()
  list_of_similar_articles = []
  duplicates = []
  for cord_uid in cord_uids:
    if isinstance(cord_uid_lda_df.loc[cord_uid], pd.DataFrame): #indicates a duplicate
      duplicates.append(cord_uid)
      continue
    topic_score = cord_uid_lda_df.loc[cord_uid].tolist()
    distances = []
    for entry in lda_output: #lda_output --> topic similarity results for each article in the corpus
      distances.append(distance.cosine(topic_score, entry))
    full_data['cosine_distance'] = np.asarray(distances)
    full_data_sorted = full_data.sort_values('cosine_distance', ascending=True)
    list_of_similar_articles.append(full_data_sorted[:20]['cord_uid'].tolist())
    del full_data_sorted
  dict_of_similar_articles[query] =  list_of_similar_articles
  # print('Duplicates: ', duplicates) #this is helpful for debugging purposes if duplicates are found after doing a search
print('Length of similar articles dict: ', [len(v) for k, v in dict_of_similar_articles.items()])
for query, list_of_similar_artilcles in dict_of_similar_articles.items():
  overlapping_articles = list_of_similar_artilcles[0]
  for articles in list_of_similar_artilcles:
    overlapping_articles = set(overlapping_articles) & set(articles)
  print(f"Query: {query} and number of overlapping articles: {len(overlapping_articles)}")
from collections import OrderedDict

all_relevant_articles = {}
print("Printing top recurring articles for each query.")
for query, list_of_similar_artilcles in dict_of_similar_articles.items():
  print(f"\nQuery: {query}")
  article_counts = {}
  for articles in list_of_similar_artilcles:  
    for article in articles:
      curValue = article_counts.get(article, 0)
      article_counts[article] = curValue + 1

  article_counts = sorted(article_counts.items(), key=lambda kv: kv[1], reverse=True)
  sorted_articles = OrderedDict(article_counts)

  relevant_articles = []
  for k, v in sorted_articles.items():
    if v > 3:
      row = full_data.loc[full_data['cord_uid'] == k]
      title = row['title' ].tolist()[0]
      print(f"{title}: {v}")
      relevant_articles.append(k)
  all_relevant_articles[query] = relevant_articles
#Number of relevant articles per precoded query
[len(v) for k, v in all_relevant_articles.items()]
temp = []
for k, v in all_search_results_dict.items():
  search_output_df = full_data[full_data['cord_uid'].apply(lambda x: x in v['cord_uid'].tolist())]
  search_output_df['source'] = 'search'
  search_output_df['query'] = k
  temp.append(search_output_df)
all_search_output = pd.concat(temp)
all_search_output.drop_duplicates(subset=['cord_uid', 'source', 'query'])
print('all_search len: ', len(all_search_output))

temp = []
for k, v in all_relevant_articles.items():
  expanded_output = full_data[full_data['cord_uid'].apply(lambda x: x in v)]
  expanded_output['source'] = 'lda'
  expanded_output['query'] = k
  temp.append(expanded_output)
relevant_search_results_lda_output = pd.concat(temp)
relevant_search_results_lda_output.drop_duplicates(subset=['cord_uid', 'source', 'query'])
print('relevant_search_results_lda_output len: ', len(relevant_search_results_lda_output))
combined_output_master = pd.concat([all_search_output,relevant_search_results_lda_output]).reset_index(drop=True)
combined_output_master = combined_output_master[['cord_uid','sha','title','abstract','publish_year','summary','url', 'source', 'query']]
print(combined_output_master.info())

#this dedupes the entries by the unique cord_uid, but the source data will be lost as a result. This shouldn't be used for analyzing where source data comes from 
combined_output_dedup = combined_output_master.drop_duplicates(subset=['cord_uid', 'query']).reset_index(drop=True)
combined_output_dedup = combined_output_dedup[['cord_uid','sha','title','abstract','publish_year','summary','url', 'source', 'query']]
print(combined_output_dedup.info())
print(combined_output_master['query'].value_counts())
combined_output_master.head()