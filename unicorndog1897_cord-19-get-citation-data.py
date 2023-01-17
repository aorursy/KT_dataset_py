import os
import json
from pprint import pprint
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import ast

import seaborn as sns
def format_name(author):
    middle_name = " ".join(author['middle'])
    
    if author['middle']:
        return " ".join([author['first'], middle_name, author['last']])
    else:
        return " ".join([author['first'], author['last']])


def format_affiliation(affiliation):
    text = []
    location = affiliation.get('location')
    if location:
        text.extend(list(affiliation['location'].values()))
    
    institution = affiliation.get('institution')
    if institution:
        text = [institution] + text
    return ", ".join(text)

def format_authors(authors, with_affiliation=False):
    name_ls = []
    
    for author in authors:
        name = format_name(author)
        if with_affiliation:
            affiliation = format_affiliation(author['affiliation'])
            if affiliation:
                name_ls.append(f"{name} ({affiliation})")
            else:
                name_ls.append(name)
        else:
            name_ls.append(name)
    
    return ", ".join(name_ls)

def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    
    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"
    
    return body

def format_bib(bibs):
    if type(bibs) == dict:
        bibs = list(bibs.values())
    bibs = deepcopy(bibs)
    formatted = []
    
    for bib in bibs:
        bib['authors'] = format_authors(
            bib['authors'], 
            with_affiliation=False
        )
        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]
        formatted.append(", ".join(formatted_ls))
        

    return "; ".join(formatted)


#use this to structure sources into nested lists, to more easily create a pandas dataframe later
def bib_nested_lists(bibs):
    #bibs_df = pd.Dataframe(columns = ['article_title', 'article_year', 'cited_article', 'cited_article_year', 'data_source'])
    bib_data_rows = []
    if type(bibs) == dict:
        bibs = list(bibs.values())
    else:
        bibs = bibs.replace("\'", "\"")
        bibs = json.loads(bibs)
        bibs = list(bibs.values())
        
    bibs = deepcopy(bibs)
    
    #print('BIBS')
    #print(bibs)
    
    for bib in bibs:
        formatted_ls = [str(bib[k]) for k in ['title', 'year']]
        bib_data_rows.append(formatted_ls)
        
        #bibs_df = pd.DataFrame(bib_data_rows, columns = ['cited_article', 'cited_article_year'])
    return bib_data_rows
def load_files(dirname):
    filenames = os.listdir(dirname)
    raw_files = []

    for filename in tqdm(filenames):
        filename = dirname + filename
        file = json.load(open(filename, 'rb'))
        raw_files.append(file)
    
    return raw_files

def generate_clean_df(all_files):
    cleaned_files = []
    
    for file in tqdm(all_files):
        features = [
            file['paper_id'],
            file['metadata']['title'],
            format_authors(file['metadata']['authors']),
            format_authors(file['metadata']['authors'], 
                           with_affiliation=True),
            format_body(file['abstract']),
            format_body(file['body_text']),
            bib_nested_lists(file['bib_entries']),
            file['metadata']['authors'],
            file['bib_entries']
        ]

        cleaned_files.append(features)

    col_names = ['paper_id', 'title', 'authors',
                 'affiliations', 'abstract', 'text', 
                 'bibliography','raw_authors','raw_bibliography']

    clean_df = pd.DataFrame(cleaned_files, columns=col_names)
    clean_df.head()
    
    return clean_df
biorxiv_dir = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/'
filenames = os.listdir(biorxiv_dir)
print("Number of articles retrieved from biorxiv:", len(filenames))
all_files = []

for filename in filenames:
    filename = biorxiv_dir + filename
    file = json.load(open(filename, 'rb'))
    all_files.append(file)
import os
import json
from pprint import pprint
from copy import deepcopy

import numpy as np
import pandas as pd
file = all_files[0]
print("Dictionary keys:", file.keys())
cleaned_files = []

for file in tqdm(all_files):
    features = [
        file['paper_id'],
        file['metadata']['title'],
        format_authors(file['metadata']['authors']),
        format_authors(file['metadata']['authors'], 
                       with_affiliation=True),
        format_body(file['abstract']),
        format_body(file['body_text']),
        bib_nested_lists(file['bib_entries']),
        file['metadata']['authors'],
        file['bib_entries']
    ]
    
    cleaned_files.append(features)
col_names = [
    'paper_id', 
    'title', 
    'authors',
    'affiliations', 
    'abstract', 
    'text', 
    'bibliography',
    'raw_authors',
    'raw_bibliography'
]

clean_df = pd.DataFrame(cleaned_files, columns=col_names)
clean_df.head()
clean_df.to_csv('biorxiv_nested_clean.csv', index=False)
clean_df['bibliography'][0]
pmc_dir = '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/'
pmc_files = load_files(pmc_dir)
pmc_df = generate_clean_df(pmc_files)
pmc_df.to_csv('clean_nested_pmc.csv', index=False)
pmc_df.head()
pmc_df['bibliography'][0]
comm_dir = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/'
comm_files = load_files(comm_dir)
comm_df = generate_clean_df(comm_files)
comm_df.to_csv('clean_comm_use_nested.csv', index=False)
comm_df.head()
comm_df['bibliography'][0]
noncomm_dir = '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/'
noncomm_files = load_files(noncomm_dir)
noncomm_df = generate_clean_df(noncomm_files)
noncomm_df.to_csv('clean_noncomm_use_nested.csv', index=False)
noncomm_df.head()
noncomm_df['bibliography'][0]

noncomm_df = pd.read_csv('/kaggle/input/covid19-for-citation-networks/clean_noncomm_use_nested.csv')
noncomm_df['dataset'] = 'non_comm'
noncomm_df
noncomm_df['bibliography'][0]
nest_list = ast.literal_eval(noncomm_df['bibliography'][0]) 
print(nest_list)
def bib_to_df(bib_data_rows):
    #bibs_df = pd.Dataframe(columns = ['article_title', 'article_year', 'cited_article', 'cited_article_year', 'data_source'])
    bibs_df = pd.DataFrame(ast.literal_eval(bib_data_rows), columns = ['cited_article', 'cited_article_year'])   
    return bibs_df
bib_to_df(noncomm_df['bibliography'][0])
def fill_df_rows(df, col_name, val):
    df[col_name] = val
def create_whole_citation_df(original_df):
    whole_citation_df = pd.DataFrame(columns = ['source_article','cited_article', 'cited_article_year', 'source_article_dataset'])
    for i in range(len(original_df)):
        print('Article: ', i)
        
        
        
        #user helper function to 
        citation_df_rows = bib_to_df(original_df.loc[i, 'bibliography'])
        
        
        
        
        fill_df_rows(citation_df_rows,'source_article', original_df.loc[i, 'title'])
        fill_df_rows(citation_df_rows, 'source_article_dataset', original_df.loc[i, 'dataset'])
        
        
        whole_citation_df = whole_citation_df.append(citation_df_rows)
        print('Num citations: ', len(whole_citation_df))
    return whole_citation_df

  

whole_df = create_whole_citation_df(noncomm_df)
whole_df
whole_df.to_csv('noncomm_network_data.csv')

biorxiv_df = pd.read_csv('/kaggle/input/covid19-for-citation-networks/biorxiv_nested_clean.csv')
biorxiv_df['dataset'] = 'biorxiv'
network_biorxiv = create_whole_citation_df(biorxiv_df)
network_biorxiv.head()
network_biorxiv.to_csv('network_biorxiv.csv')
whole_df = whole_df.append(network_biorxiv)
pmc_df = pd.read_csv('/kaggle/input/covid19-for-citation-networks/clean_nested_pmc.csv')
pmc_df['dataset'] = 'pmc'
network_pmc = create_whole_citation_df(pmc_df)
network_pmc
network_pmc.to_csv('network_pmc.csv')
whole_df = whole_df.append(network_pmc)
comm_df = pd.read_csv('/kaggle/input/covid19-for-citation-networks/clean_comm_use_nested.csv')
comm_df
comm_df['dataset'] = 'comm'
network_comm = create_whole_citation_df(comm_df)
network_comm.to_csv('network_comm.csv')
whole_df= whole_df.append(network_comm)
whole_df
whole_df.to_csv('network_all_datasets.csv')
whole_df = pd.read_csv('/kaggle/input/covid19-for-citation-networks/network_all_datasets.csv')
whole_df
whole_df['cited_article'].value_counts().head(100)
whole_df[whole_df['source_article_dataset']== 'pmc']['cited_article'].value_counts().head(100)
whole_df[whole_df['source_article_dataset']== 'biorxiv']['cited_article'].value_counts().head(100)
whole_df[whole_df['source_article_dataset']== 'comm']['cited_article'].value_counts().head(100)
whole_df[whole_df['source_article_dataset']== 'non_comm']['cited_article'].value_counts().head(100)

