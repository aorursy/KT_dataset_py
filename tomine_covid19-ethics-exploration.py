# BEFORE RUNNING, uncomment to install extra packages and models 





! pip install flair



! python -m spacy download en_core_web_md



! pip install scispacy



! pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_craft_md-0.2.4.tar.gz





# data manip

import json

import pandas as pd

import numpy as np

import networkx as nx

from collections import Counter

from ast import literal_eval



# visuals

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode

import plotly.io as pio

from spacy import displacy

from tqdm import tqdm_notebook

from termcolor import colored

# from tsne import bh_sne









# system nav

import os

import sys

import pickle



# nlp

from string import punctuation

import spacy

from spacy.lang.en.stop_words import STOP_WORDS



#feature engineering discarded

# from flair.embeddings import FlairEmbeddings, DocumentPoolEmbeddings, Sentence, BertEmbeddings



# ML model

import torch

from flair.embeddings import DocumentPoolEmbeddings, Sentence, WordEmbeddings

from sklearn.linear_model import LogisticRegressionCV

from sklearn.metrics import f1_score

from sklearn.decomposition import PCA

NO_NER = True
# spacy models used

EN_MODEL = "en_core_web_md"

SCI_NER = "en_ner_craft_md"



# BERTEmbeddings

SCI_BERT = "allenai/scibert_scivocab_uncased" # not used, for faster processing I've switched to glove static embeddings



# paths to CORD data

METADATA = "/kaggle/input/CORD-19-research-challenge/metadata.csv"

DATA_DIR = "/kaggle/input/CORD-19-research-challenge/"



# the list of 20 biomedical files and the directory to the manually labeled dataset

BIOMED_DATA = "/kaggle/input/extra-data/labeled_biomed_data.csv"

SOCIOETHICS_DATA_DIR = "/kaggle/input/arxiv-data"



# outdirectory for models and saved model input

OUTDIR = "/kaggle/working/"

INDIR = "/kaggle/input/"

LIST_DF_DIR = "list-dfs-3"

EXTRA_DATA = "extra-data"





# we won't consider some of the numerical entity or person types of the generic pipeline

UNWANTED_ENTS = ["CARDINAL", "PERCENT", "PERSON", "DATE", "TIME", "MONEY", "ORDINAL", "QUANTITY"]







# if running from PC

if not os.path.isdir(os.path.join(OUTDIR, LIST_DF_DIR)):

    os.mkdir(os.path.join(OUTDIR, LIST_DF_DIR))

if not os.path.isdir(os.path.join(OUTDIR, EXTRA_DATA)):

    os.mkdir(EXTRA_DATA)
print(len(os.listdir(os.path.join(INDIR, LIST_DF_DIR))))

print(os.listdir(os.path.join(INDIR, EXTRA_DATA)))
if NO_NER:

    with open(os.path.join(INDIR, EXTRA_DATA, "bio_sci_ner.pkl"), "rb") as f:

        bio_sci_ner = pickle.load(f)

    with open(os.path.join(INDIR, EXTRA_DATA, "bio_gen_ner.pkl"), "rb") as f:

        bio_gen_ner = pickle.load(f)

    with open(os.path.join(INDIR, EXTRA_DATA, "ethics_sci_ner.pkl"), "rb") as f:

        ethics_sci_ner = pickle.load(f)

    with open(os.path.join(INDIR, EXTRA_DATA, "ethics_gen_ner.pkl"), "rb") as f:

        ethics_gen_ner = pickle.load(f)

    with open(os.path.join(INDIR, EXTRA_DATA, "ex_sci_ner.pkl"), "rb") as f:

        ex_sci_ner = pickle.load(f)

    with open(os.path.join(INDIR, EXTRA_DATA, "ex_gen_ner.pkl"), "rb") as f:

        ex_gen_ner = pickle.load(f)

    with open(os.path.join(INDIR, EXTRA_DATA, "df_ner.pkl"), "rb") as f:

        df_ner = pickle.load(f)

    with open(os.path.join(INDIR, EXTRA_DATA, "df_sci_ner.pkl"), "rb") as f:

        df_sci_ner = pickle.load(f)

    with open(os.path.join(INDIR, EXTRA_DATA, "df_lemmas.pkl"), "rb") as f:

        df_lemmas = pickle.load(f)

    with open(os.path.join(INDIR, EXTRA_DATA, "df_tok_counts.pkl"), "rb") as f:

        df_tok_counts = pickle.load(f)

if not NO_NER:

    NLP = spacy.load(EN_MODEL)

    NER = spacy.load(SCI_NER)
df_meta = pd.read_csv(METADATA)

df_meta["full_text_file"].isna().sum()
print("Shape before reduced data ", df_meta.shape)

df_meta = df_meta.dropna(subset = ['abstract', "full_text_file", "sha"])

# df = df.loc[df['has_full_text'] == True]  # this seems to have been taken out of the original data

df_meta['publish_year'] = df_meta['publish_time'].apply(lambda x: x[:4] 

                                              if pd.isna(x) == False 

                                              else None)

df_meta = df_meta.loc[df_meta['publish_year'].isin(['2020', '2019'])]

print("Shape after reduced ", df_meta.shape)

FULL_TEXT_ARTICLES = [os.path.join(DATA_DIR, indir, indir, 'pdf_json', split.strip())+'.json' 

                      for indir, sha in zip(df_meta['full_text_file'].to_list(), 

                                            df_meta['sha'].to_list()) 

                      for split in sha.split(';')]

paths_not_exist = [x for x in FULL_TEXT_ARTICLES if not os.path.exists(x)]

if len(paths_not_exist) == 0:

    print('All of the total', 

          len(FULL_TEXT_ARTICLES), 

          'paths exist in the directory')

else:

    print(len(paths_not_exist), 

          ' paths will be removed from total ', 

          len(FULL_TEXT_ARTICLES))

    FULL_TEXT_ARTICLES = set(FULL_TEXT_ARTICLES) - set(paths_not_exist)

print('First ten article paths:')

FULL_TEXT_ARTICLES[:10]
biomed_df = pd.read_csv(BIOMED_DATA, sep = ',')

BIOMED_ARTICLES = biomed_df['path'].to_list()

for title in biomed_df.title.to_list():

    print(title)

BIOMED_ARTICLES
SOCIOETHICS_ARTICLES = [os.path.join(SOCIOETHICS_DATA_DIR, path) for path in os.listdir(SOCIOETHICS_DATA_DIR) if path.endswith('.txt')]



SOCIOETHICS_ARTICLES
df_journals = df_meta['journal'].dropna()

journal_counts = Counter(df_journals).most_common(20)

labels = [x[0] for x in journal_counts]

df_journals_pieplot = pd.DataFrame(data= [{"journal": journal, "count": count} for journal, count in journal_counts], index = labels)

df_journals_pieplot.plot.pie(y = 'count', figsize = (15,15))

df_journals.describe()
# get spacy doc for all abstracts in metadata subset

if not NO_NER:

    df_meta['abstract_spacy_doc'] = df_meta['abstract'].apply(lambda x: NLP(x) if pd.isna(x) == False else None)
if not NO_NER:

    df_meta['tok_count'] = df_meta['abstract_spacy_doc'].apply(lambda x: len(x))

    df_tok_counts = df_meta['tok_count'].to_list()



sns.distplot( df_tok_counts)

if not NO_NER:

    df_meta['lemmas'] =  df_meta['abstract_spacy_doc'].apply(lambda x: [_.lemma_ for _ in x if _.lemma_ not in STOP_WORDS and _.text not in punctuation])

    all_lemmas = Counter([x for k, v in df['lemmas'].iteritems() for x in v])

    df_lemmas = pd.DataFrame(data = [{"token" : tok, "count" : count} for tok, count in all_lemmas.most_common(20)])

df_lemmas

#taking a look at normal spaCy pipeline NER



if not NO_NER:

    df_meta['ner'] = df_meta['abstract_spacy_doc'].apply(lambda x: [(_.text, _.label_) for _ in x.ents if _.label_ not in UNWANTED_ENTS])

    all_ner = Counter([x for k, v in df['ner'].iteritems() for x in v])

    df_ner = pd.DataFrame(data = [{"Entity" : tok[0], "Type": tok[1], "Count" : count} for tok, count in all_ner.most_common(20)])

df_ner
# now let's try sciSpaCy NER

if not NO_NER:

    df_meta['sci_ner'] = df_meta['abstract'].apply(lambda x : [(_.text, _.label_) for _ in NER(x).ents] )
if not NO_NER:

    sci_ner = Counter([x for k, v in df['sci_ner'].iteritems() for x in v])

    df_sci_ner = pd.DataFrame(data = [{"Entity" : tok[0], "Type" : tok[1], "Count" : count} for tok, count in sci_ner.most_common(20)])

df_sci_ner
labeled_data_extract = """In the age of social media, disasters and epidemics usher not only a devastation and affliction in the physical world, but also prompt a deluge of information, opinions, prognoses and advice to billions of internet users. The coronavirus epidemic of 2019-2020, or COVID-19, is no exception, with the World Health Organization warning of a possible "infodemic" of fake news. In this study, we examine the alternative narratives around the coronavirus outbreak through advertisements promoted on Facebook, the largest social media platform in the US. Using the new Facebook Ads Library, we discover advertisers from public health and non-profit sectors, alongside those from news media, politics, and business, incorporating coronavirus into their messaging and agenda. We find the virus used in political attacks, donation solicitations, business promotion, stock market advice, and animal rights campaigning. Among these, we find several instances of possible misinformation, ranging from bioweapons conspiracy theories to unverifiable claims by politicians. As we make the dataset available to the community, we hope the advertising domain will become an important part of quality control for public health communication and public discourse in general."""



if not NO_NER:

    ex_gen_ner = NLP(labeled_data_extract)

displacy.render(ex_gen_ner, style='ent')

if not NO_NER:

    ex_sci_ner = NER(labeled_data_extract)

displacy.render(ex_sci_ner, style='ent')
def process_bibs(all_jsons, t):

    """Get indexed dictionary of set of bib_refs per article and index map to reference title strig

    """

    all_bib_set = set()

    titles = set()

    article_bib_set = dict()

    counts = dict()

    title2path = dict()

    # get set of title strings, bib refs and the dictionary of bib ref (string) per article 

    for file in all_jsons:

        with open(file) as f:

            article = json.load(f)

            bibs = article['bib_entries']

            title = article['metadata']['title'].lower()

            if title not in title2path:

                title2path[title] = [file]

            else:

                title2path[title].append(file)

            titles.add(title)

            article_bib_set[title] = set()

            for entry in bibs:

                article_bib_set[article['metadata']['title'].lower()].add(bibs[entry]['title'].lower())

                all_bib_set.add(bibs[entry]['title'].lower())

                if bibs[entry]['title'].lower() not in counts:

                    counts[bibs[entry]['title'].lower()] = 1

                else:

                    counts[bibs[entry]['title'].lower()] +=1

    #first index CORD article titles                           

    bib2id = {bib:idx for idx, bib in enumerate(titles)}

    # to index bib refs, first subtract titles from bib refs

    all_bib_set = all_bib_set-titles

    # add indexed bib refs

    bib2id.update({bib:idx+len(titles) for idx, bib in enumerate(all_bib_set)})

    # index the dictionary of sets of bib refs

    indexed_article_bib_set = {bib2id[k]:{bib2id[x] for x in v if counts[x]>t} for k, v in article_bib_set.items() }

    # return inverse index map for the interactive plot legend

    id2bib = {v:k for k,v in bib2id.items()}

    return indexed_article_bib_set, id2bib, title2path





def get_edges(bibset):

    edges = []

    for i, vs in bibset.items():

        for j, vals in bibset.items():

            if i in vals:

                edges.append((j, i)) #be sure that directed graph points towards article being referenced

    return edges





# this code is from tutorial https://plotly.com/ipython-notebooks/network-graphs/

def plot_interactive_graph(G, pos, id2bib, renderer = "browser"):

    edge_x = []

    edge_y = []

    for edge in G.edges():

        x0, y0 = pos[edge[0]]

        x1, y1 = pos[edge[1]]

        edge_x.append(x0)

        edge_x.append(x1)

        edge_x.append(None)

        edge_y.append(y0)

        edge_y.append(y1)

        edge_y.append(None)



    edge_trace = go.Scatter(

        x=edge_x, y=edge_y,

        line=dict(width=0.5, color='#888'),

        hoverinfo='none',

        mode='lines')



    node_x = []

    node_y = []

    node_text = []

    for node in G.nodes():

        x, y = pos[node]

        node_x.append(x)

        node_y.append(y)

        node_text.append(id2bib[node])

        

    node_trace = go.Scatter(

        x=node_x, y=node_y,

        mode='markers',

        hoverinfo='text',

        marker=dict(

            showscale=True,

            colorscale='Viridis',

            reversescale=True,

            color=[],

            size=10,

            colorbar=dict(

                thickness=15,

                title='Bib References',

                xanchor='left',

                titleside='right'

            ),

            line_width=2))



    node_adjacencies = []

    for node, adjacencies in enumerate(G.adjacency()):

        node_adjacencies.append(len(adjacencies[1]))



    node_trace.marker.color = node_adjacencies

    node_trace.text = node_text

    if renderer == "notebook_mode":

        init_notebook_mode(connected=True)

    if renderer == 'browser':

        pio.renderers.default = "browser"



    fig = go.Figure(data=[edge_trace, node_trace],

             layout=go.Layout(

                title='<br>Bibliographic references',

                titlefont_size=16,

                showlegend=False,

                hovermode='closest',

                margin=dict(b=20,l=5,r=5,t=40),

                annotations=[ dict(

                    text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> plotly.com </a>",

                    showarrow=False,

                    xref="paper", yref="paper",

                    x=0.005, y=-0.002 ) ],

                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),

                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

                )

    fig.show()

    

def get_top_edges(edges, id2bib, title2path, all_jsons, top_k = 30):

    # we get the set of ids which are also in our reduced corpus

    title_ids = [idx for idx in id2bib if id2bib[idx] in title2path]

    # get only ends of nodes as these are articles being refered TO

    total_counts = [x[1] for x in edges if x[1] in title_ids]

    # then get counts in our bib graph and return top k

    counter = Counter(total_counts)

#     print(counter.most_common(top_k))

    return [(id2bib[x[0]],title2path[id2bib[x[0]]]) for x in counter.most_common(top_k)]

        

# set threshold for number of occurences in within the bib references

# I'll set it to 6 to have a simpler looking plot

t = 6



# if this is run from offline, set renderer kwarg as "browser"

# if run on kaggle, set renderer kwarg as "notebook_mode"

renderer = "notebook_mode"



bibset, id2bib, title2path = process_bibs(FULL_TEXT_ARTICLES, t)

edges = get_edges(bibset)

# use the networkx package to instanciate a plot and get positions

G = nx.Graph()

G.add_edges_from(edges)

pos=nx.spring_layout(G)



#plot interactive graph via plotly package

plot_interactive_graph(G, pos, id2bib, renderer = renderer)


top_k = get_top_edges(edges, id2bib, title2path, FULL_TEXT_ARTICLES, top_k = 30)

# top_k is a list of tuples of the title, and list of paths assiciated

# as some articles have multiple paths indicated in metadata

top_k

# define  a function to easily recover and read any article



def get_paragraphs(file, read_only = True):

    # this method is used to recover and read top_k referenced files 

    # saved as a list of tuples --> [('title':['path1,', 'path2'])]

    # path is accesses as such

    if read_only:

        path = file[1][0]

    # if not used to read, the function takes just a path

    if not read_only:

        path = file

    text = []

    with open(path) as f:

        article = json.load(f)

        text.append(article['metadata']['title'])

        for para in article['abstract']:

            text.append(para['text'])

        for para in article['body_text']:

            text.append(para['text'])

    if read_only == True:

        for para in text:

            print(para, '\n')

    if read_only == False:

        return text


# get_paragraphs(top_k[2], read_only = True)
# ATTENTION this list might be different from the current data . It is based on the data I have downloaded,

# the paths and titles I've saved in the file directory BIOMED_DATA = "/kaggle/input/extradata2/labeld_biomed_data.csv"



# 25 articles which are purely biomedical talk

biomedical_articles = [top_k[0], 

                       top_k[2],

                       top_k[4],

                       top_k[5],

                       top_k[6],

                       top_k[7],

                       top_k[8],

                       top_k[9],

                       top_k[10],

                       top_k[13],

                       top_k[14],

                       top_k[15],

                       top_k[16],

                       top_k[17],

                       top_k[18],

                       top_k[19],

                       top_k[20],

                       top_k[21],

                       top_k[22],

                       top_k[23],

                       top_k[24],

                       top_k[26],

                       top_k[27],

                       top_k[28],

                       top_k[29]

                      ]

import pprint



pp = pprint.PrettyPrinter( indent = 4 )

print(' total number of articles : ', len(biomedical_articles))

pp.pprint(biomedical_articles)
# DONT RUN UNLESS ARTICLES READ / SELECTED . 



# THIS IS ALREADY DONE ON DOWNLOADED DATA AND INDICES WILL BE DIFFERE



# df_biomed_paths = pd.DataFrame(data = [{"title" :x[0], "path":x[1][0]} for x in biomedical_articles])

# df_biomed_paths.to_csv(os.path.join(OUTDIR, "labeled_biomed_data.csv"), sep = ',')
# to be looked at for further annotation after first roundof experiments

articles_containing_socioethical_talk = [

    # But as shown in this study, it is still crucial to isolate 

    # patients and trace and quarantine contacts as early as possible 

    top_k[1], 

    # further international seeding and subsequent local establishment of epidemics might become inevitable

    top_k[3],

    # heightened public awareness and impressively strong interventional...

    top_k[11],

    # quarantine and integrated interventions will have a major impact on its future trend

    top_k[12],

    # importance for healthcare settings and for travelers

    top_k[25],

]




def get_paragraphs_(path):

    neg_labeled = []

    pos_labeled = []

    with open(path, 'r', encoding = 'utf-8') as f:

        article = literal_eval(f.read())

        pos_labeled.append(article['metadata']['abstract'][0]['text'])

        pos_labeled.extend([x['text'] for x in article['metadata']['body_text'] if x['label']])

        neg_labeled.extend([x['text'] for x in article['metadata']['body_text'] if not x['label']])

    return pos_labeled, neg_labeled
pos_text, neg_text = get_paragraphs_("/kaggle/input/arxiv-data/arxiv01.txt")

print('Example of the first three paragraphs labeled text:')

pos_text[:3]
def get_entity_dfs(articles_paragraphs):

    ents_sci = [ent.text  for article in articles_paragraphs for para in article for ent in NER(para).ents]

    counts_sci = Counter(ents_sci)

    sci_ner = pd.DataFrame(data = [{"entity": ent, "count":count} for ent, count in counts_sci.most_common(15) ])

    ents_gen = [ent.text for article in articles_paragraphs for para in article for ent in NLP(para).ents if ent.label_ not in UNWANTED_ENTS]

    counts_gen = Counter(ents_gen)

    gen_ner = pd.DataFrame(data= [{"entity": ent, "count":count} for ent, count in counts_gen.most_common(15) ])

    return sci_ner, gen_ner


ethics_articles = [get_paragraphs_(path)[0] for path in SOCIOETHICS_ARTICLES]
if not NO_NER:

    ethics_sci_ner, ethics_gen_ner = get_entity_dfs(ethics_articles)
ethics_sci_ner
ethics_gen_ner 
if not NO_NER:

    biomed_articles = [get_paragraphs(path, read_only = False) for path in BIOMED_ARTICLES]



    bio_sci_ner, bio_gen_ner = get_entity_dfs(biomed_articles)
bio_sci_ner
bio_gen_ner
### the following code was used to save the dataframes used for demo



if not NO_NER:

    dfs_to_save = {"bio_sci_ner.pkl":bio_sci_ner, 

                   "bio_gen_ner.pkl": bio_gen_ner, 

                   "ethics_sci_ner.pkl":ethics_sci_ner, 

                   "ethics_gen_ner.pkl":ethics_gen_ner, 

                   "ex_sci_ner.pkl": ex_sci_ner, # these are spacy docs

                   "ex_gen_ner.pkl": ex_gen_ner, # these are spacy docs

                   "df_ner.pkl" : df_ner, 

                   "df_sci_ner.pkl":df_sci_ner, 

                   "df_lemmas.pkl":df_lemmas,

                   "df_tok_counts.pkl":df_tok_counts  # this is actually a list of ints...

                  }

    def save_pickle(fn, obj):

        with open(os.path.join(OUTDIR, EXTRA_DATA, fn), "wb") as f:

            pickle.dump(obj, f)





    for fn, obj in dfs_to_save.items():

        save_pickle(fn, obj)
def get_row_data(path):

    """return list of dictionaries one per paragraph with doc index, text, and doc path

    

    doc_index is the index within the document that the paragraph falls. 

    Title is doc index 0

    abstract paragraphs start at doc index 1 and paragraphs extend this indexation. 

    

    """

    row_data = []

    count = 0

    with open(path) as f:

        article = json.load(f)

        paragraph = dict()

        paragraph["text"] = article['metadata']['title']

        paragraph["doc_index"] = count

        paragraph["path"] = path

        row_data.append(paragraph)

        count +=1

        for para in article['abstract']:

            paragraph = dict()

            paragraph["text"] = para['text']

            paragraph["doc_index"] = count

            paragraph["path"] = path

            row_data.append(paragraph)

            count += 1

        for para in article['body_text']:

            paragraph = dict()

            paragraph["text"] = para['text']

            paragraph["doc_index"] = count

            paragraph["path"] = path

            row_data.append(paragraph)

            count += 1

    return row_data





def get_unlabeled_df(paths):

    all_data = []

    for path in paths:

        row_data = get_row_data(path)

        all_data.extend(row_data)

    df = pd.DataFrame(data = all_data)

    df = df.sample(frac=1) # we don't reset the index 

    return df





def split_unlabeled(unlabeled_df, num_split = 10):

    unlabeled_df

    seq_len = round(len(unlabeled_df)/num_split)

    list_dfs = []

    offset = 0

    for _ in range(num_split):

        if _ == num_split-1:

            list_dfs.append(unlabeled_df.iloc[offset:]) ## .reset_index()# witholding reset index

        else:

            popped_df = unlabeled_df.iloc[offset:offset+seq_len, :]  #.reset_index()  #holding aside reset index

            list_dfs.append(popped_df)

            offset += seq_len

    return list_dfs



def get_sci_ents(document):

    sci_ents = [ent.text for ent in NER(document).ents]

    # not sure if it's a great idea to do this but it seems the best mitigation

    # we need to have vectors of the same length, and worst case this will just pull

    # together texts in final feature vector space which have no interesting entities

    if not sci_ents:

        return  None

    return sci_ents

    

    

def get_gen_ents(document):   

    gen_ents = [ent.text for ent in NLP(document).ents if ent.label_ not in UNWANTED_ENTS]

    if not gen_ents:

        return None

    return gen_ents



def get_train_data(bio_paths, ethics_paths):

    neg_labeled = []

    pos_labeled = []

    for path in bio_paths:

        neg = get_paragraphs(path, read_only = False)

        neg_labeled.extend(neg)

    for path in ethics_paths:

        pos, neg = get_paragraphs_(path) ## function defined in 1.2.6

        pos_labeled.extend(pos)

        neg_labeled.extend(neg)

    len_pos = len(pos_labeled)

    len_neg = len(neg_labeled)

    # here we return the entitites for the labeled train data 

    # data structure is a list of tupples of 2 lists of entities

    # one list for scispacy entities and one list of general entities

    x_train = pos_labeled

    x_train.extend(neg_labeled)

    y_train = [1 for _ in range(len_pos)]

    y_train.extend([0 for _ in range(len_neg)])

    return x_train, y_train

if not NO_NER:

    unlabeled_paths = set(FULL_TEXT_ARTICLES) - set(BIOMED_ARTICLES)

    print('Total number of unlabeled articles : ', len(unlabeled_paths))







    selected = sample(list(unlabeled_paths), 15)



    unlabeled_paths = unlabeled_paths - set(selected)



    UNLABELED_DF = get_unlabeled_df(unlabeled_paths)

    DF_FOR_TEST = get_unlabeled_df(selected)

    LEN_UNLABELED = len(UNLABELED_DF)

    print('Total number of paragraphs in unlabeled data : ', LEN_UNLABELED)
if not NO_NER:

    x_train, y_train = get_train_data(BIOMED_ARTICLES, SOCIOETHICS_ARTICLES)

    # convert to numpy array and save y_train data

    y_train = np.array(y_train)

    np.save(os.path.join(OUTDIR, EXTRA_DATA, 'y_train.npy'), y_train)

    # instanciate global variable length to retrieve after vstack with unlabeled + PCA

    LEN_X_TRAIN = len(x_train)

    print("Total number of paragraphs in labeled train data : ", LEN_X_TRAIN)

if NO_NER:

    y_train = np.load(os.path.join(INDIR, EXTRA_DATA, 'y_train.npy'))
if not NO_NER:

    x_train_sci = [get_sci_ents(text) for text in x_train]

    x_train_gen = [get_gen_ents(text) for text in x_train]
if not NO_NER:

    with open(os.path.join(OUTDIR, EXTRA_DATA, "x_train_sci.pkl"), "wb") as f:

        pickle.dump(x_train_sci, f)

    with open(os.path.join(OUTDIR, EXTRA_DATA, "x_train_gen.pkl"), "wb") as f:

        pickle.dump(x_train_gen, f)
# define the number of times we would like to split the data

if not NO_NER:

    num_split = 100



    LIST_DFS = split_unlabeled(UNLABELED_DF, num_split = num_split)
def get_string_int(i):

    i += 1

    if i in list(range(10)):

        return '00'+str(i)

    if i in [i+10 for i in range(90)]:

        return '0'+str(i)

    if i >= 100:

        return str(i)





if not NO_NER:

    for i, df in enumerate(LIST_DFS):

        print("processing ulabeled dataframe ", i+1, '\n', "="*10)

        df["sci_ner"] = df['text'].apply(get_sci_ents)

        df["gen_ner"] = df['text'].apply(get_gen_ents)

        df = df.dropna(subset = ['sci_ner', 'gen_ner'])

        string_int = get_string_int(i)

        df.to_csv(ox.path.join(OUTDIR, LIST_DF_DIR, '{}_unlabeled_ner.csv'.format(string_int)), sep = ',')
if not NO_NER:

    DF_FOR_TEST["sci_ner"] = DF_FOR_TEST['text'].apply(get_sci_ents)

    DF_FOR_TEST["gen_ner"] = DF_FOR_TEST['text'].apply(get_gen_ents)

    DF_FOR_TEST.to_csv(os.path.join(OUTDIR, LIST_DF_DIR, 'DF_FOR_TEST_ner.csv'), sep = ',')


    ## discarded document embedding models for general entities and scientific entities

    ## this configuration is ideal in terms of domain adaptation, but was not running properly on kaggle

# sci_embeddings = BertEmbeddings(SCI_BERT)

# gen_embeddings = BertEmbeddings()# default goes to bert-large-uncased



# flair_embedding_forward = FlairEmbeddings('news-forward')

# flair_embedding_backward = FlairEmbeddings('news-backward')





# SCI_DOC_EMBED = DocumentPoolEmbeddings([sci_embeddings, flair_embedding_backward, flair_embedding_forward])

# GEN_DOC_EMBED = DocumentPoolEmbeddings([gen_embeddings, flair_embedding_backward, flair_embedding_forward])







#     document pooling operation to create a document level embedding

embeddings = WordEmbeddings('glove') 

DOC_EMBED = DocumentPoolEmbeddings([embeddings], fine_tune_mode='nonlinear')



## define the PCA model  ## for this version, we use PCA for 2d visualisation, setting components to 2 

PCA_MODEL = PCA(n_components = 2)









def embed_doc(document, doc_embed_model):

    sentence = ' '.join(document)

    sentence = Sentence(sentence)

    doc_embed_model.embed(sentence)

    embedding = sentence.get_embedding()

    return embedding.cpu().numpy()



    

def pca_doc(pca_model, all_embed):

    X = pca_model.fit_transform(all_embed)

    return pca_model, X
if NO_NER:

    with open(os.path.join(INDIR, EXTRA_DATA, "x_train_sci"), "rb") as f:

        x_train_sci = pickle.load(f)

    with open(os.path.join(INDIR, EXTRA_DATA, "x_train_gen"), "rb") as f:

        x_train_gen = pickle.load(f)
def drop_x_train_vals(x_sci, x_gen, y_train):

    x_train_sci = []

    x_train_gen = []

    y_train_ = []

    for s, g, y in zip(x_sci, x_gen, y_train):

        if s and g :

            x_train_sci.append(s)

            x_train_gen.append(g)

            y_train_.append(y)

    return x_train_sci, x_train_gen, y_train_



x_train_sci, x_train_gen, y_train = drop_x_train_vals(x_train_sci, x_train_gen, y_train)
LEN_X_TRAIN = len(x_train_sci)

print("length train data after reduce : ", len(y_train))
with torch.no_grad():

    x_train_sci_embed = [embed_doc(ents, DOC_EMBED) for ents in x_train_sci if ents]
with torch.no_grad():

        x_train_gen_embed = [embed_doc(ents, DOC_EMBED) for ents in x_train_gen if ents]
np.save(os.path.join(OUTDIR, EXTRA_DATA, "x_train_sci_embed.pkl"), np.array(x_train_sci_embed))

np.save(os.path.join(OUTDIR, EXTRA_DATA, "x_train_gen_embed.pkl"), np.array(x_train_gen_embed))
if NO_NER:

    LIST_DFS = [pd.read_csv(os.path.join(INDIR, LIST_DF_DIR, path)) 

                for path in sorted(os.listdir(os.path.join(INDIR, LIST_DF_DIR))) 

                if path.endswith('.csv')

                and not path.startswith('DF')]

# reset the lengths of dfs to 

LENGTHS = [len(df) for df in LIST_DFS]

LENGTHS
for i, df in enumerate(LIST_DFS):

    print("processing ulabeled dataframe {}/{}  ".format(i+1,len(LIST_DFS)), '\n', "="*30, '\n')

    with torch.no_grad(): # necessary to be able to save as numpy array as well as not take up GPU memory, as the doc pooling does not need bookkeeping

        sci_ner_embed = [embed_doc(x, DOC_EMBED) if x else np.array([]) for x in df["sci_ner"].to_list() ]

        gen_ner_embed = [embed_doc(x, DOC_EMBED) if x else np.array([]) for x in df["gen_ner"].to_list()]

    string_int = get_string_int(i)

    sci_ner_embed = np.array(sci_ner_embed)

    np.save(os.path.join(OUTDIR, LIST_DF_DIR, '{}unlabeled_sci_embed.npy'.format(string_int)), sci_ner_embed) 

    np.save(os.path.join(OUTDIR, LIST_DF_DIR, '{}unlabeled_gen_embed.npy'.format(string_int)), gen_ner_embed)

    del sci_ner_embed

    del gen_ner_embed

    torch.cuda.empty_cache()
sci_embeddings = [os.path.join(OUTDIR, LIST_DF_DIR, path) for path in sorted(os.listdir(os.path.join(OUTDIR, LIST_DF_DIR))) if path.endswith("sci_embed.npy")]

gen_embeddings = [os.path.join(OUTDIR, LIST_DF_DIR, path) for path in sorted(os.listdir(os.path.join(OUTDIR, LIST_DF_DIR))) if path.endswith("gen_embed.npy")]


all_sci = []

all_gen = []

for sci_embed, gen_embed in zip(sci_embeddings, gen_embeddings):

    sci_embed_ = np.load(sci_embed)

    gen_embed_ = np.load(gen_embed)

    all_sci.extend(sci_embed_)

    all_gen.extend(gen_embed_)

EMBEDDED_UNLABELED = np.hstack((np.array(all_sci), np.array(all_gen)))

all_sci.extend(x_train_sci_embed)

all_gen.extend(x_train_gen_embed)



all_sci = np.array(all_sci)

print(all_sci.shape)

all_gen = np.array(all_gen)

print(all_gen.shape)



all_embed = np.hstack((all_sci, all_gen))               

np.save(os.path.join(OUTDIR, EXTRA_DATA, 'all_embed.npy'), all_embed)

# for now only using PCA for plot, it would have been interesting otherwise to PCA output from Flair contextual pooled embeddings, as dimensions are very high --> ~4k

PCA_MODEL_, plot_points = pca_doc(PCA_MODEL, all_embed)
def plot_pca(X, y_train):

    x = [_[0] for _ in X[sum(LENGTHS):]]

    y = [_[1] for _ in X[sum(LENGTHS):]]

    sns.scatterplot(x = x, y= y, hue = y_train, alpha = .5)    

    
plot_pca(plot_points, y_train)


X = list(all_embed)



if len(X) != LEN_X_TRAIN+sum(LENGTHS):

    print("dimension difference", len(X), "is unequal to ", LEN_X_TRAIN+sum(LENGTHS))





X_train = X[sum(LENGTHS):]

X_train = np.array(X_train)

np.save(os.path.join(OUTDIR, 'X_train.npy'), X_train)

# liblinear solver is good for small datasets according to documentation



# f1 scoring in order to get a better idea of if our model 



# class_weight should be set at values representative of the data, but for now this is the only solution to get model to not only predict one of two classes





ml_model = LogisticRegressionCV(cv = 5, max_iter = 100, solver = 'liblinear', class_weight = {0:0.5, 1:0.5}, scoring = 'f1')
def semi_sup(model, x_train, y_train, embedded_unlabeled, batch_size = 200, threshold = .65):

    """Semisupervised training

    

    :param model: the sklearn compatible machine learning model used

    :param x_train: the labeled train input, list or array of np arrays (features)

    :param y_trian: the train labels np array of labels

    :param x_test: the labeled test input, list or array of np arrays (features)

    :param embedded_unlabeled: all unlabled feature vectors

    :param batch_size: number of times to split the data

    :param threshold: the threshold above which we decide to keep a predicted label as a pseudo label

    

    :return:

        best semi supervised ML model

        prints of scores

    

    """

    

    stacked = False

    num_batches = round(len(embedded_unlabeled)/batch_size)

    split_embedded_unlabeled = [embedded_unlabeled[i*batch_size : i*batch_size+batch_size] for i in range(num_batches)]

    original_x_train = x_train

    original_y_train = y_train

    f1_scores = []

    models_to_evaluate = []

    for num_iter, unlabeled_x in enumerate(split_embedded_unlabeled):

        # we have already droped Nones so to get the input data we just need to convert to np array

        model.fit(x_train, y_train)

        preds = model.predict(unlabeled_x)

        probas = model.predict_proba(unlabeled_x)

        pseudo_labeled_x = []

        pseudo_labeled_y = []

        for i, (pred, proba) in enumerate(zip(preds, probas)):

            # getting the probability is just the locating in the probas the index of the prediction 

            if proba[pred] > threshold:

                pseudo_labeled_x.append(unlabeled_x[i])

                pseudo_labeled_y.append(pred)

        x_train = np.vstack((x_train, np.array(pseudo_labeled_x)))

        y_train = np.hstack((y_train, np.array(pseudo_labeled_y)))

        # here's something perhaps a bit uncouth... I keep pseudo labels as train data for final evaluation

        # considering that there is not enough labeled data to withold for a test set...

        if stacked == False:

            if len(pseudo_labeled_y) > 1:

                pseudo_x_train_for_eval = np.array(pseudo_labeled_x)

                pseudo_y_train_for_eval = np.array(pseudo_labeled_y)

                stacked = True

        if stacked == True:

            pseudo_x_train_for_eval = np.vstack((pseudo_x_train_for_eval, np.array(pseudo_labeled_x)))

            pseudo_y_train_for_eval = np.hstack((pseudo_y_train_for_eval, np.array(pseudo_labeled_y)))

        model_to_evaluate = model

        if num_iter % 2 == 0:

            print(" iteration # : ", num_iter+1)

            print("size of batch : ", unlabeled_x.shape)

            print("train size : ", x_train.shape)

            if len([x for x in pseudo_y_train_for_eval if x  == 1]) > 10 and len([x for x in pseudo_y_train_for_eval if x == 0]) > 10:

                score = evaluate(model, pseudo_x_train_for_eval, pseudo_y_train_for_eval, original_x_train, original_y_train, num_iter)

                f1_scores.append(score)

                models_to_evaluate.append(model_to_evaluate)

                best_score = 0

                score_decrease = 0

                for score, model_to_eval in zip(f1_scores, models_to_evaluate):

                    if score > best_score:

                        best_score = score

                        best_model = model_to_eval

                        score_decrease = 0

                    else:

                        score_decrease += best_score - score

                if score_decrease > .15:

                    print("training stopped, cumulative score decrease greater than 15 % : ", score_decrease)

                    return best_model

                if num_iter > 9 and max(f1_scores) in f1_scores[:int(num_iter/2)] :

                    print("training stopped, max score is in first half iterations  ", num_iter+1, "iterations")

                    return best_model

            else:

                print(pseudo_y_train_for_eval)

                print("Not enough predictions in one of the two classes for eval, set threshold to at most .60")

    return best_model



              

              

def evaluate(model, pseudo_x, pseudo_y, x_train, y_train, num_iter):

    model.fit(pseudo_x, pseudo_y)         

    y_pred = model.predict(x_train)

    f1 = f1_score(y_train, y_pred, average = 'binary', zero_division = 0 )

    print('F1 measure after ', num_iter+1, 'iterations : ', f1 )

    return f1


def get_prediction_probabilities(unlabeled_x, x_train, y_train):

#     unlabeled_x = np.array(df["doc_embed"].tolist(), dtype=float)

    unlabeled_index = {i:idx for i, idx in enumerate(df.index.to_list())}

    ml_model.fit(x_train, y_train)

    preds = ml_model.predict(unlabeled_x)

    probas = ml_model.predict_proba(unlabeled_x)

    proba_dist = []

    proba_dist_pos = []

    proba_dist_neg = []

    for pred, proba in zip(preds, probas):

        proba_dist.append(proba[pred])

        if pred:

            proba_dist_pos.append(proba[pred])

        if not pred:

            proba_dist_neg.append(proba[pred])

    return proba_dist, proba_dist_pos, proba_dist_neg

def plot_proba(proba_dist, dist_name):

    lower_q = np.percentile(np.array(proba_dist), 25)

    upper_q = np.percentile(np.array(proba_dist), 75)

    threshold = np.percentile(np.array(proba_dist), 80)

    sns.distplot(proba_dist)

    print("""It seems like a lot of predicitons are between {}% and {}%... 

          We don't want to keep these 'unsure data' so, we could try {} 

          as a threshold for {} labels""".format(round(lower_q, 2), round(upper_q, 2), round(threshold, 2), dist_name))

    return threshold
proba_dist, proba_dist_pos, proba_dist_neg = get_prediction_probabilities(EMBEDDED_UNLABELED[:1000], X_train, y_train)
thresh_all = plot_proba(proba_dist, "all")
thresh_pos = plot_proba(proba_dist_pos, "positive")
thresh_neg = plot_proba(proba_dist_neg, "negative")
ml_model_trained = semi_sup(ml_model, X_train, y_train, EMBEDDED_UNLABELED, batch_size = 900, threshold = min([thresh_all, thresh_pos, thresh_neg, 0.6]))

with open(os.path.join(OUTDIR, EXTRA_DATA, "Semi_Sup_Log_Reg"), "wb") as f:   

    pickle.dump(ml_model_trained, f)
def transform_paragraph(paragraphs):

    sci_ner = []

    gen_ner = []

    sci_ner_embed = []

    gen_ner_embed = []

    for para in paragraphs:

        sci_ner.append(get_sci_ents(para))

        gen_ner.append(get_gen_ents(para))

    for sci, gen in zip(sci_ner, gen_ner):

        sci_ner_embed.append(embed_doc(sci, DOC_EMBED))

        gen_ner_embed.append(embed_doc(gen, DOC_EMBED))

    

    sci_ner_embed = np.array(sci_ner_embed)

    gen_ner_embed = np.array(gen_ner_embed)

    all_embed = np.hstack((sci_ner_embed, gen_ner_embed))

    return all_embed









def read_predicted(article_path, model):

    paragraphs = get_paragraphs_(article_path, read_only = False)

    doc_reps = transform_paragraph(paragraphs)

    preds = model.predict(doc_reps)

    

    for para, pred in zip(paragraphs, preds):

        if not pred:

            print(para, '\n')

        if pred:

            print(colored(para, color = None, on_color = 'on_yellow'), '\n')

    

def kaggle_demo(article_path, example_sci_ents, example_gen_ents,paragraphs, model):

    sci_ner_embed = []

    gen_ner_embed = []

    para_predicted = []

    for sci, gen, para in zip(example_sci_ents, example_gen_ents, paragraphs):

        if sci and gen:

            with torch.no_grad():

                sci_ner_embed.append(embed_doc(sci, DOC_EMBED))

                gen_ner_embed.append(embed_doc(gen, DOC_EMBED))

                para_predicted.append(para)

    sci_ner_embed = np.array(sci_ner_embed)

    gen_ner_embed = np.array(gen_ner_embed)

    all_embed = np.hstack((sci_ner_embed, gen_ner_embed))

    

    preds = model.predict(all_embed)

    for para, pred in zip(para_predicted, preds):

        if not pred:

            print(para, '\n\n')

        

        if pred:

            print(colored(para, color = None, on_color = 'on_yellow'), '\n\n')

    
article_path = "/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/02201e4601ab0eb70b6c26480cf2bfeae2625193.json"

## some processing done on my machine to run NER pipelines

if not NO_NER:

    paragraphs = get_paragraphs(article_path, read_only = False)



    example_sci_ents = get_sci_ents(paragraphs)

    example_gen_ents = get_gen_ents(paragraphs)



    with open(os.path.join(OUTDIR, "example_sci_ents.pkl"), "wb") as f:

        pickle.dump(example_sci_ents, f)

    with open(os.path.join(OUTDIR, "example_gen_ents.pkl"), "wb") as f:

        pickle.dump(example_gen_ents, f)
if NO_NER:

    with open(os.path.join(INDIR, EXTRA_DATA, "example_sci_ents.pkl"), "rb") as f:

        example_sci_ents = pickle.load(f)

    with open(os.path.join(INDIR, EXTRA_DATA, "example_gen_ents.pkl"), "rb") as f:

        example_gen_ents = pickle.load(f)

        

        

paragraphs = get_paragraphs(article_path, read_only = False)

kaggle_demo(article_path, example_sci_ents, example_gen_ents, paragraphs, ml_model_trained)
df_for_kaggle_demo = pd.read_csv(os.path.join(INDIR, LIST_DF_DIR, "DF_FOR_TEST_ner.csv"))

article_paths = set(df_for_kaggle_demo["path"].to_list())
article_paths
for article in article_paths:

    print("processing article, ", article)

    df_article = df_for_kaggle_demo.loc[df_for_kaggle_demo["path"] == article].sort_values("doc_index", axis = 0)

    df_article = df_article.dropna(subset = ["sci_ner", "gen_ner"])

    sci_ents = [literal_eval(x) for x in df_article.sci_ner.to_list()]

    gen_ents = [literal_eval(x) for x in df_article.gen_ner.to_list()]

    paragraphs = [x for x in df_article.text.to_list()]

    kaggle_demo(article_path, sci_ents, gen_ents, paragraphs, ml_model_trained)

    print("===================================================================", '\n'*3)

    