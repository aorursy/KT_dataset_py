import numpy as np

import pandas as pd

from pathlib import Path, PurePath

import pandas as pd

import requests

from requests.exceptions import HTTPError, ConnectionError

from ipywidgets import interact

import ipywidgets as widgets

import nltk

from nltk.corpus import stopwords

nltk.download("punkt")

pd.options.display.max_colwidth = 500
input_dir = PurePath('../input/CORD-19-research-challenge/2020-03-13')



metadata = pd.read_csv(input_dir / 'all_sources_metadata_2020-03-13.csv', 

                      dtype={'Microsoft Academic Paper ID': str,

                             'pubmed_id': str})



def doi_url(d): return f'http://{d}' if d.startswith('doi.org') else f'http://doi.org/{d}'

metadata.doi = metadata.doi.fillna('').apply(doi_url)



metadata.abstract = metadata.abstract.fillna(metadata.title)
duplicate_paper = ~(metadata.title.isnull() | metadata.abstract.isnull()) & (metadata.duplicated(subset=['title', 'abstract']))

metadata = metadata[~duplicate_paper].reset_index(drop=True)
def get(url, timeout=6):

    try:

        r = requests.get(url, timeout=timeout)

        return r.text

    except ConnectionError:

        print(f'Cannot connect to {url}')

        print(f'Remember to turn Internet ON in the Kaggle notebook settings')

    except HTTPError:

        print('Got http error', r.status, r.text)



class DataHolder:

    def __init__(self, data: pd.DataFrame):

        self.data = data

        

    def __len__(self): return len(self.data)

    def __getitem__(self, item): return self.data.loc[item]

    def head(self, n:int): return DataHolder(self.data.head(n).copy())

    def tail(self, n:int): return DataHolder(self.data.tail(n).copy())

    def _repr_html_(self): return self.data._repr_html_()

    def __repr__(self): return self.data.__repr__()





class ResearchPapers:

    

    def __init__(self, metadata: pd.DataFrame):

        self.metadata = metadata

        

    def __getitem__(self, item):

        return Paper(self.metadata.iloc[item])

    

    def __len__(self):

        return len(self.metadata)

    

    def head(self, n):

        return ResearchPapers(self.metadata.head(n).copy().reset_index(drop=True))

    

    def tail(self, n):

        return ResearchPapers(self.metadata.tail(n).copy().reset_index(drop=True))

    

    def abstracts(self):

        return self.metadata.abstract.dropna()

    

    def titles(self):

        return self.metadata.title.dropna()

        

    def _repr_html_(self):

        return self.metadata._repr_html_()

    

class Paper:

    

    def __init__(self, item):

        self.paper = item.to_frame().fillna('')

        self.paper.columns = ['Value']

    

    def doi(self):

        return self.paper.loc['doi'].values[0]

    

    def html(self):

        text = get(self.doi())

        return widgets.HTML(text)

    

    def text(self):

        text = get(self.doi())

        return text

    

    def abstract(self):

        return self.paper.loc['abstract'].values[0]

    

    def title(self):

        return self.paper.loc['title'].values[0]

    

    def authors(self, split=False):

        authors = self.paper.loc['authors'].values[0]

        if not authors:

            return []

        if not split:

            return authors

        if authors.startswith('['):

            authors = authors.lstrip('[').rstrip(']')

            return [a.strip().replace("\'", "") for a in authors.split("\',")]



        return [a.strip() for a in authors.split(';')]

        

    def _repr_html_(self):

        return self.paper._repr_html_()

    



papers = ResearchPapers(metadata)
!pip install rank_bm25

from rank_bm25 import BM25Okapi
english_stopwords = list(set(stopwords.words('english')))



def tokenize(text):

    words = nltk.word_tokenize(text)

    return list(set([word for word in words if word.isalnum() 

                                             and not word in english_stopwords

                                             and not (word.isnumeric() and len(word) < 4)]))

def preprocess(string):

    return tokenize(string.lower())



class SearchResults:

    

    def __init__(self, 

                 data: pd.DataFrame,

                 columns = None):

        self.results = data

        if columns:

            self.results = self.results[columns]

            

    def __getitem__(self, item):

        return Paper(self.results.loc[item])

    

    def __len__(self):

        return len(self.results)

        

    def _repr_html_(self):

        return self.results._repr_html_()

    

    def __return_data__(self):

        return self.results



SEARCH_DISPLAY_COLUMNS = ['title', 'abstract', 'doi', 'authors', 'journal']



class WordTokenIndex:

    

    def __init__(self, 

                 corpus: pd.DataFrame, 

                 columns=SEARCH_DISPLAY_COLUMNS):

        self.corpus = corpus

        raw_search_str = self.corpus.abstract.fillna('') + ' ' + self.corpus.title.fillna('')

        self.index = raw_search_str.apply(preprocess).to_frame()

        self.index.columns = ['terms']

        self.index.index = self.corpus.index

        self.columns = columns

    

    def search(self, search_string):

        search_terms = preprocess(search_string)

        result_index = self.index.terms.apply(lambda terms: any(i in terms for i in search_terms))

        results = self.corpus[result_index].copy().reset_index().rename(columns={'index':'paper'})

        return SearchResults(results, self.columns + ['paper'])
class RankBM25Index(WordTokenIndex):

    

    def __init__(self, corpus: pd.DataFrame, columns=SEARCH_DISPLAY_COLUMNS):

        super().__init__(corpus, columns)

        self.bm25 = BM25Okapi(self.index.terms.tolist())

        

    def search(self, search_string, n=4):

        search_terms = preprocess(search_string)

        doc_scores = self.bm25.get_scores(search_terms)

        ind = np.argsort(doc_scores)[::-1][:n]

        results = self.corpus.iloc[ind][self.columns]

        results['Score'] = doc_scores[ind]

        results = results[results.Score > 0]

        return SearchResults(results.reset_index(), self.columns + ['Score'])
bm25_index = RankBM25Index(metadata.head(10_000))
tasks = [('What is known about transmission, incubation, and environmental stability?', 

        'transmission incubation environment coronavirus'),

        ('What do we know about COVID-19 risk factors?', 'risk factors'),

        ('What do we know about virus genetics, origin, and evolution?', 'genetics origin evolution'),

        ('What has been published about ethical and social science considerations','ethics ethical social'),

        ('What do we know about diagnostics and surveillance?','diagnose diagnostic surveillance'),

        ('What has been published about medical care?', 'medical care'),

        ('What do we know about vaccines and therapeutics?', 'vaccines vaccine vaccinate therapeutic therapeutics')] 

tasks = pd.DataFrame(tasks, columns=['Task', 'Keywords'])
from html.parser import HTMLParser

import urllib.request as urllib2

class MyHTMLParser(HTMLParser):



    #Initializing lists

    lsStartTags = list()

    lsEndTags = list()

    lsStartEndTags = list()

    lsComments = list()



    #HTML Parser Methods

    def handle_starttag(self, startTag, attrs):

        self.lsStartTags.append(startTag)



    def handle_endtag(self, endTag):

        self.lsEndTags.append(endTag)



    def handle_startendtag(self,startendTag, attrs):

        self.lsStartEndTags.append(startendTag)



    def handle_comment(self,data):

        self.lsComments.append(data)
!pip install html2text
import html2text

html2text.html2text(bm25_index.search('What do we know about virus genetics, origin, and evolution?',n=200).__getitem__(0)._repr_html_())
bm25_index.search('What do we know about virus genetics, origin, and evolution?',n=200).__getitem__(0).abstract()
import bs4 as bs

import urllib.request

import re



scraped_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Artificial_intelligence')

article = scraped_data.read()



parsed_article = bs.BeautifulSoup(article,'lxml')



paragraphs = parsed_article.find_all('p')



article_text = ""



for p in paragraphs:

    article_text += p.text
article_text
bm25_index.search('What do we know about virus genetics, origin, and evolution?',n=200).__getitem__(0).abstract()