import numpy as np 

import pandas as pd 



json_file_paths = []  # Intializing empty list to save full paths of json files to read them later on



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)

    for filename in filenames:

        a = os.path.join(dirname, filename)

        if a.endswith('.json') and 'biorxiv_medrxiv' in a:

            json_file_paths.append(a)

        elif a.endswith('.csv'):

            print(a)
# Number of json files

len(json_file_paths)
!pip install git+https://github.com/boudinfl/pke.git

!python -m nltk.downloader stopwords

!python -m nltk.downloader universal_tagset

!python -m spacy download en
!pip install scispacy

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_bc5cdr_md-0.2.4.tar.gz
import json, spacy, scispacy, en_ner_bc5cdr_md

import pandas as pd

from pke.unsupervised import TopicRank, TextRank, YAKE
def read_jsons(file_name):

    with open(file_name, 'r') as f:

        json_data = json.load(f)

    return json_data
data = pd.DataFrame([read_jsons(i) for i in  json_file_paths])
data = data.fillna(' ')
def process_abstract_body(value):

    if value != ' ':

        return " ".join(i['text'] for i in value)

    else:

        return ' '
def get_title(value):

    try:

        return value.get('title')

    except:

        return ' '
data.abstract = data.abstract.apply(process_abstract_body)
data.body_text = data.body_text.apply(process_abstract_body)
data["title"] = data.metadata.apply(get_title)
data = data[['paper_id', 'title', 'abstract', 'body_text']]
data.head()
def use_textrank(text, num_keyphrases=5):

    extractor = TextRank()

    extractor.load_document(text)

    extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'})

    extractor.candidate_weighting(window=2, top_percent=10)

    keys = []

    scores = []

    for (keyphrase, score) in extractor.get_n_best(n=num_keyphrases):

        keys.append(keyphrase)

        scores.append(score)

    return keys, scores
for i,j in zip(data.body_text[:5],data.title[:5]):

    keys, scores = use_textrank(i)

    key_data = pd.DataFrame({"keyphrases":keys, "scores":scores})

    print("Title is "+j)

    print(key_data)
def use_topicrank(text, num_keyphrases=10):

    extractor = TopicRank()

    extractor.load_document(text)

    extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'})

    extractor.candidate_weighting(threshold=0.7,

                                  method='average')

    keys = []

    scores = []

    for (keyphrase, score) in extractor.get_n_best(n=num_keyphrases):

        keys.append(keyphrase)

        scores.append(score)

    return keys, scores
for i,j in zip(data.body_text[:5],data.title[:5]):

    keys, scores = use_topicrank(i)

    key_data = pd.DataFrame({"keyphrases":keys, "scores":scores})

    print("Title is "+j)

    print(key_data)
nlp = en_ner_bc5cdr_md.load()
doc = nlp(data.body_text[0])

covid_ents = list(set([(i.text,i.label_) for i in doc.ents if i.label_ == 'DISEASE']))

covid_ents