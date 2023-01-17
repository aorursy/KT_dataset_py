!pip install -U stanza
import os

import json

import stanza

import hashlib

import numpy as np

import pandas as pd
stanza.download('en')
nlp = stanza.Pipeline(processors='tokenize', lang='en', use_gpu=True)
!git clone https://github.com/COVIEWED/coviewed_web_scraping
!pip install -r coviewed_web_scraping/requirements.txt
EXAMPLE_URL = 'https://www.euronews.com/2020/04/01/the-best-way-prevent-future-pandemics-like-coronavirus-stop-eating-meat-and-go-vegan-view'

print(EXAMPLE_URL)
!cd coviewed_web_scraping/ && python3 src/scrape.py -u={EXAMPLE_URL}
data_path = 'coviewed_web_scraping/data/'

fname = [f for f in os.listdir(data_path) if f.endswith('.txt')][0]

print(fname)

with open(os.path.join(data_path, fname), 'r') as my_file:

    txt_data = my_file.readlines()

txt_data = [line.strip() for line in txt_data if line.strip()]

len(txt_data)
article_url = txt_data[0]

print(article_url)

article_published_datetime = txt_data[1]

print(article_published_datetime)
article_title = txt_data[2]

print(article_title)
article_text = "\n\n".join(txt_data[3:])

print(article_text)
ALL_SENTENCES = []

txt = [p.strip() for p in article_text.split('\n') if p.strip()]

file_id = fname.split('.')[0]

print(file_id)

print()

for i, paragraph in enumerate(txt):

    doc = nlp(paragraph)

    for sent in doc.sentences:

        S = ' '.join([w.text for w in sent.words])

        sH = hashlib.md5(S.encode('utf-8')).hexdigest()

        print(sH)

        print(S)

        print()

        ALL_SENTENCES.append([file_id, sH, S])

fname = file_id+'_sentences.tsv'

print(fname)

AS = pd.DataFrame(ALL_SENTENCES, columns=['file_id','sentenceHash','sentence'])

len(AS)
AS.sample(n=min(len(AS),3))
AS.to_csv(fname, sep='\t', index=False, index_label=False)