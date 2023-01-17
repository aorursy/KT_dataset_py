from glob import glob

import json

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

import os
dir_list = [

    '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv',

    '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset',

    '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license',

    '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset'

]

results_list = list()

for target_dir in dir_list:

    

    print(target_dir)

    

    for json_fp in glob(target_dir + '/*.json'):



        with open(json_fp) as json_file:

            target_json = json.load(json_file)



        data_dict = dict()

        data_dict['doc_id'] = target_json['paper_id']

        data_dict['title'] = target_json['metadata']['title']



        abstract_section = str()

        for element in target_json['abstract']:

            abstract_section += element['text'] + ' '

        data_dict['abstract'] = abstract_section



        full_text_section = str()

        for element in target_json['body_text']:

            full_text_section += element['text'] + ' '

        data_dict['full_text'] = full_text_section

        

        results_list.append(data_dict)

        

    

df_results = pd.DataFrame(results_list)

df_results.head()        
articles=df_results['full_text'].str.lower().values
Remdesivir=[]

for text in articles:

    for sentences in text.split('.'):

        if 'remdesivir' in sentences:

            Remdesivir.append(sentences)          
stopwords = set(STOPWORDS)

wordcloud = WordCloud(stopwords =stopwords, width=1000, height=500).generate("+".join(Remdesivir))

plt.figure(figsize=(15,8))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
Chloroquine=[]

for text in articles:

    for sentences in text.split('.'):

        if 'chloroquine' in sentences:

            Chloroquine.append(sentences)         
wordcloud = WordCloud(stopwords =stopwords, width=1000, height=500).generate("+".join(Chloroquine))

plt.figure(figsize=(15,8))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
hydroxychloroquine=[]

for text in articles:

    for sentences in text.split('.'):

        if 'hydroxychloroquine' in sentences:

            hydroxychloroquine.append(sentences)            
wordcloud = WordCloud(stopwords =stopwords, width=1000, height=500).generate("+".join(hydroxychloroquine))

plt.figure(figsize=(15,8))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
lopinavir=[]

for text in articles:

    for sentences in text.split('.'):

        if 'lopinavir' in sentences:

            lopinavir.append(sentences)           
wordcloud = WordCloud(stopwords =stopwords, width=1000, height=500).generate("+".join(lopinavir))

plt.figure(figsize=(15,8))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
Ritonavir=[]

for text in articles:

    for sentences in text.split('.'):

        if 'ritonavir' in sentences:

            Ritonavir.append(sentences)          
wordcloud = WordCloud(stopwords =stopwords, width=1000, height=500).generate("+".join(Ritonavir))

plt.figure(figsize=(15,8))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
interferon_beta=[]

for text in articles:

    for sentences in text.split('.'):

        if 'interferon-beta' in sentences:

            interferon_beta.append(sentences) 
wordcloud = WordCloud(stopwords =stopwords, width=1000, height=500).generate("+".join(interferon_beta))

plt.figure(figsize=(15,8))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()