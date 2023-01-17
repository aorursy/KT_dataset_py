# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



datafiles = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        ifile = os.path.join(dirname, filename)

        if ifile.split(".")[-1] == "json":

            datafiles.append(ifile)

        #print(ifile)



# Any results you write to the current directory are saved as output.
len(datafiles)
datafiles[0:2]
with open(datafiles[0], 'r')as f1:

    sample = json.load(f1)
for key,value in sample.items():

    print(key)
print(sample['metadata'].keys())

print('abstract: ',sample['abstract'][0].keys())

print('body_text: ',sample['body_text'][0].keys())

print('bib_entries: ',sample['bib_entries'].keys())

print('ref_entries: ', sample['ref_entries'].keys())

print('back_matter: ',sample['back_matter'][0].keys())
id2title = []

for file in datafiles:

    with open(file,'r')as f:

        doc = json.load(f)

    id = doc['paper_id'] 

    title = doc['metadata']['title']

    id2title.append({id:title})
id2title[0:3]
#with open('id2title.json','w')as f2:

#    json.dump(id2title,f2)
id2abstract = []

for file in datafiles:

    with open(file,'r')as f:

        doc = json.load(f)

    id = doc['paper_id'] 

    abstract = ''

    for item in doc['abstract']:

        abstract = abstract + item['text']

        

    id2abstract.append({id:abstract})
id2abstract[0]
#with open('id2abstract.json','w')as f3:

#   json.dump(id2abstract,f3)
id2bodytext = []

for file in datafiles:

    with open(file,'r')as f:

        doc = json.load(f)

    id = doc['paper_id'] 

    bodytext = ''

    for item in doc['body_text']:

        bodytext = bodytext + item['text']

        

    id2bodytext.append({id:bodytext})
#id2bodytext[0]
#with open('id2bodytext.json','w')as f4:

#    json.dump(id2bodytext,f4)
bibEntries = []

for key,value in sample['bib_entries'].items():

    refid = key

    title = value['title']

    year = value['year']

    venue = value['venue']

    try:

        DOI = value['other_ids']['DOI'][0]

    except:

        DOI = 'NA'

        

    bibEntries.append({"refid": refid,\

                      "title":title,\

                      "year": year,\

                      "venue":venue,\

                      "DOI": DOI})
bibEntries[0:2]
import networkx as nx
G = nx.Graph()

G.add_node(sample['paper_id'])

for item in bibEntries:

    G.add_node(item["refid"], title = item['title'], year = item['year'], venue = item['venue'])

    G.add_edge(sample['paper_id'], item["refid"])  
len(G.nodes())
import matplotlib.pyplot as plt



plt.figure(figsize = [10,8])

pos = nx.spring_layout(G)

nx.draw(G,with_labels=True, node_size =1500, node_color = 'lightblue')

plt.savefig('ref.png')
for item in list(G.nodes().data('venue')):

    print(item)
for item in list(G.nodes().data('title')):

    print(item)
for item in list(G.nodes().data('year')):

    print(item)