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
id2bib = []

for file in datafiles:

    '''id and title of a single document'''

    with open(file,'r')as f:

        doc = json.load(f)

    id = doc['paper_id']

    title = doc['metadata']['title']

    

    '''collect bib-entries of a single document'''

    bibEntries = []

    for key,value in doc['bib_entries'].items():

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

    id2bib.append({"id": id, "bib": bibEntries,"title": title})
id2bib[0]
import networkx as nx



G = nx.Graph()

for item in id2bib:

    '''iterate over each doc'''

    G.add_node(item['id'],title = item['title'])

    for ref in item['bib']:

        '''iterate over each reference'''

        G.add_node(ref['DOI'], title = ref['title'], year = ref['year'], venue = ref['venue'])

        G.add_edge(item['id'], ref['DOI'], value = ref['refid'])  
'''How many nodes are there in my network?'''

#155339 nodes

len(G.nodes())
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
Gs = nx.Graph()

for item in id2bib[0:200]:

    for ref in item['bib']:

        '''select the title which includes -virus- term'''

        if ref['title'].find('virus'):

            Gs.add_node(item['id'])

            Gs.add_node(ref['DOI'])

            Gs.add_edge(item['id'], ref['DOI'])  
import seaborn as sns

sns.set()
plt.figure(figsize = [12,10]) 

pos = nx.spring_layout(Gs) 

nx.draw(Gs, with_labels=False, node_size = 1, node_color = 'lightblue') 

plt.savefig('cite.png')
len(Gs.nodes())
Ga = nx.Graph()

for item in id2bib[0:10]:

    Ga.add_node(item['id'],title = item['title'])

    for ref in item['bib']:

        Ga.add_node(ref['DOI'], title = ref['title'], year = ref['year'], venue = ref['venue'])

        Ga.add_edge(item['id'], ref['DOI'], value = ref['refid'])  
i = 0

for item in Ga.nodes().data():

    i += 1

    print(item)

    if i>10:

        break
DegreeCentrality = nx.degree_centrality(Ga)

ID = max(DegreeCentrality)

ID, DegreeCentrality[ID]
'''the node title which has maximum paper cited on virus'''

Ga.nodes[ID]
'''the titles of the cited papers'''

for item in Ga.neighbors(ID):

    print(Ga.nodes[item])