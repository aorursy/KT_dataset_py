import os



dirs_with_jsons = [

    '../data/raw/2020-03-13/biorxiv_medrxiv/biorxiv_medrxiv',

    '../data/raw/2020-03-13/comm_use_subset/comm_use_subset',

    '../data/raw/2020-03-13/noncomm_use_subset/noncomm_use_subset',

    '../data/raw/2020-03-13/pmc_custom_license/pmc_custom_license'

]



json_list = []



for dir_to_parse in dirs_with_jsons:

    for file in os.scandir(dir_to_parse):

        if file.name.split('.')[-1] == 'json':

            json_list.append(dir_to_parse + '/' + file.name)
len(json_list)
import json



with open(json_list[0], 'r') as file:

    data = json.load(file)
data['body_text'][0]['text']
import re



regex = "\([\d, i]+\)"

re.sub(regex, '', data['body_text'][0]['text'])
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize



sent_tokenize(data['body_text'][0]['text'])
from tqdm.auto import tqdm



index = []

docs = []

for doc_file in tqdm(json_list, unit='Papers'):

    with open(doc_file, 'r') as file:

        data = json.load(file)

        

        for parragraph_data in data['body_text']:

            parragraph = re.sub("\([\d, i]+\)", '', parragraph_data['text']) + '\n'

            

            for sentence in sent_tokenize(parragraph):

                index.append([len(docs), data['paper_id'], data['metadata']['title'], sentence])

                docs.append(sentence)
import pandas as pd



sentence_index = pd.DataFrame(index, columns=['sentence_id', 'paper_id', 'paper_title', 'sentence'])
sentence_index.sample(n=5)
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer



vectorizer = CountVectorizer()

counts = vectorizer.fit_transform(docs)
transformer = TfidfTransformer()

pcount = transformer.fit_transform(counts)
dcounts = counts.sum(axis=0)[0]
word_usages = pd.DataFrame({'wcount': dcounts.tolist()[0], 'word': vectorizer.get_feature_names()})
import seaborn as sns

import matplotlib.pyplot as plt



def plot_dist(row, limit = None, title = None):

    fig = plt.figure(figsize=(16.5, 8))

    sns.distplot(row, bins=50)

    

    if limit:

        plt.axvline(limit, label="", color='tab:green')

        stats = '{} < {} < {}'.format(sum(row < limit), limit, sum(row > limit))

        ylim = fig.axes[0].get_ylim()

        yloc = ((ylim[1] - ylim[0]) * .8) - ylim[0]

        plt.annotate(stats, (limit, yloc), fontsize='large', horizontalalignment='center')

        

    if title:

        plt.title(title)

    

    plt.show()



    

plot_dist(word_usages.query('wcount < 10').wcount)
word_usages.query('2 < wcount').sort_values(by='wcount')
# Exploring json structure



def print_keys(dict_to_traverse: dict, indent = ''):

    for key, value in dict_to_traverse.items():

        print(f'{indent}{key}')

        if type(value) == dict:

            print_keys(value, f'-{indent}')



print_keys(data)