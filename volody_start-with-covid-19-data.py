import numpy as np

import pandas as pd

import nltk
metadata = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv",

                        na_values=[], keep_default_na=False)

metadata.head()
metadata.info()
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

from nltk.corpus import stopwords 

from string import punctuation



stop_words = set(stopwords.words('english'))

p_chars = set([char for char in punctuation])



def extract(items):

    records = []

    for x in items:

        if(isinstance(x, str)):

            records.append(x)

    return records



def tokenize(items):

    records = []

    for x in items:

        for word in nltk.word_tokenize(x):

            word = word.lower()

            if not word in (stop_words|p_chars):

                records.append(word)

    return records



def ShowCloud(text, title):

    # build

    wc = WordCloud(max_font_size=50, background_color="white", 

                   collocations=False,

                   max_words=100, stopwords=STOPWORDS)

    wc.generate(" ".join(text))

    # plot

    plt.figure(figsize=(20,10))

    plt.axis("off")

    plt.title(title, fontsize=20)

    plt.imshow(wc, interpolation="bilinear")

    plt.show()    
titles = extract(metadata.title)

title_tokens = tokenize(titles)

print(f"total words in title: {len(title_tokens)}")
ShowCloud(title_tokens, "Frequent words in titles")
abstracts = extract(metadata.abstract)

abstract_tokens = tokenize(abstracts)

print(f"total words in abstract: {len(abstract_tokens)}")
ShowCloud(abstract_tokens, "Frequent words in abstract")
from nltk.probability import FreqDist



def builtFreq(data):

    porter = nltk.PorterStemmer()

    lemma = nltk.WordNetLemmatizer()

    stems = [porter.stem(t) for t in data]

    words = [lemma.lemmatize(t) for t in stems]

    return FreqDist(words)
title_dist = builtFreq(title_tokens)

title_dist.most_common()[:10]
import matplotlib.pyplot as plt



def plotFreq(dist, count, title):

    plt.figure(figsize=(20, 8))

    plt.title(title, size = 40)

    plt.xticks(size = 20)

    plt.yticks(size = 20)

    dist.plot(count)
plotFreq(title_dist, 50, "word frequency in title")
abstract_dist = builtFreq(abstract_tokens)

abstract_dist.most_common()[:10]
plotFreq(abstract_dist, 50, "word frequency in abstract")
import json



class FileReader:

    def __init__(self, file_path):

        with open(file_path) as file:

            content = json.load(file)

            self.paper_id = content['paper_id']

            self.abstract = []

            self.body_text = []

            # Abstract

            for entry in content['abstract']:

                self.abstract.append(entry['text'])

            # Body text

            for entry in content['body_text']:

                self.body_text.append(entry['text'])

            self.abstract = '\n'.join(self.abstract)

            self.body_text = '\n'.join(self.body_text)

    def __repr__(self):

        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'



def get_breaks(content, length):

    data = ""

    words = content.split(' ')

    total_chars = 0



    # add break every length characters

    for i in range(len(words)):

        total_chars += len(words[i])

        if total_chars > length:

            data = data + "<br>" + words[i]

            total_chars = 0

        else:

            data = data + " " + words[i]

    return data    

    

def process(files, meta_df):

    dic = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}

    for idx, entry in enumerate(files):

        if idx % (len(files) // 10) == 0:

            print(f'Processing index: {idx} of {len(files)}')



        content = FileReader(entry)



        # get metadata information

        meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

        # no metadata, skip this paper

        if len(meta_data) == 0:

            continue



        dic['paper_id'].append(content.paper_id)

        dic['abstract'].append(content.abstract)

        dic['body_text'].append(content.body_text)



        # also create a column for the summary of abstract to be used in a plot

        if len(content.abstract) == 0: 

            # no abstract provided

            dic['abstract_summary'].append("Not provided.")

        elif len(content.abstract.split(' ')) > 100:

            # abstract provided is too long for plot, take first 300 words append with ...

            info = content.abstract.split(' ')[:100]

            summary = get_breaks(' '.join(info), 40)

            dic['abstract_summary'].append(summary + "...")

        else:

            # abstract is short enough

            summary = get_breaks(content.abstract, 40)

            dic['abstract_summary'].append(summary)



        # get metadata information

        meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]



        try:

            # if more than one author

            authors = meta_data['authors'].values[0].split(';')

            if len(authors) > 2:

                # more than 2 authors, may be problem when plotting, so take first 2 append with ...

                dic['authors'].append(". ".join(authors[:2]) + "...")

            else:

                # authors will fit in plot

                dic['authors'].append(". ".join(authors))

        except Exception as e:

            # if only one author - or Null valie

            dic['authors'].append(meta_data['authors'].values[0])



        # add the title information, add breaks when needed

        try:

            title = get_breaks(meta_data['title'].values[0], 40)

            dic['title'].append(title)

        # if title was not provided

        except Exception as e:

            dic['title'].append(meta_data['title'].values[0])



        # add the journal information

        dic['journal'].append(meta_data['journal'].values[0])

    return dic
import glob



all_json = glob.glob('/kaggle/input/CORD-19-research-challenge/**/*.json', recursive=True)

len(all_json)
# get sample

first_row = FileReader(all_json[0])

print(first_row)
dict_ = process(all_json, metadata)

df_covid = pd.DataFrame(dict_)

df_covid.head()