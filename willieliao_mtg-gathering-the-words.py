keeps = ['name', 'colorIdentity', 'colors', 'type', 'types', 'subtypes', 'supertypes', 'cmc', 'power', 'toughness', 'flavor', 'text', 'legalities']

colorIdentity_map = {'B': 'Black', 'G': 'Green', 'R': 'Red', 'U': 'Blue', 'W': 'White'}

plt_colors = ['k', 'b', '0.5', 'g', 'r', 'w', 'm']
import pandas as pd

import numpy as np

from numpy.random import random

from math import ceil



from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from nltk.stem.porter import PorterStemmer

from nltk import RegexpTokenizer

from nltk.corpus import stopwords



from wordcloud import WordCloud

import matplotlib.pyplot as plt

import seaborn as sns
raw = pd.read_json('../input/AllSets-x.json')

raw.shape
# Some data-fu to get all cards into a single table along with a couple of release columns.

mtg = []

for col in raw.columns.values:

    release = pd.DataFrame(raw[col]['cards'])

    release = release.loc[:, keeps]

    release['releaseName'] = raw[col]['name']

    release['releaseDate'] = raw[col]['releaseDate']

    mtg.append(release)

mtg = pd.concat(mtg)

del release, raw   

mtg.shape
# remove promo cards that aren't used in normal play

# Edit 2016-09-30: Could be null because it's too new to have a ruling on it.

mtg_nulls = mtg.loc[mtg.legalities.isnull()]

mtg = mtg.loc[~mtg.legalities.isnull()]



# remove tokens without types

mtg = mtg.loc[~mtg.types.apply(lambda x: isinstance(x, float))]



# Power and toughness that depends on board state or mana cannot be resolved

mtg[['power', 'toughness']] = mtg[['power', 'toughness']].apply(lambda x: pd.to_numeric(x, errors='coerce'))



# Fill

mtg.flavor.fillna('', inplace=True)

mtg.text.fillna('', inplace=True)

mtg.shape
# Combine colorIdentity and colors

mtg.loc[(mtg.colors.isnull()) & (mtg.colorIdentity.notnull()), 'colors'] = mtg.loc[(mtg.colors.isnull()) & (mtg.colorIdentity.notnull()), 'colorIdentity'].apply(lambda x: [colorIdentity_map[i] for i in x])

mtg['colorsCount'] = 0

mtg.loc[mtg.colors.notnull(), 'colorsCount'] = mtg.colors[mtg.colors.notnull()].apply(len)

mtg.loc[mtg.colors.isnull(), 'colors'] = ['Colorless']

mtg['colorsStr'] = mtg.colors.apply(lambda x: ''.join(x))



# Include colorless and multi-color.

mtg['manaColors'] = mtg['colorsStr']

mtg.loc[mtg.colorsCount>1, 'manaColors'] = 'Multi'



# Materialize color columns

mono_colors = np.sort(mtg.colorsStr[mtg.colorsCount<=1].unique()).tolist()



for color in mono_colors:

    mtg[color] = mtg.colors.apply(lambda x: color in x)
wc = WordCloud(width=1000, height=800, max_words=200, relative_scaling=0.5)

cols = ['text', 'flavor', 'type', 'name']

f, axs = plt.subplots(len(cols), figsize=(80, 36))



for i, col in enumerate(cols):

    text = mtg[col].str.cat(sep=' ')    

    wc.generate(text)

    axs[i].imshow(wc)

    axs[i].axis("off")    

    axs[i].set_title(col.upper(), fontsize=24)



del wc, cols, f, axs
stemmer = PorterStemmer()

tokenizer = RegexpTokenizer(r'\w+')



def my_tokenizer(s):

    return [stemmer.stem(t.lower()) for t in tokenizer.tokenize(s) if t.lower() not in stopwords.words('english')]    
count_vect = CountVectorizer(tokenizer=my_tokenizer)

flavors = count_vect.fit_transform(mtg.flavor)



flavors_tf = [(w, i, flavors.getcol(i).sum()) for w, i in count_vect.vocabulary_.items()]

flavors_tf = sorted(flavors_tf, key=lambda x: -x[2])[0:99]

flavors.shape
flavors_tf[:9]
flavors_words = [i for (i, j, k) in flavors_tf]

flavors_indices = [j for (i, j, k) in flavors_tf]



flavors_pivot = []

for color in mono_colors:

    f = flavors[np.where(mtg.manaColors==color)[0], :].tocsc()[:, flavors_indices]

    flavors_pivot.append(f.sum(axis=2).getA1())



f = flavors[np.where(mtg.colorsCount>1)[0], :].tocsc()[:, flavors_indices]

flavors_pivot.append(f.sum(axis=2).getA1())



flavors_pivot = pd.DataFrame(flavors_pivot, index=mono_colors + ['Multi'], columns=flavors_words)

del flavors, flavors_tf, flavors_words, flavors_indices, f            
plt.figure(figsize=(8, 24))

sns.heatmap(flavors_pivot.transpose())
flavors_pivot.transpose().plot(kind='barh', figsize=(8, 24), title='Flavors Stacked Barchart', stacked=True, color=plt_colors)
### Count Vectorize Texts

count_vect = CountVectorizer(tokenizer=my_tokenizer)

texts = count_vect.fit_transform(mtg.text)



texts_tf = [(w, i, texts.getcol(i).sum()) for w, i in count_vect.vocabulary_.items()]

texts_tf = sorted(texts_tf, key=lambda x: -x[2])[0:99]

texts.shape
texts_tf[0:9]
texts_words = [i for (i, j, k) in texts_tf]

texts_indices = [j for (i, j, k) in texts_tf]



texts_pivot = []

for color in mono_colors:

    t = texts[np.where(mtg.manaColors==color)[0], :].tocsc()[:, texts_indices]

    texts_pivot.append(t.sum(axis=2).getA1())



t = texts[np.where(mtg.colorsCount>1)[0], :].tocsc()[:, texts_indices]

texts_pivot.append(t.sum(axis=2).getA1())



texts_pivot = pd.DataFrame(texts_pivot, index=mono_colors + ['Multi'], columns=texts_words)

del texts, texts_tf, texts_words, texts_indices, t
plt.figure(figsize=(8, 24))

sns.heatmap(texts_pivot.transpose())
texts_pivot.transpose().plot(kind='barh', figsize=(8, 24), title='Texts Stacked Barchart', stacked=True, color=plt_colors)