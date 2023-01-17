import numpy as np

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

import re

import nltk

from nltk.tokenize import word_tokenize

from collections import Counter

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/cartier-jewelry-catalog/cartier_catalog.csv')
data.head()
fig, axes = plt.subplots(1, 2, figsize=(12,4))

sns.countplot(data.categorie, ax=axes[0])

sns.boxplot(x='categorie', y='price', data=data, ax=axes[1])

plt.ylim(0,100000)

plt.suptitle('Count and price distribution by categories')

plt.show()
avg_by_cat = data.groupby('categorie').price.mean()



def base_est(X):

    """returns average price for each category"""

    

    X['predict'] = X.categorie.apply(lambda x: avg_by_cat[x])

    return X.predict.values    
r2_score(data.price, base_est(data))
corpus = []

for i in data.index:

    tokens = nltk.word_tokenize(data.title[i])

    corpus += list(tokens)

    

word_counts = Counter(corpus)

word_counts.most_common(20)
common_words = ['Cartier', 'Panthère', 'wedding', 'Love', 'Trinity', 'Juste']

for word in common_words:

    data[word] = data.title.apply(lambda x: int(word in nltk.word_tokenize(x)))
data['descr_tok'] = data.description.apply(lambda x: nltk.word_tokenize(x))

data['descr_pos'] = data.descr_tok.apply(lambda x: nltk.pos_tag(x))
def extract(string, nouns, kind='count', following=True):

    """extracts a number such that noun follows it, i.e. 3 beautiful diamonds.

       accepts list of nouns(need plural and singular)

       kind - mostly used for count, but for some cases we need to use mean"""

    

    count = []



    for num, word in enumerate(string):

        if word[1]=='CD':

            if find_next_noun(string, num, following) in nouns:

                if word[0] in ['one', 'One']:

                    numerical = [1]

                elif word[0] in ['two', 'Two']:

                    numerical = [2]

                elif word[0] in ['three', 'Three']:

                    numerical = [3]

                elif word[0] in ['four', 'Four']:

                    numerical = [4]

                elif word[0] in ['five', 'Five']:

                    numerical = [5]

                else:    

                    numerical =  re.findall('\d+.\d+|\d+', word[0])

                if len(numerical):

                    try:

                        count.append(float(numerical[0]))

                    except ValueError:

                        pass

        if (word[0] in ['a', 'one']) and following==True:

            if find_next_noun(string, num, following) in nouns:

                count.append(1)                    

                

    if len(count)==0:

        count=[0]

    if kind=='count':

        return np.sum(count)

    if kind=='mean':

        return np.mean(count)
def find_next_noun(string, position, following=True):

    """finds next noun appearing after position in pos_tagged string, 

       if following==False, finds preceding noun """

    

    step = 1

    if following == False:

        step=-1

    i=position+step

    while (i<len(string)) and (i>-1):

        if string[i][1] in ['NN', 'NNS', 'NNP', 'NNPS']:

            return string[i][0]

        i=i+step
# Extract the numerical info



data['num_diamonds'] = data.descr_pos.apply(lambda x: extract(x, ['diamond', 'diamonds'], kind='count'))

data['carat'] = data.descr_pos.apply(lambda x: extract(x, ['carat', 'carats'], kind='count'))

data['purity'] = data.descr_pos.apply(lambda x: extract(x, ['gold'], kind='mean'))

data['width'] = data.descr_pos.apply(lambda x: extract(x, ['width', 'Width'], kind='mean', following=False))

data['num_garnets'] = data.descr_pos.apply(lambda x: extract(x, ['garnet', 'garnets'], kind='count'))

data['num_sapphires'] = data.descr_pos.apply(lambda x: extract(x, ['sapphire', 'sapphires'], kind='count'))

data['num_emeralds'] = data.descr_pos.apply(lambda x: extract(x, ['emerald', 'emeralds'], kind='count'))
data.purity.unique()
data.drop(columns=['purity'], inplace=True)
# Extract sizes



small = ['small', 'Small', 'S', 'XS', 'xs']

medium = ['medium', 'Medium', 'M']

large = ['large', 'Large', 'L', 'XL', 'big', 'Big']



def size_encoder(string, size):

    return  int(len(set(size) & set(nltk.word_tokenize(string)))>0)



data['small'] = data.description.apply(lambda x: size_encoder(x, small))

data['medium'] = data.description.apply(lambda x: size_encoder(x, medium))

data['large'] = data.description.apply(lambda x: size_encoder(x, large))
# Obtaining a list of most common tags from the data



tag_set = []

for tags in data.tags.values:

    tag_set += tags.split(',')



tag_count = Counter(tag_set)    

common_tags = [ word for (word, count) in tag_count.most_common(15)]    
# One hot encoding of tags

for tag in common_tags:

    data[tag] = data.tags.apply(lambda x: int(tag in x))
# One hot encoding of categories.



data = pd.get_dummies(data=data, columns=['categorie'], drop_first=True)

#data['rings'] = data.categorie.apply(lambda x: int(x=='rings'))

#data['bracelets'] = data.categorie.apply(lambda x: int(x=='bracelets'))

#data['necklaces'] = data.categorie.apply(lambda x: int(x=='necklaces'))
df = data.drop(columns=['ref', 'title', 'tags', 'description', 'image', 'descr_tok', 'descr_pos'])



X_train, X_test, y_train, y_test = train_test_split(

                    df.drop(columns=['price']), df.price, test_size=0.3, random_state=42)
regr = RandomForestRegressor(n_estimators=100, max_depth=20)

regr.fit(X_train,y_train)

regr.score(X_train,y_train)
regr.score(X_test,y_test)