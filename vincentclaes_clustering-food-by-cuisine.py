# I used the set https://www.kaggle.com/alonalevy/whats-cooking/cultural-diffusion-by-recipes 

# as basis for this notebook.

# My goal is to cluster food by cuisine.  I would like to see what cuisines 

# relate more to each other in regard to all available cuisines.



import numpy as np

import pandas as pd

from pandas import DataFrame

import  json


# open training file and parse recipies



with open('../input/train.json') as data_file:    

    data = json.load(data_file)



def collect_recipies(recipies):

    ret_val = []

    for _object in recipies:

        ingredients = _object.get('ingredients')

        cuisine = _object.get('cuisine')

        _id = _object.get('id')

        for i in ingredients:

            _dict = dict()

            _dict['ingredients'] = i

            _dict['cuisine'] = cuisine

            _dict['id'] = _id

            ret_val.append(_dict)

    return ret_val



parsed_data = collect_recipies(data)



# create a dataframe with cuisine, id and ingredients as columns



df_data = DataFrame(parsed_data)

cuisines = list(set(df_data.cuisine))

ingredients = list(set(df_data.ingredients))





df_data.head(3)

# count ingredients per cuisine

def count_ingredients_per_cuisine():

    ret_list = []

    ret_dict = dict()

    grouped = df_data.groupby('cuisine')

    for cuisine, sub_df in grouped:

        ret_list.append((cuisine,sub_df.groupby('ingredients')['ingredients'].count()))

        ret_dict[cuisine] = set(sub_df['ingredients'])

    return ret_list, ret_dict



ingredients_per_cuisine, ingredients_per_cuisine_set = count_ingredients_per_cuisine()

print('cuisine : ' + ingredients_per_cuisine[0][0])

print(ingredients_per_cuisine[0][1][:10])

# create a counts matrix using a dataframe. 1 column with all the ingredients. 

# The other columns are cuisines



def create_counts_matrix_per_cuisine():

    for cuisine, ingredients in ingredients_per_cuisine:

        for ingredient in ingredients:

            df_counts[cuisine][ingredient] = ingredient



df_counts = DataFrame(columns=cuisines)

df_counts['ingredients'] = pd.Series(list(ingredients))

create_counts_matrix_per_cuisine()

df_counts = df_counts.set_index('ingredients')

df_counts.head(3)

## create sparse matrix



from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import PCA

from scipy import sparse



# we transpose our counts matrix so that the samples (cuisines) are along axis 0

# and the features are along axis 1



sparse_counts = df_counts.T.to_sparse(fill_value=0)

def tf_idf_from_count_matrix(countsMatrix):

    

    countsMatrix = sparse.csr_matrix(countsMatrix)

    transformer = TfidfTransformer()

    

    # we do fit_transform so that we perform tf-idf

    # 

    # tf 

    # measures the frequency of a term wrt to the total terms used in the corpus.

    # 

    # idf 

    # measures the inverse of the frequency of the term in all the documents. 

    # If used in all documents idf = 1. if used in few documents idf -> 0.

    tfidf = transformer.fit_transform(countsMatrix) 

    tfidf.toarray() 



    return tfidf.toarray()

    

tfIdf_Matrix = tf_idf_from_count_matrix(sparse_counts)
## principal component analysis

# we divide our cuisines over 2 components so that we can map this on a 2 dimensional space.

pca = PCA(n_components=2)

reduced_data = pca.fit_transform(tfIdf_Matrix)



pca2dataFrame = pd.DataFrame(reduced_data)

pca2dataFrame.columns = ['PC1', 'PC2']
## cluster our data over 3 cuisines



from sklearn.cluster import KMeans

def kmeans_cultures(numOfClusters):

    

    kmeans = KMeans(init='k-means++', n_clusters=numOfClusters, n_init=10)

    kmeans.fit(reduced_data)

    return kmeans.predict(reduced_data)

labels = kmeans_cultures(3)

print(labels[:350])
### JACCARD SIMILARITY

#### 

#### this provides us of a measure of how much ingredients of this cuisine relates 

#### to the total number of ingredients in the cluster where this cuisine belongs.



effect_on_cluster = [0] * len(cuisines)

for i, cuisineA in enumerate(cuisines):

    ingrA = set(ingredients_per_cuisine_set[cuisineA])

    not_A_ingredients_in_cluster = []

    

    for j, cuisineB in enumerate(cuisines):

        if cuisineA != cuisineB:

            if labels[i] == labels[j]:

                not_A_ingredients_in_cluster.extend(set(ingredients_per_cuisine_set[cuisineB]))

    intersect = ingrA.intersection(set(not_A_ingredients_in_cluster))

    union = ingrA.union(set(not_A_ingredients_in_cluster))

    jaccard = len(intersect) / len(union)

    effect_on_cluster[i] = jaccard



print('jaccard similarity for all the cuisines : {0}'.format(dict(zip(cuisines, effect_on_cluster))))
from pylab import *

from scipy import *

import matplotlib.pyplot as plt



# reading the data from a csv file

rdata = reduced_data

print(len(rdata))

print(len(labels))

figureRatios = (15,20)

x = []

y = []

color = []

area = []



#creating a color palette:

colorPalette = ['#009600','#2980b9', '#ff6300','#2c3e50', '#660033'] 

# green,blue, orange, grey, purple



plt.figure(1, figsize=figureRatios)



for index, data in enumerate(rdata):

    x.append(data[0]) 

    y.append(data[1])  

    color.append(colorPalette[labels[index]]) 

    area.append(effect_on_cluster[index]*27000) # magnifying the bubble's sizes (all by the same unit)

    # plotting the name of the cuisine:

    text(data[0], data[1], cuisines[index], size=10.6,horizontalalignment='center', fontweight = 'bold', color='w')



plt.scatter(x, y, c=color, s=area, linewidths=2, edgecolor='w', alpha=0.80) 



plt.axis([-0.45,0.65,-0.55,0.55])

plt.axes().set_aspect(0.8, 'box')



plt.xlabel('PC1')

plt.ylabel('PC2')

plt.axis('off') # removing the PC axes



plt.show()