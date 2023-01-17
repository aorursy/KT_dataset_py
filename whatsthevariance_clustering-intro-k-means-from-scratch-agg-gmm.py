# Author: Caleb Woy



import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt # plotting

import seaborn as sb # plotting

import os # Reading data

import matplotlib.pylab as plt # plotting hyperparamter cost curves

from sklearn import preprocessing # scaling features

import random # random centroid generation
path_to_data = "/kaggle/input/"



# Loading the training and test data sets into pandas

small_xy = pd.read_csv(path_to_data + "/small_Xydf.csv", header=0)

small_xy = small_xy.drop(columns=["Unnamed: 0"])



large_xy = pd.read_csv(path_to_data + "/large_Xydf.csv", header=0)

large_xy = large_xy.drop(columns=["Unnamed: 0"])



red_wine = pd.read_csv(path_to_data + "/winequality-red.csv", sep=';', 

                       header=0)

red_wine['y'] = red_wine.apply(lambda x: int(x['quality'] - 3), axis=1)
# A two dimensional data set of 100 observations.

# y is the label representing the true clustering.

small_xy.head()
# A two dimensional data set of 1000 observations.

# y is the label representing the true clustering.

large_xy.head()
# An eleven dimensional data set of 1600 observations.

# quality is the label representing the true clustering.

# I've added the y label to make feeding the data into 

# my own algorithm easier.

red_wine.head()
"""

Implementing K-means clustering algorithm from scratch.



data: pandas dataframe, X data to be used for clustering

k: int, the number of centroids to assign

verbose: Boolean, default is true, if false nothing is printed

"""

def Kmeans(data, k, verbose = True):

    random.seed(k)

    if verbose:

        print(f'TESTING FOR K = {k}')

    # scale the x data uniformly

    x = data.values # Get data as np array

    standard_scaler = preprocessing.StandardScaler()

    x_scaled = standard_scaler.fit_transform(x)

    data = pd.DataFrame(x_scaled)

    # calculating the max and min values for each column and storing them in 

    # an array

    min_max = data.apply(lambda x:

        pd.Series(index=['min','max'],data=[x.min(),x.max()]))

    min_max_list = min_max.T.values.tolist()

    # Randomly initialize centroids within max and min range of feature vectors

    centroids = {}

    for i in range(k):

        centroids[i] = [ 

            random.uniform(x[0], x[1]) for x in min_max_list]

    # Create arrays for assignments to check for convergence

    curr = np.array([0 for i in range(data.shape[0])])

    prev = np.array([1 for i in range(data.shape[0])])

    # iterate until assignments don't change

    while not (curr == prev).all():

        # copy current into prev

        prev = np.copy(curr)

        if verbose:

            print(f'.', end=' ')

        # Assignment step

        # iterate over each row in the data 

        for index, row in data.iterrows():

            # set min container and flag

            min_dist, min_centroid = float("inf"), 0

            # iterate over each centroid

            for l in range(k):

                # get the centroid vector

                a = np.array(centroids[l])

                # get the row vector

                b = np.array(row.T.values.tolist())

                # calculate euclidean distance

                dist = np.linalg.norm(a - b)

                # set min distance if min

                if dist < min_dist:

                    min_dist, min_centroid = dist, l

            # assign current datum

            curr[index] = min_centroid

        # Getting the counts of each cluster assignment

        curr_counter = np.array(curr)

        unique, counts = np.unique(curr_counter, return_counts=True)

        counts = dict(zip(unique, counts))

        # init container for new centroids

        new_centroids = {}

        for i in range(k):

            new_centroids[i] = [0 for x in min_max_list]

        # Update step

        # iterate over all data

        for index, row in data.iterrows():

            # retrieve row

            row = row.T.values.tolist()

            # retrieve centroid

            cent = new_centroids[curr[index]]

            # iterate over each dimension of the centroid and accumulate the

            # avg distance

            for j in range(len(cent)):

                cent[j] += row[j] / counts[curr[index]]

            new_centroids[curr[index]] = cent

        # reassign centroid

        centroids = new_centroids

    if verbose:    

        print('FINISHED')

    return curr
"""

Prints the cluster SSE.



data: pandas dataframe, X data

verbose: Boolean, default is true, if false nothing is printed

"""

def cluster_sse(data, verbose = True):

    # calculating cluster centroids

    labels = data['y']

    counts = labels.value_counts()

    data = data.drop(columns=['y'])

    centroids = {}

    for i in range(len(counts)):

        centroids[i] = [0 for j in range(data.shape[1])]

    for index, row in data.iterrows():

        point = np.array(row.T.values.tolist())

        cent = centroids[labels[index]]

        for j in range(len(cent)):

            cent[j] += row[j] / counts[labels[index]]

        centroids[labels[index]] = cent

    # calculating cluster sse

    sse = [0 for j in range(len(counts))]

    for index, row in data.iterrows():

        a = np.array(centroids[labels[index]])

        b = np.array(row.T.values.tolist())

        dist = np.linalg.norm(a - b)

        sse[labels[index]] += dist

    if verbose:

        print(f'CLUSTER SSE IS: {sse}')

    return sse
"""

Prints the total SSE.



data: pandas dataframe, X data

verbose: Boolean, default is true, if false nothing is printed

"""

def total_sse(data, verbose = True):

    # calculating cluster centroids

    labels = data['y']

    counts = labels.value_counts()

    data = data.drop(columns=['y'])

    centroids = {}

    for i in range(len(counts)):

        centroids[i] = [0 for j in range(data.shape[1])]

    for index, row in data.iterrows():

        point = np.array(row.T.values.tolist())

        cent = centroids[labels[index]]

        for j in range(len(cent)):

            cent[j] += row[j] / counts[labels[index]]

        centroids[labels[index]] = cent

    # calculating cluster sse

    sse = 0

    for index, row in data.iterrows():

        a = np.array(centroids[labels[index]])

        b = np.array(row.T.values.tolist())

        dist = np.linalg.norm(a - b)

        sse += dist

    if verbose:

        print(f'TOTAL SSE IS: {sse}')

    return sse
"""

Prints the cluster SSB.



data: pandas dataframe, X data

verbose: Boolean, default is true, if false nothing is printed

"""

def cluster_ssb(data, verbose = True):

    # calculating cluster centroids

    labels = data['y']

    counts = labels.value_counts()

    data = data.drop(columns=['y'])

    centroids = {}

    for i in counts.iteritems():

        centroids[i[0]] = [0 for j in range(data.shape[1])]

    for index, row in data.iterrows():

        point = np.array(row.T.values.tolist())

        cent = centroids[labels[index]]

        for j in range(len(cent)):

            cent[j] += row[j] / counts[labels[index]]

        centroids[labels[index]] = cent

     # calculating global centroid

    glob_centroid = [0 for j in range(data.shape[1])]

    for index, row in data.iterrows():

        point = np.array(row.T.values.tolist())

        for j in range(len(glob_centroid)):

            glob_centroid[j] += row[j] / data.shape[0]

    # calculating total ssb

    ssb = 0

    a = np.array(glob_centroid)

    for key, b in centroids.items():

        b = np.array(b)

        dist = np.linalg.norm(a - b) * counts[key]

        ssb += dist

    if verbose:

        print(f'TOTAL SSB IS: {ssb}')

    return ssb
def test_Kmeans_small_large(data, k_collection, k_special = 0):

    x = data.drop(columns=['y'])

    for k in k_collection:

        x['y'] = Kmeans(x, k)

        # Calculating test SSE, SSB

        cluster_sse(x)

        sse = total_sse(x)

        ssb = cluster_ssb(x)

        print(f'SSE RATIO: {sse / (sse + ssb)}')

        if k == k_special:

            plt.scatter(x['X0'], x['X1'], c=x['y'])

            plt.show()

        x = x.drop(columns=['y'])

        print()
def test_Kmeans_redwine(data, k, k_special = 0):

    x = data.drop(columns=['quality', 'y'])

    x['y'] = Kmeans(x, k)

    # Calculating test SSE, SSB

    cluster_sse(x)

    sse = total_sse(x)

    ssb = cluster_ssb(x)

    print(f'SSE RATIO: {sse / (sse + ssb)}')

    if k == k_special:

        sb.pairplot(x, 

                    vars=x.loc[:, x.columns != 'y'], 

                    hue ='y',

                    diag_kind = 'hist')

    x = x.drop(columns=['y'])

    print()
def Eval_Kmeans_scree(data, k_collection):

    x = data.drop(columns=['y'])

    scores = [0 for i in k_collection]

    for index, k in enumerate(k_collection):

        x['y'] = Kmeans(x, k, False)

        # Calculating test SSE, SSB

        sse = total_sse(x, False)

        ssb = cluster_ssb(x, False)

        scores[index] = (k, sse / (sse + ssb))

        x = x.drop(columns=['y'])

    plt.plot(*zip(*scores))

    plt.title("Scree Plot")

    plt.ylabel("sse / (sse + ssb)")

    plt.xlabel("k")

    plt.show()
# Calculating true cluster SSE

print('TRUE ', end='')

sse = cluster_sse(small_xy)
# Calculating total SSE

print('TRUE ', end='')

sse = total_sse(small_xy)
# Calculating true cluster SSB

print('TRUE ', end='')

ssb = cluster_ssb(small_xy)
# plotting true clustering

plt.scatter(small_xy['X0'], small_xy['X1'], c=small_xy['y'])
k_collection = [x for x in range(2, 10)]

Eval_Kmeans_scree(small_xy, k_collection)
k_collection = [4]

test_Kmeans_small_large(small_xy, k_collection, 4)
# Calculating true cluster SSE

print('TRUE ', end='')

sse = cluster_sse(large_xy)
# Calculating total SSE

print('TRUE ', end='')

sse = total_sse(large_xy)
# Calculating true cluster SSB

print('TRUE ', end='')

ssb = cluster_ssb(large_xy)
# plotting true clustering

plt.scatter(large_xy['X0'], large_xy['X1'], c=large_xy['y'])
k_collection = [x for x in range(2, 10)]

Eval_Kmeans_scree(large_xy, k_collection)
k_collection = [5]

test_Kmeans_small_large(large_xy, k_collection, 5)
# Calculating true cluster SSE

print('TRUE ', end='')

red_wine_no_qual = red_wine.drop(columns=['quality'])

sse = cluster_sse(red_wine_no_qual)
# Calculating total SSE

print('TRUE ', end='')

sse = total_sse(red_wine_no_qual)
# Calculating true cluster SSB

print('TRUE ', end='')

ssb = cluster_ssb(red_wine_no_qual)
# plotting true clustering

sb.pairplot(red_wine_no_qual, 

            hue='y', 

            vars=red_wine_no_qual.columns[:-1],

            diag_kind = 'hist')
# Testing on wine dataset, had to split them out of the loop to get

# the seaborn plot to show in proper order. It takes a while to load.
k_collection = [x for x in range(2, 10)]

Eval_Kmeans_scree(red_wine.drop(columns=['quality']), k_collection)
test_Kmeans_redwine(red_wine, 5, 5)
from sklearn import cluster as cl # for testing K-means
def Eval_SKLearn_Kmeans_scree(data, k_collection):

    x = data.drop(columns=['y'])

    scores = [0 for i in k_collection]

    for index, k in enumerate(k_collection):

        kmeans = cl.KMeans(n_clusters = k)

        x['y'] = kmeans.fit_predict(x)

        # Calculating test SSE, SSB

        sse = total_sse(x, False)

        ssb = cluster_ssb(x, False)

        scores[index] = (k, sse / (sse + ssb))

        x = x.drop(columns=['y'])

    plt.plot(*zip(*scores))

    plt.title("Scree Plot")

    plt.ylabel("sse / (sse + ssb)")

    plt.xlabel("k")

    plt.show()
def test_SKLearn_Kmeans_small_large(data, k_collection, k_special = 0):

    x = data.drop(columns=['y'])

    for k in k_collection:

        kmeans = cl.KMeans(n_clusters = k)

        x['y'] = kmeans.fit_predict(x)

        # Calculating test SSE, SSB

        cluster_sse(x)

        sse = total_sse(x)

        ssb = cluster_ssb(x)

        print(f'SSE RATIO: {sse / (sse + ssb)}')

        if k == k_special:

            plt.scatter(x['X0'], x['X1'], c=x['y'])

            plt.show()

        x = x.drop(columns=['y'])

        print()
def test_SKLearn_Kmeans_redwine(data, k, k_special = 0):

    x = data.drop(columns=['quality', 'y'])

    kmeans = cl.KMeans(n_clusters = k)

    x['y'] = kmeans.fit_predict(x)

    # Calculating test SSE, SSB

    cluster_sse(x)

    sse = total_sse(x)

    ssb = cluster_ssb(x)

    print(f'SSE RATIO: {sse / (sse + ssb)}')

    if k == k_special:

        sb.pairplot(x, 

                    vars=x.loc[:, x.columns != 'y'], 

                    hue ='y',

                    diag_kind = 'hist')

    x = x.drop(columns=['y'])

    print()
k_collection = [x for x in range(2, 10)]

Eval_SKLearn_Kmeans_scree(small_xy, k_collection)
test_SKLearn_Kmeans_small_large(small_xy, [4], k_special = 4)
k_collection = [x for x in range(2, 10)]

Eval_SKLearn_Kmeans_scree(large_xy, k_collection)
test_SKLearn_Kmeans_small_large(large_xy, [3], k_special = 3)
k_collection = [x for x in range(2, 10)]

Eval_SKLearn_Kmeans_scree(red_wine.drop(columns=['quality']), k_collection)
test_SKLearn_Kmeans_redwine(red_wine, 5, k_special = 5)
from sklearn.mixture import GaussianMixture as GMM
def Eval_SKLearn_GMM_scree(data, k_collection):

    x = data.drop(columns=['y'])

    scores = [0 for i in k_collection]

    for index, k in enumerate(k_collection):

        gmm = GMM(n_components = k)

        x['y'] = gmm.fit_predict(x)

        # Calculating test SSE, SSB

        sse = total_sse(x, False)

        ssb = cluster_ssb(x, False)

        scores[index] = (k, sse / (sse + ssb))

        x = x.drop(columns=['y'])

    plt.plot(*zip(*scores))

    plt.title("Scree Plot")

    plt.ylabel("sse / (sse + ssb)")

    plt.xlabel("k")

    plt.show()
def test_SKLearn_GMM_small_large(data, k_collection, k_special = 0):

    x = data.drop(columns=['y'])

    for k in k_collection:

        gmm = GMM(n_components = k)

        x['y'] = gmm.fit_predict(x)

        # Calculating test SSE, SSB

        cluster_sse(x)

        sse = total_sse(x)

        ssb = cluster_ssb(x)

        print(f'SSE RATIO: {sse / (sse + ssb)}')

        if k == k_special:

            plt.scatter(x['X0'], x['X1'], c=x['y'])

            plt.show()

        x = x.drop(columns=['y'])

        print()
def test_SKLearn_GMM_redwine(data, k, k_special = 0):

    x = data.drop(columns=['quality', 'y'])

    gmm = GMM(n_components = k)

    x['y'] = gmm.fit_predict(x)

    # Calculating test SSE, SSB

    cluster_sse(x)

    sse = total_sse(x)

    ssb = cluster_ssb(x)

    print(f'SSE RATIO: {sse / (sse + ssb)}')

    if k == k_special:

        sb.pairplot(x, 

                    vars=x.loc[:, x.columns != 'y'], 

                    hue ='y',

                    diag_kind = 'hist')

    x = x.drop(columns=['y'])

    print()
k_collection = [x for x in range(2, 10)]

Eval_SKLearn_GMM_scree(small_xy, k_collection)
test_SKLearn_GMM_small_large(small_xy, [5], k_special = 5)
k_collection = [x for x in range(2, 10)]

Eval_SKLearn_GMM_scree(large_xy, k_collection)
test_SKLearn_GMM_small_large(large_xy, [3], k_special = 3)
k_collection = [x for x in range(2, 10)]

Eval_SKLearn_GMM_scree(red_wine.drop(columns=['quality']), k_collection)
test_SKLearn_GMM_redwine(red_wine, 5, k_special = 5)
from sklearn.cluster import AgglomerativeClustering as AGG
def Eval_SKLearn_agg_scree(data, k_collection):

    x = data.drop(columns=['y'])

    scores = [0 for i in k_collection]

    for index, k in enumerate(k_collection):

        agg = AGG(n_clusters = k)

        x['y'] = agg.fit_predict(x)

        # Calculating test SSE, SSB

        sse = total_sse(x, False)

        ssb = cluster_ssb(x, False)

        scores[index] = (k, sse / (sse + ssb))

        x = x.drop(columns=['y'])

    plt.plot(*zip(*scores))

    plt.title("Scree Plot")

    plt.ylabel("sse / (sse + ssb)")

    plt.xlabel("k")

    plt.show()
def test_SKLearn_agg_small_large(data, k_collection, k_special = 0):

    x = data.drop(columns=['y'])

    for k in k_collection:

        agg = AGG(n_clusters = k)

        x['y'] = agg.fit_predict(x)

        # Calculating test SSE, SSB

        cluster_sse(x)

        sse = total_sse(x)

        ssb = cluster_ssb(x)

        print(f'SSE RATIO: {sse / (sse + ssb)}')

        if k == k_special:

            plt.scatter(x['X0'], x['X1'], c=x['y'])

            plt.show()

        x = x.drop(columns=['y'])

        print()
def test_SKLearn_agg_redwine(data, k, k_special = 0):

    x = data.drop(columns=['quality', 'y'])

    agg = AGG(n_clusters = k)

    x['y'] = agg.fit_predict(x)

    # Calculating test SSE, SSB

    cluster_sse(x)

    sse = total_sse(x)

    ssb = cluster_ssb(x)

    print(f'SSE RATIO: {sse / (sse + ssb)}')

    if k == k_special:

        sb.pairplot(x, 

                    vars=x.loc[:, x.columns != 'y'], 

                    hue ='y',

                    diag_kind = 'hist')

    x = x.drop(columns=['y'])

    print()
k_collection = [x for x in range(2, 10)]

Eval_SKLearn_agg_scree(small_xy, k_collection)
test_SKLearn_agg_small_large(small_xy, [6], k_special = 6)
k_collection = [x for x in range(2, 10)]

Eval_SKLearn_agg_scree(large_xy, k_collection)
test_SKLearn_agg_small_large(large_xy, [5], k_special = 5)
k_collection = [x for x in range(2, 10)]

Eval_SKLearn_agg_scree(red_wine.drop(columns=['quality']), k_collection)
test_SKLearn_agg_redwine(red_wine, 4, k_special = 4)