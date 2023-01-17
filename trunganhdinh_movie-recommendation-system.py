# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from IPython.display import Image

import itertools
from surprise import AlgoBase, NormalPredictor, Dataset, Reader, KNNBaseline, NormalPredictor, accuracy, PredictionImpossible, SVD
from surprise.model_selection import train_test_split, LeaveOneOut
from collections import defaultdict

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt # show graph
import math
import heapq

from ast import literal_eval

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# pd.options.display.max_colwidth = 300
linkDF = pd.read_csv('/kaggle/input/the-movies-dataset/links.csv')
linkDF.head(5)
print(linkDF.shape, len(linkDF.movieId.unique()), len(linkDF.imdbId.unique()), len(linkDF.tmdbId.unique()))
print(linkDF.isna().sum())
linkDF.describe(include='all')
ratingsDF = pd.read_csv('/kaggle/input/the-movies-dataset/ratings_small.csv')
ratingsDF.describe(include='all')
ratingsDF.head(5)
ratingsDF.drop_duplicates(subset=['userId', 'movieId'], inplace=True)
ratingsDF.describe(include='all')
movieDF = pd.read_csv('/kaggle/input/the-movies-dataset/movies_metadata.csv', low_memory=False)
movieDF.head(5)
movieDF.describe(include='all')
print(movieDF.shape, movieDF.imdb_id.isna().sum())
imdb_id_value_counts = movieDF.imdb_id.value_counts()
duplicate_imdb_id = imdb_id_value_counts[imdb_id_value_counts > 1]
print(duplicate_imdb_id)
movieDF[movieDF.imdb_id == '0']
movieDF = movieDF[movieDF.imdb_id != '0']
movieDF[movieDF.imdb_id.isin(duplicate_imdb_id.index)].sort_values(by=['imdb_id'])
movieDF.drop_duplicates(subset=['id'], keep='first', inplace=True)
def getNameCol(x):
    return [i['name'] for i in x] if isinstance(x, list) else []

movieDF.belongs_to_collection = movieDF.belongs_to_collection.fillna('{}').apply(literal_eval).apply(lambda x: x['name'] if 'name' in x else '')

movieDF.genres = movieDF.genres.fillna('[]').apply(literal_eval).apply(getNameCol)
movieDF.production_companies = movieDF.production_companies.fillna('[]').apply(literal_eval).apply(getNameCol)
movieDF.production_countries = movieDF.production_countries.fillna('[]').apply(literal_eval).apply(getNameCol)
movieDF.spoken_languages = movieDF.spoken_languages.fillna('[]').apply(literal_eval).apply(getNameCol)

movieDF.imdb_id.dropna(inplace=True) # I just do not care about movie without imdb_id
movieDF['imdbId'] = movieDF.imdb_id.str[2:].astype(int)
movieDF.drop(['imdb_id'],axis = 1,inplace = True)

movieDF.id = movieDF.id.astype(int)
movieDF.adult = movieDF.adult.apply(lambda x : True if (x == 'True') else False)

movieDF.loc[movieDF.overview == 'No overview found.', 'overview'] = ''
movieDF.overview = movieDF.overview.fillna('')
movieDF.popularity = movieDF.popularity.astype(float)
movieDF.budget = movieDF.budget.fillna(0).astype(int)

movieDF.describe(include='all')
keywordsDF = pd.read_csv('/kaggle/input/the-movies-dataset/keywords.csv')
keywordsDF.head(5)
keywordsDF.describe(include='all')
keywordsDF.drop_duplicates(inplace=True)
print(keywordsDF.shape, len(keywordsDF.id.unique()))
keywordsDF.keywords = keywordsDF.keywords.apply(literal_eval).apply(getNameCol)
keywordsDF.head(5)
creditsDF = pd.read_csv('/kaggle/input/the-movies-dataset/credits.csv')
creditsDF.describe()
creditsDF.head(5)
creditsDF.drop_duplicates(subset=['id'], inplace=True)
creditsDF.shape, len(creditsDF.id.unique())
def extract_by_job(x, job):
    for crew_mem in x:
        if crew_mem['job'] == job:
            return crew_mem['name']
        else:
            return np.nan

def extract_director(x):
    return extract_by_job(x, 'Director')

creditsDF['director'] = creditsDF.crew.apply(literal_eval).apply(extract_director).fillna('')
creditsDF.drop(['crew'],axis = 1,inplace = True)

creditsDF.cast = creditsDF.cast.fillna('[]').apply(literal_eval).apply(getNameCol)
creditsDF.head(5)
Image("../input/moviedatasetstructure/MovieDatasetRelation.jpg")
movieDF = movieDF.merge(creditsDF, on = 'id', how = 'left')
movieDF = movieDF.merge(keywordsDF, on = 'id', how = 'left')
movieDF = movieDF.merge(linkDF, on = 'imdbId', how = 'inner')
movieDF.head()
# Remove unused data to optimize memory
ratingsDF = ratingsDF[ratingsDF.movieId.isin(movieDF.movieId)]
movieDF = movieDF[movieDF.movieId.isin(ratingsDF.movieId)]
movieDF.index = np.arange(0, len(movieDF)) # reindexing
def RMSE(predictions):
    return accuracy.rmse(predictions, verbose=False)
def AverageReciprocalHitRate(topNPredicted, leftOutPredictions):
    summation = 0
    total = 0
    for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
        hitRank = 0
        rank = 0
        for movieID, predictedRating in topNPredicted[int(userID)]:
            rank = rank + 1
            if (int(leftOutMovieID) == movieID):
                hitRank = rank
                break
        if (hitRank > 0) :
            summation += 1.0 / hitRank

        total += 1

    return summation / total
def Coverage(topNPredicted, numUsers, ratingThreshold=0):
    hits = 0
    for userID in topNPredicted.keys():
        hit = False
        for movieID, predictedRating in topNPredicted[userID]:
            if (predictedRating >= ratingThreshold):
                hit = True
                break
        if (hit):
            hits += 1

    return hits / numUsers
def Diversity(topNPredicted, simFunc):
    n = 0
    total = 0
    for userID in topNPredicted.keys():
        pairs = itertools.combinations(topNPredicted[userID], 2)
        for pair in pairs:
            movie1 = pair[0][0]
            movie2 = pair[1][0]
            similarity = simFunc(movie1, movie2)
            total += similarity
            n += 1

    S = total / n if n != 0 else 1
    return (1-S)
def Novelty(topNPredicted, rankings):
    n = 0
    total = 0
    for userID in topNPredicted.keys():
        for rating in topNPredicted[userID]:
            movieID = rating[0]
            rank = rankings[movieID]
            total += rank
            n += 1
            
    return total / n if n != 0 else 0.0
movieDF['popularity_rank'] = movieDF.popularity.rank(ascending=False, na_option='bottom', method='first').astype(int)
popularityRankings = dict(zip(movieDF.movieId, movieDF.popularity_rank))
movieDF.drop(columns=['popularity_rank'], inplace=True)
ratingsDF['adjusted_rating'] = ratingsDF.rating * 2 # increase scale form (0.5, 5) to (1, 10)
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratingsDF[['userId', 'movieId', 'adjusted_rating']], reader)

trainSet, testSet = train_test_split(data, test_size=.25, random_state=1)

loocv = LeaveOneOut(n_splits=1, random_state=1)
for train, test in loocv.split(data):
    loocvTrain = train
    loocvTest = test
loocvAntiTestSet = loocvTrain.build_anti_testset()
sim_options = {'name': 'cosine', 'user_based': False}
simsAlgo = KNNBaseline(sim_options=sim_options) 
simsAlgo.fit(loocvTrain)
simsMatrix = simsAlgo.compute_similarities()

def similarityFunc(movie1, movie2): # TODO reconsider this similarity metric
    innerID1 = simsAlgo.trainset.to_inner_iid(movie1)
    innerID2 = simsAlgo.trainset.to_inner_iid(movie2)
    return simsMatrix[innerID1][innerID2]
def getTopN(predictions, n=10, minimumRating=4.0):
    topN = defaultdict(list)
    for userID, movieID, actualRating, estimatedRating, _ in predictions:
        if (estimatedRating >= minimumRating):
            topN[userID].append((movieID, estimatedRating))
    for userID, ratings in topN.items():
        topN[userID] = heapq.nlargest(n, ratings, key=lambda x: x[1])
    return topN

def evaluate(algorithm):
    metrics = {}
    n = 10
    
    algorithm.fit(trainSet)
    predictions = algorithm.test(testSet)
    metrics["RMSE"] = RMSE(predictions)
    
    algorithm.fit(loocvTrain)
    leftOutPredictions = algorithm.test(loocvTest)
    allPredictions = algorithm.test(loocvAntiTestSet)
    topNPredicted = getTopN(allPredictions, n)
    metrics["ARHR"] = AverageReciprocalHitRate(topNPredicted, leftOutPredictions)
    # Print user coverage with a minimum predicted rating of 4.0:
    metrics["Coverage"] = Coverage(topNPredicted, loocvTrain.n_users, ratingThreshold=4.0)
    # Measure diversity of recommendations:
    metrics["Diversity"] = Diversity(topNPredicted, similarityFunc)
    # Measure novelty (average popularity rank of recommendations):
    metrics["Novelty"] = Novelty(topNPredicted, popularityRankings)
    
    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format("RMSE", "ARHR", "Coverage", "Diversity", "Novelty"))
    print("{:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(metrics["RMSE"], metrics["ARHR"],metrics["Coverage"], metrics["Diversity"], metrics["Novelty"]))
    
    # TODO print some real cases

    
# evaluate(NormalPredictor())
def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

class DemographicFiltering(AlgoBase):
    def __init__(self, movieDF):
        AlgoBase.__init__(self)
        
        # Calculation of IMDB weighted rating
        vote_counts = movieDF[movieDF.vote_count.notnull()].vote_count.astype(int)
        vote_averages = movieDF[movieDF.vote_average.notnull()].vote_average.astype(int)
        C = vote_averages.mean()
        m = vote_counts.quantile(0.95)
        qualifiedImdb = movieDF[movieDF.vote_count >= m]
        score = qualifiedImdb.apply(weighted_rating, args=(m, C), axis=1)
        
        self.movieRating = dict(zip(qualifiedImdb.movieId, score / 2.5  + 1))
        self.movieGenres = dict(zip(movieDF.movieId, movieDF.genres))
        
    def default_prediction(self):
        return 1.0 # Avoid return what is impossible to predict
        
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        
        # Find most favorite genre for each user
        self.userFavorGenre = {}
        for iuid in self.trainset.all_users():
            genreCount = {}
            for iiid, r in self.trainset.ur[iuid]:
                if r <= 3.0:
                    continue
                for genre in self.movieGenres[self.trainset.to_raw_iid(iiid)]:
                    if genre not in genreCount:
                        genreCount[genre] = 1
                    else:
                        genreCount[genre] += 1
            if bool(genreCount):
                result = max(genreCount, key=genreCount.get)
            else:
                result = ''
            self.userFavorGenre[self.trainset.to_raw_uid(iuid)] = result
            
        return self
        
    def estimate(self, iuid, iiid):
        if not (self.trainset.knows_user(iuid) and self.trainset.knows_item(iiid)):
            raise PredictionImpossible('User and/or item is unknown.')
        iid = self.trainset.to_raw_iid(iiid)
        uid = self.trainset.to_raw_uid(iuid)
        if not (uid in self.userFavorGenre and iid in self.movieGenres):
            raise PredictionImpossible('User and/or movieGenre is unknown.')
        if self.userFavorGenre[uid] == '': # User has no ratings before
            return self.movieRating[iid] # Return best movie from all genre
        if self.userFavorGenre[uid] not in self.movieGenres[iid]:
            raise PredictionImpossible("User's favorite genre is unknown.")
        if iid not in self.movieRating:
            raise PredictionImpossible('Movie without rating.')
        else:
            return self.movieRating[iid]

evaluate(DemographicFiltering(movieDF))
class ContentBasedFiltering(AlgoBase):
    def __init__(self, movieDF):
        AlgoBase.__init__(self)
        self.k = 40
        
        tfidf = TfidfVectorizer(stop_words='english')
        #Construct the required TF-IDF matrix by fitting and transforming the data
        tfidf_matrix = tfidf.fit_transform(movieDF.overview)

        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        self.indices = dict(zip(movieDF.movieId, movieDF.index))
        
    def default_prediction(self):
        return 1.0 # Avoid return what is impossible to predict
    
    def estimate(self, iuid, iiid):
        if not (self.trainset.knows_user(iuid) and self.trainset.knows_item(iiid)):
            raise PredictionImpossible('User and/or item is unknown.')
        
        # Build up similarity scores between this item and everything the user rated
        neighbors = []
        iindex = self.indices[self.trainset.to_raw_iid(iiid)]
        for other_iiid, r in self.trainset.ur[iuid]:
            other_iid = self.trainset.to_raw_iid(other_iiid)
            genreSimilarity = self.cosine_sim[iindex][self.indices[other_iid]]
            neighbors.append((genreSimilarity, r))
        
        # Extract the top-K most-similar ratings
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
        
        # Compute average sim score of K neighbors weighted by user ratings
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
            if (simScore > 0):
                simTotal += simScore
                weightedSum += simScore * rating
            
        if (simTotal == 0):
            raise PredictionImpossible('No neighbors')

        predictedRating = weightedSum / simTotal

        return predictedRating
        
evaluate(ContentBasedFiltering(movieDF))
evaluate(SVD())
# -*- coding: utf-8 -*-
# """
# Created on Fri May  4 13:08:25 2018
# @author: Frank
# """

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class RBM(object):

    def __init__(self, visibleDimensions, epochs=20, hiddenDimensions=50, ratingValues=10, learningRate=0.001, batchSize=100):
        self.visibleDimensions = visibleDimensions
        self.epochs = epochs
        self.hiddenDimensions = hiddenDimensions
        self.ratingValues = ratingValues
        self.learningRate = learningRate
        self.batchSize = batchSize
        
    def Train(self, X):
        ops.reset_default_graph()

        self.MakeGraph()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        for epoch in range(self.epochs):
            np.random.shuffle(X)
            
            trX = np.array(X)
            for i in range(0, trX.shape[0], self.batchSize):
                self.sess.run(self.update, feed_dict={self.X: trX[i:i+self.batchSize]})

#             print("Trained epoch ", epoch)


    def GetRecommendations(self, inputUser):
        hidden = tf.nn.sigmoid(tf.matmul(self.X, self.weights) + self.hiddenBias)
        visible = tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(self.weights)) + self.visibleBias)

        feed = self.sess.run(hidden, feed_dict={ self.X: inputUser} )
        rec = self.sess.run(visible, feed_dict={ hidden: feed} )
        return rec[0]       

    def MakeGraph(self):
        tf.set_random_seed(0)
        
        # Create variables for the graph, weights and biases
        self.X = tf.placeholder(tf.float32, [None, self.visibleDimensions], name="X")
        
        # Initialize weights randomly
        maxWeight = -4.0 * np.sqrt(6.0 / (self.hiddenDimensions + self.visibleDimensions))
        self.weights = tf.Variable(tf.random_uniform([self.visibleDimensions, self.hiddenDimensions], minval=-maxWeight, maxval=maxWeight), tf.float32, name="weights")
        
        self.hiddenBias = tf.Variable(tf.zeros([self.hiddenDimensions], tf.float32, name="hiddenBias"))
        self.visibleBias = tf.Variable(tf.zeros([self.visibleDimensions], tf.float32, name="visibleBias"))
        
        # Perform Gibbs Sampling for Contrastive Divergence, per the paper we assume k=1 instead of iterating over the 
        # forward pass multiple times since it seems to work just fine
        
        # Forward pass
        # Sample hidden layer given visible...
        # Get tensor of hidden probabilities
        hProb0 = tf.nn.sigmoid(tf.matmul(self.X, self.weights) + self.hiddenBias)
        # Sample from all of the distributions
        hSample = tf.nn.relu(tf.sign(hProb0 - tf.random_uniform(tf.shape(hProb0))))
        # Stitch it together
        forward = tf.matmul(tf.transpose(self.X), hSample)
        
        # Backward pass
        # Reconstruct visible layer given hidden layer sample
        v = tf.matmul(hSample, tf.transpose(self.weights)) + self.visibleBias
        
        # Build up our mask for missing ratings
        vMask = tf.sign(self.X) # Make sure everything is 0 or 1
        vMask3D = tf.reshape(vMask, [tf.shape(v)[0], -1, self.ratingValues]) # Reshape into arrays of individual ratings
        vMask3D = tf.reduce_max(vMask3D, axis=[2], keepdims=True) # Use reduce_max to either give us 1 for ratings that exist, and 0 for missing ratings
        
        # Extract rating vectors for each individual set of 10 rating binary values
        v = tf.reshape(v, [tf.shape(v)[0], -1, self.ratingValues])
        vProb = tf.nn.softmax(v * vMask3D) # Apply softmax activation function
        vProb = tf.reshape(vProb, [tf.shape(v)[0], -1]) # And shove them back into the flattened state. Reconstruction is done now.
        # Stitch it together to define the backward pass and updated hidden biases
        hProb1 = tf.nn.sigmoid(tf.matmul(vProb, self.weights) + self.hiddenBias)
        backward = tf.matmul(tf.transpose(vProb), hProb1)
    
        # Now define what each epoch will do...
        # Run the forward and backward passes, and update the weights
        weightUpdate = self.weights.assign_add(self.learningRate * (forward - backward))
        # Update hidden bias, minimizing the divergence in the hidden nodes
        hiddenBiasUpdate = self.hiddenBias.assign_add(self.learningRate * tf.reduce_mean(hProb0 - hProb1, 0))
        # Update the visible bias, minimizng divergence in the visible results
        visibleBiasUpdate = self.visibleBias.assign_add(self.learningRate * tf.reduce_mean(self.X - vProb, 0))

        self.update = [weightUpdate, hiddenBiasUpdate, visibleBiasUpdate]
# -*- coding: utf-8 -*-
# """
# Created on Fri May  4 13:08:25 2018
# @author: Frank
# """

class RBMAlgorithm(AlgoBase):

    def __init__(self, epochs=100, hiddenDim=100, learningRate=0.001, batchSize=100, sim_options={}):
        AlgoBase.__init__(self)
        self.epochs = epochs
        self.hiddenDim = hiddenDim
        self.learningRate = learningRate
        self.batchSize = batchSize
        
    def default_prediction(self):
        return 1.0 # Avoid return what is impossible to predict
        
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        numUsers = trainset.n_users
        numItems = trainset.n_items
        
        trainingMatrix = np.zeros([numUsers, numItems, 10], dtype=np.float32)
        
        for (uid, iid, rating) in trainset.all_ratings():
            trainingMatrix[int(uid), int(iid), int(rating) - 1] = 1
        
        # Flatten to a 2D array, with nodes for each possible rating type on each possible item, for every user.
        trainingMatrix = np.reshape(trainingMatrix, [trainingMatrix.shape[0], -1])
        
        # Create an RBM with (num items * rating values) visible nodes
        rbm = RBM(trainingMatrix.shape[1], hiddenDimensions=self.hiddenDim, learningRate=self.learningRate, batchSize=self.batchSize, epochs=self.epochs)
        rbm.Train(trainingMatrix)

        self.predictedRatings = np.zeros([numUsers, numItems], dtype=np.float32)
        for uiid in range(trainset.n_users):
#             if (uiid % 50 == 0):
#                 print("Processing user ", uiid)
            recs = rbm.GetRecommendations([trainingMatrix[uiid]])
            recs = np.reshape(recs, [numItems, 10])
            
            for itemID, rec in enumerate(recs):
                # The obvious thing would be to just take the rating with the highest score:                
                #rating = rec.argmax()
                # ... but this just leads to a huge multi-way tie for 5-star predictions.
                # The paper suggests performing normalization over K values to get probabilities
                # and take the expectation as your prediction, so we'll do that instead:
                normalized = self.softmax(rec)
                rating = np.average(np.arange(10), weights=normalized)
                self.predictedRatings[uiid, itemID] = rating + 1
        
        return self


    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')
        
        rating = self.predictedRatings[u, i]
        
        if (rating < 0.001):
            raise PredictionImpossible('No valid prediction exists.')
            
        return rating

evaluate(RBMAlgorithm())