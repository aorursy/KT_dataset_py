from IPython.display import Image

Image("/kaggle/input/heuristic-portfolio-selection/Lesson3GoalHeaderImage.png")
from IPython.display import Image

Image("/kaggle/input/heuristic-portfolio-selection/Lesson3Fig3_1.png")

from IPython.display import Image

Image("/kaggle/input/heuristic-portfolio-selection/Lesson3Fig3_2.png")

from IPython.display import Image

Image("/kaggle/input/heuristic-portfolio-selection/Lesson3Equation3_1.png")



from IPython.display import Image

Image("/kaggle/input/heuristic-portfolio-selection/Lesson3Fig3_3.png")

from IPython.display import Image

Image("/kaggle/input/heuristic-portfolio-selection/Lesson3Fig3_4.png")

#read stock prices from a cleaned DJIA dataset 



#Dependencies 

import numpy as np

import pandas as pd

from sklearn.cluster import KMeans 



#input stock prices data set 

stockFileName = '/kaggle/input/heuristic-portfolio-selection/DJIA_Apr112014_Apr112019.csv'

originalRows = 1259   #excluding header

originalColumns = 29  #excluding date

clusters = 15



#read stock dataset into a dataframe 

df = pd.read_csv(stockFileName,  nrows= originalRows)



#extract asset labels

assetLabels = df.columns[1:originalColumns+1].tolist()

print(assetLabels)



#extract stock prices excluding header and trading dates

dfStockPrices = df.iloc[0:, 1:]



#store stock prices as an array

arStockPrices = np.asarray(dfStockPrices)

[rows, cols]= arStockPrices.shape

print(rows, cols)

print(arStockPrices)

#function for Stock Returns computing 

def StockReturnsComputing(StockPrice, Rows, Columns):

    

    import numpy as np

    

    StockReturn = np.zeros([Rows-1, Columns])

    for j in range(Columns):  # j: Assets

        for i in range(Rows-1):     #i: Daily Prices

            StockReturn[i,j]=((StockPrice[i+1, j]-StockPrice[i,j])/StockPrice[i,j])



    return StockReturn

#compute daily returns of all stocks in the mini universe

arReturns = StockReturnsComputing(arStockPrices, rows, cols)

print('Size of the array of daily returns of stocks:\n', arReturns.shape)

print('Array of daily returns of stocks\n',  arReturns)
#compute mean returns and variance covariance matrix of returns

meanReturns = np.mean(arReturns, axis = 0)

print('Mean returns:\n', meanReturns)

covReturns = np.cov(arReturns, rowvar=False)

#set precision for printing results

np.set_printoptions(precision=5, suppress = True)

print('Size of Variance-Covariance matrix of returns:\n', covReturns.shape)

print('Variance-Covariance matrix of returns:\n', covReturns)
#prepare asset parameters for k-means clustering

#reshape for concatenation

meanReturns = meanReturns.reshape(len(meanReturns),1)

assetParameters = np.concatenate([meanReturns, covReturns], axis = 1)

print('Size of the asset parameters for clustering:\n', assetParameters.shape)

print('Asset parameters for clustering:\n', assetParameters)
#kmeans clustering of assets using the characteristic vector of 

#mean return and variance-covariance vector of returns



assetsCluster= KMeans(algorithm='auto',  max_iter=600, n_clusters=clusters)

print('Clustering of assets completed!') 

assetsCluster.fit(assetParameters)

centroids = assetsCluster.cluster_centers_

labels = assetsCluster.labels_



print('Centroids:\n', centroids)

print('Labels:\n', labels)
#fixing asset labels to cluster points

print('Stocks in each of the clusters:\n',)

assets = np.array(assetLabels)

for i in range(clusters):

    print('Cluster', i+1)

    clt  = np.where(labels == i)

    assetsCluster = assets[clt]

    print(assetsCluster)

    
from IPython.display import Image

Image("/kaggle/input/heuristic-portfolio-selection/Lesson3Fig3_5.png")

from IPython.display import Image

Image("/kaggle/input/heuristic-portfolio-selection/Lesson3ExitTailImage.png")