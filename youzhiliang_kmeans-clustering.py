import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def ReadFile(fileName):
    fileIn = open(fileName, 'r')
    dataSet = []
    for line in fileIn.readlines():
        temp=[]
        lineArr = line.strip().split('\t')
        temp.append(float(lineArr[0]))
        temp.append(float(lineArr[1]))
        dataSet.append(temp)
    fileIn.close()
    return np.mat(dataSet)

dataSet = ReadFile('../input/data.txt')
#print(dataSet)
plt.scatter(np.array(dataSet)[:, 0], np.array(dataSet)[:, 1])
plt.show()
print('Data set size:', len(dataSet))
# calculate Euclidean distance
# you can use this function to calculate distance or use function cdist
def euclDistance(vector1, vector2):
    #print("dis is:")
    #print(np.sqrt(np.sum(np.power(vector2 - vector1, 2))))
    return np.sqrt(np.sum((vector1-vector2)**2))
    #return np.sqrt(np.sum(np.power(vector2 - vector1, 2)))

# show your cluster (only available with 2-D data) 
def showCluster(dataSet, k, centroids, clusterAssment):  
    numSamples, dim = dataSet.shape
    if dim != 2:
        print ("Sorry! I can not draw because the dimension of your data is not 2!")  
        return 1  
  
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  
    if k > len(mark):
        print ("Sorry! Your k is too large!")  
        return 1

    # draw all samples
    for i in range(numSamples):
        # assign colors for samples
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']  
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)  

    plt.show()
k = 4
clusteringResult = KMeans(n_clusters=k).fit(dataSet)
clusterAssment = np.reshape(clusteringResult.labels_, [-1, 1])
centroids = clusteringResult.cluster_centers_

print('show the result of sklearn ...')
showCluster(dataSet, k, centroids, clusterAssment)
# 1. finish function my_k_means;
# 2. Plot the results of the first five iterations (use function showCluster)
def my_k_means(dataSet, k):
    #centroids = list()
    #clusterAssment = list()
    #for i in range(len(dataSet)):
        #clusterAssment.append([np.random.randint(k)])
    #for i in range(k):
        #centroids.append(np.random.random(2))
    
    # TO DO
    iteration = 0
    m,n = np.shape(dataSet)
    centroids = np.zeros((k,n))
    clusterAssment = np.mat(np.zeros((m,2)))
    for i in range(k):
        index= int(np.random.uniform(0,m))
        centroids[i,:] = dataSet[index,:]

    clusterChange = True
    while clusterChange:
        iteration+=1
        clusterChange = False
        #遍历所有的样本
        for i in range(m):
            minDist,minIndex = 1000000.0,-1
            #遍历所有的质心
            for j in range(k):
                dist = euclDistance(centroids[j,:], np.array(dataSet)[i,:])
                if dist < minDist:
                    minDist = dist
                    minIndex = j
            #已经找到需要更新的质心，更新该样本所属的簇
            if clusterAssment [i,0] != minIndex:
                clusterChange = True
                clusterAssment[i,:] = minIndex, minDist**2
        #更新质心
        for j in range(k):
            pointsinCluster = []
            for w in range(m):
                if(clusterAssment[w,0]==j):
                    pointsinCluster.append(np.array(dataSet)[w,:])
            #print(clusterAssment)
        
        sum = 0
        for r in range(np.shape(pointsinCluster)[0]):
            sum += pointsinCluster[r][0]
        mean1 = sum/np.shape(pointsinCluster)[0]
        sum = 0
        for r in range(np.shape(pointsinCluster)[0]):
            sum += pointsinCluster[r][1]
        mean2 = sum/np.shape(pointsinCluster)[0]
        centroids[j,0] = mean1
        centroids[j,1] = mean2
        if iteration <=5:
            print('show the result of the',iteration,' iteration of my_k_means ...')
            showCluster(dataSet, k, centroids, clusterAssment)
        
    #print(clusterAssment)
    #print(centroids)

    return centroids, clusterAssment

k = 4
centroids, clusterAssment = my_k_means(dataSet, k)
print('show the final result of my_k_means ...')
showCluster(dataSet, k, centroids, clusterAssment)
