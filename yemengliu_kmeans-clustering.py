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

plt.scatter(np.array(dataSet)[:, 0], np.array(dataSet)[:, 1])

plt.show()

print('Data set size:', len(dataSet))
# calculate Euclidean distance

# you can use this function to calculate distance or use function cdist

def euclDistance(vector1, vector2):

    return np.sqrt(np.sum(np.power(vector2 - vector1, 2)))



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



def Cal_Distance(vector1, vector2):

    vec_a = np.array(vector1)

    vec_b = np.array(vector2)

    [n,_] = vector1.shape

    return np.sqrt(np.power(vec_b[0:n,0] - vec_a[0:n,0], 2)+np.power(vec_b[0:n,1] - vec_a[0:n,1], 2))



def my_k_means(dataSet, k,iterations):

    centroids = list()

    clusterAssment = list()

    for i in range(len(dataSet)):

        clusterAssment.append([np.random.randint(k)])

    for i in range(k):

        centroids.append(np.random.random(2))



    for i in range(iterations):

        point = [[] for i in range(len(centroids))]

        new_clusterAssment = []

        new_centroids = []

        for j in range(len(dataSet)):

            data_index = np.ones((len(centroids),1))*dataSet[j,:]

            data_center_dis = Cal_Distance(data_index, centroids)

            dis_index = np.argmin(data_center_dis)

            point[dis_index].append(j)

            new_clusterAssment.append(dis_index)

        for j in range(len(centroids)):

            if len(point[j]) > 0:

                temp = np.mean(dataSet[point[j]],0)

                new_centroids.append([temp[0,0],temp[0,1]])

            else:

                new_centroids.append(centroids[j])

        centroids=np.mat(new_centroids)

        clusterAssment=new_clusterAssment

        clusterAssment = np.transpose(clusterAssment)

        clusterAssment = np.mat([[clusterAssment[i]] for i in range(len(dataSet))])

        print('show the result of my_k_means, iterations:',i+1,'times')

        showCluster(dataSet, k, centroids, clusterAssment)



k = 4

iterations = 5

my_k_means(dataSet, k,iterations)