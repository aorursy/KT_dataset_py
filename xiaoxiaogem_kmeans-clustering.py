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

print(centroids)

print('show the result of sklearn ...')

showCluster(dataSet, k, centroids, clusterAssment)

# 1. finish function my_k_means;

# 2. Plot the results of the first five iterations (use function showCluster)

def euclDistance0(vector1, vector2):

    vector1_a = np.array(vector1)

    vector2_a = np.array(vector2)

    [size_x,size_y] = vector1.shape

    return np.sqrt(np.power(vector2_a[0:size_x,0] - vector1_a[0:size_x,0], 2)+np.power(vector2_a[0:size_x,1] - vector1_a[0:size_x,1], 2))

def point_assignment(dataSet,centroids):

    k_center = len(centroids)

    point_assign = [[] for i in range(k_center)]

    data_num = len(dataSet)

    point_assign_index = []

    for index in range(data_num):

        data_index = np.ones((k_center,1))*dataSet[index,:]

        data_center_dis = euclDistance0(data_index, centroids)

        dis_index = np.argmin(data_center_dis)

        point_assign[dis_index].append(index)

        point_assign_index.append(dis_index)

    return point_assign,point_assign_index

def ensure_center(dataSet,point_assign,centroids):

    center_len = len(point_assign)

    center_new = []

    for index in range(center_len):

        if len(point_assign[index]) > 0:

            center_ensure = np.mean(dataSet[point_assign[index]],0)

            center_new.append([center_ensure[0,0],center_ensure[0,1]])

        else:

            center_new.append(centroids[index])

    return center_new    



    



def my_k_means(dataSet, k,times):

    centroids = list()

    clusterAssment = list()

    for i in range(k):

        centroids.append((np.random.random(2)-0.5)*5)

    for i in range(times):

        if(i < 6 and i>0):

            clusterAssment0 = np.transpose(clusterAssment)

            clusterAssment0 = [[clusterAssment0[j]] for j in range(len(dataSet))]

            showCluster(dataSet, k, np.mat(centroids), np.mat(clusterAssment0))

           

        point_assign,clusterAssment = point_assignment(dataSet,centroids)

        centroids = ensure_center(dataSet,point_assign,centroids)

        #print(centroids)

     

    clusterAssment = np.transpose(clusterAssment)

    clusterAssment = [[clusterAssment[i]] for i in range(len(dataSet))]

    # TO DO

    #return centroids, clusterAssment

    return np.mat(centroids), np.mat(clusterAssment)



k = 4

centroids, clusterAssment = my_k_means(dataSet, k,50)

print('show the result of my_k_means ...')

print(centroids)

showCluster(dataSet, k, centroids, clusterAssment)