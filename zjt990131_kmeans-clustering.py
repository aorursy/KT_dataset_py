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

def my_k_means(dataSet, k):

    centroids = list()

    clusterAssment = list()

    for i in range(len(dataSet)):

        clusterAssment.append([np.random.randint(k)])

    for i in range(k):

        centroids.append(np.random.random(2))

    # TO DO

    data=dataSet.tolist()

    flag=1

    while flag!=0:

        s0=[0]*k#s0是横坐标求和

        s1=[0]*k#s1是纵坐标求和

        l=[0]*k#每一类共有多少个

        for i in range(len(dataSet)):

            dist=euclDistance(dataSet[i],centroids[0])

            tag=0

            for j in range(1,k):

                tmp=euclDistance(dataSet[i],centroids[j])#求每个点到k个中心向量的距离

                if tmp<dist:

                    dist=tmp

                    tag=j#打擂台的方式求出最小值，并更新

            clusterAssment[i]=[tag]

            s0[tag]+=data[i][0]

            s1[tag]+=data[i][1]

            l[tag]+=1

        flag=0#flag用于判断中心向量是否发生了变化

        for i in range(k):

            if l[i]==0:

                l[i]=1

            tmp=np.array([s0[i]/l[i],s1[i]/l[i]])

            showCluster(dataSet, k, np.mat(centroids), np.mat(clusterAssment))

            if any(tmp!=centroids[i]):#只要横纵中任意一个坐标不同就需要更新，且flag+1

                flag+=1

                centroids[i]=tmp

    return np.mat(centroids), np.mat(clusterAssment)



k = 4



centroids, clusterAssment = my_k_means(dataSet, k)

print('show the result of my_k_means ...')

print(centroids)

print(clusterAssment)

showCluster(dataSet, k, centroids, clusterAssment)