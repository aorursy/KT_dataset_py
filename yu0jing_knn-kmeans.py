#KNN

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

import numpy as np



#處理數據

import csv

import random

def loaddataset(filename,split,trainingset=[],testset=[]):

    with open(filename) as csvfile:

        lines = csv.reader(csvfile)

        dataset = list(lines)

        del dataset[0]

        #檢查test是否為空陣列

        while len(testset) == 0:

            for x in range(len(dataset)):

                for y in range(2):

                    dataset[x][y] = float(dataset[x][y])

                if random.random() < split:

                    trainingset.append(dataset[x])

                else:

                    testset.append(dataset[x])

        

                

#相似度 - 計算兩個資料之間的相似度，已獲得最相似的N個資料做出分類

import math

def euclideanDistance(instance1,instance2,length):

    distance = 0

    #feature數 

    for x in range(length):

        distance += pow((instance1[x]-instance2[x]),2)

    return math.sqrt(distance)



import operator

def getNeighbors(trainingset,testinstance,k):

    distances = []

    length = len(trainingset[0])-1

    for x in range(len(trainingset)):

        dist = euclideanDistance(testinstance,trainingset[x],length)

        distances.append((trainingset[x],dist))

    distances.sort(key = operator.itemgetter(1))

    neighbors = []

    for x in range(k):

        neighbors.append(distances[x][0])

    return neighbors



#結果 - 基於最近的資料得到預測結果(讓附近的資料對預測屬性投票，最多為結果)

def getresponse (neighbors):

    return neighbors[0][2]

    '''classvotes = {}

    for x in range(len(neighbors)):

        response = neighbors[x][-1]

        if response in classvotes:

            classvotes[response] += 1

        else:

            classvotes[response] = 1

    

    sortedvotes = sorted(classvotes.items(),key = operator.itemgetter(1),reverse = True )

    print("sortedvotes = ",sortedvotes)

    #檢查是否平手

    for i in range(len(sortedvotes)):

        if sortedvotes[i][1] == sortedvotes[i + 1 ][1]:

            sortedvotes.append("none")

            return sortedvotes[-1]

        else :

            return sortedvotes[0][0]'''

    

#準確率 - 評估演算法的分類準確率

def getaccuracy(testset,predictions):

    correct = 0

    for x in range(len(testset)):

        if testset[x][-1] is predictions[x]:

            correct +=1

    return(correct/float(len(testset)))*100.0



def main():

    #準備資料

    trainingset = []

    testset = []

    split = 0.7

    loaddataset("../input/knn_test.csv",split,trainingset,testset)

    print("train = " , *trainingset)

    print("test:" , *testset)

    #產生預測

    predictions = []

    k = 3

    for x in range(len(testset)):

        neighbors = getNeighbors(trainingset,testset[x],k)

        print("neighbors" , neighbors)

        result = getresponse(neighbors)

        print("result" , result)

        predictions.append(result)

        print("predicted = " + repr(result) + ", actual = " + repr(testset[x][-1]))

    accuracy = getaccuracy(testset,predictions)

    print("accuracy = " + repr(accuracy) + "%")
main()
#Kmeans

%matplotlib inline

from copy import deepcopy

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (16, 9)

plt.style.use('ggplot')



#匯入資料

data = pd.read_csv("../input/knn_test.csv")

print(data.shape)

data
#取值並圖示

f1 = data["temperature"].values

f2 = data["humidity"].values

x = np.array(list(zip(f1,f2)))

plt.scatter(f1,f2,c="black",s=100)
#計算Euclidean距離 (np.lianlg.norm 計算兩點距離)

def dist(a,b,ax = 1):

    return np.linalg.norm(a-b,axis = ax)



#分類數

k = 2



import random

points = [x[j] for j in range(len(x))]

temp_centroids = np.mean(points, axis=0,dtype = int)



#隨機產生重心點的x座標

x1_temperature = temp_centroids[0] + random.randint(-3, 3)

x2_temperature = temp_centroids[0] + random.randint(-3, 3)

C_x = [x1_temperature,x2_temperature]

#隨機產生重心點的y座標

y1_humidity =  temp_centroids[1] + random.randint(-3,3)

y2__humidity = temp_centroids[1] + random.randint(-3,3)

C_y = [y1_humidity,y2__humidity]

C = np.array(list(zip(C_x, C_y)))

print(C)
# 畫出重心

plt.scatter(f1, f2, c='#050505', s=100)

plt.scatter(C_x, C_y, marker='*', s=500, c='g')
#更新的時候儲存重心

C_old = np.zeros(C.shape)

print("C_old = ",C_old)

# Cluster Lables(0, 1) 

clusters = np.zeros(len(x))

# Error func. - Distance between new centroids and old centroids

print(C)

error = dist(C, C_old, None)

print("error = ",error)

# Loop will run till the error becomes zero

while error != 0:

    # Assigning each value to its closest cluster

    for i in range(len(x)):

        distances = dist(x[i], C)

        print("x[i]",x[i])

        cluster = np.argmin(distances)

        clusters[i] = cluster

        print("clusters[i]",clusters[i])

    # Storing the old centroid values

    C_old = deepcopy(C)

    print('C_old',C_old)

    # Finding the new centroids by taking the average value

    for i in range(k):

        points = [x[j] for j in range(len(x)) if clusters[j] == i]

        print("points" , points)

        C[i] = np.mean(points, axis=0)

        print("c[i]",C[i])

    error = dist(C, C_old, None)

    

colors = ['r', 'g', 'b', 'y', 'c', 'm']

fig, ax = plt.subplots()

for i in range(k):

        points = np.array([x[j] for j in range(len(x)) if clusters[j] == i])

        ax.scatter(points[:, 0], points[:, 1], s=100, c=colors[i])

ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')