import pandas as pd

import numpy as np
data = pd.read_csv('cities.csv')

total = data.shape[0]
#check prime

def isSu(num):

    

    if num > 1:

        

        for i in range(2, num//2+1):

            if (num % i) == 0:

                return False

                break

        else:

            return True

    

    else:

        return False

isSu(7)
#checks that the adjacency matrix is constructed correctly

data_test = data.head()



#calculate distance

def distance(tuple1, tuple2):

    

    return np.sqrt((tuple1[0]-tuple2[0])**2 + (tuple1[1]-tuple2[1])**2)

    



#Adjacency matrix and mapping (dictionary) are used to store distance respectively

def find_distance(data):

    #initialize

    temp = data.values

    matrix = [([0] * data.shape[0]) for i in range(data.shape[0])]

    matrix_dic = {}

    for i in range(data.shape[0]):

        for j in range(i+1,data.shape[0]):

            dis = distance((temp[i][1],temp[i][2]),(temp[j][1],temp[j][2]))

            

            #every 10 cities, if prime, increase 10%

            #every 10 cities

            if i % 10 == 0 or j % 10 == 0:

                #check if prime

                if isSu(i) or isSu(j):

                    dis = dis * 1.1 

            

            #Construct the adjacency matrix

            matrix[i][j] = dis

            matrix[j][i] = dis

            matrix_dic[(i,j)] = dis

            matrix_dic[(j,i)] = dis

            

    return matrix, matrix_dic



dis,matrix_dic = find_distance(data_test)

dis
matrix_dic
#Build the adjacency list store path (edge) as follows

#graph = {0: [1,2,3],

#        1: [0,2,3],

#        2: [0,1,3],

#        3: [0,1,2]}



def build_graph(data):

    graph = {}

    values = set(data['CityId'].values)

    for i in values:

        neighborhoods = values.copy()

        neighborhoods.remove(i)

        graph[i] = list(neighborhoods)

    

    return graph



build_graph(data_test)


graph = build_graph(data_test)



def total_distance(path,dis):

    total = 0.

    for item in range(len(path) - 1):

        total = total + dis[path[item]][path[item+1]]

    return total



# Find all the paths from start 0 to end 0, DFS

def find_all_path_dsf(graph, start, end,total, path=[],isFirst = True):

    path = path + [start]

    if start == end and not isFirst and total == len(path):

        return [path]





    paths = []  # store all path

    for node in graph[start]:

        

        if node not in path or (len(path) == total -1 and (node == 0)):

            newpaths = find_all_path_dsf(graph, node, end, total, path, False)

            for newpath in newpaths:

                paths.append(newpath)

    return paths



def find_all_path_dis(allpath,dis):

    distances = []  # total distance

    for item in allpath:

        distances.append(total_distance(item,dis))

    return distances



allpath = find_all_path_dsf(graph,0,0,data_test.shape[0]+1)



print('\nall path：',allpath)



distances = find_all_path_dis(allpath,dis)

print('\ntotal distance：',distances)

#Using heap sort generation order, find the minimum

def heapify(arr, n, i): 

    largest = i  

    l = 2 * i + 1     # left = 2*i + 1 

    r = 2 * i + 2     # right = 2*i + 2 

  

    if l < n and arr[i] < arr[l]: 

        largest = l 

  

    if r < n and arr[largest] < arr[r]: 

        largest = r 

  

    if largest != i: 

        arr[i],arr[largest] = arr[largest],arr[i]  # 

  

        heapify(arr, n, largest) 



def heapSort(arr): 

    n = len(arr) 

  

    # Build the maximum heap. 

    for i in range(n, -1, -1): 

        heapify(arr, n, i) 

  

    

    for i in range(n-1, 0, -1): 

        arr[i], arr[0] = arr[0], arr[i]   # 

        heapify(arr, i, 0) 



origin = distances.copy()

heapSort(distances) 

n = len(distances) 

print ("After sorting, the first 3") 

distances[:3]
print ("Min distance：" ,distances[0])



n = -1;

for i,item in enumerate(origin):

    if item == distances[0]:

        n = i

        break

print ("Min distance path：" ,allpath[n])

import sys

#Set the maximum recursive depth

sys.setrecursionlimit(90000000)



#read 19W cities

data = pd.read_csv('cities.csv')

#

data = data.iloc[:197772,:]

print("End of reading")

#1.Read 19W city for distance calculation, and store distance with adjacency matrix and dictionary respectively

dis, matrix_dic = find_distance(data)

print("Step 1 done")

#2.Build the path adjacency list

graph = build_graph(data)

print("Step 2 done")

#3.DFS find all possible paths, starting at 0 and ending at 0

allpath = find_all_path_dsf(graph,0,0,data.shape[0]+1)

print("Step 3 done")

#4.Calculate distance

distances = find_all_path_dis(allpath,dis)

print("Step 4 done")

#5.Heap sort

origin = distances.copy()

heapSort(distances)

print("Step 5 done")

#6.Find path of minimum distance

print ("Minimum  distance：" ,distances[0])

n = -1;

for i,item in enumerate(origin):

    if item == distances[0]:

        n = i

        break

#There could be multiple paths of the same distance, but I'm just going to take one

print ("Final path：" ,allpath[n])

print("Step 6 done")