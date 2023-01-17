# import pandas as pd

import numpy as np

import pandas as pd

import seaborn as sns

# import library for visualizing data 

import matplotlib.pyplot as plt

import os

print(os.getcwd())

print(os.listdir("../input"))
#read in csv file into 

cities = pd.read_csv('../input/traveling-santa-2018-prime-paths/cities.csv')



# avoid run time errors - bug fix for display formats

pd.set_option('display.float_format', lambda x:'%f'%x)

cities = cities.apply(pd.to_numeric, errors='coerce')
#function to get 10% data from original dataset

def divide_record(n):

    #create list to store Id,X,Y of each city

    CityId=["null" for i in range(n+1)]

    City_X=["null" for i in range(n+1)]

    City_Y=["null" for i in range(n+1)]

    #adding data from cities file to list

    for i in range(n+1):

        CityId[i]=i

        City_X[i]=cities['X'][i]

        City_Y[i]=cities['Y'][i]

    #initialized a dictionary to with key and value from 3 lists 

    dict = {'CityId': CityId, 'X': City_X, 'Y': City_Y}  

    df = pd.DataFrame(dict) 

    #write data from dataframe to csv file

    df.to_csv('mini_cites.csv', index=False)
#divide cities file to 10% 

divide_record(max(cities.CityId)//10+1)
#read in csv file into 

mini_cities = pd.read_csv('mini_cites.csv', low_memory=False,dtype={'X': np.float64, 'Y': np.float64}) #increase efficiency





# avoid run time errors - bug fix for display formats

pd.set_option('display.float_format', lambda x:'%f'%x)

mini_cities = mini_cities.apply(pd.to_numeric, errors='coerce')
mini_cities.head()
#function to find which are prime cities

def find_primes(n):

    primes = [True for i in range(n+1)] # Start assuming all numbers are primes

    primes[0] = False # 0 is not a prime

    primes[1] = False # 1 is not a prime

    for i in range(2,int(np.sqrt(n)) + 1):

        if primes[i]:

            k = 2

            while i*k <= n:

                primes[i*k] = False

                k += 1

    return(primes)

mini_cities['is_prime'] = find_primes(max(mini_cities.CityId))

prime_cities = find_primes(max(mini_cities.CityId))

mini_cities.head()
%matplotlib notebook

fig = plt.figure(figsize=(10,10))

plt.scatter(mini_cities[mini_cities['CityId']==0].X , mini_cities[mini_cities['CityId']==0].Y, s= 200, color = 'red')

plt.scatter(mini_cities[mini_cities['is_prime']==True].X , mini_cities[mini_cities['is_prime']==True].Y, s= 0.8, color = 'purple')

plt.scatter(mini_cities[mini_cities['is_prime']==False].X , mini_cities[mini_cities['is_prime']==False].Y, s= 0.1)

plt.grid(False)

#function to calculate distance between two cities by using euclidean distance

def pair_distance(x,y):

    x1 = (mini_cities.X[x] - mini_cities.X[y]) ** 2

    x2 = (mini_cities.Y[x] - mini_cities.Y[y]) ** 2

    return np.sqrt(x1 + x2)
def total_distance(path):

    distance = [pair_distance(path[x], path[x+1]) + 0.1 * pair_distance(path[x], path[x+1])

    if (x+1)%10 == 0 and mini_cities.is_prime[path[x]] == False else pair_distance(path[x], path[x+1]) for x in range(len(path)-1)]

    return np.sum(distance)
dumbest_path = mini_cities['CityId'].values

#add North Pole add the end of trip

dumbest_path =  np.append(dumbest_path,0)
print('Total distance with the sorted city path is '+ "{:,}".format(total_distance(dumbest_path)))
## import file

os.chdir("../input/external/")

from linked_binary_tree import LinkedBinaryTree

from collections import deque

#create list contain all of CityId and passing to queue

path=[]

for x in range(max(mini_cities.CityId)+1):

        path.append(x)

array = deque(path)

#remove the CityId=0 

array.popleft()

#Create list of CityId from tree node

tree_path=[0]

def tree_build():

    #list to store node position

    node=[]

    tree = LinkedBinaryTree()

    node.append(tree._add_root(array.popleft()))

    #add node to the tree

    for i in range(len(path)):

        try:

            node.append(tree._add_left(node[i],array.popleft()))

            node.append(tree._add_right(node[i],array.popleft()))

        except:

            pass

    #get element of each node and add to final path

    if not tree.is_empty():

      for p in tree._subtree_inorder(tree.root()):

        tree_path.append(p.element())

      tree_path.append(0)

tree_build()
print('Total distance with the sorted city path is '+ "{:,}".format(total_distance(tree_path)))
sortx_path=[]

for x in range(1,max(mini_cities.CityId)+1):

        sortx_path.append(x)
City_X=[]

for x in range(max(mini_cities.CityId)+1):

    City_X.append(mini_cities['X'][x])
def insertionSort(arr): 

    # Traverse through 1 to len(arr) 

    for i in range(1,len(arr)): 

        key = arr[i] 

        j = i-1

        while j >=0 and City_X[key] < City_X[arr[j]] : 

                arr[j+1] = arr[j] 

                j -= 1

        arr[j+1] = key 
insertionSort(sortx_path)
#create a path for calculating total distance

sortedx_path=[0]

for each in range(len(sortx_path)):

    sortedx_path.append(sortx_path[each])

sortedx_path.append(0)
print('Total distance with the sorted city path is '+ "{:,}".format(total_distance(sortedx_path)))
sorty_path=[]

for x in range(1,max(mini_cities.CityId)+1):

        sorty_path.append(x)
City_Y=[]

for x in range(max(mini_cities.CityId)+1):

    City_Y.append(mini_cities['Y'][x])
def selectionsort(alist):



   for i in range(len(alist)):



      # Find the minimum element in remaining

       minPosition = i



       for j in range(i+1, len(alist)):

           if City_Y[alist[minPosition]] > City_Y[alist[j]]:

               minPosition = j

                

       # Swap the found minimum element with minPosition       

       temp = alist[i]

       alist[i] = alist[minPosition]

       alist[minPosition] = temp

selectionsort(sorty_path)
#create a path for calculating total distance

sortedy_path=[0]

for each in range(len(sorty_path)-1):

    sortedy_path.append(sorty_path[each])

sortedy_path.append(0)
print('Total distance with the sorted city path is '+ "{:,}".format(total_distance(sortedy_path)))
def nearest_neighbour():

    ids = mini_cities.CityId.values[1:]

    xy = np.array([mini_cities.X.values, mini_cities.Y.values]).T[1:]

    path = [0,]

    while len(ids) > 0:

        last_x, last_y = mini_cities.X[path[-1]], mini_cities.Y[path[-1]]

        dist = ((xy - np.array([last_x, last_y]))**2).sum(-1)

        nearest_index = dist.argmin()

        path.append(ids[nearest_index])

        ids = np.delete(ids, nearest_index, axis=0)

        xy = np.delete(xy, nearest_index, axis=0)

    path.append(0)

    return path



nnpath = nearest_neighbour()
print('Total distance with the Nearest Neighbor path '+  "is {:,}".format(total_distance(nnpath)))
print(os.getcwd())

os.chdir("/kaggle/working")
def submission():

    dict = {'Path': nnpath}  

    df = pd.DataFrame(dict) 

    #write data from dataframe to csv file

    df.to_csv('Final_Submission.csv', index=False)
submission()
#Visualize the traveling path of Nearest Neighbor algorithm

%matplotlib notebook

df_path = pd.DataFrame({'CityId':nnpath}).merge(mini_cities,how = 'left')

fig, ax = plt.subplots(figsize=(10,10))

ax.plot(df_path['X'], df_path['Y'])