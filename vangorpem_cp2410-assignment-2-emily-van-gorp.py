import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import time

import os

import sympy as sp

import sys

from scipy.spatial import distance

from math import isinf
df_cities = pd.read_csv('../input/cities.csv')
#Determine which cities are prime numbers

def sieve_of_eratosthenes(n):

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
df_cities['is_prime'] = sieve_of_eratosthenes(max(df_cities.CityId))
prime_cities = sieve_of_eratosthenes(max(df_cities.CityId))
#Visualising the map with the prime cities highlighted.

%matplotlib notebook

fig = plt.figure(figsize=(10,10))

plt.scatter(df_cities[df_cities['CityId']==0].X , df_cities[df_cities['CityId']==0].Y, s= 200, color = 'red')

plt.scatter(df_cities[df_cities['is_prime']==True].X , df_cities[df_cities['is_prime']==True].Y, s= 0.8, color = 'purple')

plt.scatter(df_cities[df_cities['is_prime']==False].X , df_cities[df_cities['is_prime']==False].Y, s= 0.1)

plt.grid(False)
start_time = time.time()

def pair_distance(x,y):

    x1 = (df_cities.X[x] - df_cities.X[y]) ** 2

    x2 = (df_cities.Y[x] - df_cities.Y[y]) ** 2

    return np.sqrt(x1 + x2)

end_time = time.time()

dumbest_elapsed_time = end_time - start_time
print("Total elapsed time of dumbest path algorithm: ", dumbest_elapsed_time)
def total_distance(path):

    distance = [pair_distance(path[x], path[x+1]) + 0.1 * pair_distance(path[x], path[x+1])

    if (x+1)%10 == 0 and df_cities.is_prime[path[x]] == False else pair_distance(path[x], path[x+1]) for x in range(len(path)-1)]

    return np.sum(distance)
dumbest_path = df_cities['CityId'].values

#add North Pole add the end of trip

dumbest_path =  np.append(dumbest_path,0)
print('Total distance with the paired city path is '+ "{:,}".format(total_distance(dumbest_path)))
sys.setrecursionlimit(500000)

City_X=[]

for x in range(max(df_cities.CityId)+1):

    City_X.append(df_cities['X'][x])

City_Y=[]

for x in range(max(df_cities.CityId)+1):

    City_Y.append(df_cities['Y'][x])
path=[]

for x in range(1,max(df_cities.CityId)+1):

        path.append(x)
def partition(arr,low,high): 

    i = ( low-1 )         # index of smaller element 

    pivot = arr[high]     # pivot 

  

    for j in range(low , high): 

  

        # If current element is smaller than or 

        # equal to pivot 

        if   City_X[arr[j]] <= City_X[pivot]: 

          

            # increment index of smaller element 

            i = i+1 

            arr[i],arr[j] = arr[j],arr[i] 

  

    arr[i+1],arr[high] = arr[high],arr[i+1] 

    return ( i+1 ) 
start_time = time.time()

def quickSort(arr,low,high): 

    if low < high: 

  

        # pi is partitioning index, arr[p] is now 

        # at right place 

        pi = partition(arr,low,high) 

  

        # Separately sort elements before 

        # partition and after partition 

        quickSort(arr, low, pi-1) 

        quickSort(arr, pi+1, high) 

end_time = time.time()

quicksort_elapsed_time = end_time - start_time
print("Total elapsed time of quicksort algorithm: ", quicksort_elapsed_time)
quicksort_path=[]

for x in range(1,max(df_cities.CityId)+1):

        quicksort_path.append(x)
quickSort(quicksort_path,0,len(quicksort_path)-1)
quicksorted_path=[0]

for each in range(len(quicksort_path)):

    quicksorted_path.append(quicksort_path[each])

quicksorted_path.append(0)
print('Total distance with the quick sorted cities based on X path is '+ "{:,}".format(total_distance(quicksorted_path)))
matrix=[]

def generate_graph():

    for i in range(20):

        #get array of list X,Y value from dataset

        coordinates = np.array([df_cities.X.values, df_cities.Y.values]).T[0:20]

        #calculate distance of all city from last city in path list

        dist = ((coordinates - np.array([City_X[i], City_Y[i]]))**2).sum(-1)

        matrix.append(dist.tolist())

generate_graph()
def size(int_type):

   length = 0

   count = 0

   while (int_type):

       count += (int_type & 1)

       length += 1

       int_type >>= 1

   return count



def length(int_type):

   length = 0

   count = 0

   while (int_type):

       count += (int_type & 1)

       length += 1

       int_type >>= 1

   return length
def generateSubsets(n):

    l = []

    for i in range(2**n):

        l.append(i)

    return sorted(l, key = lambda x : size(x) )
def inSubset(i, s):

    while i > 0 and s > 0:

        s = s >> 1

        i -= 1

    cond = s & 1

    return cond



def remove(i, s):

    x = 1

    x = x << i

    l = length(s)

    l = 2 ** l - 1

    x = x ^ l

    #print ( "i - %d x - %d  s - %d x&s -  %d " % (i, x, s, x & s) )

    return x & s

def findPath(p):

    n = len(p[0])

    number = 2 ** n - 2

    prev = p[number][0]

    path = []

    while prev != -1:

        path.append(prev)

        number = remove(prev, number)

        prev = p[number][prev]

    reversepath = [str(path[len(path)-i-1]+1) for i in range(len(path))]

    reversepath.append("1")

    reversepath.insert(0, "1")

    return reversepath
start_time = time.time()

def tsp():

    n=len(matrix) 

    l = generateSubsets(n)

    cost = [ [-1 for city in range(n)] for subset in l]

    p = [ [-1 for city in range(n)] for subset in l]

    count = 1

    total = len(l)

    

    for subset in l:

        for dest in range(n):

            if not size(subset):

                cost[subset][dest] = matrix[0][dest]

            elif (not inSubset(0, subset)) and (not inSubset(dest, subset)):

                mini = float("inf")

                for i in range(n):

                    if inSubset(i, subset):

                        modifiedSubset = remove(i, subset)

                        val = matrix[i][dest] + cost[modifiedSubset][i]

                        

                        if val < mini:

                            mini = val

                            p[subset][dest] = i

                            

                if not isinf(mini):

                    cost[subset][dest] = mini

        count += 1

    path = findPath(p)

    print(" => ".join(path))

    Cost = cost[2**n-2][0]

    print("Total distance with dynamic programing using graph:",Cost)

tsp()  

end_time = time.time()

matrix_elapsed_time = end_time - start_time
print("Total elapsed time of matrix algorithm: ", matrix_elapsed_time)
def final_output():

    dict = {'Path': tree_path}  

    df = pd.DataFrame(dict) 

    df.to_csv('Final_Submission.csv', index=False)