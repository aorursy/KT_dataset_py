import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
cities = pd.read_csv("cities.csv")
cities.head()
def is_prime(n):

    """Determines if a positive integer is prime."""



    if n > 2:

        i = 2

        while i ** 2 <= n:

            if n % i:

                i += 1

            else:

                return False

    elif n != 2:

        return False

    return True
#Create a column within the cities dataframe to flag prime cities

cities['is_prime'] = cities.CityId.apply(is_prime)

fig = plt.figure(figsize=(18,18))

plt.scatter(cities.X, cities.Y, c=cities['is_prime'], marker=".", alpha=.5);
from array_stack import ArrayStack

import math

import sys

sys.setrecursionlimit(190000)

Citys = open('cities.csv')

f = Citys.readlines()

Citys.close()

cities = ArrayStack()

for i in range(1, len(f)//100):

    city = f[i].strip('\n').split(',')

    NewCity = [int(city[0]), float(city[1]), float(city[2])]

    cities.push(NewCity)





def cal_distance(x1, y1, x2, y2):

    x_distance = x1 - x2

    y_distance = y1 - y2

    distance = math.sqrt(x_distance **2 + y_distance **2)

    return distance





def tot_distance(data,i):

    if i == 0:

        distance = cal_distance(data[0][1], data[0][2], data[-1][1], data[-1][2])

    elif i%10==0 and data[i-1][0]!=0:

        distance = tot_distance(data, i - 1) + cal_distance(data[i - 1][1], data[i - 1][2], data[i][1], data[i][2])*1.1

    else:

        distance = tot_distance(data, i - 1) + cal_distance(data[i - 1][1], data[i - 1][2], data[i][1], data[i][2])

    return distance



def final_city(c,i):

    if i <=2:

        return c[:3]

    else:

        c1=final_city(c,i-1)

        c2=c1[:1]+[c[i]]+c1[1:]

        c3 = c2

        min_distance=tot_distance(c2,len(c2)-1)

        for j in range(1,i+1):

            c2=c1[:j]+[c[i]]+c1[j:]

            if tot_distance(c2,len(c2)-1)<min_distance:

                min_distance=tot_distance(c2,len(c2)-1)

                c3=c2

        return c3

final=open('city.csv','w')

for i in range(0,19000):

    final.write(str(final_city(cities._data,19000)[i][0])+'\n')

    print(final_city(cities._data,19000)[i][0])

final.write(str(cities._data[0][0])+'\n')

print(cities._data[0][0])

final.close()
import pandas as pd

cities = pd.read_csv("../input/cities.csv")