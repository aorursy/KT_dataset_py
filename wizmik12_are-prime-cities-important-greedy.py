import numpy as np
import pandas as pd
import time
from sympy import isprime 

df_cities = pd.read_csv('../input/cities.csv')
df_cities.loc[:,'prime'] = df_cities.loc[:,'CityId'].apply(isprime)

# calculate the value of the objective function (total distance)
def pair_distance(x,y):
    x1 = (df_cities.X[x] - df_cities.X[y]) ** 2
    x2 = (df_cities.Y[x] - df_cities.Y[y]) ** 2
    return np.sqrt(x1 + x2)

def total_distance(path):
    distance = [pair_distance(path[x], path[x+1]) + 0.1 * pair_distance(path[x], path[x+1])
                if (x+1)%10 == 0 and df_cities.prime[path[x]] == False else pair_distance(path[x], path[x+1]) for x in range(len(path)-1)]
    return np.sum(distance)

def make_submission(name, path):
    assert path[0] == path[-1] == 0
    assert len(set(path)) == len(path) - 1 == 197769
    pd.DataFrame({'Path': path}).to_csv(f'{name}.csv', index=False)
def nearest_neighbour():
    cities = df_cities.copy()
    ids = cities.CityId.values[1:]
    xy = np.array([cities.X.values, cities.Y.values]).T[1:]
    path = [0,]
    while len(ids) > 0:
        last_x, last_y = cities.X[path[-1]], cities.Y[path[-1]]
        dist = ((xy - np.array([last_x, last_y]))**2).sum(-1)
        nearest_index = dist.argmin()
        path.append(ids[nearest_index])
        ids = np.delete(ids, nearest_index, axis=0)
        xy = np.delete(xy, nearest_index, axis=0)
    path.append(0)
    make_submission('nearest_neighbour.csv', path)
    return path

def nearest_neighbour_prim():
    cities = df_cities.copy()
    ids = cities.CityId.values[1:]
    prim = cities.prime.values[1:]
    xy = np.array([cities.X.values, cities.Y.values]).T[1:]
    path = [0,]
    step = 1
    while len(ids) > 0:
        last_x, last_y = cities.X[path[-1]], cities.Y[path[-1]]
        dist = np.sqrt(((xy - np.array([last_x, last_y]))**2).sum(-1))
        if step % 10 == 0:
            dist = np.array([dist[x] if prim[x] else dist[x] * 1.1 for x in range(len(dist))])
        nearest_index = dist.argmin()
        path.append(ids[nearest_index])
        ids = np.delete(ids, nearest_index, axis=0)
        prim = np.delete(prim, nearest_index, axis=0)
        xy = np.delete(xy, nearest_index, axis=0)
        step +=1 
    path.append(0)
    make_submission('nearest_neighbour_prime.csv', path)
    return path

path = nearest_neighbour()
path_prime = nearest_neighbour_prim()
distance = total_distance(path)
prime_distance = total_distance(path_prime)
print('The distance of NN is', distance, 'but taking into account the prime cities:', prime_distance)