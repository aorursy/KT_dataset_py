import numpy as np
import pandas as pd
df_cities = pd.read_csv('../input/cities.csv')
df_cities.tail()
from sympy import isprime
df_cities.loc[:,'prime'] = df_cities.loc[:,'CityId'].apply(isprime)
df_cities.head()
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 10))
plt.scatter(df_cities[df_cities['prime']==False].X , df_cities[df_cities['prime']==False].Y, s= 0.1)
plt.scatter(df_cities[df_cities['CityId']==0].X , df_cities[df_cities['CityId']==0].Y, s= 200, color = 'yellow')
plt.scatter(df_cities[df_cities['prime']==True].X , df_cities[df_cities['prime']==True].Y, s= 0.5, color = 'red')
plt.grid(False)
plt.show()
print('How many tenth cities will not have been assigned to a prime', (len(df_cities.index)/10) - df_cities['prime'].value_counts()[1])
print('How much bigger is the total amount of cities related to the prime cities', len(df_cities.index)/df_cities['prime'].value_counts()[1])
plt.title('Number of primes')
df_cities['prime'].value_counts().plot(kind='bar')
# calculate the value of the objective function (total distance)
def pair_distance(x,y):
    x1 = (df_cities.X[x] - df_cities.X[y]) ** 2
    x2 = (df_cities.Y[x] - df_cities.Y[y]) ** 2
    return np.sqrt(x1 + x2)

def total_distance(path):
    distance = [pair_distance(path[x], path[x+1]) + 0.1 * pair_distance(path[x], path[x+1])
                if (x+1)%10 == 0 and df_cities.prime[path[x]] == False else pair_distance(path[x], path[x+1]) for x in range(len(path)-1)]
    return np.sum(distance)
#Path following the Ids, every solution we think will have to beat it
path = df_cities['CityId'].values
path =  np.append(path, 0)
total_distance(path)
