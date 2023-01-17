import numpy as np
import pandas as pd
from sympy import sieve
import matplotlib.pyplot as plt
%matplotlib inline
cities = pd.read_csv("../input/cities.csv", index_col=['CityId'])
print(cities.head(3))
print(cities.shape)
sieve.primerange(0, cities.shape[0]) #取CityID為質數
pnums = list(sieve.primerange(0, cities.shape[0]))
cities['isprime'] = cities.index.isin(pnums)
cities.head(10)
from sklearn.mixture import GaussianMixture
mclusterer = GaussianMixture(n_components=350, tol=0.01, random_state=66, verbose=1) #生成350個點
cities['mclust'] = mclusterer.fit_predict(cities[['X', 'Y']].values)
cities.head()
centers = cities.groupby('mclust')['X', 'Y'].agg('mean').reset_index()
centers.head()
df = centers.drop("mclust",axis = 1) # Drop CityID
centers_array=df.values #轉成Array
type(centers_array)
class GA(object):
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, ):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        self.pop = np.vstack([np.random.permutation(DNA_size) for _ in range(pop_size)]) 
        # 隨機生成 500*350 的陣列: 多維隨機打亂數列[1 2 ...], 並且沿著垂直方向堆疊

    def translateDNA(self, DNA, city_position):     
        line_x = np.empty_like(DNA, dtype=np.float64)
        line_y = np.empty_like(DNA, dtype=np.float64)
        for i, d in enumerate(DNA):
            city_coord = city_position[d]
            line_x[i, :] = city_coord[:, 0]
            line_y[i, :] = city_coord[:, 1]
        return line_x, line_y # 生成2個 500*350 矩陣

    def get_fitness(self, line_x, line_y):
        total_distance = np.empty((line_x.shape[0],), dtype=np.float64) #創造(500,1) 隨機empty矩陣
        for i, (xs, ys) in enumerate(zip(line_x, line_y)):              #將line_x,line_y merge為ZIP, 在分別取出index & (x,y)值
            total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys)))) #算出1~500分別的總距離
        fitness = np.exp(self.DNA_size * 22850 / 2 / total_distance) #計算fitness, DNA_Size=350 (取指數分配, 拉大x軸差距, 創造出差異性)
        return fitness, total_distance
    
    def select(self, fitness):
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return self.pop[idx]

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)                        # select another individual from pop
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)   # choose crossover points
            keep_city = parent[~cross_points]                                       # find the city number
            swap_city = pop[i_, np.isin(pop[i_].ravel(), keep_city, invert=True)]
            parent[:] = np.concatenate((keep_city, swap_city))
        return parent

    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                swap_point = np.random.randint(0, self.DNA_size)
                swapA, swapB = child[point], child[swap_point]
                child[point], child[swap_point] = swapB, swapA
        return child

    def evolve(self, fitness):
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop
class TravelSalesPerson(object):
    def __init__(self, n_cities):
        #self.city_position = np.random.rand(n_cities, 2)
        self.city_position = centers_array
        plt.ion()

    def plotting(self, lx, ly, total_d):
        plt.cla()
        plt.scatter(self.city_position[:, 0].T, self.city_position[:, 1].T, s=100, c='k')
        plt.plot(lx.T, ly.T, 'r-')
        plt.text(-400, -400, "Total distance=%.2f" % total_d, fontdict={'size': 20, 'color': 'red'})
        plt.xlim((-500, 5500))
        plt.ylim((-500, 3500))
        plt.pause(0.01)
N_CITIES = 350  # DNA size
CROSS_RATE = 0.1
MUTATE_RATE = 0.005
POP_SIZE = 500
N_GENERATIONS = 1000

ga = GA(DNA_size=N_CITIES, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)

#env = cities_array
env = TravelSalesPerson(N_CITIES) #隨機生成 N_CITIES 個點

fit_list = []
dist_list = []

for generation in range(N_GENERATIONS):
    lx, ly = ga.translateDNA(ga.pop, env.city_position) #得到500組初始座標連線 (初始解?)
    fitness, total_distance = ga.get_fitness(lx, ly) #得到初始fitness(distance去指數分配) & 總距離長度
    ga.evolve(fitness) #啟動基因演算法, 含交配 與 突變
    best_idx = np.argmax(fitness) #得到演算過後的最佳值
    fit_list.append(fitness[best_idx])
    dist_list.append(total_distance[best_idx])
    #print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx],)
    #print('Gen:', generation, '| total distance: %.2f' % total_distance[best_idx],)
    # env.plotting(lx[best_idx], ly[best_idx], total_distance[best_idx]) # 劃出行軍圖與總距離長度

plt.ioff()
plt.show()
plt.figure(figsize=(10,6))
plt.plot(fit_list,color="blue", linewidth=2, linestyle="-")
plt.xlabel('generations', fontsize=12)
plt.title("Fittness Trend in 1000 generations", fontsize=14)
plt.show()
plt.figure(figsize=(10,6))
plt.plot(dist_list,color="red", linewidth=2, linestyle="-")
plt.xlabel('generations', fontsize=12)
plt.title("Fittness Trend in 1000 generations", fontsize=14)
plt.show()