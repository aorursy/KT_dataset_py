import numpy as np
import matplotlib.pyplot as plt
import random
import math
abscissa = np.arange(0, 1.01, 0.01)
abscissa
ordinata = np.sin(abscissa *  math.pi / 2)
ordinata
plt.plot(abscissa, ordinata)
uniform = np.random.rand(30)
plt.hist(uniform)
sample = np.arcsin(uniform) * 2 / math.pi
plt.hist(sample)
edf = []
sample = np.array(sorted(sample))
for i in range(len(sample)):
    edf.append(sample[ sample <= sample[i] ].size / sample.size)
edf = np.array(edf)
#plt.scatter(sample, edf)
plt.plot(sample, edf, drawstyle="steps-post")
from scipy import stats
def my_sin(x):
    return np.sin(x *  math.pi / 2)
stats.kstest(sample, my_sin)
def find_epsilon(n):
    return (np.log(2 / 0.05) / (2 * n)) ** 0.5
n_array = np.arange(1, 200)
epsilons = find_epsilon(n_array)
plt.plot(n_array, epsilons)
def gen_emp_sample(count):
    uniform = np.random.rand(count)
    return np.arcsin(uniform) * 2 / math.pi
def gen_ecdf(my_sample):
    sample = np.array(my_sample)
    def ecdf(x):
        return sample[ sample <= x ].size / sample.size
    return ecdf
def get_distance(sample, my_ecdf, real_function):
    points = np.array(sample)
    values_2 = real_function(points)
    values_1 = np.array([my_ecdf(point) for point in points])
    #values_1 = np.array(list(map(my_ecdf, points)))
    return (np.absolute(values_1 - values_2)).max()
n_array = np.arange(1, 200)
n_samples = []
n_ecdf = []
for i in n_array:
    sample = gen_emp_sample(i)
    n_samples.append(sample)
    n_ecdf.append(gen_ecdf(sample))
n_distance = []
for i in range(len(n_samples)):
    n_distance.append(get_distance(n_samples[i], n_ecdf[i], my_sin))
n_distance = np.array(n_distance)
plt.plot(n_array, epsilons)
plt.plot(n_array, n_distance)
n_samples
plt.hist(n_samples[-1])
stats.kstest(n_samples[-1], my_sin)
get_distance(n_samples[-1], n_ecdf[-1], my_sin)
map.__doc__
plt.scatter(n_samples[-1], list(map(n_ecdf[-1], n_samples[-1])))
