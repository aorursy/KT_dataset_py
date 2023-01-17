import random

import progressbar as pb



def gen_data(n,k):

    return [random.randint(0,n) for i in range(k)]





n = 122

k = 30

results = []

for test in pb.progressbar(range(1_000)):

    max_value = max(gen_data(n,k))

    prediction = max_value + (max_value/k) - 1

    results.append(abs(n-prediction))
import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

plt.figure(figsize=(23,5))

plt.plot(np.array(results))

plt.title("Results for 1,000 German Tank Problem Experiments")

plt.xlabel('Experiment #')

plt.ylabel('Absolute Difference Between Target & Formula Prediction')
np.array(results).mean()
n = 342

k = 10

values = [[0 for i in range(15)] for i in range(15)]

index1 = 0

for n in pb.progressbar(range(1,1_500,100)):

    index2 = 0

    for k in range(5,50,3):

        results = []

        for i in range(1_000):

            max_value = max(gen_data(n,k))

            prediction = max_value + (max_value/k) - 1

            results.append(abs(n-prediction))

        values[index1][index2] = np.array(results).mean()

        index2 += 1

    index1 += 1
nlist = [i for i in range(1,1_500,100)]

klist = [i for i in range(5, 50, 3)]

plt.figure(figsize=(18,6))

sns.heatmap(values,annot=True,xticklabels=klist,yticklabels=nlist)

plt.xlabel('# of Serial Codes Obtained')

plt.ylabel('# of True German Tanks Produced')

plt.title('Average Error of Formula by True # of German Tanks vs. # of Serial Codes Obtained')