import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import random





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/datafile2/ACCIDENTS_GU_BCN_2014.csv', encoding = "ISO-8859-1")

data['Date'] = data[u'Dia de mes'].apply(lambda x: str(x)) + '-' + data[u'Mes de any'].apply(lambda x: str(x)) #создает колонку день-месяц(где-то сохраняется случай)

accidents = data.groupby(['Date']).size()#группирует

print(accidents.mean())#выводит среднее
df = accidents.to_frame() #переводит в дата-фрейм

N_test = 10000

elements = 200

# mean array of samples

# mean array of samples

means = [0] * N_test

# sample generation

for i in range (N_test):

    rows = np.random.choice(df.index.values, elements)

    sampled_df = df.loc[rows]

    means[i] = sampled_df.mean()
rows = np.random.choice(df.index. values , 200)

sampled_df = df.loc[rows]

est_sigma_mean = sampled_df.std()/ math.sqrt(200)

print('Direct estimation of SE from one sample of 200 elements:', est_sigma_mean[0])

print('Estimation of the SE by simulating 10000 samples of 200 elements:', np.array(means).std())
def meanBootstrap(X, numberb):

    x = [0]* numberb

    for i in range (numberb):

        sample = [X[j]

            for j in np.random.randint(len(X), size=len(X))]

        x[i] = np.mean(sample)

    return x

m = meanBootstrap(accidents, 10000)

print('Mean estimate:', np.mean(m))
m = accidents.mean()

se = accidents.std() / math.sqrt(len(accidents))

ci = [m - se * 1.96, m + se * 1.96]

print('Confidence interval:', ci)
m = meanBootstrap(accidents , 10000)

sample_mean = np.mean(m)

sample_se = np.std(m)

print('Mean estimate:', sample_mean)

print('SE of the estimate:', sample_se)

ci = [np.percentile(m, 2.5), np. percentile(m, 97.5)]

print('Confidence interval:', ci)
data = pd.read_csv('../input/data-file/ACCIDENTS_GU_BCN_2015.csv', encoding = "ISO-8859-1")

data['Date'] = data['Dia de mes'].apply(lambda x: str(x)) + '-' + data['Mes de any'].apply(lambda x: str(x))

data2 = data['Date']

counts2010 = data['Date'].value_counts()

print('2010: Mean', counts2010.mean())

data = pd.read_csv('/kaggle/input/accidentsbcn/ACCIDENTS_GU_BCN_2013.csv', encoding = "ISO-8859-1")

data['Date'] = data['Dia de mes'].apply(lambda x: str(x)) + '-' + data['Mes de any'].apply(lambda x: str(x))

data2 = data['Date']

counts2013 = data['Date'].value_counts()

print('2013: Mean', counts2013. mean())
n = len(counts2015)

mean = counts2015.mean()

s = counts2015.std()

ci = [mean - s * 1.96/ np.sqrt(n), mean + s * 1.96 / np.sqrt(n)]

print('2014 accident rate estimate:', counts2014.mean())

print('2015 accident rate estimate:', counts2015.mean())

print('CI for 2015:', ci)
m = len(counts2014)

n = len(counts2015)

p = (counts2015.mean() - counts2014.mean())

print('m: ', m, 'n: ', n)

print('mean difference: ', p)
x = counts2010

y = counts2013

pool = np.concatenate([x, y])

np.random.shuffle(pool)

N = 10000 # number of samples

diff = list(range(N))

for i in range(N):

     p1 = [random.choice(pool) for _ in range(n)]

     p2 = [random.choice(pool) for _ in range(n)]

     diff[i] = (np.mean(p1) - np.mean(p2))
diff2 = np.array(diff)

w1 = np.where(diff2 > p)[0]

print('p-value (Simulation)=', len(w1)/ float (N), '(', len(w1) / float(N) * 100 ,'%)', 'Difference =', p)

if (len(w1)/ float (N)) < 0.05:

     print('The effect is likely')

else:

     print('The effect is not likely')