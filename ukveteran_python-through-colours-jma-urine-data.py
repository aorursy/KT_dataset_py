import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

import seaborn as sns

sns.set(style="whitegrid")

from collections import Counter

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings

warnings.filterwarnings('ignore')



dat = pd.read_csv('../input/urine-analysis-data/urine.csv')

dat
plt.style.use(style='ggplot')

plt.rcParams['figure.figsize'] = (10, 6)
dat.plot(kind='scatter', x='urea', y='calc', alpha=0.5, color='mediumorchid', figsize = (12,9))

plt.title('Urea vs. Calc')

plt.xlabel("Urea")

plt.ylabel("Calc")

plt.show()
dat.plot(kind='scatter', x='ph', y='calc', alpha=0.5, color='chartreuse', figsize = (12,9))

plt.title('Ph vs. Calc')

plt.xlabel("Ph")

plt.ylabel("Calc")

plt.show()
print ("Skew is:", dat.urea.skew())

plt.hist(dat.urea, color='purple')

plt.show()
dat.plot(kind='scatter', x='osmo', y='calc', alpha=0.5, color='lightcoral', figsize = (12,9))

plt.title('Osmo vs. Calc')

plt.xlabel("Osmo")

plt.ylabel("Calc")

plt.show()
ax = sns.scatterplot(x="urea", y="calc", \

                     hue="r", legend="full", palette='RdPu', data=dat)
ax = sns.scatterplot(x="urea", y="calc", \

                     hue="r", legend="full", palette='BuGn', data=dat)
ax = sns.scatterplot(x="urea", y="calc", \

                     hue="r", legend="full", palette='GnBu', data=dat)
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(dat['urea'], dat['calc'], dat['osmo'], c='deeppink', s=60)

ax.view_init(30, 185)

plt.show()
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(dat['urea'], dat['calc'], dat['osmo'], c='darkolivegreen', s=60)

ax.view_init(30, 185)

plt.show()
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(dat['urea'], dat['calc'], dat['osmo'], c='orangered', s=60)

ax.view_init(30, 185)

plt.show()
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(dat['urea'], dat['calc'], dat['osmo'], c='midnightblue', s=60)

ax.view_init(30, 185)

plt.show()
cmap = sns.cubehelix_palette(as_cmap=True)



f, ax = plt.subplots(figsize=(20,10))

points = ax.scatter(dat['urea'],dat['calc'] , c=dat['r'],s=20, cmap='rainbow')

#plt.xticks(np.arange(0, 400,20))

#plt.axis('scaled')

f.colorbar(points)

#plt.legend(labels=Cover_Type_Dict.values(),loc='upper left')

plt.show()