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





dat = pd.read_csv('../input/sparrow-measurements/Sparrows.csv')

dat
plt.style.use(style='ggplot')

plt.rcParams['figure.figsize'] = (10, 6)
dat.plot(kind='scatter', x='Weight', y='WingLength', alpha=0.5, color='mediumorchid', figsize = (12,9))

plt.title('Weight vs. Wing Length')

plt.xlabel("Weight")

plt.ylabel("Wing Length")

plt.show()
dat.plot(kind='scatter', x='Weight', y='WingLength', alpha=0.5, color='chartreuse', figsize = (12,9))

plt.title('Weight vs. Wing Length')

plt.xlabel("Weight")

plt.ylabel("Wing Length")

plt.show()
print ("Skew is:", dat.Weight.skew())

plt.hist(dat.Weight, color='purple')

plt.show()
print ("Skew is:", dat.WingLength.skew())

plt.hist(dat.WingLength, color='purple')

plt.show()
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(dat['Weight'], dat['WingLength'], c='deeppink', s=60)

ax.view_init(30, 185)

plt.show()
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(dat['Weight'], dat['WingLength'], c='darkolivegreen', s=60)

ax.view_init(30, 185)

plt.show()
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(dat['Weight'], dat['WingLength'], c='orangered', s=60)

ax.view_init(30, 185)

plt.show()
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(dat['Weight'], dat['WingLength'], c='midnightblue', s=60)

ax.view_init(30, 185)

plt.show()