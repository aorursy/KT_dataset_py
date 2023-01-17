# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")
m_scores = list(df["math score"])

r_scores = list(df["reading score"])
from matplotlib import pyplot as plt

plt.scatter(m_scores, r_scores, s = 0.8)
from sklearn.cluster import KMeans

X = list(zip(m_scores, r_scores))
model = KMeans(n_clusters=2, random_state=0).fit(X)
cluster = model.predict(X)
cluster[:10]
colors = {0: "red", 1: "blue"}
scatter_colors = list(map(lambda x: colors.get(x), cluster))
scatter_colors[:10]
plt.scatter(m_scores, r_scores, s = 0.8, c = scatter_colors, label="good student")

plt.ylabel("reading score")

plt.xlabel("math score")

plt.legend()
df.head()
X = df[["math score", "reading score", "writing score"]].values
model = KMeans(n_clusters=6, random_state=0).fit(X)
y = model.predict(X)
colors = {0: "red", 1: "blue", 2: "orange", 3: "yellow", 4: "black", 5: "green"}
m_colors = list(map(lambda x: colors.get(x), y))
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(list(df["math score"]), list(df["reading score"]), list(df["writing score"]), c=m_colors) 