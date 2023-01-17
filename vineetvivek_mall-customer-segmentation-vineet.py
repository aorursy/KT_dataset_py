import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.datasets import make_blobs

from sklearn.model_selection import train_test_split

from sklearn.mixture import GaussianMixture

from sklearn.manifold import TSNE



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.metrics import silhouette_score

from yellowbrick.cluster import SilhouetteVisualizer



import re

import warnings

warnings.filterwarnings("ignore")

import os
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
os.chdir('/kaggle/input/customer-segmentation-tutorial-in-python/')
os.listdir()
df=pd.read_csv("Mall_Customers.csv")
df.head()
df.describe()
df.shape

df.dtypes
df.rename({'CustomerID':'Customer_ID',

           'Annual Income (k$)':'Annual_Income',

           'Spending Score (1-100)':'Spending_Score'},

           axis=1,

           inplace=True)
df.columns
df.drop(columns={'Customer_ID'}, inplace=True)
df.shape

df.columns
df.Gender.value_counts()
df.Gender[df.Gender == 'Male'] = 1

df.Gender[df.Gender == 'Female'] = 0

# Male=1, Female=0

df.head()

df.describe()
df["Age_cat"] = pd.cut(

                       df['Age'],

                       bins = [0,35,50,80],

                       labels= ["y", "m", "s"]

                      )
df["Annual_Income_cat"] = pd.cut(

                               df['Annual_Income'],

                               bins = [0,40,80,150],

                               labels= ["l", "m", "h"]

                               )
df["Spending_Score_cat"] = pd.cut(

                               df['Spending_Score'],

                               bins = 3,

                               labels= ["Ls", "Ms", "Hs"]

                               )
df.sample(n=10)
columns = ['Gender', 'Age', 'Annual_Income', 'Spending_Score']

fig = plt.figure(figsize = (10,10))

for i in range(len(columns)):

    plt.subplot(2,2,i+1)

    sns.distplot(df[columns[i]])
fig = plt.figure(figsize = (10,8))

sns.barplot(x = 'Gender',

            y = 'Spending_Score',

            hue = 'Age_cat',       # Age-cat wise plots

            estimator = np.mean,

            ci = 68,

            data =df)

sns.boxplot(x = 'Age',                 

            y = 'Spending_Score',

            data = df

            )

sns.boxplot(x = 'Annual_Income',

            y = 'Age', 

            data = df

            )
sns.jointplot(df.Age, df.Spending_Score,kind = "kde")
sns.jointplot(df.Age, df.Annual_Income,kind="hex")
sns.barplot(x = 'Annual_Income',

            y = 'Spending_Score',

            estimator = np.mean,

            ci = 95,

            data =df

            )

df.columns
grouped = df.groupby(['Gender', 'Age_cat'])

df_wh = grouped['Spending_Score'].sum().unstack()

df_wh



sns.heatmap(df_wh)
grouped = df.groupby(['Gender', 'Age_cat'])

df_wh = grouped['Annual_Income'].sum().unstack()

df_wh



sns.heatmap(df_wh)
grouped = df.groupby(['Age_cat','Spending_Score_cat'])

df_wq = grouped['Annual_Income'].sum().unstack()

sns.heatmap(df_wq, cmap = plt.cm.Spectral)
sns.catplot(x = 'Spending_Score',

            y = 'Age', 

            row = 'Spending_Score_cat',

            col = 'Age_cat' ,

            kind = 'box',

            estimator = np.sum,

            data = df)
sns.relplot(x = 'Annual_Income',

            y = 'Spending_Score', 

            col = 'Age_cat' ,

            kind = 'line',

            estimator = np.sum,

            data = df)
df.dtypes

df.shape
y=df['Spending_Score'].values
num1=df.select_dtypes('int64').copy()
num1.shape

num1.head()
ss=StandardScaler()
ss.fit(num1)
X=ss.transform(num1)
X[:5,]
gm=GaussianMixture(n_components=3,

                   n_init=10,

                   max_iter=100)
gm.fit(X)
gm.means_
gm.converged_
gm.n_iter_
gm.predict(X)
gm.weights_
np.unique(gm.predict(X), return_counts = True)[1]/len(X)
gm.sample()
fig=plt.figure()
plt.scatter(X[:,0],X[:,1],c=gm.predict(X),s=2)
plt.scatter(gm.means_[:, 0], gm.means_[:, 1],

            marker='v',

            s=5,               # marker size

            linewidths=5,      # linewidth of marker edges

            color='red'

            )

plt.show()
densities=gm.score_samples(X)
densities
density_threshold=np.percentile(densities,4)
density_threshold
bic = []

aic = []
for i in range(8):

    gm = GaussianMixture(

                     n_components = i+1,

                     n_init = 10,

                     max_iter = 100)

    gm.fit(X)

    bic.append(gm.bic(X))

    aic.append(gm.aic(X))
fig = plt.figure()

plt.plot([1,2,3,4,5,6,7,8], aic)

plt.plot([1,2,3,4,5,6,7,8], bic)

plt.show()
tsne = TSNE(n_components = 2)

tsne_out = tsne.fit_transform(X)

plt.scatter(tsne_out[:, 0], tsne_out[:, 1],

            marker='x',

            s=50,              # marker size

            linewidths=5,      # linewidth of marker edges

            c=gm.predict(X)   # Colour as per gmm

            )
anomalies=X[densities<density_threshold]
anomalies
plt.scatter(X[:, 0], X[:, 1], c = gm.predict(X))
plt.scatter(anomalies[:, 0], anomalies[:, 1],

            marker='x',

            s=50,               # marker size

            linewidths=5,      # linewidth of marker edges

            color='red'

            )

plt.show()
unanomalies = X[densities >= density_threshold]
unanomalies.shape
df_anomalies = pd.DataFrame(anomalies, columns = ['w','x', 'y'])

df_anomalies['z'] = 'anomalous'   # Create a IIIrd constant column

df_normal = pd.DataFrame(unanomalies, columns = ['w','x','y'])

df_normal['z'] = 'unanomalous'
sns.distplot(df_anomalies['w'])

sns.distplot(df_normal['w'])
df = pd.concat([df_anomalies,df_normal])
sns.boxplot(x = df['z'], y = df['x'])

sns.boxplot(x = df['z'], y = df['w'])