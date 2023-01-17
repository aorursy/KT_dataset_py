import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer['DESCR'])
df =  pd.DataFrame(data=cancer.data, columns = cancer.feature_names)
df.head(2)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df)
scaled_data = scaler.transform(df)

type(scaled_data)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

x_pca = pca.fit_transform(scaled_data)
x_pca.shape
plt.figure(figsize=(10,10))

plt.scatter(x=x_pca[:,0],y=x_pca[:,1],c=cancer.target,cmap='coolwarm')

plt.xlabel('First Component')

plt.ylabel('Second Component')
df_pca = pd.DataFrame(pca.components_,columns=cancer.feature_names)
df_pca.head()
plt.figure(figsize=(15,5))

sns.heatmap(df_pca,cmap='plasma')