import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import os

import math
df = pd.read_csv('/kaggle/input/iris/Iris.csv')
df.head()
df_dtype = pd.DataFrame(columns=['Column','Data Type'])



for i,col in enumerate(df.columns):

    temp=[col,str(df[col].dtype)]

    df_dtype.loc[i,:]=temp

df_dtype
print("DataFrame has "+str(df.shape[0])+" rows and "+str(df.shape[1])+" columns")
print("Dataset has "+str(len(df['Species'].unique())) +" unique species")
print("Dataset has "+str(sum(df.isnull().sum())) +" missing values")
df['SepalAreaCm2'] = df['SepalLengthCm']*df['SepalWidthCm']

df['PetalAreaCm2'] = df['PetalLengthCm']*df['PetalWidthCm']
df.head()
df['SepalLengthCm'].hist()
df['SepalWidthCm'].hist()
df['PetalLengthCm'].hist()
df['PetalWidthCm'].hist()
df['SepalAreaCm2'].hist()
df['PetalAreaCm2'].hist()
df['Species'].unique()
df_Irissetosa = df[df['Species']=='Iris-setosa']

df_versicolor = df[df['Species']=='Iris-versicolor']

df_Irisvirginica = df[df['Species']=='Iris-virginica']
g = sns.FacetGrid(df, col="Species", margin_titles=True)

g.map(plt.hist, "SepalAreaCm2", color="steelblue")
g = sns.FacetGrid(df, col="Species", margin_titles=True)

g.map(plt.hist, "PetalAreaCm2", color="steelblue")
g = sns.FacetGrid(df, col="Species", margin_titles=True)

g.map(plt.hist, "SepalLengthCm", color="steelblue")
g = sns.FacetGrid(df, col="Species", margin_titles=True)

g.map(plt.hist, "SepalWidthCm", color="steelblue")
g = sns.FacetGrid(df, col="Species", margin_titles=True)

g.map(plt.hist, "PetalLengthCm", color="steelblue")
g = sns.FacetGrid(df, col="Species", margin_titles=True)

g.map(plt.hist, "PetalWidthCm", color="steelblue")