from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd
df = pd.read_csv("../input/dataset.csv")

df.head()
df.info()
df.describe()
df['Data_Value_Type'].value_counts()
df['Data_Value_Type'].value_counts(normalize=True)
df['Break_out'].value_counts()
df['Break_out'].value_counts(normalize=True)
df['Confidence_Limit_Low'].hist();
df['Confidence_Limit_High'].hist();
sns.boxplot(df['Confidence_Limit_Low'])
sns.boxplot(df['Confidence_Limit_High'])
plt.scatter(df['Confidence_Limit_High'], df['Confidence_Limit_Low']);