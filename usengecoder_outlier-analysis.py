import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns #visualization tools



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
ad=pd.read_csv("../input/advertising/advertising.csv")

df = ad.copy()

df = df.select_dtypes(include = ['float64', 'int64']) #we chose only numeric variables

df.head()
df_table = df["Area Income"].copy()
sns.boxplot(x = df_table)
Q1 = df_table.quantile(0.25)

Q3 = df_table.quantile(0.75)

IQR = Q3 - Q1



lower_bound = Q1- 1.5*IQR

upper_bound = Q3 + 1.5*IQR

print("lower bound is " + str(lower_bound))

print("upper bound is " + str(upper_bound))

print(Q1)

print(Q3)
(df_table < (lower_bound)) | (df_table > (upper_bound))
outliers_vector = (df_table < (lower_bound)) | (df_table > (upper_bound) )

outliers_vector
outliers = df_table[outliers_vector]
outliers.index #we obtained indexes of outlier
df_table.shape
clean_df_table = df_table[~((df_table<(lower_bound)) | (df_table > (upper_bound)))] 

# We only hold data within boundary conditions
clean_df_table.shape
df_table = df["Area Income"].copy()
sns.boxplot(x= df_table)
df_table.mean()
df_table.describe()
df_table[outliers_vector] = df_table.mean()
df_table[outliers_vector].head()
df_table.describe()
df_table = df["Area Income"].copy()

print(df_table.min())

print(df_table.max())
table_min = df_table.min()

table_max = df_table.max()

for e in range(len(df_table)):

    if df_table.iloc[e] < lower_bound:

        df_table.iloc[e] = lower_bound

    

    elif df_table.iloc[e] > upper_bound:

        df_table.iloc[e] = upper_bound

        
df_table.min()
df_table.max()
outliers_lower_vector = (df_table < (lower_bound))
outliers_upper_vector = (df_table > (upper_bound))
df_table[outliers_lower_vector] = lower_bound

df_table[outliers_upper_vector] = upper_bound
df_table.max()
df_table.min()