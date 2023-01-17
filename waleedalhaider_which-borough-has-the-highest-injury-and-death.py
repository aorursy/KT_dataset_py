import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/database.csv')
print (df.head(n=3))
df.columns
a = df['BOROUGH'].value_counts()

print (a)
df['COUNTER']=1

group1 = df.groupby(['BOROUGH','PERSONS INJURED'])['COUNTER'].sum()

print (group1)
group1.plot.barh(fontsize=7, color='yellow', width=1, figsize=(8,10))
df['COUNTER']=1

group2 = df.groupby(['BOROUGH','PERSONS KILLED'])['COUNTER'].sum()

print (group2)
group2.plot.barh(fontsize=6, color='yellow', width=1, figsize=(8,10))
df['COUNTER']=1

group3 = df.groupby(['BOROUGH','PEDESTRIANS INJURED'])['COUNTER'].sum()

print (group3)
group3.plot.barh(fontsize=6, color='yellow', width=1, figsize=(8,10))
df['COUNTER']=1

group4 = df.groupby(['BOROUGH','PEDESTRIANS KILLED'])['COUNTER'].sum()

print (group4)
group4.plot.barh(fontsize=6,color='yellow', width=1, figsize=(8,10))
df['COUNTER']=1

group5 = df.groupby(['BOROUGH','CYCLISTS INJURED'])['COUNTER'].sum()

print (group5)
group5.plot.barh(fontsize=6, color='yellow', width=1, figsize=(8,10))
df['COUNTER']=1

group6 = df.groupby(['BOROUGH','CYCLISTS KILLED'])['COUNTER'].sum()

print (group6)
group6.plot.barh(fontsize=6, color='yellow', width=1, figsize=(8,10))
df['COUNTER']=1

group7 = df.groupby(['BOROUGH','MOTORISTS INJURED'])['COUNTER'].sum()

print (group7)
group7.plot.barh(fontsize=6, color='yellow', width=1, figsize=(8,10))
df['COUNTER']=1

group8 = df.groupby(['BOROUGH','MOTORISTS KILLED'])['COUNTER'].sum()

print (group8)
group8.plot.barh(fontsize=6, color='yellow', width=1, figsize=(8,10))