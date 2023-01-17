import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





df = pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")
df.head()
df.tail()
df.corr()
fig_dims = (15, 10)

fig, ax = plt.subplots(figsize=fig_dims)

sns.heatmap(data=df.corr(),annot=True).set_title('Correlation between Scores')
fig_dims = (15, 10)

fig, ax = plt.subplots(figsize=fig_dims)

sns.boxplot(data=df)
df['total score']=df['math score']+df['reading score']+df['writing score'] #Calculating total score
fig_dims = (15, 10)

fig, ax = plt.subplots(figsize=fig_dims)

sns.boxplot(x=df['gender'],y=df['total score']).set_title('Male vs Female Score')
fig_dims = (15, 10)

fig, ax = plt.subplots(figsize=fig_dims)

sns.boxplot(x=df['gender'],y=df['math score']).set_title('Male vs Female Score')
fig_dims = (15, 10)

fig, ax = plt.subplots(figsize=fig_dims)

sns.boxplot(x=df['race/ethnicity'],y=df['total score']).set_title('Race-wise Score')
fig_dims = (15, 10)

fig, ax = plt.subplots(figsize=fig_dims)

sns.boxplot(x=df['parental level of education'],y=df['total score'])
fig_dims = (15, 10)

fig, ax = plt.subplots(figsize=fig_dims)

sns.boxplot(x=df['lunch'],y=df['total score'])
fig_dims = (15, 10)

fig, ax = plt.subplots(figsize=fig_dims)

sns.boxplot(x=df['test preparation course'],y=df['total score'])