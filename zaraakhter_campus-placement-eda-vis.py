#import necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#import data

data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data.head()
data.info()
#check data for nulls

data.isnull().sum()
#setting sl_no as index

data = data.set_index('sl_no')
data.describe()
#identifying unique values 

print(data['gender'].unique())

print(data['ssc_b'].unique())

print(data['hsc_s'].unique())

print(data['degree_t'].unique())

print(data['specialisation'].unique())
#changing data type to category for necessary columns

data['gender']=data['gender'].astype('category')

data['status']=data['status'].astype('category')

data['workex']=data['workex'].astype('category')

data['hsc_b']=data['hsc_b'].astype('category')

data['ssc_b']=data['ssc_b'].astype('category')

data['specialisation']=data['specialisation'].astype('category')

data['degree_t']=data['degree_t'].astype('category')

data['hsc_s']=data['hsc_s'].astype('category')
data.head(5)
data.dtypes
#Placement status

sns.set(style="darkgrid")

sns.catplot(x = 'status', data=data, kind='count',height=7,aspect=1.2, palette = 'Spectral_r')
#placement status pie chart

labels = ['Placed', 'Not Placed']

fig, ax = plt.subplots(figsize=(5,5))

ax.pie(data.status.value_counts(normalize=True), labels=labels, autopct='%2.2f%%', startangle=90, colors = ['salmon', 'darkturquoise'] )

ax.axis('equal')

plt.show()
#Male:Female

sns.set(style="darkgrid")

sns.catplot(x = 'gender', data=data, kind='count',height=7, aspect = 1.2, palette = 'Spectral_r')
labels = ['Male', 'Female']

fig, ax = plt.subplots(figsize=(5,5))

ax.pie(data.gender.value_counts(normalize=True), labels=labels, autopct='%2.2f%%', startangle=90, colors = ['salmon', 'darkturquoise'] )

ax.axis('equal')

plt.show()
#placements based on gender

status_gender = data.groupby('status')

status_gender.gender.value_counts()
labels = ['Placed', 'Not Placed']

M = [100, 39]

F = [48, 28]

x = np.arange(len(labels))

width = 0.35

fig, ax = plt.subplots(figsize = (8,8))

rects1 = ax.bar(x - width/2, M, width, label='Male')

rects2 = ax.bar(x + width/2, F, width, label='Female')

ax.set_ylabel('Status Count')

ax.set_title('Status of placement by gender')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()
#correlation matrix

plt.figure(figsize=(15,8))

sns.heatmap(data.corr(), annot = True)
#correlation of degree percentages to placement

data['status_cat'] = list(map(lambda x : 1 if x == 'Placed' else 0, data['status']))



print('placement status against Secondary Education: ', data['status_cat'].corr(data['ssc_p']))

print('placement status against Higher Secondary Education: ',data['status_cat'].corr(data['hsc_p']))

print('placement status against Degree: ',data['status_cat'].corr(data['degree_p']))

print('placement status against Employability Test: ',data['status_cat'].corr(data['etest_p']))

print('placement status against Post Grad Degree: ',data['status_cat'].corr(data['mba_p']))

#average percentages

values = [(data['ssc_p'].mean()) , (data['hsc_p'].mean()) , (data['mba_p'].mean()) , (data['degree_p'].mean())]

fig, ax = plt.subplots(figsize = (15,10))

names = ['ssc_p','hsc_p','mba_p','degree_p']

ax.set_ylabel('Average percentages')

ax.bar(names,values)

plt.show()
#work experience based on gender

work_gender = data.groupby('workex')

work_gender.gender.value_counts()
labels = ['Existing Work Exp.', 'No Work Exp.']

M = [52, 87]

F = [22, 54]

x = np.arange(len(labels))

width = 0.35

fig, ax = plt.subplots(figsize = (8,8))

rects1 = ax.bar(x - width/2, M, width, label='Male')

rects2 = ax.bar(x + width/2, F, width, label='Female')

ax.set_title('Work Experience by gender')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()