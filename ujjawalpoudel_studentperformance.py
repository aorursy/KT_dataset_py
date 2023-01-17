import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt
# Read CSV file

df=pd.read_csv('../input/StudentsPerformance.csv')

df.head()
# Check is there any null or not

df.isnull().sum()
# Gives shape of given dataset

df.shape
# Statistics Description of given data

df.describe()
# Plotting test preparation scores against the maths,reading and writing scores to see which gender perfroms better

plt.figure(figsize=(18,8))

plt.subplot(1,4,1)

sns.barplot(x='test preparation course',y='math score',data=df,hue='gender',palette="Blues_d")

plt.title('Maths Scores')



plt.subplot(1, 4, 2)

sns.barplot(x='test preparation course',y='reading score',data=df,hue='gender',palette="Blues_d")

plt.title('Reading scores')



plt.subplot(1, 4, 3)

sns.barplot(x='test preparation course',y='writing score',data=df,hue='gender',palette="Blues_d")

plt.title('Writing scores')

plt.show()
#Plotting lunch against the maths,reading and writing scores to see which gender perfroms better



plt.figure(figsize=(18,8))

plt.subplot(1,4,1)

sns.barplot(x='lunch',y='math score',data=df,hue='gender',palette="Blues_d")

plt.title('Maths Scores')



plt.subplot(1, 4, 2)

sns.barplot(x='lunch',y='reading score',data=df,hue='gender',palette="Blues_d")

plt.title('Reading scores')



plt.subplot(1, 4, 3)

sns.barplot(x='lunch',y='writing score',data=df,hue='gender',palette="Blues_d")

plt.title('Writing scores')

plt.show()
#plotting a graph with parental level of education against count to see how important is parent's education

fig,ax=plt.subplots()

sns.countplot(x='parental level of education',data=df)

plt.tight_layout()

fig.autofmt_xdate()
# Calculate average value from 3 columns and consider this as pass marks

total=(df['math score'].values).mean()+(df['reading score'].values).mean()+(df['writing score'].values).mean()



#calculating the average

passMarks=total/3



print("Pass marks is {}".format(round(passMarks,2)))
#maths passing score

print("Numbers of pass students in math:- {}".format(len(df[df['math score']>=passMarks])))

print("Numbers of fail students in math:- {}".format(len(df[df['math score']<passMarks])))
#reading passing score

print("Numbers of pass students in reading:- {}".format(len(df[df['reading score']>=passMarks])))

print("Numbers of fail students in reading:- {}".format(len(df[df['reading score']<passMarks])))
#writing passing score

print("Numbers of pass students in writing:- {}".format(len(df[df['writing score']>=passMarks])))

print("Numbers of fail students in writing:- {}".format(len(df[df['writing score']<passMarks])))
# Overall passing score

print("Numbers of pass students in all categories:- {}".format(len(df[(df['math score']>=passMarks) & (df['writing score']>=passMarks) & (df['reading score']>=passMarks)])))

print("Numbers of fail students in all categories:- {}".format(len(df)-len(df[(df['math score']>=passMarks) & (df['writing score']>=passMarks) & (df['reading score']>=passMarks)])))
# new column of overAllPass is added for pass 'P' and fail 'F'

df['overAllPass']=np.where((df['math score']>=passMarks)&(df['writing score']>=passMarks)&(df['reading score']>=passMarks),'P','F')

df.head()
#Plotting a graph to see which group has more number of pass students

fig,ax=plt.subplots()

sns.countplot(x='race/ethnicity', data = df, hue='overAllPass', palette='Blues_d')

plt.tight_layout()

fig.autofmt_xdate()
# Top 10 students with their performance

df['percentage']=df.apply(lambda x : round((x['math score']+x['writing score']+x['reading score'])/3,2),axis=1)

dfPercentage=df.sort_values(by='percentage',ascending=False)

dfPercentage[:10]
# Top 10 students with their worst performance

dfPercentage[-10:][::-1]