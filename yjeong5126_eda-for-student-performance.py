# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats
# Importing dataset

df = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
df.head()
df.shape
# Checking Missing Values

df.isnull().sum()
# Greating Grade Columns

scores = [('math score', 'math grade'), ('reading score', 'reading grade'), ('writing score', 'writing grade')]



for score_col, grade_col in scores:

    df[grade_col] = 0

    df.loc[df[score_col]>=90, grade_col] ='A'

    df.loc[(df[score_col]<90)&(df[score_col]>=80), grade_col] = 'B'

    df.loc[(df[score_col]<80)&(df[score_col]>=70), grade_col] = 'C'

    df.loc[(df[score_col]<70)&(df[score_col]>=60), grade_col] = 'D'

    df.loc[df[score_col]<60, grade_col] = 'E'
grade_order = ['E','D','C','B','A']

two_palette=['#bf00ff', '#26d9d9']
# Scores by Gender

male_math = df[df['gender']=='male']['math score']

female_math = df[df['gender']=='female']['math score']



male_reading = df[df['gender']=='male']['reading score']

female_reading = df[df['gender']=='female']['reading score']



male_writing = df[df['gender']=='male']['writing score']

female_writing = df[df['gender']=='female']['writing score']
# Style

sns.set_style("whitegrid")
# Gender Distribution

fig, ax = plt.subplots(1,2, figsize=(14,7))



df['gender'].value_counts().plot.pie(explode=[0,0.1], colors=['m','c'], autopct='%1.1f%%', ax=ax[0], shadow=True)

sns.countplot('gender', data=df, palette=two_palette, ax=ax[1])



ax[0].set_title('Gender')

ax[0].set_ylabel('')

ax[1].set_title('Gender')

ax[1].set_ylabel('Counts')

ax[1].set_xlabel('')
fig, ax = plt.subplots(1,3, figsize=(18,5))



sns.distplot(male_math, hist=False, color='c', label='Male', ax=ax[0])

sns.distplot(female_math, hist=False, color='m', label='Female', ax=ax[0])



sns.distplot(male_reading, hist=False, color='c', label='Male', ax=ax[1])

sns.distplot(female_reading, hist=False, color='m', label='Female', ax=ax[1])



sns.distplot(male_writing, hist=False, color='c', label='Male', ax=ax[2])

sns.distplot(female_writing, hist=False, color='m', label='Female', ax=ax[2])



ax[0].set_title('Math Score Distribution by Gender')

ax[0].set_xlabel('Math Score')

ax[0].set_ylabel('Density')



ax[1].set_title('Reading Score Distribution by Gender')

ax[1].set_xlabel('Reading Score')

ax[1].set_ylabel('Density')



ax[2].set_title('Writing Score Distribution by Gender')

ax[2].set_xlabel('Writing Score')

ax[2].set_ylabel('Density')
fig, ax = plt.subplots(1,3, figsize=(18,5))



sns.countplot('math grade', data=df,hue='gender',palette=two_palette, order=grade_order, ax=ax[0])

sns.countplot('reading grade', data=df,hue='gender',palette=two_palette, order=grade_order, ax=ax[1])

sns.countplot('writing grade', data=df,hue='gender',palette=two_palette, order=grade_order, ax=ax[2])



ax[0].set_title('Math Grade Distribution by Gender')

ax[0].set_xlabel('Math Grade')

ax[0].set_ylabel('Counts')



ax[1].set_title('Reading Grade Distribution by Gender')

ax[1].set_xlabel('Reading Grade')

ax[1].set_ylabel('Counts')



ax[2].set_title('Writing Grade Distribution by Gender')

ax[2].set_xlabel('Writing Grade')

ax[2].set_ylabel('Counts')
group_order = ['group A', 'group B', 'group C', 'group D', 'group E']

five_palette=['#ff0000', '#0000ff', '#3d8f52', '#d9d926', '#bf00ff']
# Scores by Race

A_math = df[df['race/ethnicity']=='group A']['math score']

B_math = df[df['race/ethnicity']=='group B']['math score']

C_math = df[df['race/ethnicity']=='group C']['math score']

D_math = df[df['race/ethnicity']=='group D']['math score']

E_math = df[df['race/ethnicity']=='group E']['math score']



A_reading = df[df['race/ethnicity']=='group A']['reading score']

B_reading = df[df['race/ethnicity']=='group B']['reading score']

C_reading = df[df['race/ethnicity']=='group C']['reading score']

D_reading = df[df['race/ethnicity']=='group D']['reading score']

E_reading = df[df['race/ethnicity']=='group E']['reading score']



A_writing = df[df['race/ethnicity']=='group A']['writing score']

B_writing = df[df['race/ethnicity']=='group B']['writing score']

C_writing = df[df['race/ethnicity']=='group C']['writing score']

D_writing = df[df['race/ethnicity']=='group D']['writing score']

E_writing = df[df['race/ethnicity']=='group E']['writing score']
# Race Distribution

fig, ax = plt.subplots(1,2, figsize=(14,7))



df['race/ethnicity'].value_counts().plot.pie(colors=['g','y','b','m','r'], autopct='%1.1f%%', ax=ax[0], shadow=True)

sns.countplot('race/ethnicity', data=df, order=group_order, palette=five_palette, ax=ax[1])



ax[0].set_title('Race')

ax[0].set_ylabel('')

ax[1].set_title('Race')

ax[1].set_ylabel('Counts')

ax[1].set_xlabel('')
figure,ax = plt.subplots(1,3,figsize=(20,7))



sns.distplot(A_math, hist=False, color='r', label='group A', ax=ax[0] )

sns.distplot(B_math, hist=False, color='b', label='group B', ax=ax[0] )

sns.distplot(C_math, hist=False, color='g', label='group C', ax=ax[0] )

sns.distplot(D_math, hist=False, color='y', label='group D', ax=ax[0] )

sns.distplot(E_math, hist=False, color='m', label='group E', ax=ax[0] )



ax[0].set_title('Math Score Distribution by Race')

ax[0].set_xlabel('Math Score')

ax[0].set_ylabel('Density')



sns.distplot(A_reading, hist=False, color='r', label='group A', ax=ax[1] )

sns.distplot(B_reading, hist=False, color='b', label='group B', ax=ax[1] )

sns.distplot(C_reading, hist=False, color='g', label='group C', ax=ax[1] )

sns.distplot(D_reading, hist=False, color='y', label='group D', ax=ax[1] )

sns.distplot(E_reading, hist=False, color='m', label='group E', ax=ax[1] )



ax[1].set_title('Reading Score Distribution by Race')

ax[1].set_xlabel('Reading Score')

ax[1].set_ylabel('Density')



sns.distplot(A_writing, hist=False, color='r', label='group A', ax=ax[2] )

sns.distplot(B_writing, hist=False, color='b', label='group B', ax=ax[2] )

sns.distplot(C_writing, hist=False, color='g', label='group C', ax=ax[2] )

sns.distplot(D_writing, hist=False, color='y', label='group D', ax=ax[2] )

sns.distplot(E_writing, hist=False, color='m', label='group E', ax=ax[2] )



ax[2].set_title('Writing Score Distribution by Race')

ax[2].set_xlabel('Writing Score')

ax[2].set_ylabel('Density')
figure,ax = plt.subplots(1,3,figsize=(20,7))



sns.countplot('math grade', data=df, hue='race/ethnicity', order=grade_order, hue_order=group_order, palette=five_palette, ax=ax[0])

sns.countplot('reading grade', data=df, hue='race/ethnicity', order=grade_order, hue_order=group_order, palette=five_palette, ax=ax[1])

sns.countplot('writing grade', data=df, hue='race/ethnicity', order=grade_order, hue_order=group_order, palette=five_palette, ax=ax[2])



ax[0].set_title('Math Grade Distribution by Race')

ax[0].set_xlabel('Math Grade')

ax[0].set_ylabel('Counts')



ax[1].set_title('Reading Grade Distribution by Race')

ax[1].set_xlabel('Reading Grade')

ax[1].set_ylabel('Counts')



ax[2].set_title('Writing Grade Distribution by Race')

ax[2].set_xlabel('Writing Grade')

ax[2].set_ylabel('Counts')
six_palette = ['#ff0000', '#0000ff', '#3d8f52', '#d9d926', '#bf00ff', '#f57e0f']

school_order=['high school', 'some high school', 'bachelor\'s degree', 'some college', 'master\'s degree', 'associate\'s degree']
# Scores by Parents Education Level

bachelor_math = df[df['parental level of education']=='bachelor\'s degree']['math score']

somecollege_math = df[df['parental level of education']=='some college']['math score']

master_math = df[df['parental level of education']=='master\'s degree']['math score']

associate_math = df[df['parental level of education']=='associate\'s degree']['math score']

high_math = df[df['parental level of education']=='high school']['math score']

somehigh_math = df[df['parental level of education']=='some high school']['math score']



bachelor_reading = df[df['parental level of education']=='bachelor\'s degree']['reading score']

somecollege_reading = df[df['parental level of education']=='some college']['reading score']

master_reading = df[df['parental level of education']=='master\'s degree']['reading score']

associate_reading = df[df['parental level of education']=='associate\'s degree']['reading score']

high_reading = df[df['parental level of education']=='high school']['reading score']

somehigh_reading = df[df['parental level of education']=='some high school']['reading score']



bachelor_writing = df[df['parental level of education']=='bachelor\'s degree']['writing score']

somecollege_writing = df[df['parental level of education']=='some college']['writing score']

master_writing = df[df['parental level of education']=='master\'s degree']['writing score']

associate_writing = df[df['parental level of education']=='associate\'s degree']['writing score']

high_writing = df[df['parental level of education']=='high school']['writing score']

somehigh_writing = df[df['parental level of education']=='some high school']['writing score']
# Parent Educational Level Distribution

fig, ax = plt.subplots(1,2, figsize=(14,7))



df['parental level of education'].value_counts().plot.pie(colors=['y','orange','r','b','g','m'], autopct='%1.1f%%', ax=ax[0], shadow=True)

sns.countplot('parental level of education', data=df, order=school_order, palette=six_palette, ax=ax[1])



ax[0].set_title('Parent\'s Education Level')

ax[0].set_ylabel('')

ax[1].set_title('Parent\'s Education Level')

ax[1].set_ylabel('Counts')

ax[1].set_xlabel('')

ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)
figure,ax = plt.subplots(1,3,figsize=(20,7))



sns.distplot(bachelor_math, hist=False, color='g', label='bachelor\'s degree', ax=ax[0] )

sns.distplot(somecollege_math, hist=False, color='y', label='some college', ax=ax[0] )

sns.distplot(master_math, hist=False, color='m', label='master\'s degree', ax=ax[0] )

sns.distplot(associate_math, hist=False, color='orange', label='associate\'s degree', ax=ax[0] )

sns.distplot(high_math, hist=False, color='r', label='high school', ax=ax[0] )

sns.distplot(somehigh_math, hist=False, color='b', label='some high school', ax=ax[0] )



ax[0].set_title('Math Score Distribution by Parent\'s Education Level')

ax[0].set_xlabel('Math Score')

ax[0].set_ylabel('Density')



sns.distplot(bachelor_reading, hist=False, color='g', label='bachelor\'s degree', ax=ax[1] )

sns.distplot(somecollege_reading, hist=False, color='y', label='some college', ax=ax[1] )

sns.distplot(master_reading, hist=False, color='m', label='master\'s degree', ax=ax[1] )

sns.distplot(associate_reading, hist=False, color='orange', label='associate\'s degree', ax=ax[1] )

sns.distplot(high_reading, hist=False, color='r', label='high school', ax=ax[1] )

sns.distplot(somehigh_reading, hist=False, color='b', label='some high school', ax=ax[1] )



ax[1].set_title('Reading Score Distribution by Parent\'s Education Level')

ax[1].set_xlabel('Reading Score')

ax[1].set_ylabel('Density')



sns.distplot(bachelor_writing, hist=False, color='g', label='bachelor\'s degree', ax=ax[2] )

sns.distplot(somecollege_writing, hist=False, color='y', label='some college', ax=ax[2] )

sns.distplot(master_writing, hist=False, color='m', label='master\'s degree', ax=ax[2] )

sns.distplot(associate_writing, hist=False, color='orange', label='associate\'s degree', ax=ax[2] )

sns.distplot(high_writing, hist=False, color='r', label='high school', ax=ax[2] )

sns.distplot(somehigh_writing, hist=False, color='b', label='some high school', ax=ax[2] )



ax[2].set_title('Writing Score Distribution by Parent\'s Education Level')

ax[2].set_xlabel('Writing Score')

ax[2].set_ylabel('Density')
figure,ax = plt.subplots(1,3,figsize=(20,7))



sns.countplot('math grade', data=df, hue='parental level of education', order=grade_order, hue_order=school_order, palette=six_palette, ax=ax[0])

sns.countplot('reading grade', data=df, hue='parental level of education', order=grade_order, hue_order=school_order, palette=six_palette, ax=ax[1])

sns.countplot('writing grade', data=df, hue='parental level of education', order=grade_order, hue_order=school_order, palette=six_palette, ax=ax[2])



ax[0].set_title('Math Grade Distribution by Parents Education Level')

ax[0].set_xlabel('Math Grade')

ax[0].set_ylabel('Counts')



ax[1].set_title('Reading Grade Distribution by Parents Education Level')

ax[1].set_xlabel('Reading Grade')

ax[1].set_ylabel('Counts')



ax[2].set_title('Writing Grade Distribution by Parents Education Level')

ax[2].set_xlabel('Writing Grade')

ax[2].set_ylabel('Counts')
# Scores by Lunch

standard_math = df[df['lunch']=='standard']['math score']

free_math = df[df['lunch']=='free/reduced']['math score']



standard_reading = df[df['lunch']=='standard']['reading score']

free_reading = df[df['lunch']=='free/reduced']['reading score']



standard_writing = df[df['lunch']=='standard']['writing score']

free_writing = df[df['lunch']=='free/reduced']['writing score']
fig, ax = plt.subplots(1,2, figsize=(14,7))



df['lunch'].value_counts().plot.pie(explode=[0,0.1], colors=['m','c'], autopct='%1.1f%%', ax=ax[0], shadow=True)

sns.countplot('lunch', data=df, palette=two_palette, ax=ax[1])



ax[0].set_title('Lunch')

ax[0].set_ylabel('')

ax[1].set_title('Lunch')

ax[1].set_ylabel('Counts')

ax[1].set_xlabel('')
fig, ax = plt.subplots(1,3, figsize=(20,7))



sns.distplot(standard_math, hist=False, color='m', label='standard', ax=ax[0])

sns.distplot(free_math, hist=False, color='c', label='free/reduced', ax=ax[0])



sns.distplot(standard_reading, hist=False, color='m', label='standard', ax=ax[1])

sns.distplot(free_reading, hist=False, color='c', label='free/reduced', ax=ax[1])



sns.distplot(standard_writing, hist=False, color='m', label='standard', ax=ax[2])

sns.distplot(free_writing, hist=False, color='c', label='free/reduced', ax=ax[2])



ax[0].set_title('Math Score Distribution by Lunch')

ax[0].set_xlabel('Math Score')

ax[0].set_ylabel('Density')



ax[1].set_title('Reading Score Distribution by Lunch')

ax[1].set_xlabel('Reading Score')

ax[1].set_ylabel('Density')



ax[2].set_title('Writing Score Distribution by Lunch')

ax[2].set_xlabel('Writing Score')

ax[2].set_ylabel('Density')
fig, ax = plt.subplots(1,3, figsize=(20,7))



sns.countplot('math grade', data=df, hue='lunch', palette=two_palette, order=grade_order, ax=ax[0])

sns.countplot('reading grade', data=df, hue='lunch', palette=two_palette, order=grade_order, ax=ax[1])

sns.countplot('writing grade', data=df, hue='lunch', palette=two_palette, order=grade_order, ax=ax[2])



ax[0].set_title('Math Grade Distribution by Lunch')

ax[0].set_xlabel('Math Grade')

ax[0].set_ylabel('Counts')



ax[1].set_title('Reading Grade Distribution by Lunch')

ax[1].set_xlabel('Reading Grade')

ax[1].set_ylabel('Counts')



ax[2].set_title('Writing Grade Distribution by Lunch')

ax[2].set_xlabel('Writing Grade')

ax[2].set_ylabel('Counts')
# Score by Test Preparation

none_math = df[df['test preparation course']=='none']['math score']

completed_math = df[df['test preparation course']=='completed']['math score']



none_reading = df[df['test preparation course']=='none']['reading score']

completed_reading = df[df['test preparation course']=='completed']['reading score']



none_writing = df[df['test preparation course']=='none']['writing score']

completed_writing = df[df['test preparation course']=='completed']['writing score']
fig, ax = plt.subplots(1,2, figsize=(14,7))



df['test preparation course'].value_counts().plot.pie(explode=[0,0.1], colors=['m','c'], autopct='%1.1f%%', ax=ax[0], shadow=True)

sns.countplot('test preparation course', data=df, palette=two_palette, ax=ax[1])



ax[0].set_title('Test Preparation')

ax[0].set_ylabel('')

ax[1].set_title('Test Preparation')

ax[1].set_ylabel('Counts')

ax[1].set_xlabel('')
fig, ax = plt.subplots(1,3, figsize=(20,7))



sns.distplot(none_math, hist=False, color='m', label='none', ax=ax[0])

sns.distplot(completed_math, hist=False, color='c', label='completed', ax=ax[0])



sns.distplot(none_reading, hist=False, color='m', label='none', ax=ax[1])

sns.distplot(completed_reading, hist=False, color='c', label='completed', ax=ax[1])



sns.distplot(none_writing, hist=False, color='m', label='none', ax=ax[2])

sns.distplot(completed_writing, hist=False, color='c', label='completed', ax=ax[2])



ax[0].set_title('Math Score Distribution by Test Preparation')

ax[0].set_xlabel('Math Score')

ax[0].set_ylabel('Density')



ax[1].set_title('Reading Score Distribution by Test Preparation')

ax[1].set_xlabel('Reading Score')

ax[1].set_ylabel('Density')



ax[2].set_title('Writing Score Distribution by Test Preparation')

ax[2].set_xlabel('Writing Score')

ax[2].set_ylabel('Density')
fig, ax = plt.subplots(1,3, figsize=(20,7))



sns.countplot('math grade', data=df,hue='test preparation course',palette=two_palette, order=grade_order, ax=ax[0])

sns.countplot('reading grade', data=df,hue='test preparation course',palette=two_palette, order=grade_order, ax=ax[1])

sns.countplot('writing grade', data=df,hue='test preparation course',palette=two_palette, order=grade_order, ax=ax[2])



ax[0].set_title('Math Grade Distribution by Test Preparation')

ax[0].set_xlabel('Math Grade')

ax[0].set_ylabel('Counts')



ax[1].set_title('Reading Grade Distribution by Test Preparation')

ax[1].set_xlabel('Reading Grade')

ax[1].set_ylabel('Counts')



ax[2].set_title('Writing Grade Distribution by Test Preparation')

ax[2].set_xlabel('Writing Grade')

ax[2].set_ylabel('Counts')