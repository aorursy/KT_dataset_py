# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from scipy import stats

from itertools import permutations, combinations

import matplotlib as mpl

sns.set(color_codes=True)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
student_per = pd.read_csv('../input/StudentsPerformance.csv')
#Reformatting the columns to make them pandas sytanx friendly

cols = student_per.columns

old_cols = student_per.columns

new_cols = [i.replace(' ', '_') for i in student_per.columns]

col_dict = {}

for i in range(len(cols)):

    col_dict[old_cols[i]] = new_cols[i]

col_dict
student_per.rename(col_dict, axis=1, inplace=True)
#Creating a list containing the names of the columns that contain scores

math, reading, writing = student_per.math_score,student_per.reading_score,student_per.writing_score

scores = student_per.columns[-3:]

scores
#A simple built-in pandas method that returns the useful descriptie statistic for columns that contain numerical values

student_per.describe()
#Creating a boxplot to see the averages and deviations among scores. Averages don't seem too far off!

fig,ax = plt.subplots(1,1, figsize=(10,10))

sns.set(style="ticks", palette="pastel", font_scale=1.6)

sns.boxplot(x=['math','reading' ,'writing'],

    y=[math, reading,student_per.writing_score])

sns.despine(offset=10, trim=True)
#Getting the normal distribution of the scores, using random SAMPLING. This is not how the data looks raw.

sns.set(color_codes=True)

fig,axes = plt.subplots(1,3,figsize=(20,10))

#s = np.random.normal(mu, sigma, 2000)

colors = ['b','darkorange','g']

for test_score, ax, i in zip(scores,axes, range(3)):

    mu, sigma = student_per[test_score].mean(), student_per[test_score].std()

    s = np.random.normal(mu, sigma, 2000)

    ax.hist(s, color=colors[i])

    ax.set_title(test_score.replace('_',' ')+", mean: {}".format(round(mu,2)), size=16)

    ax.grid(True)
#plotting scores on a scatter

fig, axes = plt.subplots(1,3, figsize=(25,6))

#sns.scatterplot(reading, math);

ax1, ax2, ax3 = axes

ax1.scatter(writing, math,edgecolors='black')

ax2.scatter(reading, writing,edgecolors='black')

ax3.scatter(reading, math, edgecolors='black')

for ax in axes:

    ax.grid(True)

ax1.set_xlabel('Writing');

ax1.set_ylabel('Math');

ax2.set_xlabel('Reading');

ax2.set_ylabel('Writing');

ax3.set_xlabel('Reading');

ax3.set_ylabel('Math');



sns.set(color_codes=True)

parent_level_score = (student_per.groupby('parental_level_of_education')[['writing_score','math_score','reading_score']].mean()).sort_values(by='writing_score')

fig, ax = plt.subplots(1,1,figsize=(16,8))

width = .25

x = np.arange(6)

read = ax.bar(x-width, height=parent_level_score.reading_score, width=width, bottom=0)

write = ax.bar(x, height=parent_level_score.writing_score, width=width,)

mat = ax.bar(x+width, height=parent_level_score.math_score, width=width,)

ind = list(parent_level_score.index)

ind.insert(0,'a')

ax.set_xticklabels(ind, size=13);

ax.legend(['reading','writing','math'], loc='best', fontsize='large');

ax.set_title('Averages for Scores ',size=15);

# for bar, num in zip(read,x):

#     #print(bar.get_height())

#     print(num-width)

#     ax.text(num-width, y=round(bar.get_height())+2, s=round(bar.get_height()))
#Does completing the test preparation course boost actual scores?

scores = ['reading_score','writing_score','math_score']

for score in scores:

    print(student_per.groupby('test_preparation_course')[score].mean())

    print('for {}\n'.format(score))

test = student_per.groupby('test_preparation_course')[scores].mean()
mpl.rcParams["font.size"] = 15

fig, axes = plt.subplots(1,2, figsize=(20,15))

ax1, ax2 = axes

test_prep_gender = student_per.groupby('gender').test_preparation_course.value_counts(normalize=True)

female, male = test_prep_gender['female'], test_prep_gender['male']

ax1.pie(female, labels=['none','completed'], autopct='%1.1f%%', textprops={'size':15});

ax1.set_title('Female Test Prep', size=17)

ax2.set_title('Male Test Prep', size=17)

ax2.pie(male, labels=['none','completed'], autopct='%1.1f%%', colors=['r','g'],textprops={'size':15});
student_per.groupby('gender')[scores].mean()

student_per.groupby('lunch')[scores].mean()
(student_per.groupby(['test_preparation_course','lunch'])['reading_score','writing_score','math_score'].mean())

student_per.groupby(['lunch','gender'])[scores].mean()

plt.figure(figsize=(12,12))

(student_per.groupby('parental_level_of_education').lunch.value_counts(normalize=True)).plot(kind='bar');

student_per.groupby('race/ethnicity')[scores].mean()
#creating buckets to categorize the scores by letter grade

bucket = [0,60,70,80,90,101]

bucket_grade = ['F','D','C','B','A']
#Creating three additionals columns containing letter grades for the corresponding scores

student_per['math_grade'] = pd.cut(math,bins=bucket, labels=bucket_grade);

student_per['reading_grade'] = pd.cut(reading,bins=bucket, labels=bucket_grade);

student_per['writing_grade'] = pd.cut(writing,bins=bucket, labels=bucket_grade);

failed_all_exams = (student_per[(student_per.math_grade=='F')&(student_per.writing_grade=='F')&(student_per.reading_grade=='F')])

len(failed_all_exams)
failed_one_exam = (student_per[(student_per.math_grade=='F')|(student_per.writing_grade=='F')|(student_per.reading_grade=='F')])

len(failed_one_exam)

#failed_all_exams
aced_all_exams = (student_per[(student_per.math_grade=='A')&(student_per.writing_grade=='A')&(student_per.reading_grade=='A')])

len(aced_all_exams)
passed_one = (student_per[(student_per.math_score>=70)|(student_per.writing_score>=70)|(student_per.reading_score>=70)])

len(passed_one)
print(aced_all_exams.lunch.value_counts(normalize=True))

failed_all_exams.lunch.value_counts(normalize=True)
print(aced_all_exams.parental_level_of_education.value_counts(normalize=True))

failed_all_exams.parental_level_of_education.value_counts(normalize=True)
print(aced_all_exams.test_preparation_course.value_counts(normalize=True))

failed_all_exams.test_preparation_course.value_counts(normalize=True)
#Althogh there is some notable differences in the output here. Not knowing the groups for race/ethnicity makes his information hard to use.

# print(aced_all_exams['race/ethnicity'].value_counts(normalize=True))

# failed_all_exams['race/ethnicity'].value_counts(normalize=True)