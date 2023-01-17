# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
students = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
students.columns
students.head()
students.shape
students.isnull().sum()
sns.distplot(students['math score'], bins=20, color='orange')

plt.grid()
sns.distplot(students['reading score'], bins=20, color='orange')

plt.grid()
sns.distplot(students['writing score'], bins=20, color='orange')

plt.grid()
sns.catplot(y="gender",  kind="count", height=6, aspect=2, data=students);
students['gender'].value_counts(normalize=True) * 100
sns.catplot(y="race/ethnicity",  kind="count", height=6, aspect=2, data=students);
students['race/ethnicity'].value_counts(normalize=True) * 100
sns.catplot(y="parental level of education",  kind="count", height=6, aspect=2, data=students);
students['parental level of education'].value_counts(normalize=True) * 100
sns.catplot(y="lunch",  kind="count", height=6, aspect=2, data=students);
students['lunch'].value_counts(normalize=True) * 100
sns.catplot(y="test preparation course",  kind="count", height=6, aspect=2, data=students);
students['test preparation course'].value_counts(normalize=True) * 100
sns.catplot(x="test preparation course", y="math score", data=students, height=6, aspect=2, kind='swarm', hue='gender');

plt.grid()
sns.catplot(x="test preparation course", y="math score", kind='violin', hue='gender', split='true', data=students, height=6, aspect=2);

plt.grid()
sns.catplot(x="test preparation course", y="reading score", data=students, height=6, aspect=2, kind='swarm', hue='gender');

plt.grid()
sns.catplot(x="test preparation course", y="reading score", kind='violin', hue='gender', split='true', data=students, height=6, aspect=2);

plt.grid()
sns.catplot(x="test preparation course", y="writing score", data=students, height=6, aspect=2, kind='swarm', hue='gender');

plt.grid()
sns.catplot(x="test preparation course", y="writing score", kind='violin', hue='gender', split='true', data=students, height=6, aspect=2);

plt.grid()
sns.catplot(y="parental level of education", x="math score", data=students, height=6, aspect=2, kind='swarm', hue='gender');

plt.grid()
sns.catplot(y="parental level of education", x="math score", kind='violin', hue='gender', split='true', data=students, height=6, aspect=2);

plt.grid()
# fig, ax = plt.subplots()

# fig.set_size_inches(11.7, 8.27)

sns.catplot(y="parental level of education", x="reading score", data=students, height=5, aspect=2, kind='swarm', hue='gender');

plt.grid()
sns.catplot(y="parental level of education", x="reading score", kind='violin', hue='gender', split='true', data=students, height=5, aspect=2);

plt.grid()
sns.catplot(y="parental level of education", x="writing score", data=students, kind='swarm', height=6, aspect=2, hue='gender');

plt.grid()
sns.catplot(y="parental level of education", x="writing score", kind='violin', hue='gender', split='true', data=students, height=6, aspect=2);

plt.grid()
sns.pairplot(students, hue="gender", palette="Set2", diag_kind="kde", height=5)