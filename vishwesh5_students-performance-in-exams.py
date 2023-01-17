import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import probplot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Filename of the input CSV file

filename = "/kaggle/input/students-performance-in-exams/StudentsPerformance.csv"
# Read the data into a dataframe

data = pd.read_csv(filename)
# Check first 5 rows

data.head()
# Start off by having a look at the shape of the data

data.shape
# Check the datatypes

data.info()
# Describe the numerical attributes

data.describe()
# Boxplot for math score

sns.boxplot(data["math score"])

plt.show()
# 25% quantile

Q1 = np.quantile(data["math score"],0.25)

# 75% quantile

Q3 = np.quantile(data["math score"],0.75)

# Inter-quantile range

IQR = Q3-Q1

# Outliers on the lower end

data[data["math score"] < Q1-1.5*IQR]
# Boxplot for reading score

sns.boxplot(data["reading score"])

plt.show()
# 25% quantile

Q1 = np.quantile(data["reading score"],0.25)

# 75% quantile

Q3 = np.quantile(data["reading score"],0.75)

# Inter-quantile range

IQR = Q3-Q1

# Outliers on the lower end

data[data["reading score"] < Q1-1.5*IQR]
# Boxplot for writing score

sns.boxplot(data["writing score"])

plt.show()
# 25% quantile

Q1 = np.quantile(data["writing score"],0.25)

# 75% quantile

Q3 = np.quantile(data["writing score"],0.75)

# Inter-quantile range

IQR = Q3-Q1

# Outliers on the lower end

data[data["writing score"] < Q1-1.5*IQR]
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)

sns.distplot(data["math score"])

plt.subplot(1,3,2)

sns.distplot(data["reading score"])

plt.subplot(1,3,3)

sns.distplot(data["writing score"])

plt.show()
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)

probplot(data["math score"],dist="norm",plot=plt);

plt.subplot(1,3,2)

probplot(data["reading score"],dist="norm",plot=plt);

plt.subplot(1,3,3)

probplot(data["writing score"],dist="norm",plot=plt);

plt.show()
sns.countplot(data["gender"])

plt.show()
data["gender"].value_counts(normalize=True)
sns.countplot(data["race/ethnicity"])

plt.show()
data["race/ethnicity"].value_counts(normalize=True)
plt.figure(figsize=(10,5))

sns.countplot(data["parental level of education"])

plt.show()



data["parental level of education"].value_counts(normalize=True)
sns.countplot(data["lunch"])

plt.show()



data["lunch"].value_counts(normalize=True)
sns.countplot(data["test preparation course"])

plt.show()



data["test preparation course"].value_counts(normalize=True)
# parental level of education one hot encoding

data = pd.concat([data,

                 pd.get_dummies(data["parental level of education"],

                               prefix="parent_education")],

                axis=1)

# drop parental level of education

data.drop("parental level of education",

         axis=1,

         inplace=True)
# race/ethnicity one hot encoding

data = pd.concat([data,

                 pd.get_dummies(data["race/ethnicity"],

                               prefix="race")],

                axis=1)

# drop race/ethnicity

data.drop("race/ethnicity",

         axis=1,

         inplace=True)
data.head()
# Binary encoding for 2 level categorical variables

data["gender"] = data["gender"].apply(lambda x:0 if x=="male" else 1)
data["lunch"] = data["lunch"].apply(lambda x:0 if x=="standard" else 1)

data["test preparation course"] = data["test preparation course"].apply(lambda x: 0 if x=="none" else 1)
data.head()
sns.pairplot(data[["math score","reading score","writing score"]])

plt.show()
plt.figure(figsize=(15,3))

plt.subplot(1,3,1)

sns.scatterplot(x="math score",

             y="writing score",

             data = data,

             hue = "gender")

plt.subplot(1,3,2)

sns.scatterplot(x="reading score",

             y="writing score",

             data = data,

             hue = "gender")

plt.subplot(1,3,3)

sns.scatterplot(x="math score",

             y="reading score",

             data = data,

             hue = "gender")

plt.show()
plt.figure(figsize=(15,3))

plt.subplot(1,3,1)

sns.scatterplot(x="math score",

             y="writing score",

             data = data,

             hue = "lunch")

plt.subplot(1,3,2)

sns.scatterplot(x="reading score",

             y="writing score",

             data = data,

             hue = "lunch")

plt.subplot(1,3,3)

sns.scatterplot(x="math score",

             y="reading score",

             data = data,

             hue = "lunch")

plt.show()
plt.figure(figsize=(15,3))

plt.subplot(1,3,1)

sns.scatterplot(x="math score",

             y="writing score",

             data = data,

             hue = "test preparation course")

plt.subplot(1,3,2)

sns.scatterplot(x="reading score",

             y="writing score",

             data = data,

             hue = "test preparation course")

plt.subplot(1,3,3)

sns.scatterplot(x="math score",

             y="reading score",

             data = data,

             hue = "test preparation course")

plt.show()
# Race groups

race_groups = []

for col in data.columns:

    if col.startswith("race_group"):

        race_groups.append(col)
plt.figure(figsize=(15,18))

for i in range(len(race_groups)):

    plt.subplot(len(race_groups),3,i*3+1)

    sns.scatterplot(x="math score",

                 y="writing score",

                 data = data,

                 hue = race_groups[i])

    plt.subplot(len(race_groups),3,i*3+2)

    sns.scatterplot(x="reading score",

                 y="writing score",

                 data = data,

                 hue = race_groups[i])

    plt.subplot(len(race_groups),3,i*3+3)

    sns.scatterplot(x="math score",

                 y="reading score",

                 data = data,

                 hue = race_groups[i])

plt.show()
# Parent education groups

parent_edu_groups = []

for col in data.columns:

    if col.startswith("parent_education_"):

        parent_edu_groups.append(col)
plt.figure(figsize=(15,24))

for i in range(len(parent_edu_groups)):

    plt.subplot(len(parent_edu_groups),3,i*3+1)

    sns.scatterplot(x="math score",

                 y="writing score",

                 data = data,

                 hue = parent_edu_groups[i])

    plt.subplot(len(parent_edu_groups),3,i*3+2)

    sns.scatterplot(x="reading score",

                 y="writing score",

                 data = data,

                 hue = parent_edu_groups[i])

    plt.subplot(len(parent_edu_groups),3,i*3+3)

    sns.scatterplot(x="math score",

                 y="reading score",

                 data = data,

                 hue = parent_edu_groups[i])

plt.show()