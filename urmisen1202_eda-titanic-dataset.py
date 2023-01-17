# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Data visualisation & images

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
#Load the train and test data from the dataset

df_train=pd.read_csv('../input/titanic/train.csv')

df_test=pd.read_csv('../input/titanic/test.csv')
# Join all data into one file

ntrain = df_train.shape[0]

ntest = df_test.shape[0]



# Creating y_train variable; we'll need this when modelling, but not before

y_train = df_train['Survived'].values



# Saving the passenger ID's ready for our submission file at the very end

passId = df_test['PassengerId']



# Create a new all-encompassing dataset

data = pd.concat((df_train, df_test))



# Printing overall data shape

print("data size is: {}".format(data.shape))
df_train.info()
df_test.info()
df_train.head()
df_test.head()
# Returning descriptive statistics of the train dataset

df_train.describe(include = 'all')
# Initiate correlation matrix

corr = df_train.corr()  # Pandas dataframe.corr() is used to find the pairwise correlation of all columns in the dataframe. 

# Set-up mask

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

# Set-up figure

plt.figure(figsize=(14, 8))

# Title

plt.title('Overall Correlation of Titanic Features', fontsize=18)

# Correlation matrix

sns.heatmap(corr, mask=mask, annot=False,cmap='RdYlGn', linewidths=0.2, annot_kws={'size':20})

plt.show()
# Plot for survived

fig = plt.figure(figsize = (10,5))

sns.countplot(x='Survived', data = df_train)

print(df_train['Survived'].value_counts())
# Bar chart of each Pclass type

fig = plt.figure(figsize = (10,10))

ax1 = plt.subplot(2,1,1)

ax1 = sns.countplot(x = 'Pclass', hue = 'Survived', data = df_train)

ax1.set_title('Ticket Class Survival Rate')

ax1.set_xticklabels(['1 Upper','2 Middle','3 Lower'])

ax1.set_ylim(0,400)

ax1.set_xlabel('Ticket Class')

ax1.set_ylabel('Count')

ax1.legend(['No','Yes'])



# Pointplot Pclass type

ax2 = plt.subplot(2,1,2)

sns.pointplot(x='Pclass', y='Survived', data=df_train)

ax2.set_xlabel('Ticket Class')

ax2.set_ylabel('Percent Survived')

ax2.set_title('Percentage Survived by Ticket Class')


# Bar chart of age mapped against sex. For now, missing values have been dropped and will be dealt with later

survived = 'survived'

not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

women = df_train[df_train['Sex']=='female']

men = df_train[df_train['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=20, label = survived, ax = axes[0], kde =False)

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=20, label = not_survived, ax = axes[0], kde =False)

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=20, label = survived, ax = axes[1], kde = False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=20, label = not_survived, ax = axes[1], kde = False)

ax.legend()

_ = ax.set_title('Male')


# Plotting survival rate vs Siblings or Spouse on board

fig = plt.figure(figsize = (10,12))

ax1 = plt.subplot(2,1,1)

ax1 = sns.countplot(x = 'SibSp', hue = 'Survived', data = df_train)

ax1.set_title('Survival Rate with Total of Siblings and Spouse on Board')

ax1.set_ylim(0,500)

ax1.set_xlabel('# of Sibling and Spouse')

ax1.set_ylabel('Count')

ax1.legend(['No','Yes'],loc = 1)



# Plotting survival rate vs Parents or Children on board

ax2 = plt.subplot(2,1,2)

ax2 = sns.countplot(x = 'Parch', hue = 'Survived', data = df_train)

ax2.set_title('Survival Rate with Total Parents and Children on Board')

ax2.set_ylim(0,500)

ax2.set_xlabel('# of Parents and Children')

ax2.set_ylabel('Count')

ax2.legend(['No','Yes'],loc = 1)


# Bar chart of each Fare type

fig = plt.figure(figsize = (10,10))

ax1 = sns.countplot(x = 'Pclass', hue = 'Survived', data = df_train)

ax1.set_title('Ticket Class Survival Rate with respect to fare')

ax1.set_xticklabels(['1 Upper','2 Middle','3 Lower'])

ax1.set_xlabel('Ticket Class')

ax1.set_ylabel('Fare')

ax1.legend(['No','Yes'])
# Graph to display fare paid per the three ticket types

fig = plt.figure(figsize = (10,5))

sns.swarmplot(x="Pclass", y="Fare", data=df_train, hue='Survived')
print("TRAIN DATA:")

df_train.isnull().sum()
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

print("TEST DATA:")

df_test.isnull().sum()
sns.heatmap(df_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
