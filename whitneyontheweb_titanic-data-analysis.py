# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Read data from CSV into Pandas DataFrame

titanic_df = pd.read_csv('../input/train.csv') #titanic-data.csv
# Preview DataFrame Head

titanic_df.head()
# Preview DataFrame Tail

titanic_df.tail()
# Get DataFrame Data Types

titanic_df.info()
#Descriptive statistics on numerical data

titanic_df.describe()
columns = list(titanic_df[['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])



def describe_data(data, col):

    print ('\n\n', col)

    print ('_' * 40)

    print ('Mean:', np.mean(data)) #NumPy Mean

    print ('STD:', np.std(data))   #NumPy STD

    print ('Min:', np.min(data))   #NumPy Min

    print ('Max:', np.max(data))   #NumPy Max



for c in columns:

    describe_data(titanic_df[c], c)
#Descriptive statistics on categorical data

titanic_df.describe(include=['O'])
# Select Sex and Survived Columns

# Group data by sex

# Show index for row

# Return mean of values for male/female

# Sort by highest survival rate first

titanic_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Set plot style

sns.set(style='ticks', color_codes=True)



# Plot passenger age distribution

age_hist = sns.FacetGrid(titanic_df)

age_hist.map(plt.hist, 'Age', bins=20)
# Plot histogram of survival by age

age_hist = sns.FacetGrid(titanic_df, col='Survived', hue='Survived')

age_hist.map(plt.hist, 'Age', bins=15)
# Create AgeRange with 16 ranges

titanic_df['AgeRange'] = pd.cut(titanic_df['Age'], 16)



# Calculate proportion of surviors for each AgeRange

titanic_df[['AgeRange', 'Survived']].groupby(['AgeRange'], as_index=False).mean().sort_values(by='AgeRange', ascending=True)
# Plot histogram by survival, sex, and age

age_sex_hist = sns.FacetGrid(titanic_df, col='Survived', row='Sex', hue='Survived')

age_sex_hist.map(plt.hist, 'Age', bins=20)
# Calculate proportion of surviors for each AgeRange

titanic_df[['Sex', 'AgeRange', 'Survived']].groupby(['Sex', 'AgeRange'], as_index=False).mean().sort_values(by='AgeRange', ascending=True)
#Create Passenger column to plot total passengers

titanic_df['Passenger'] = 'Passenger'

# Create Class column with string values for class

titanic_df['Class'] = titanic_df['Pclass'].map( {1: 'Upper', 2: 'Middle', 3: 'Lower'} )



# Create PointPlot for Passengers by Class

bp = sns.pointplot(x='Passenger', y='Survived', hue='Class', data=titanic_df, hue_order=['Lower', 'Middle', 'Upper'])

bp.set(ylabel='% Survivors', xlabel='Passenger Class')
# Create PointPlot for Passengers by Class and Sex

bps = sns.barplot(x='Sex', y='Survived', hue='Class', data=titanic_df, hue_order=['Lower', 'Middle', 'Upper'])

bps.set(ylabel='% Survivors', xlabel='Passenger Sex by Class')
# Recreate AgeRanges

titanic_df['AgeRange'] = pd.cut(titanic_df['Age'], 6)



# Create PointPlot for Passengers by Class and Age

pp = sns.pointplot(x='AgeRange', y='Survived', hue='Class', data=titanic_df, hue_order=['Lower', 'Middle', 'Upper'])

pp.set(ylabel='% Survivors', xlabel='Passenger Age by Class')
#Pearsons R for class and survival

np.corrcoef(x=titanic_df['Pclass'], y=titanic_df['Survived'])
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train_df.head()