# import packages

import numpy as np

import pandas as pd



# include data files in this environment

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# read in titanic training data as pandas dataframe

train_filepath = '../input/titanic/train.csv'

trainingdata = pd.read_csv(train_filepath)
# What columns does this data include?

trainingdata.columns
# some summary statistics

trainingdata.describe()
# What percentage of passengers were single passengers (no family aboard)?

singles = trainingdata[(trainingdata['SibSp']==0) & (trainingdata['Parch']==0)]

singles.describe()
print('Percentage of passengers with no family aboard:',537/891)
# get the three data subsets based on class

firstclass = trainingdata[trainingdata['Pclass']==1]

secondclass = trainingdata[trainingdata['Pclass']==2]

thirdclass = trainingdata[trainingdata['Pclass']==3]
# grouping by class, plot histogram for fare

firstclass['Fare'].plot(kind='hist',title='First Class Fares')
secondclass['Fare'].plot(kind='hist',title='Second Class Fares')
thirdclass['Fare'].plot(kind='hist',title='Third Class Fares')
print('First Class Subpopulation')

firstclass.describe()
print('Second Class Subpopulation')

secondclass.describe()
print('Third Class Subpopulation')

thirdclass.describe()
males = trainingdata[trainingdata['Sex']=='male']

females = trainingdata[trainingdata['Sex']=='female']
print('Male Subpopulation')

males.describe()
print('Female Subpopulation')

females.describe()
print('Percentage of male passengers = ',577/891)

print('Percentage of female passengers = ',314/891)