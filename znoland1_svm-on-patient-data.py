# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import scale





df = pd.read_csv('../input/No-show-Issue-Comma-300k.csv')
#box plot

plt.figure(figsize=(12,5))

sns.boxplot(df)
# pair plot

sns.pairplot(df, hue='Status')
df.head()
# Create lable: Show-up = 0, Now-Show = 1

options = {'Show-Up':0, 'No-Show':1}

y = df.Status.replace(options)

y.value_counts()
# Day of the week value counts

df.DayOfTheWeek.value_counts()
# Gender value counts

df.Gender.value_counts()
# Drop Date and time values for now -- REVISIT and add more detailed date/time features

del df['AppointmentRegistration']

del df['ApointmentData']



# Remove Label

del df['Status']



# Scale AwaitingTime and Age

df['AwaitingTime'] = scale(df['AwaitingTime'])

df['Age'] = scale(df['Age'])



# Create dummy variables for categorical variables (Gender, DayOfThe Week)

df_gender = pd.get_dummies(df['Gender'])

df_DayOfTheWeek = pd.get_dummies(df['DayOfTheWeek'])

df = pd.concat([df,df_gender, df_DayOfTheWeek])



# Drop categorical variables

del df['Gender']

del df['DayOfTheWeek']



# Drop dummy variables not needed (M, Sunday)

del df['M']

del df['Sunday']



# Fill NA

df = df.fillna(0)



df.head()
from sklearn.model_selection import train_test_split