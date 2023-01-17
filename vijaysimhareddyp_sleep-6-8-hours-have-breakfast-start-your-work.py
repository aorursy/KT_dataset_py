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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_csv('/kaggle/input/sleepstudypilot/SleepStudyPilot.csv')

df
df1=pd.read_csv('/kaggle/input/sleepstudypilot/SleepStudyData.csv')

df1
df1.groupby('Tired').mean()
df1.groupby('Breakfast')['Tired'].mean()
HoursSleep=pd.DataFrame(df1.groupby('Hours')['Tired'].mean())

HoursSleep
Enough_SleepYes=df1[df1["Enough"]=="Yes"][["Hours","Breakfast","PhoneReach","PhoneTime","Tired"]].sort_values("Hours",ascending=False)

Enough_SleepYes
Enough_SleepNo=df1[df1["Enough"]=="No"][["Hours","Breakfast","PhoneReach","PhoneTime","Tired"]].sort_values("Hours",ascending=False)

Enough_SleepNo
HoursSleep
plt.figure(figsize=(8,5))

sns.countplot(df1['Hours'],data=HoursSleep)  # Count is No. of people

plt.title('CountPlot')
plt.figure(figsize=(15,8))

sns.violinplot(x=df1['Hours'], y=df1['Tired'], data=df1)
plt.figure(figsize=(5,5))

sns.violinplot(x=df1['Enough'], y=df1['Tired'], data=df1)

# The below Plot shows the Person with Enough Sleep has Less Tired Rate than Who doesn't Sleep Enough Hours.
df1.corr()
sns.heatmap(df1.corr(), annot=True, cmap='Spectral')
plt.figure(figsize=(15,8))

sns.boxplot(y=df1['Hours'], x=df1['Tired'], data=df1)

# The below Plot shows the Person Who sleeps 6 to 8 hours has less Tired Rate

# And People who Sleep More than that and less than that have more Tired Rate.
plt.figure(figsize=(8,5))

sns.boxplot(x=df1['Enough'], y=df1['Tired'], data=df1,hue='Enough')

# From the below Plot we can see No Enough sleep has More Tired Rate Compared to Enough Sleep People
sns.boxplot(x=df1['Breakfast'], y=df1['Tired'], data=df1,hue='Breakfast')

# From the below Plot we can see No Breakfast has More Tired Rate Compared to People who had their Breakfast.

# Both the Outliers in the plot says that only some(about 1-3%) are viceversa to the above statement.
sns.jointplot(x=df1['Tired'], y=df1['Hours'], data = df1, kind='hex')

# From thee below Plot we can see,average Tired% is between 2 and 3 because the Thickness is more.
sns.jointplot(x=df1['Tired'], y=df1['Hours'], data = df1, kind='reg')

# From the Below Plot we can see the Regression Line Passing in Between 6 & 8 Hours.