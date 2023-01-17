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
df = pd.read_csv('../input/train.csv')
df.describe()
df.apply(lambda x: sum(x.isnull()),axis=0) 
df = df.drop(['PassengerId','Name','Ticket'], axis=1)
df.apply(lambda x: sum(x.isnull()),axis=0) 
temp1 = df['Embarked'].value_counts(ascending=True)

print(temp1)
df['Embarked'].fillna("S", inplace=True)
temp1 = df['Embarked'].value_counts(ascending=True)

print(temp1)
temp2 = df.pivot_table(values='Survived',index=['Embarked'],aggfunc=lambda x: x.mean())

print(temp2)



print('/n Probablity of sutviving based on Port of Embarkation')

fig = plt.figure(figsize=(8,4))

ax1 = fig.add_subplot(121)

ax1.set_xlabel('Embarked')

ax1.set_ylabel('Survived')

ax1.set_title("Survival by Port of Embarkation")

temp2.plot(kind='bar')
fig = plt.figure(figsize=(8,4))

ax1 = fig.add_subplot(121)

ax1.set_xlabel('Gender')

ax1.set_ylabel('Count of Applicants')

ax1.set_title("Applicants by Gender")

temp1.plot(kind='bar')