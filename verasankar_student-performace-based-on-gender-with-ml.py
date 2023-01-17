# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

students=open('../input/StudentsPerformance.csv','r')



# Any results you write to the current directory are saved as output.
print('welcome')

print(students.read())
df = pd.read_csv('../input/StudentsPerformance.csv')

print(df.head(5))
df.info()
df.nunique().sort_values()
df.describe()
print(df.groupby('gender').agg(['mean', 'count' ,'max','min'])['math score'])
df[['gender','math score']].head(10)