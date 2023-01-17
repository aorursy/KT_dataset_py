# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as snb

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
Cities = pd.read_csv('../input/cities_r2.csv')
Cities.shape
Cities.columns
Cities.info()
Cities.head(10)
Cities.describe()
Cities.isnull().sum()
Cities.notnull().sum()
Cities[['state_name','population_total']].groupby('state_name').sum().sort_values('population_total', ascending=False).head(5)
Cities[['name_of_city','population_total']].sort_values('population_total', ascending=False).head(5)
Cities[['name_of_city','male_graduates']].sort_values('male_graduates', ascending=False).head(5)
Cities[['name_of_city','male_graduates']].sort_values('male_graduates', ascending=True).head(5)
Statewise_Population = Cities[['state_name', 'population_total']].groupby('state_name').sum().sort_values('population_total', ascending=False)

print (Statewise_Population )

Statewise_Population.plot(kind = 'bar', legend=False)

plt.show()
Cities[['name_of_city','female_graduates']].sort_values('female_graduates', ascending=False).head(5)
Cities[['name_of_city','female_graduates']].sort_values('female_graduates', ascending=True).head(5)
Cities['grad_pert'] = (Cities['total_graduates']/Cities['population_total'])*100

Statewise_Graduation_Pert = Cities[['state_name', 'grad_pert']].groupby('state_name').mean().sort_values('grad_pert', ascending=False)

Statewise_Graduation_Pert.plot(kind='bar', legend=False)

plt.show()