# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
raw_data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

raw_data.head(10)
raw_data.corr()
raw_data.describe()
raw_data.dtypes
print("Unique values in Gender are",raw_data.gender.unique())

print("Unique values in SSC Board are",raw_data.ssc_b.unique())

print("Unique values in HSC Board are",raw_data.hsc_b.unique())

print("Unique values in Degree type are",raw_data.degree_t.unique())

print("Unique values in Work Experience are",raw_data.workex.unique())

print("Unique values in Degree Specialisation are",raw_data.specialisation.unique())

print("Unique values in Placement status are",raw_data.status.unique())
droped_data = raw_data.drop(['sl_no'],axis = 1)

droped_data['Gender'] = droped_data.gender.map({'M': 0,'F' : 1})

droped_data['SSC_board'] = droped_data.ssc_b.map({'Others': 0,'Central' : 1})

droped_data['HSC_board'] = droped_data.hsc_b.map({'Others': 0,'Central' : 1})

droped_data['Degree_type'] = droped_data.degree_t.map({'Sci&Tech': 0,'Comm&Mgmt' : 1, 'Others': 2})

droped_data['Work_exp'] = droped_data.workex.map({'No': 0,'Yes' : 1})

droped_data['Specialisation'] = droped_data.specialisation.map({'Mkt&HR': 0,'Mkt&Fin' : 1})                                    

droped_data['Status'] = droped_data.status.map({'Placed': 1,'Not Placed' : 0}) 

droped_data.head()
ready_data = droped_data.drop(['gender','ssc_b','hsc_b','hsc_s','degree_t','status','workex','specialisation'], axis = 1)

ready_data['Salary'] = ready_data['salary'].fillna(0)

ready_data = ready_data.drop(['salary'], axis = 1)

ready_data.head()
#The test scores that the candidate has scored in his academics.

test_scores = ready_data[['ssc_p','hsc_p','degree_p','etest_p','mba_p']]

test_scores.head()
#The final verdict and salary that the candidate achivied based on his/her scores.

verdict = ready_data[['ssc_p','hsc_p','degree_p','etest_p','mba_p','Status','Salary']]

verdict.head()
salary = verdict['Salary']

salary = salary[salary != 0] #Remove all unplaced students



salary.hist(bins = 100)

plt.grid()
sns.boxplot(salary)
g = sns.PairGrid(verdict)

g.map(plt.scatter);