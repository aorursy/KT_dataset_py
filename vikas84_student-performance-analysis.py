# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
students_df = pd.read_csv("../input/StudentsPerformance.csv")
students_df.head()
score_limits = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90,100]
score_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
students_df.rename(columns={'race/ethnicity':'group'},inplace=True)
students_df['avg_score'] = (students_df['math score'] + students_df['reading score'] + students_df['writing score'])/3
students_df['math_bin'] = pd.cut(students_df['math score'], score_limits, labels=score_labels, right=False, include_lowest = True)
students_df['reading_bin'] = pd.cut(students_df['reading score'], score_limits, labels=score_labels, right=False, include_lowest = True)
students_df['writing_bin'] = pd.cut(students_df['writing score'], score_limits, labels=score_labels, right=False, include_lowest = True)
students_df['avg_score_bin'] = pd.cut(students_df['avg_score'], score_limits, labels=score_labels, right=False, include_lowest = True)
students_df.head()
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

def isNaN(num):
    return num != num
## Analyzing student perfromance based on gender and math scores 
sns.set()
sns.set_context("poster")
g=sns.catplot("math_bin", col="gender", col_wrap=2,
                    data=students_df[['math_bin','gender']],
                    kind="count", height=8, aspect=1.6 )

for plt in g.axes:
    total = 1000
    #print total

    for i in plt.patches :
        x = i.get_height()
        if isNaN(x):
            val = 0
        else :
            val = float(x)
        normalized_val = val/total 
        plt.text(i.get_x(), val+5, str(round((normalized_val)*100, 2))+'%', fontsize=15)
        
g.set_xticklabels(rotation=30)
sns.set()
sns.set_context("poster")
g=sns.catplot("reading_bin", col="gender", col_wrap=2,
                    data=students_df[['reading_bin','gender']],
                    kind="count", height=8, aspect=1.6 )

for plt in g.axes:
    total = 1000
    #print total

    for i in plt.patches :
        x = i.get_height()
        if isNaN(x):
            val = 0
        else :
            val = float(x)
        normalized_val = val/total 
        plt.text(i.get_x(), val+5, str(round((normalized_val)*100, 2))+'%', fontsize=15)
        
g.set_xticklabels(rotation=30)
sns.set()
sns.set_context("poster")
g=sns.catplot("writing_bin", col="gender", col_wrap=2,
                    data=students_df[['writing_bin','gender']],
                    kind="count", height=8, aspect=1.6 )

for plt in g.axes:
    total = 1000
    #print total

    for i in plt.patches :
        x = i.get_height()
        if isNaN(x):
            val = 0
        else :
            val = float(x)
        normalized_val = val/total 
        plt.text(i.get_x(), val+5, str(round((normalized_val)*100, 2))+'%', fontsize=15)
        
g.set_xticklabels(rotation=30)
sns.set()
sns.set_context("poster")
g=sns.catplot("avg_score_bin", col="gender", col_wrap=2,
                    data=students_df[['avg_score_bin','gender']],
                    kind="count", height=8, aspect=1.6 )

for plt in g.axes:
    total = 1000
    #print total

    for i in plt.patches :
        x = i.get_height()
        if isNaN(x):
            val = 0
        else :
            val = float(x)
        normalized_val = val/total 
        plt.text(i.get_x(), val+5, str(round((normalized_val)*100, 2))+'%', fontsize=15)
        
g.set_xticklabels(rotation=30)
sns.set()
sns.set_context("poster")
g=sns.catplot("avg_score_bin", col="group", col_wrap=2,
                    data=students_df[['avg_score_bin','group']],
                    kind="count", height=8, aspect=1.6 )

for plt in g.axes:
    total = 1000
    #print total

    for i in plt.patches :
        x = i.get_height()
        if isNaN(x):
            val = 0
        else :
            val = float(x)
        normalized_val = val/total 
        plt.text(i.get_x(), val+5, str(round((normalized_val)*100, 2))+'%', fontsize=15)
        
g.set_xticklabels(rotation=30)
sns.set()
sns.set_context("poster")
g=sns.catplot("avg_score_bin", col="parental level of education", col_wrap=2,
                    data=students_df[['avg_score_bin','parental level of education']],
                    kind="count", height=8, aspect=1.6 )

for plt in g.axes:
    total = 1000
    #print total

    for i in plt.patches :
        x = i.get_height()
        if isNaN(x):
            val = 0
        else :
            val = float(x)
        normalized_val = val/total 
        plt.text(i.get_x(), val+2, str(round((normalized_val)*100, 2))+'%', fontsize=15)
        
g.set_xticklabels(rotation=30)

