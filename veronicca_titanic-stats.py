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

#reading titanic data

ts = pd.read_csv("../input/train.csv")

ts.head(10)
ts.count()
ts["sex_group"] = ts["Sex"].map({'male': 1,'female': 0})#mapping
total_male =ts[ts['Sex']=='male'].Pclass.count()
total_male
ts.count()
female_count =ts[ts['Sex']=='female'].Pclass.count()
female_count
female_survived = ts[(ts['Sex']=='female') & (ts['Survived']==1)].Pclass.count()#manual calculation
female_survived
male_survived = ts[(ts['Sex']=='male') & (ts['Survived']==1)].Pclass.count()
male_survived
survived_male_ratio = male_survived / total_male
survived_male_ratio
survived_female_ratio = female_survived / female_count
survived_female_ratio
total_children= ts[ts["Age"]<16].Pclass.count()
total_children
children_survived = ts[(ts["Age"]<16) & (ts['Survived']==1)].Pclass.count()
children_survived
children_survived_ratio = children_survived/total_children
children_survived_ratio
missing_ages = ts[ts['Age'].isnull()]



mean_age = ts.groupby(['Sex','Pclass'])['Age'].mean()

#calculating mean age by sex,pclass



def remove_na_ages(row):

    #defining function for removing missing ages

    

    if pd.isnull(row['Age']):

        return mean_age[row['Sex'],row['Pclass']]

    else:

        return row['Age']



ts['Age'] =ts.apply(remove_na_ages, axis=1)
missing_ages

mean_age
def get_title(name):

    if '.' in name:

        return name.split(',')[1].split('.')[0].strip()

    else:

        return 'Unknown'



def title_map(title):

    if title in ['Mr']:

        return 1

    elif title in ['Master']:

        return 3

    elif title in ['Ms','Mlle','Miss']:

        return 4

    elif title in ['Mme','Mrs']:

        return 5

    else:

        return 2

    

ts['title'] = ts['Name'].apply(get_title).apply(title_map)   

title_xt = pd.crosstab(ts['title'], ts['Survived'])

title_xt_pct = title_xt.div(title_xt.sum(1).astype(float), axis=0)



title_xt_pct.plot(kind='bar', 

                  stacked=True, 

                  title='Survival Rate by title')

plt.xlabel('title')

plt.ylabel('Survival Rate')