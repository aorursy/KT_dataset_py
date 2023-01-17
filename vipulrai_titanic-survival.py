# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

import sklearn

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

#print(train_df)



train_df.head(10)
total_male = train_df[train_df['Sex'] == 'male'].Pclass.count()

total_female = train_df[train_df['Sex'] == 'female'].PassengerId.count()
total_male
total_female
male_female_ratio = total_female / total_male
male_female_ratio
survived_male = train_df[(train_df['Sex'] == 'male') & (train_df['Survived'] == 1)].PassengerId.count()
survived_male
#getting the info  about df



train_df.info()
def get_title(name):

    if '.' in name:

        return name.split(',')[1].split('.')[0].strip()

    else:

        return "Unknown"

    
train_df['Title'] = train_df['Name'].apply(get_title)
train_df.head()
train_df['Title'].value_counts()
#converting the tile into categorical values
def map_title(title):

    if title in ['Mr' , 'Sir' , 'Col' , 'Major' , 'Capt' , 'Rev']:

        return 1

    elif title in ['Master']:

        return 3

    elif title in ['Ms','Mlle','Miss']:

        return 4

    elif title in ['Mme','Mrs']:

        return 5

    else:

        return 2
train_df['Title_Cat'] = train_df['Name'].apply(get_title).apply(map_title)

test_df['Title_Cat'] = test_df['Name'].apply(get_title).apply(map_title)

#test_df['Title_Cat'] = test_df['Name'].apply(get_title)
train_df.head()
print("-----------------------------------------------------")

test_df.head()
title_xt = pd.crosstab(train_df['Title_Cat'] , train_df['Survived'])
title_xt
title_xt_pct = title_xt.div(title_xt.sum(1).astype(float), axis=0)
title_xt_pct
title_xt_pct.plot(kind='bar')

plt.xlabel('Title')

plt.ylabel('Survived')

train_df = train_df.drop(["PassengerId" , "Name" ,"Ticket" ,"Title"] , axis=1)
train_df.head()
train_df['Embarked'].value_counts()
train_df['Embarked'] = train_df['Embarked'].fillna('S')
train_df['Embarked'].value_counts()
sns.factorplot('Embarked','Survived', data=train_df,size=4,aspect=3)