# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import pip

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



%matplotlib inline



# Use ggplot style

matplotlib.style.use('ggplot')
#sorted(["%s==%s" % (i.key, i.version) for i in pip.get_installed_distributions()])
# Read Training Data "train.csv"

df = pd.read_csv('../input/train.csv')

df.info()
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

df
median_ages = np.zeros((2,3))

median_ages
for i in range(0, 2):

    for j in range(0, 3):

        median_ages[i,j] = df[(df['Gender'] == i) & \

                              (df['Pclass'] == j+1)]['Age'].dropna().median()

 

median_ages
df['AgeFill'] = df['Age']

df
df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)
for i in range(0, 2):

    for j in range(0, 3):

        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\

                'AgeFill'] = median_ages[i,j]

df
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

df
df['FamilySize'] = df['SibSp'] + df['Parch']

df
df['Age*Class'] = df.AgeFill * df.Pclass

df
df['Age*Class'].plot.hist()
df.dtypes[df.dtypes.map(lambda x: x=='object')]
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1) 

df = df.drop(['Age'], axis=1)

df = df.dropna()

df
train_data = df.iloc[0:, 1:].values

train_data
train_data[0:,1:]
train_data[0::,0]
# Create the random forest object which will include all the parameters

# for the fit

forest = RandomForestClassifier(n_estimators = 100)
# Fit the training data to the Survived labels and create the decision trees

forest = forest.fit(train_data[0::,1::],train_data[0::,0])