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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestClassifier



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



all_ = pd.concat([train.loc[:,'Pclass':'Embarked'],test.loc[:,'Pclass':'Embarked']])

test_id = test['PassengerId']

y = train['Survived']

print(all_.head)
all_['Group'] = ''

for i in range(1, all_.shape[0]):

    name = all_.Name.irow(i)

    if 'Mrs' in name:

        all_['Group'].iloc[i] = 'Mrs'

    elif 'Miss' in name:

        all_['Group'].iloc[i] = 'Miss'

    elif 'Mr' in name:

        all_['Group'].iloc[i] = 'Mr'

    elif 'Master' in name:

        all_['Group'].iloc[i] = 'Master'

    elif 'Dr' in name:

        all_['Group'].iloc[i] = 'Dr'

    else:

        all_['Group'].iloc[i] = 'Other'

print(all_['Group'].head)
mr_age = all_[all_['Group'] == 'Mr']['Age'].dropna().mean()

mrs_age = all_[all_['Group'] == 'Mrs']['Age'].dropna().mean()

miss_age = all_[all_['Group'] == 'Miss']['Age'].dropna().mean()

master_age = all_[all_['Group'] == 'Master']['Age'].dropna().mean()

dr_age = all_[all_['Group'] == 'Dr']['Age'].dropna().mean()

other_age = all_[all_['Group'] == 'Other']['Age'].dropna().mean()

print([mr_age,mrs_age,miss_age,master_age,dr_age,other_age])
# Replace nan age with the mean of corresponding group

mean_group = {'Mr':mr_age,'Mrs':mrs_age,'Miss':miss_age,'Master':master_age,'Dr':dr_age,'Other':other_age}

for i in range(1, all_.shape[0]):

    if all_['Age'].iloc[i] == None:

        if all_['Group'].iloc[i] == 'Mr':

            all_['Age'].iloc[i] = mean_group['Mr']

        elif all_['Group'].iloc[i] == 'Mrs':

            all_['Age'].iloc[i] = mean_group['Mrs']

        elif all_['Group'].iloc[i] == 'Miss':

            all_['Age'].iloc[i] = mean_group['Miss']

        elif all_['Group'].iloc[i] == 'Master':

            all_['Age'].iloc[i] = mean_group['Master']

        elif all_['Group'].iloc[i] == 'Dr':

            all_['Age'].iloc[i] = mean_group['Dr']

        else:

            all_['Age'].iloc[i] = mean_group['Other']
# Dummy variables and replace category nan value with its mean

all_ = pd.get_dummies(all_)

all_ = all_.fillna(all_.mean())



#y = train['Survived']

train = all_[:train.shape[0]]

test = all_[train.shape[0]:]



clf = RandomForestClassifier(random_state=1, n_estimators=50, max_depth=9,min_samples_split=6, min_samples_leaf=4)

clf.fit(train,y)

y_test = clf.predict(test)

print(clf.score(train,y))
# Output

output = {'PassengerId':test_id,'Survived':y_test}

output = pd.DataFrame.from_dict(output)

output.to_csv('submission_bae.csv')