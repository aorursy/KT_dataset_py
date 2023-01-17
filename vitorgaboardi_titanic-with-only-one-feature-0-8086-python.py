# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Train and test data

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")



# To make the final submission

holdout_ids = test["PassengerId"] 
train.head()
# Get all the young (under 16) male and store in "male"   

male = train.loc[(train.Sex=='male') & (train.Age < 16)]



# Get all the Pclass 3 female and store in "female"

female = train.loc[(train.Sex=='female') & (train.Pclass==3)]



print(male['Survived'].value_counts())

print(female['Survived'].value_counts())
# First, let's extract the Title of each passenger

train['Title'] = 0

train['Title'] = train.Name.str.extract('([A-Za-z]+)\.') #lets extract the title



titles = {

    "Mr":       "man",

    "Mme":      "woman",

    "Ms":       "woman",

    "Mrs":      "woman",

    "Master":   "boy",

    "Mlle":     "woman",

    "Miss":     "woman",

    "Capt":     "man",

    "Col":      "man",

    "Major":    "man",

    "Dr":       "man",

    "Rev":      "man",

    "Jonkheer": "man",

    "Don":      "man",

    "Sir":      "man",

    "Countess": "woman",

    "Dona":     "woman",

    "Lady":     "woman"

}



train["Title"] = train["Title"].map(titles)
train.head()
# Let's extract the Surname of each passenger

train['Surname'] = 0

train['Surname'] = train.Name.str.extract('([A-Za-z]+)\,')





# All the men will be labelled as DeadMan (we only want to test the boys and the woman)

train.loc[(train.Title=='man'), 'Surname'] = 'DeadMan'





# Get women/boys that are alone (no repeated Surname and without considering Man)

unique_surname = (train['Surname'].value_counts().index)[ (train['Surname'].value_counts() == 1) ]



# List that will have all the SINGLE passengers that survived and died

passenger_lived = []

passenger_died = []



for surname in unique_surname:

  single_df = train.loc[(train.Surname == surname)]

  lived = single_df['Survived'].loc[(single_df.Survived == 1)].sum()

  if(lived == 1):

    passenger_lived.append(surname)

  else:

    passenger_died.append(surname)





# Get women/boys that are NOT alone.

non_unique_surname = (train['Surname'].value_counts().index)[ train['Surname'].value_counts() != 1 ].drop('DeadMan')



# List that will have all the FAMILIES that survived and died.

families_lived = []

families_dead = []



# Test if the majority of the family members (> 0.5) lived or died

for surname in non_unique_surname:

  surname_df = train.loc[(train.Surname == surname)]

  number_family = len(surname_df)

  number_lived = surname_df['Survived'].loc[(surname_df.Survived == 1)].sum()

  #print("Family: %s, Size: %d, Lived: %d " % (surname,number_family,number_lived))

  if(number_lived/number_family >= 0.5):

    families_lived.append(surname)

  else:

    families_dead.append(surname)





## Replace the 'Surname' feature according to the classification just made

train['Surname'] = train['Surname'].replace(passenger_lived, 'LivedPassenger')

train['Surname'] = train['Surname'].replace(passenger_died, 'DeadPassenger')

train['Surname'] = train['Surname'].replace(families_lived, 'LivedFamily')

train['Surname'] = train['Surname'].replace(families_dead, 'DeadFamily')

train['Surname'].value_counts()
# Define new feature

train['SurnameSurvival'] = 0



train.loc[(train.Surname == 'LivedPassenger'), 'SurnameSurvival'] = 1

train.loc[(train.Surname == 'LivedFamily'), 'SurnameSurvival'] = 1
family_lived = train.loc[(train.Surname == 'LivedFamily')] 

print(family_lived['Survived'].value_counts())                  
prediction_train = train.SurnameSurvival

true_train = train.Survived



print(accuracy_score(true_train, prediction_train))
# Combine train/test data to apply transformations simultaneously

X = pd.read_csv("../input/titanic/train.csv").drop('Survived', axis=1)

df = pd.concat([test,X],ignore_index=True,sort=False)



# Get the index to later separate train and test data.

test_index = test.index
# Let's extract the Title of each passenger

df['Title'] = 0

df['Title'] = df.Name.str.extract('([A-Za-z]+)\.') #lets extract the title



titles = {

    "Mr":       "man",

    "Mme":      "woman",

    "Ms":       "woman",

    "Mrs":      "woman",

    "Master":   "boy",

    "Mlle":     "woman",

    "Miss":     "woman",

    "Capt":     "man",

    "Col":      "man",

    "Major":    "man",

    "Dr":       "man",

    "Rev":      "man",

    "Jonkheer": "man",

    "Don":      "man",

    "Sir":      "man",

    "Countess": "woman",

    "Dona":     "woman",

    "Lady":     "woman"

}



df["Title"] = df["Title"].map(titles)
df.head()
# Extract the Surname of each passenger

df['Surname'] = 0

df['Surname'] = df.Name.str.extract('([A-Za-z]+)\,')



# All the men will be labelled as DeadMan (we only want to test the boys and the woman)

df.loc[(df.Title=='man'), 'Surname'] = 'DeadMan'





# All previously single passengers that were labeled as lived will be considered lived.

for surname in passenger_lived:

  df['Surname'] = df['Surname'].replace(surname, 'LivedPassenger')





# All previously single passengers that were labeled as dead will be considered dead.

for surname in passenger_died:

  df['Surname'] = df['Surname'].replace(surname, 'DeadPassenger')





# All previously families that were labeled as lived will be considered lived.

for surname in families_lived:

  df['Surname'] = df['Surname'].replace(surname, 'LivedFamily')





# All previosly families that were labeled as dead will be considered dead.

for surname in families_dead:

  df['Surname'] = df['Surname'].replace(surname, 'DeadFamily')

df['Surname'].value_counts()
unique_surname = (df['Surname'].value_counts().index)[ df['Surname'].value_counts() == 1 ] 

unique_surname
# Get the single passenger information

single_passenger = df[df['Surname'].isin(unique_surname.to_list())]



# Separate into single_woman and single_boy DataFrames

single_woman = single_passenger['Surname'].loc[(single_passenger.Title=='woman')]

single_boy = single_passenger['Surname'].loc[(single_passenger.Title=='boy')]



# All the women alone will be labelled as LivedPassenger

df['Surname'] = df['Surname'].replace(single_woman, 'LivedPassenger')



# All the boys alone will be labelled as DeadPassenger

df['Surname'] = df['Surname'].replace(single_boy, 'DeadPassenger')

df['Surname'].value_counts()
new_families = df.loc[(df.Surname != 'DeadMan') & (df.Surname != 'LivedPassenger') & (df.Surname != 'DeadPassenger') & (df.Surname != 'LivedFamily') & (df.Surname != 'DeadFamily')]

new_families_list = (df['Surname'].value_counts().index)[ df['Surname'].value_counts() < 5 ]

new_families
df['Surname'] = df['Surname'].replace('Billiard', 'DeadFamily')

df['Surname'] = df['Surname'].replace(new_families_list, 'LivedFamily') #the highest value is this one where everyone here is dead!



df['Surname'].value_counts()
# Define new feature

df['SurnameSurvival'] = 0



df.loc[(df.Surname == 'LivedPassenger'), 'SurnameSurvival'] = 1

df.loc[(df.Surname == 'LivedFamily'), 'SurnameSurvival'] = 1
prediction = df['SurnameSurvival'].loc[test_index]

prediction
submission_df = {"PassengerId": holdout_ids,

                 "Survived": prediction}



submission = pd.DataFrame(submission_df)

submission.to_csv("submission.csv",index=False)