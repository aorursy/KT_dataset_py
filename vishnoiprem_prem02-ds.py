# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory







# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="whitegrid", color_codes=True)

sns.set(font_scale=1)

from subprocess import check_output

print(check_output(["ls", "../input/"]).decode("utf8"))

#print(check_output(["ls", "/Users/vishnoiprem/Udemy/kaggle/titanic"]).decode("utf8"))





#genderclassmodel.csv
#titanic.isnull().any()



#test=pd.read_csv('/Users/vishnoiprem/Udemy/kaggle/titanic/test.csv', header = 0, dtype={'Age': np.float64})

#df=pd.read_csv('/Users/vishnoiprem/Udemy/kaggle/titanic/train.csv', header = 0, dtype={'Age': np.float64})

df = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})

test  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})



#df.head(19)

full_data=[test, df]



df.head(10)

#full_data
df.columns[df.isnull().any()]
def mr_mrs_other(name):

    if 'Mr.' in name:

        return 'Mr'

    elif 'Mrs.' in name:

        return 'Mrs'

    elif 'Miss.' in name:

        return 'Miss' 

    else: 

        return 'Other'



df['Gen_Clas']=df['Name'].apply(mr_mrs_other)

fill_na=df[df['Gen_Clas']=='Other'][['Age','Pclass']].groupby('Pclass').mean()

#fill_na

a=np.array(fill_na)

print(a[0][0],a[1][0],a[2][0])

#df[(df['Gen_Clas']=='Other') & (df['Pclass']==1)]['Age'].fillna(a[0][0],inplace=False)

#df[(df['Gen_Clas']=='Other') & (df['Pclass']==2)]['Age'].fillna(a[1][0],inplace=True)

df[(df['Gen_Clas']=='Other') & (df['Pclass']==3)]['Age'].fillna(value=a[2][0], inplace=True)

df[(df['Gen_Clas']=='Other') & (df['Pclass']==3)][['Age','Pclass','Gen_Clas']].head(10)



#fill_na=df[df['Gen_Clas']=='Mr'][['Age','Pclass']].groupby('Pclass').mean()

#df[(df['Gen_Clas']=='Mr') & (df['Pclass']==1)]['Age'].fillna(fill_na.loc[1],inplace=True)

#df[(df['Gen_Clas']=='Mr') & (df['Pclass']==2)]['Age'].fillna(fill_na.loc[2],inplace=True)

#df[(df['Gen_Clas']=='Mr') & (df['Pclass']==3)]['Age'].fillna(fill_na.loc[3],inplace=True)

#

#

#fill_na=df[df['Gen_Clas']=='Mrs'][['Age','Pclass']].groupby('Pclass').mean()

#df[(df['Gen_Clas']=='Mrs') & (df['Pclass']==1)]['Age'].fillna(fill_na.loc[1],inplace=True)

#df[(df['Gen_Clas']=='Mrs') & (df['Pclass']==2)]['Age'].fillna(fill_na.loc[2],inplace=True)

#df[(df['Gen_Clas']=='Mrs') & (df['Pclass']==3)]['Age'].fillna(fill_na.loc[3],inplace=True)

#

#

#

#fill_na=df[df['Gen_Clas']=='Miss'][['Age','Pclass']].groupby('Pclass').mean()

#df[(df['Gen_Clas']=='Miss') & (df['Pclass']==1)]['Age'].fillna(fill_na.loc[1],inplace=True)

#df[(df['Gen_Clas']=='Miss') & (df['Pclass']==2)]['Age'].fillna(fill_na.loc[2],inplace=True)

#df[(df['Gen_Clas']=='Miss') & (df['Pclass']==3)]['Age'].fillna(fill_na.loc[3],inplace=True)

#

#

#df[['Age','Gen_Clas','Pclass']]

fill_na=df[df['Gen_Clas']=='Mr'][['Age','Pclass']].groupby('Pclass').mean()

fill_na.loc[3]

#df[(df['Gen_Clas']=='Mr') & (df['Pclass']==1)]['Age'].fillna(fill_na.loc[1],inplace=True)

#df[(df['Gen_Clas']=='Mr') & (df['Pclass']==2)]['Age'].fillna(fill_na.loc[2],inplace=True)

df[(df['Gen_Clas']=='Mr') & (df['Pclass']==3)]['Age'].fillna(fill_na.loc[3],inplace=False).head(2)



print (df.info())
df[['Pclass', 'Survived']].groupby(['Pclass'],as_index=False,axis=0).mean()
df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

dataset[['FamilySize','Survived']].groupby(['FamilySize'], as_index=False).mean()

for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



#dataset.head(10)

#dataset.loc[dataset['FamilySize'] == 1, 'IsAlone']

#dataset['FamilySize']







dataset[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

dataset.head(2)

df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(df['Fare'].median())

    #print(dataset['Fare'])

    

df['CategoricalFare'] = pd.qcut(df['Fare'], 4)

#df

df[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean()
def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    #print(title_search,name)

    #If the title exists, extract and return it.

    

    if title_search:

        return title_search.group(1)

    return ""



for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)



#dataset

#pd.crosstab(df['Title'], df['Sex'])



dataset[['Title','Survived']].groupby('Title',as_index=False).mean()
def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    #print(title_search,name)

    #If the title exists, extract and return it.

    

    if title_search:

        return title_search.group(1)

    return ""



for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)



#dataset

pd.crosstab(df['Title'], df['Sex'])
for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


#train=pd.read_csv('/Users/vishnoiprem/Udemy/kaggle/titanic/train.csv', header = 0, dtype={'Age': np.float64})

train = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})





def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    #print(title_search,name)

    #If the title exists, extract and return it.

    

    if title_search:

        return title_search.group(1)

    return ""



for dataset in [train]:

    dataset['Title'] = dataset['Name'].apply(get_title)



for dataset in [train]:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

#train.head(10)

#pd.crosstab(df['Title'], df['Sex'])



for dataset in [train]:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



for dataset in [train]:

    age_avg   = dataset['Age'].mean()

    age_std   = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    #print(age_avg,age_std,age_null_count)

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    #print(age_null_random_list)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)







for dataset in [train]:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

 

 # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4





drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',\

                 'Parch']

train = train.drop(drop_elements, axis = 1)





train.head(10)
train
#train=pd.read_csv('/Users/vishnoiprem/Udemy/kaggle/titanic/train.csv')

#test=pd.read_csv('/Users/vishnoiprem/Udemy/kaggle/titanic/train.csv')

train= pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head(3)



sns.countplot(train['Survived'])
sns.factorplot('Pclass', 'Survived', data=train, hue='Sex')
train['Age'].fillna(train['Age'].median(), inplace=True)

sns.countplot(train['Survived'], hue=train['Sex'])
sns.countplot(train['Embarked'])
sns.boxplot(train['Survived'],train['Fare'], hue= train['Embarked'])
train[train['Embarked'].isnull()]