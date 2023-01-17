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
test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")
full=test.append(train)

full['Title'] = full.Name.str.extract(', (.*?)\.')
full.groupby(['Sex','Title']).count()['PassengerId']
rare_title = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']

full.loc[full['Title'] == 'Mlle','Title'] = 'Miss'

full.loc[full['Title'] == 'Ms','Title'] = 'Miss'

full.loc[full['Title'] == 'Mme','Title'] = 'Mrs'

full.loc[full['Title'].apply(lambda x: x in rare_title),'Title'] = 'Rare Title'
full.groupby(['Sex','Title']).count()['PassengerId']
import re

full["Surname"] = full["Name"].apply(lambda x: re.split(",|\.",x)[0])

num_surnames = full["Surname"].drop_duplicates().count()

print("There are %d unique surnames." % num_surnames)
full['Fsize'] = full['SibSp'] + full['Parch'] + 1

full['Family'] = full['Surname'] + "_" +full['Fsize'].apply(lambda x:str(x))
import matplotlib.pyplot as plt

import matplotlib.axes as axes
fig, ax = plt.subplots()

width = 0.35

rects1 = ax.bar(full[full["Survived"] == 0].groupby('Fsize').count()["Surname"].index,full[full["Survived"] == 0].groupby('Fsize').count()["Surname"],width,color='c',edgecolor='w')

rects1 = ax.bar(full[full["Survived"] == 1].groupby('Fsize').count()["Surname"].index+width,full[full["Survived"] == 1].groupby('Fsize').count()["Surname"],width,color='y',edgecolor='w')

ax.set_xticks(range(1,13))

plt.legend(['Survived','Unsurvived'])
fig, ax = plt.subplots()

width = 0.35

rects1 = ax.bar(full[full["Survived"] == 0].groupby('Fsize').count()["Surname"].index,full[full["Survived"] == 0].groupby('Fsize').count()["Surname"]/891,width,color='c',edgecolor='w')

rects1 = ax.bar(full[full["Survived"] == 1].groupby('Fsize').count()["Surname"].index+width,full[full["Survived"] == 1].groupby('Fsize').count()["Surname"]/891,width,color='y',edgecolor='w')

ax.set_xticks(range(1,13))

plt.legend(['Survived','Unsurvived'])
full.loc[full['Fsize'] == 1, 'FsizeD'] = 'singleton'

full.loc[full["Fsize"].apply(lambda x:  x > 1 and x < 5), 'FsizeD'] = 'small'

full.loc[full['Fsize'] > 4, 'FsizeD'] = 'large'
from statsmodels.graphics.mosaicplot import mosaic

data_m = full.groupby(["FsizeD","Survived"]).count()

mosaic(data_m.PassengerId)
def deck_fun(x):

    if isinstance(x, str):

        return x[0]

    return x

full['Deck'] = full['Cabin'].apply(deck_fun)
full['Deck'].value_counts()
full['Deck'].value_counts().sort_index().plot(kind='bar')

plt.grid(False)
full[full["PassengerId"] == 62]

full[full["PassengerId"] == 830]
embark_fare = full[full['PassengerId'].notnull()]
import seaborn as sns

sns.boxplot(x='Embarked',y='Fare',hue='Pclass',data=embark_fare)
full['Deck'].value_counts().sort_index().plot(kind='bar')

plt.grid(False)
full[full["PassengerId"] == 62]

full[full["PassengerId"] == 830]
sns.pairplot(embark_fare[['Pclass',"Fare",'Embarked']], hue="Embarked")
full.loc[full["PassengerId"] == 62, 'Embarked'] = 'C'

full.loc[full["PassengerId"] == 830, 'Embarked'] = 'C'
sns.kdeplot(full.loc[(full['Pclass'] == 3) & (full['Embarked'] == 'S'),:]['Fare'].dropna())

plt.axvline(full.loc[(full['Pclass'] == 3) & (full['Embarked'] == 'S'),:]['Fare'].median(),color='y',ls='--')
sns.distplot(full.loc[(full['Pclass'] == 3) & (full['Embarked'] == 'S'),:]['Fare'].dropna(), hist=True, color = 'y',bins=50,kde_kws={"shade": True})

plt.axvline(full.loc[(full['Pclass'] == 3) & (full['Embarked'] == 'S'),:]['Fare'].median(),color='k',ls='--',linewidth=0.5)
full.loc[full['PassengerId'] == 1044] = full.loc[(full['Pclass'] == 3) & (full['Embarked'] == 'S'),:]['Fare'].median()
len(full.loc[full['Age'].isnull(),'Age'])
factor_vars = ['PassengerId','Pclass','Sex','Embarked','Title','Surname','Family','FsizeD']

d_factors = ['PassengerId','Name','Ticket','Cabin','Family','Surname','Survived']
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

fit_ans = imp.fit_transform(full[[i for i in full.columns if i not in (d_factors + ['Embarked', 'Sex', 'Title','FsizeD','Deck'])]])
plt.subplot(121)

full.Age.hist(bins=18)

plt.subplot(122)

plt.hist(fit_ans[:,0],bins=18)
full['Age_new'] = fit_ans[:,0]
uncompleted = full[full['Age'].isnull()].index

completed = full['Age'].dropna()

full['Age_new2'] = full['Age_new'].copy()

for i in uncompleted:

    value = completed.sample()

    full.loc[i,'Age_new2'] = value.values[0]

full['Age'] = full['Age_new2']
plt.subplot(121)

full.Age.hist(bins=18)

plt.subplot(122)

plt.hist(full.Age_new2,bins=18)
uncompleted = full[full['Deck'].isnull()].index

completed = full['Deck'].dropna()

full['Deck_new'] = full['Deck'].copy()

for i in uncompleted:

    value = completed.sample()

    full.loc[i,'Deck_new'] = value.values[0]

full['Deck'] = full['Deck_new']
plt.subplot(121)

plt.hist(full.loc[(full['Survived']==0) & (full['Sex']=="female"),:]['Age'],bins=30)

plt.hist(full.loc[full['Survived']==1 & (full['Sex']=="female"),:]['Age'],bins=30,alpha=0.5)

plt.legend(['Unsurvived','Survived'])

plt.title('Female')

plt.subplot(122)

plt.hist(full.loc[(full['Survived']==0) & (full['Sex']=="male"),:]['Age'],bins=30)

plt.hist(full.loc[full['Survived']==1 & (full['Sex']=="male"),:]['Age'],bins=30,alpha=0.5)

plt.legend(['Unsurvived','Survived'])

plt.title('Male')
full.loc[full['Age'] < 18,'Child'] = 'Child'

full.loc[full['Age'] >= 18,'Child'] = 'Adult'
full.loc[:,"Mother"] = 'Not Mother'

full.loc[(full['Sex'] == 'female') & (full['Parch'] > 0) & (full['Age'] > 18) & (full['Title'] != 'Miss'),"Mother"] = "Mother"

full = full.drop(152)

full = full.drop('Age_new',axis=1)

full = full.drop('Age_new2',axis=1)
full.index = full.PassengerId

embark_dummies_titanic  = pd.get_dummies(full['Embarked'])

embark_dummies_titanic.drop(['S'], axis=1, inplace=True)

full.drop(['Embarked'], axis=1,inplace=True)

full = full.join(embark_dummies_titanic)
sex_dummies_titanic  = pd.get_dummies(full['Sex'])

sex_dummies_titanic.drop(['female'], axis=1, inplace=True)

full.drop(['Sex'], axis=1,inplace=True)

full = full.join(sex_dummies_titanic)
fsized_dummies_titanic  = pd.get_dummies(full['FsizeD'])

fsized_dummies_titanic.drop(['singleton'], axis=1, inplace=True)

full = full.join(fsized_dummies_titanic)

full.drop(['FsizeD'], axis=1,inplace=True)
child_dummies_titanic  = pd.get_dummies(full['Child'])

child_dummies_titanic.drop(['Child'], axis=1, inplace=True)

full.drop(['Child'], axis=1,inplace=True)

full = full.join(child_dummies_titanic)
mother_dummies_titanic  = pd.get_dummies(full['Mother'])

mother_dummies_titanic.drop(['Mother'], axis=1, inplace=True)

full = full.join(mother_dummies_titanic)

full.drop(['Mother'], axis=1,inplace=True)
train = full.loc[full['Survived'].notnull(),['Pclass','male','Age','SibSp','Parch','Fare','C','Q','Adult','Not Mother']]
test_0 = full.loc[full['Survived'].isnull(),['Pclass','male','Age','SibSp','Parch','Fare','C','Q','Adult','Not Mother',"PassengerId"]]

test = test_0.drop('PassengerId',axis=1)
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(train, full.loc[full['Survived'].notnull(),'Survived'])
Y_pred = random_forest.predict(test)
random_forest.score(train, full.loc[full['Survived'].notnull(),'Survived'])
submission = pd.DataFrame({

        "PassengerId": test_0.loc[:,"PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)