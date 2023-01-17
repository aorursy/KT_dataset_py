%matplotlib inline
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('dark')

plt.style.use('fivethirtyeight')
titanic = pd.read_csv("../input/titanic/train.csv", sep=',', index_col='PassengerId', parse_dates=True)
titanic.head(10)
titanic.shape
group_by_sex = titanic.groupby('Sex').mean()

group_by_sex
plt.figure(figsize=(10,8))

plt.title('Survival rate ')

sns.countplot('Survived', data=titanic, hue='Survived')

plt.show()
plt.figure(figsize=(10,8))

plt.title('Survival rate based on sex ')

sns.barplot(x='Sex', y='Survived', data=titanic, hue='Sex')

plt.show()
plt.figure(figsize=(10,8))

plt.title('Gender Age Distribution')

sns.barplot(x='Survived', y='Age', data=titanic, hue='Survived')
#locate the male passengers and find all who survived

male_percent_survived = titanic.loc[(titanic.Sex == 'male') & (titanic.Survived > 0)]

male_percent_survived
#Which passenger class had the most survivals

male_percent_survived.Pclass.value_counts()
plt.figure(figsize=(8,6))

plt.title('Ranking of male that survived based on their passengerclass')

sns.countplot(x=male_percent_survived.Pclass, data=male_percent_survived, hue='Survived')
#Which Age had the most survivals

male_percent_survived.Age.value_counts()
#taking passengers from ages 1 and above

male_age_above_092 = male_percent_survived.loc[male_percent_survived.Age > 0.92]

male_age_above_092
plt.figure(figsize=(30,15))

plt.title('Ranking of male that survived based on their ages')

sns.countplot(x=male_age_above_092.Age.astype(int) , data=male_percent_survived, hue='Survived')