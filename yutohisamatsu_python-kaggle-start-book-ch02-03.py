import pandas as pd
import pandas_profiling


train = pd.read_csv('../input/titanic/train.csv')
train.profile_report()
import matplotlib.pyplot as plt
import seaborn as sns
plt.hist(train.loc[train['Survived'] == 0, 'Age'].dropna(), bins=30, alpha=0.5, label='0')
plt.hist(train.loc[train['Survived'] == 1, 'Age'].dropna(), bins=30, alpha=0.5, label='1')
plt.xlabel('Age')
plt.ylabel('count')
plt.legend(title='Survived')
plt.show()
sns.countplot(x='SibSp', hue='Survived', data=train)
plt.legend(loc='upper right', title='Survived')
plt.show()
sns.countplot(x='Parch', hue='Survived', data=train)
plt.legend(loc='upper right', title='Survived')
plt.show()
plt.hist(train.loc[train['Survived'] == 0, 'Fare'].dropna(),
         range=(0, 250), bins=25, alpha=0.5, label='0')
plt.hist(train.loc[train['Survived'] == 1, 'Fare'].dropna(),
         range=(0, 250), bins=25, alpha=0.5, label='1')
plt.xlabel('Fare')
plt.ylabel('count')
plt.legend(title='Survived')
plt.xlim(-5, 250)
plt.show()
sns.countplot(x='Pclass', hue='Survived', data=train)
plt.show()
sns.countplot(x='Sex', hue='Survived', data=train)
plt.show()
sns.countplot(x='Embarked', hue='Survived', data=train)
plt.show()
