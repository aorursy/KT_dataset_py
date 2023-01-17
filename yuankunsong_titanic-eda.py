import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

train_source = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')

test_source = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')


############################
# Make new feature 'title' #
############################

# extract title from name
title_list = train_source.Name.str.extract('([A-Za-z]+)\.', expand=False)
train_source['title'] = title_list
test_source['title'] = title_list

# change some to Miss and Mrs
train_source['title'] = train_source['title'].replace('Mlle','Miss')
train_source['title'] = train_source['title'].replace('Ms','Miss')
train_source['title'] = train_source['title'].replace('Mme','Mrs')

test_source['title'] = test_source['title'].replace('Mlle','Miss')
test_source['title'] = test_source['title'].replace('Ms','Miss')
test_source['title'] = test_source['title'].replace('Mme','Mrs')

# combine title with small counts to 'Other'
small_title = train_source.title.value_counts().index[train_source.title.value_counts() < 5]

train_source['title'] = train_source['title'].replace(small_title,'Other')
test_source['title'] = test_source['title'].replace(small_title,'Other')

train_source['title'] = train_source['title'].replace('Dr','Other')
test_source['title'] = test_source['title'].replace('Dr','Other')


train = train_source
test = test_source

train
# survival count
sns.countplot(train_source.Survived)
plt.title('Survival Count')
plt.ylabel('Count')
plt.show()
# count by gender
sns.countplot(train.Sex)
plt.title('Gender Count')
plt.ylabel('Count')
plt.show()
# survival count by gender
sns.countplot(train.Survived, hue=train.Sex)
plt.title('Survival count by gender')
plt.ylabel('Count')
plt.show()
# survival count by title
print(pd.DataFrame(train_source[['title','Survived']].groupby('title').mean()))

sns.countplot(train_source.Survived, hue = train_source.title)
plt.show()
# survival count by embarked
sns.countplot(train.Embarked, hue=train.Survived)
plt.title('Survival count by city embarked')
plt.ylabel('Count')
plt.show()
# distribution of age
sns.distplot(train.Age)
plt.show()
# plot age vs survived
aa = sns.FacetGrid(train,col = 'Survived')
aa.map(plt.hist, 'Age', bins = 20)
plt.show()
# siblings count
sns.countplot(train.SibSp)
plt.title('Siblings Count')
plt.show()
#print(train.SibSp.value_counts())

# surviveal by sibling count
sns.countplot(train.Survived, hue=train.SibSp)
plt.title('Survival by Sibling')
plt.legend(loc='upper right')
plt.show()
# parent/children count
sns.countplot(train.Parch)
plt.title('Par/Ch Count')
plt.show()
#print(train.Parch.value_counts())

# surviveal by parent/children count
sns.countplot(train.Survived, hue=train.Parch)
plt.title('Survival by Par/Ch')
plt.legend(loc='upper right')
plt.show()
# by class
sns.countplot(train.Pclass)
plt.title('Count by class')
plt.show()

sns.countplot(train.Survived, hue = train.Pclass)
plt.title('Survival Count by class')
plt.show()
# age by class / survived
aa = sns.FacetGrid(train, col = 'Survived', row = 'Pclass')
aa.map(plt.hist, 'Age', bins = 20)
plt.show()
# histogram for fare
sns.distplot(train.Fare)
plt.title('Distribution for Fare')
plt.show()

# scatterplot for fare and class
sns.swarmplot(x=train.Pclass, y=train.Fare)
plt.show()
corr = train.corr()

plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, square=True)

plt.show()