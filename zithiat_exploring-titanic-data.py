import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sb

plt.style.use('fivethirtyeight')



#ignore warnings

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
#import train and test CSV files

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



#take a look at the training data

train.describe(include="all")

train.shape

train.head(10)
test.head()
#get a list of the features within the dataset

print(train.columns)
train.isnull().sum()
sb.countplot('Survived',data=train)

plt.show()
train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()

sb.countplot('Sex', hue='Survived', data=train,)

plt.show()
plt.scatter(train['Pclass'], train['Age'])

plt.show()
plt.scatter(train['Survived'], train['Age'])

plt.show()
train.boxplot(column='Age', by=['Survived', 'Sex'])
train.boxplot(column='Age', by=['Survived','Pclass'])
pd.crosstab(train.SibSp,train.Pclass).style.background_gradient('summer_r')
from sklearn.preprocessing import LabelBinarizer

one_hot = LabelBinarizer()

one_hot.fit_transform(train['Pclass'])
one_hot.classes_
train.info()
fig, ax = plt.subplots(figsize=(9,5))

sb.heatmap(train.isnull(), cbar=False, cmap="YlGnBu_r")

plt.show()
cols = ['Survived', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']

nr_rows = 2

nr_cols = 3



fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))



for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        

        i = r*nr_cols+c       

        ax = axs[r][c]

        sb.countplot(train[cols[i]], hue=train["Survived"], ax=ax)

        ax.set_title(cols[i], fontsize=14, fontweight='bold')

        ax.legend(title="survived", loc='upper center') 

        

plt.tight_layout()   
bins = np.arange(0, 80, 5)

g = sb.FacetGrid(train, row='Sex', col='Pclass', hue='Survived', margin_titles=True, size=3, aspect=1.1)

g.map(sb.distplot, 'Age', kde=False, bins=bins, hist_kws=dict(alpha=0.6))

g.add_legend()  

plt.show()  
train['Fare'].max()
bins = np.arange(0, 550, 50)

g = sb.FacetGrid(train, row='Sex', col='Pclass', hue='Survived', margin_titles=True, size=3, aspect=1.1)

g.map(sb.distplot, 'Fare', kde=False, bins=bins, hist_kws=dict(alpha=0.6))

g.add_legend()  

plt.show()
sb.barplot(x='Pclass', y='Survived', data=train)

plt.ylabel("Survival Rate")

plt.title("Survival as function of Pclass")

plt.show()
sb.barplot(x='Sex', y='Survived', hue='Pclass', data=train)

plt.ylabel("Survival Rate")

plt.title("Survival between Pclass and Sex")

plt.show()
sb.barplot(x='Embarked', y='Survived', data=train)

plt.ylabel("Survival Rate")

plt.title("Survival as function of Embarked Port")

plt.show()
sb.barplot(x='Embarked', y='Survived', hue='Pclass', data=train)

plt.ylabel("Survival Rate")

plt.title("Survival as function of Embarked Port")

plt.show()
sb.countplot(x='Embarked', hue='Pclass', data=train)

plt.title("Count of Passengers as function of Embarked Port")

plt.show()
sb.boxplot(x='Embarked', y='Age', data=train)

plt.title("Age distribution as function of Embarked Port")

plt.show()
sb.boxplot(x='Embarked', y='Fare', data=train)

plt.title("Fare distribution as function of Embarked Port")

plt.show()
cm_surv = ["darkgrey" , "lightgreen"]

fig, ax = plt.subplots(figsize=(13,7))

sb.swarmplot(x='Pclass', y='Age', hue='Survived', split=True, data=train , palette=cm_surv, size=7, ax=ax)

plt.title('Survivals for Age and Pclass ')

plt.show()
fig, ax = plt.subplots(figsize=(13,7))

sb.violinplot(x="Pclass", y="Age", hue='Survived', data=train, split=True, bw=0.05 , palette=cm_surv, ax=ax)

plt.title('Survivals for Age and Pclass ')

plt.show()
g = sb.factorplot(x="Pclass", y="Age", hue="Survived", col="Sex", data=train, kind="swarm", split=True, palette=cm_surv, size=7, aspect=.9, s=7)
g = sb.factorplot(x="Pclass", y="Age", hue="Survived", col="Sex", data=train, kind="violin", split=True, bw=0.05, palette=cm_surv, size=7, aspect=.9, s=7)