import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df= pd.read_csv('../input/train.csv')
df.head()
#1 axis x and y

sns.catplot(x='Pclass', y='Survived', data=df, size=5)

plt.show()
#2 choose type of figure 'Bar'

sns.catplot(x='Pclass', y= 'Survived', kind='bar', data= df)

#3 choose type of figure 'Bar', non the confidence interval

sns.catplot(x='Pclass', y= 'Survived', kind='bar', data= df, ci=None)
#4 choose type of figure 'Bar', with Hue Sex male or femal

sns.catplot(x='Pclass', y= 'Survived', hue= 'Sex', kind='bar', data= df, ci=None)
#5 choose type of figure 'Bar', with Col Sex male or femal

sns.catplot(x='Pclass', y= 'Survived', col= 'Sex', kind='bar', data= df, ci=None)
#6 choose type of figure 'Bar', with hue (Sex male or femal) and  col (embarked)

sns.catplot(x='Pclass', y= 'Survived', hue='Sex', col= 'Embarked', kind='bar', data= df, ci=None)
#7 choose type of figure 'Bar', cross Sex and Embarked

sns.catplot(x='Pclass', y= 'Survived', row='Sex', col= 'Embarked', kind='bar', data= df, ci=None)
#8 choose type of figure 'violin'

sns.catplot(x='Survived', y= 'Age', hue='Sex',kind='violin', data= df, size=6)
#9 choose type of figure 'box'

sns.catplot(x='Sex', y= 'Age',kind='box', data= df, size=6)

plt.show()