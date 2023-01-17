import pandas as pd

import seaborn as sns

import numpy as np
# import dataset 

df_train = pd.read_csv("../input/titanic/train.csv")

df_train.head(10)
# age distribution

df_train['Age'].sort_index().plot.hist()
df_train['Pclass'].value_counts().sort_index().plot.bar()
grp_class=df_train.groupby('Pclass').Survived.sum().reset_index()

sp=sns.barplot(

    x='Pclass',

    y='Survived',

    data=grp_class)

sp
grp_gender=df_train.groupby('Sex').Survived.sum().reset_index()

sp1=sns.barplot(

    x='Sex',

    y='Survived',

    data=grp_gender)

sp1
# age, survived and sex

sns.lmplot(x='Age', y='Survived', hue='Sex', 

           data=df_train, 

           fit_reg=False)