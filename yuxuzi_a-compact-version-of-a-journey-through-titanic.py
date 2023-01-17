# Import libraries

# pandas numpy, matplotlib, seaborn

import pandas as pd

from pandas import Series,DataFrame

pd.options.mode.chained_assignment = None 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
# Read data

train_data = pd.read_csv("../input/train.csv")

y_train=train_data.loc[:,"Survived"].values

train_df=train_data.drop("Survived",1)

test_df = pd.read_csv("../input/test.csv")



# preview the data

train_data.head()
# Analyze and visualize data

#-- Continuous variables

cols1=['Survived','Age','Parch','SibSp','Fare']

# View the correlation matrix with heatmap 

cm = np.corrcoef(train_data[cols1].values.T)

hm = sns.heatmap(cm,

    cbar=True,

    annot=True,

    square=True,

    cmap="YlGnBu",             

    fmt='.2f',

    annot_kws={'size': 15},

    yticklabels=cols1,

    xticklabels=cols1)
# -- Categorical variables



cols2=['Sex','Parch','SibSp','Pclass','Embarked']

# flatten compounded tuples

import itertools

flatten=lambda x: list(itertools.chain.from_iterable(x))



# factorplot

fig, axes = plt.subplots(3,2,figsize=(15,8))

fig.suptitle("Factor plot -- Survived vs Factors",fontsize=18)





for v, axis in zip(cols2,flatten(axes)):

    g=sns.factorplot(v,'Survived', data=train_data,ax=axis)

    plt.close(g.fig)

# countplot

fig, axes = plt.subplots(3,2,figsize=(15,8))

fig.suptitle("Count plot -- Count vs Factors",fontsize=18)

for v, axis in zip(cols2,flatten(axes)):

    sns.countplot(x=v,hue='Survived', data=train_data,ax=axis, palette="Set1")
# print data info

train_df.info()

print("----------------------------")

test_df.info()
# filling na

# only in titanic_df, fill the two missing values with the most occurred value, which is "S".

train_data["Embarked"] = train_data["Embarked"].fillna("S")

# only for test_df, since there is a missing "Fare" values

test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

# Age

# Filling na with generated random numbers between (mean - std) & (mean + std)

def fix_age(df):

    X=df["Age"]

    m=X.mean()

    s=X.std()

    n=X.isnull().sum()

    rand=np.random.randint(m-s, m+s,n)

    X[X.isnull()]=rand

    X.astype(int)

    return df



# Family

# Add a Family variable if the passenger had any family member aboard or not,

def fix_family(df):

    df['Family'] = np.where( df["Parch"] + df["SibSp"]>0,1,0)   

    return df

# Children(age < 16) on aboard seem to have a high chances for Survival.

def fix_person(df):

    df['Person']= np.where(df['Age']<16,'child',df['Sex'])

    return df

# chain the data preparation proceures together

def fix_data(df):

    selected=['Person','Age','Family', 'Fare','Pclass','Embarked']

    return ( df.pipe(fix_age)

               .pipe(fix_family)

               .pipe(fix_person)

               .pipe(lambda X: X[selected])

                # Covert categoricals into dummy variables

               .pipe(pd.get_dummies,

                     columns=['Pclass','Person', 'Embarked'],

                     drop_first=True

                     )

    )



    



X_train=train_df.pipe(fix_data).values

X_test =test_df.pipe(fix_data).values  
# Logistic Regression



logreg = LogisticRegression()



logreg.fit(X_train, y_train)



y_pred = logreg.predict(X_test)



logreg.score(X_train, y_train)
# Random Forests



random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, y_train)



y_pred = random_forest.predict(X_test)



random_forest.score(X_train, y_train)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": y_pred

    })

submission.to_csv('titanic.csv', index=False)