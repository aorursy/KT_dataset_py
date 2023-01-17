import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from matplotlib import pyplot as plt # Graf

import seaborn as sns



from sklearn.preprocessing import LabelEncoder # Encoder



SEED = 1



import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

train.head()
test = pd.read_csv('../input/test.csv')

test.head()
test.iloc[414]
pd.Series('Oliva y Ocana, Dona. Fermina').str.extract('([A-Za-z]+)\.', expand=False)
train.info()
datasets = [train,test]
for df in datasets:

    df['Title'] = df.Name.str.extract('([A-Za-z]+)\.', expand=False)

    

train.Title.value_counts(dropna=False)
train.groupby('Title',as_index=False)['Survived'].mean().sort_values('Survived',ascending = False)
def encTitle(title):

    title = str(title)

    if title in ('Jonkheer','Don','Rev','Capt'):

        return 1

    elif title in ('Mr'):

        return 2

    elif title in ('Dr','Major','Col','Master'):

        return 3

    elif title in ('Miss','Mrs','Dona'):

        return 4

    elif title in ('Ms','Mme','Sir'):

        return 9

    elif title in ('Mlle','Lady','Countess'):

        return 10
for df in datasets:

    df['TitleEnc'] = df['Title'].apply(encTitle)

    

train.TitleEnc.value_counts(dropna=False)
for df in datasets:

    df['hasCabin'] = np.where(pd.isnull(df['Cabin']),0,1)

    df.loc[pd.isnull(df['Embarked']),'Embarked'] = 'None'

    df.drop(['Title','Name','Ticket','Cabin'],axis=1,inplace=True)

    

train.head()
np.random.seed(SEED)

le = dict()

le['Sex'] = LabelEncoder()

le['Sex'].fit(train.Sex)

le['Embarked'] = LabelEncoder()

le['Embarked'].fit(train.Embarked)



for df in datasets:

    df['Sex'] = le['Sex'].transform(df['Sex'])

    df['Embarked'] = le['Embarked'].transform(df['Embarked'])

    

train.head()
for df in datasets:

    df['Family'] = np.where((df['SibSp'] > 1),0,np.where((df['Parch'] > 1),2,1))

    

print('Sobreviventes:\n',train.groupby(['Family'])['Survived'].mean())
titles = train.TitleEnc.unique()

family = train.Family.unique()

titles.sort()

family.sort()



for f in family:

    for title in titles:

        for df in datasets:

            df.loc[(pd.isnull(df['Age'])) & 

                   (df['TitleEnc'] == title) &

                   (df['Family'] == f), 'Age'] = df[(df['TitleEnc'] == title) & (df['Family'] == f)]['Age'].mean()



for df in datasets:

    df.loc[:,'Age'] = np.round(df['Age'])

            

train.info()
for df in datasets:

    df.loc[pd.isnull(df['Fare']),'Fare'] = df['Fare'].mean()
plt.figure(figsize=(12,8))

sns.heatmap(train.corr(), annot=True)

plt.show()
print('Sobreviventes:\n',train.groupby(['TitleEnc'])['Survived'].mean())
plt.figure(figsize=(12,8))

sns.heatmap(train.corr(), annot=True)

plt.show()
def ageGroup(age):

    if age < 18:

        return 0.5

    if age < 60:

        return 0.36

    return 0.27
for df in datasets:

    df['AgeGroup'] = df['Age'].apply(ageGroup)

    

print('Sobreviventes:\n',train.groupby(['AgeGroup'])['Survived'].mean())
for df in datasets:

    df.drop(['Age','SibSp','Parch'],axis=1,inplace=True)

    

train.head()
plt.figure(figsize=(12,8))

sns.heatmap(train.corr(), annot=True)

plt.show()
train.describe()
def show_results(results):

  media = results['test_score'].mean()

  desvio_padrao = results['test_score'].std()

  print("mean Accuracy: %.2f" % (media * 100))

  print("Accuracy: [%.2f, %.2f]" % ((media - 2 * desvio_padrao)*100, (media + 2 * desvio_padrao) * 100))
x_train = train.drop(['PassengerId','Survived'],axis=1)

y_train = train['Survived']
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_validate

from sklearn.model_selection import GroupKFold

from sklearn.preprocessing import StandardScaler



tree = DecisionTreeClassifier(max_depth=3,random_state=0)

cv = GroupKFold(n_splits = 2)

results = cross_validate(tree, x_train, y_train, cv = cv, groups = train.TitleEnc.values, return_train_score=False)

show_results(results)
tree.fit(x_train,y_train)



print(tree.score(x_train,y_train))
import graphviz

from sklearn.tree import export_graphviz



features = x_train.columns



dot_data = export_graphviz(tree, out_file=None, filled = True, rounded = True,

                           feature_names = features,

                          class_names = ["Morre", "Sobrevive"])

graph = graphviz.Source(dot_data)

graph
test.info()
x_test = test.drop(['PassengerId'],axis=1)

y_pred = tree.predict(x_test)



my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_pred})



my_submission.head()
my_submission.to_csv('submission.csv', index=False)