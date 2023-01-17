# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

from imblearn.over_sampling import RandomOverSampler

from imblearn.under_sampling import TomekLinks

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier



import warnings

warnings.filterwarnings("default", category=DeprecationWarning)

%matplotlib inline
titanic_df = pd.read_csv("../input/train.csv")

titanic_df.head()
titanic_df.info()
titanic_df.groupby('Sex')['Sex'].count()
sns.catplot('Sex',data=titanic_df,kind="count")

# shows that there are more male passengers as compared to female passangers
titanic_df.groupby('Pclass').count()
# Distribution of males and females based on pclass and place of embarkment

titanic_df.groupby(['Pclass','Sex','Embarked']).size()
print (titanic_df.groupby(['Pclass','Sex']).size())

sex_age_df = pd.crosstab(index=[titanic_df['Pclass']], columns=[titanic_df['Sex']])

sex_age_df
sns.catplot('Pclass',data=titanic_df,kind='count',hue='Sex')
def male_female_child(passenger):

    age,sex = passenger

    

    if age < 16:

        return 'child'

    else:

        return sex
titanic_df['Person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis = 1)

titanic_df.head(8)
titanic_df.groupby(['Pclass','Person']).size()
sns.catplot('Pclass',data=titanic_df,kind='count',hue='Person')
# Ages of people 

titanic_df['Age'].hist(bins = 70)

# Mean is around 30
titanic_df['Age'].mean()
titanic_df['Person'].value_counts()
fig = sns.FacetGrid(titanic_df,hue='Sex',aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)



oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()
fig = sns.FacetGrid(titanic_df,hue='Person',aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()



fig.set(xlim=(0,oldest))

fig.add_legend()
# Distribution of female ages

titanic_df[titanic_df['Sex'] == 'female']['Age'].hist(bins = 70)
# Distribution of male ages

titanic_df[titanic_df['Sex'] == 'male']['Age'].hist(bins = 70)
#How age is distributed w.r.t. to class

fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()



fig.set(xlim=(0,oldest))

fig.add_legend()
# Q2 What deck the passengers on and how does it relate to their class?

titanic_df.head()
# Percentage of Null Values in all columns =

p_null= (titanic_df.isnull().sum()/len(titanic_df)) * 100

p_null
sns.catplot('Pclass', data=titanic_df, kind='count', hue='Person', order=[1,2,3], 

               hue_order=['child','female','male'], aspect=1)
# Do the same as above, but split the passengers into either survived or not

sns.catplot('Pclass', data=titanic_df, kind='count', hue='Person', col='Survived', order=[1,2,3], 

               hue_order=['child','female','male'], aspect=1, height=5)
#Where did the passengers come from i.e. Where did the passengers land into the ship from?

sns.catplot('Embarked', data=titanic_df, kind='count', hue='Pclass')
titanic_df.groupby(['Embarked','Pclass']).size()
titanic_df.Embarked.value_counts()
embarked_vs_pclass = pd.crosstab(index = [titanic_df['Embarked']], columns=[titanic_df['Pclass']],margins=True)

embarked_vs_pclass
def alone_with_family(passenger):

    parch,sibsp = passenger

    if (parch == 0) & (sibsp == 0):

        return 'alone'

    else:

        return 'with_family'
titanic_df['alone_or_with_family'] = titanic_df[['Parch','SibSp']].apply(alone_with_family,axis = 1)

titanic_df.head()
titanic_df['alone_or_with_family'].value_counts()
fg=sns.catplot('alone_or_with_family', data=titanic_df, kind='count', hue='Pclass', col='Person')
pd.crosstab(index = [titanic_df['alone_or_with_family'],titanic_df['Person']], columns=[titanic_df['Pclass']],margins=True)
def titanic_preprocessing(train, test):

    train_df = pd.read_csv("../input/"+train)

    test_df = pd.read_csv("../input/"+test)

    combine = [train_df, test_df]

    

    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

    combine = [train_df, test_df]

    

    for dataset in combine:

        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+).', expand=False)

    

    Rare_Titles = set(train_df['Title'].unique()) - set(['Mlle','Ms','Mme','Mr','Master','Miss','Mrs'])

    

    for dataset in combine:

        dataset['Title'] = dataset['Title'].replace(Rare_Titles, 'Rare', regex = True)                                                 

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

    pd.crosstab(train_df['Title'], train_df['Sex'])

    

    temp = train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

    

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    for dataset in combine:

        dataset['Title'] = dataset['Title'].map(title_mapping)

        dataset['Title'] = dataset['Title'].fillna(0)

        

    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

    test_df = test_df.drop(['Name'], axis=1)

    combine = [train_df, test_df]



    for dataset in combine:

        dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    

    guess_ages = np.zeros((2,3))



    for dataset in combine:

        for i in range(0, 2):

            for j in range(0, 3):

                guess_df = dataset[(dataset['Sex'] == i) & \

                                      (dataset['Pclass'] == j+1)]['Age'].dropna()



                age_guess = guess_df.median()

                

                guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

        

        for i in range(0, 2):

            for j in range(0, 3):

                dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                        'Age'] = guess_ages[i,j]



        dataset['Age'] = dataset['Age'].astype(int)



    for dataset in combine:

        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1







    for dataset in combine:

        dataset['IsAlone'] = 0

        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



    combine = [train_df, test_df]



    freq_port = train_df.Embarked.dropna().mode()[0]



    for dataset in combine:

        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)



    test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

    

    train_df = pd.get_dummies(train_df)

    test_df  = pd.get_dummies(test_df)



    print(test_df.head(10))

    return (train_df, test_df)
train = 'train.csv'

test = 'test.csv'

titanic_train,titanic_test = titanic_preprocessing(train, test)
print(titanic_test.shape, titanic_train.shape)

titanic_train.head()
titanic_train.head()
from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth=3)
X_train = titanic_train.iloc[:,1:]

Y_train = titanic_train['Survived']
Y_train.head()
print(X_train.shape)

print(X_train.columns)
clf = clf.fit(X_train,Y_train)
from sklearn.metrics import accuracy_score

Y_pred = clf.predict(titanic_test.iloc[:,1:])

df1 = pd.DataFrame(np.array([titanic_test.iloc[:,0],Y_pred]).T,columns=['PassengerId','Survived'])

df1.head()
def fit_predict_display_plot(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    dispaly_model_parameters(y_test, y_pred)

    

    probs = model.predict_proba(X_test)  

    probs = probs[:, 1]  

    plot_roc_curve(y_test, probs)

    

def dispaly_model_parameters(y_test, y_pred):

        print("Confusion Matrix: \n", metrics.confusion_matrix(y_test, y_pred))

        print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))

        print("Precision: ",metrics.precision_score(y_test, y_pred))

        print("Recall: ",metrics.recall_score(y_test, y_pred))

        print("f1 score: ",metrics.f1_score(y_test, y_pred))

        print("roc_auc_score: ",metrics.roc_auc_score(y_test, y_pred))

        print("classification_report: \n",metrics.classification_report(y_test, y_pred))

        print(pd.crosstab(y_test, y_pred))

        

def plot_roc_curve(y_test, probs):

    fpr, tpr, threshold = metrics.roc_curve(y_test, probs)

    roc_auc = metrics.auc(fpr, tpr)

    # method I: plt

    import matplotlib.pyplot as plt

    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()