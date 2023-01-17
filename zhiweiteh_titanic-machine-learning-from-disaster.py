import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import missingno

import seaborn as sns



%matplotlib inline



#pd.set_option('display.max_rows', None)
train = pd.read_csv('../input/titanic/train.csv')

#train = pd.read_csv('data/train.csv')

train.head()
test = pd.read_csv('../input/titanic/test.csv')

#test = pd.read_csv('data/test.csv')

test.head()
combined_data = [train, test]
### creating dataframe to hold our pre-processed data

df_train = pd.DataFrame()

df_train['Survived'] = train['Survived']

df_test = pd.DataFrame()

combined_df = [df_train, df_test]
for dataset in combined_data:

    print(dataset.info())

    print(dataset.isnull().sum())
for dataset in combined_data:

    title = dataset.Name.str.extract(r' ([A-Za-z]+)\. ', expand=False)

    dataset['title'] = title

    dataset.drop('Name', axis=1, inplace=True)

    print(pd.crosstab(dataset['title'], dataset['Sex']))
for dataset in combined_data:

    dataset['title'] = dataset['title'].replace(['Dr', 'Rev', 'Col', 'Major',  

                                         'Capt', 'Sir', 'Don', 'Lady', 

                                         'Jonkheer', 'Countess', 'Dona'], 

                                        "Rare")

    dataset['title'] = dataset['title'].replace({

                        'Mlle': 'Miss',

                        'Ms': 'Miss',

                        'Mme': 'Mrs'})

    print(dataset['title'].value_counts())
plt.figure(figsize = (30, 10))

sns.set(font_scale=2)

sns.regplot(x=train['title'], y=train['Age'], fit_reg=False)
for dataset in combined_data:

    for title in train['title'].unique():

        median_age = dataset.Age[train['title'] == title].median()

        print('{0} median age: {1}'.format(title, median_age))



        dataset.Age[(dataset['title'] == title) & dataset['Age'].isnull()] = median_age

    print('\n')  

train.Embarked.value_counts()
train['Embarked'] = train['Embarked'].fillna('S')
# extracting deck from Cabin

for dataset in combined_data:

    dataset['cabin_deck'] = dataset['Cabin'].str.get(0)

    #print(dataset.head())
for dataset in combined_data:

    print(dataset['cabin_deck'].value_counts())
# Passenger in the T deck is changed to A

idx = train[train['cabin_deck'] == 'T'].index

train.loc[idx, 'cabin_deck'] = 'A'
for dataset in combined_data:

    dataset['cabin_deck'] = dataset['cabin_deck'].fillna('Z')

    dataset.drop('Cabin', axis=1, inplace=True)
# TODO: get distribution of passengers in each deck

print(pd.crosstab(train['cabin_deck'], train['Pclass']))
#train[['Survived', 'title']].groupby(['title'], as_index = False).mean()

train[['Fare', 'Pclass', 'Embarked']].groupby(['Pclass', 'Embarked'], as_index = False).mean()
test[test['Fare'].isnull()]
test['Fare'] = test['Fare'].fillna(14.64)
for dataset in combined_data:

    print(dataset.info())
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean()
df_train['Pclass'] = train['Pclass']

df_test['Pclass'] = test['Pclass']
train[['Survived', 'title']].groupby(['title'], as_index = False).mean()
df_train['title'] = train['title']

df_test['title'] = test['title']
train[['Survived', 'Sex']].groupby(['Sex'], as_index = False).mean()
df_train['Sex'] = train['Sex']

df_test['Sex'] = test['Sex']
sns.distplot(train.Age)
plt.figure(figsize = (30, 10))

sns.set(font_scale=2)

sns.violinplot(x=train['Survived'], y=train['Age'])
for i in range (3, 8):

    train['Categorical_Age'] = pd.cut(train['Age'], i)

    print(train[['Categorical_Age', 'Survived']].groupby(['Categorical_Age'], as_index=False).mean())
train.drop('Categorical_Age', axis=1, inplace=True)
for dataset in combined_data:

    dataset['Categorical_Age'] = 0

    dataset.loc[(dataset['Age'] > 16.0) & (dataset['Age'] <= 32.0), 'Categorical_Age'] = 1

    dataset.loc[(dataset['Age'] > 32.0) & (dataset['Age'] <= 48.0), 'Categorical_Age'] = 2

    dataset.loc[(dataset['Age'] > 48.0) & (dataset['Age'] <= 64.0), 'Categorical_Age'] = 3

    dataset.loc[(dataset['Age'] > 64.0), 'Categorical_Age'] = 4
df_train['Categorical_Age'] = train['Categorical_Age']

df_test['Categorical_Age'] = test['Categorical_Age']

print(df_train.info())

print(df_test.info())
sns.distplot(train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean())
sns.distplot(train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean())
# we add 1 to account for the passenger himself

for dataset in combined_data:

    dataset['Family_size'] = dataset['SibSp'] + dataset['Parch'] + 1

    print(dataset['Family_size'].value_counts())
sns.distplot(train[['Family_size', 'Survived']].groupby(['Family_size'], as_index=False).mean())
temp_family_size = train[['Family_size', 'Survived']]

for i in range (2, 6):

    temp_family_size['Categorical_Fam_size'] = pd.cut(train['Family_size'], i)

    print(temp_family_size[['Categorical_Fam_size', 'Survived']].groupby(['Categorical_Fam_size'], as_index=False).mean())
for dataset in combined_data:

    dataset['Categorical_Fam_size'] = 0

    dataset.loc[(dataset['Family_size'] > 3.0) & (dataset['Family_size'] <= 5.0), 'Categorical_Fam_size'] = 1

    dataset.loc[(dataset['Family_size'] > 5.0), 'Categorical_Fam_size'] = 2

    print(dataset['Categorical_Fam_size'].value_counts())
df_train['Categorical_Fam_size'] = train['Categorical_Fam_size']

df_test['Categorical_Fam_size'] = test['Categorical_Fam_size']

print(df_train.info())

print(df_test.info())
sns.distplot(train.Fare)
sns.distplot(train[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean())
for i in range (2, 8):

    train['Categorical_Fare'] = pd.qcut(train['Fare'], i)

    print(train[['Categorical_Fare', 'Survived']].groupby(['Categorical_Fare'], as_index=False).mean())

    sns.catplot(x='Categorical_Fare', y='Survived', kind="bar", data=train)
sns.catplot(x='Categorical_Fare', y='Survived', kind="bar", data=train)
for dataset in combined_data:

    dataset['Categorical_Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Categorical_Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31.0), 'Categorical_Fare'] = 2

    dataset.loc[(dataset['Fare'] > 31.0), 'Categorical_Fare'] = 3

    print(dataset['Categorical_Fare'].value_counts())
df_train['Categorical_Fare'] = train['Categorical_Fare']

df_test['Categorical_Fare'] = test['Categorical_Fare']
train[['Survived', 'cabin_deck']].groupby(['cabin_deck'], as_index = False).mean()
sns.catplot(x='cabin_deck', y='Survived', kind="bar", data=train)
cabin_deck_category ={

    'A': 1,

    'B': 1,

    'C': 1,

    'D': 1,

    'E': 2,

    'F': 2,

    'G': 3,

    'Z': 4

}
for dataset in combined_data:

    dataset['Categorical_cabin_deck'] = dataset['cabin_deck'].map(cabin_deck_category)

    

train[['Survived', 'Categorical_cabin_deck']].groupby(['Categorical_cabin_deck'], as_index = False).mean()
df_train['Categorical_cabin_deck'] = train['Categorical_cabin_deck']

df_test['Categorical_cabin_deck'] = test['Categorical_cabin_deck']
g = sns.FacetGrid(train, col='Pclass', height=4, aspect=.5)

g.map(sns.barplot, "Embarked", "Survived")
g = sns.FacetGrid(train, col='Embarked', height=4, aspect=0.6)

g.map(sns.barplot, "Pclass", "Survived")
train.groupby('Embarked')['Survived'].mean()
df_train['Embarked'] = train['Embarked']

df_test['Embarked'] = test['Embarked']
for dataset in combined_df:

    print(dataset.info())
from sklearn.preprocessing import LabelEncoder
one_hot_title = pd.get_dummies(df_train['title'], prefix = 'title')

one_hot_sex = pd.get_dummies(df_train['Sex'], prefix = 'Sex')

one_hot_embarked = pd.get_dummies(df_train['Embarked'], prefix = 'Embarked')



df_train_enc = pd.concat([df_train,

                     one_hot_title,

                     one_hot_sex,

                     one_hot_embarked], axis = 1)



one_hot_title = pd.get_dummies(df_test['title'], prefix = 'title')

one_hot_sex = pd.get_dummies(df_test['Sex'], prefix = 'Sex')

one_hot_embarked = pd.get_dummies(df_test['Embarked'], prefix = 'Embarked')



df_test_enc = pd.concat([df_test,

                     one_hot_title,

                     one_hot_sex,

                     one_hot_embarked], axis = 1)
combined_df_enc = [df_train_enc, df_test_enc]
# we will drop the original features that were hot-encoded and 

# one of the newly encoded categories for each feature to avoid

# dummy variable trap

features_to_drop = ['title', 'Sex', 

                    #'Combined_pclass_embarked',

                    'Embarked',

                   'title_Rare', 'Sex_male', 'Embarked_C']



for dataset in combined_df_enc:

    dataset.drop(features_to_drop, axis = 1, inplace=True)
print(df_train_enc.info(), df_test_enc.info())
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
X_train = df_train_enc.drop('Survived', axis=1)

y_train = df_train_enc.Survived
X_train.shape
y_train.shape
df_test_enc.shape
classifiers = [

    KNeighborsClassifier(),

    SVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]



log_cols = ["Classifier", "Accuracy"]

log = pd.DataFrame(columns = log_cols)
acc_dict = {}



for classifier in classifiers:

    scores = cross_val_score(classifier, X_train, y_train, cv = 10, scoring = 'accuracy')

    name = classifier.__class__.__name__

    acc_dict[name] = scores.mean()

    

acc_dict
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)] 

max_features = ['auto', 'sqrt'] 

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)] 

max_depth.append(None) 

min_samples_split = [2, 5, 10] 

min_samples_leaf = [1, 2, 4] 

bootstrap = [True, False]



random_grid = {'n_estimators': n_estimators, 

               'max_features': max_features, 

               'max_depth': max_depth, 

               'min_samples_split': min_samples_split, 

               'min_samples_leaf': min_samples_leaf, 

               'bootstrap': bootstrap}



rf = RandomForestClassifier()



rf_random = RandomizedSearchCV(estimator = rf, 

                               param_distributions=random_grid, 

                               n_iter = 100, 

                               cv = 10, 

                               verbose =2, 

                               random_state = 42, 

                               n_jobs = -1) 



rf_random.fit(X_train, y_train) 

print(rf_random.best_score_) 

print(rf_random.best_params_)
rf = RandomForestClassifier(n_estimators=800, 

                            min_samples_split=5, 

                            min_samples_leaf=1, 

                            max_features='sqrt', 

                            max_depth=90, 

                            bootstrap=False)

rf.fit(X_train, y_train)
y_pred = rf.predict(df_test_enc)
y_pred.shape
submission = pd.DataFrame({

    "PassengerId": test["PassengerId"],

    "Survived": y_pred

})

submission
submission.to_csv('submission.csv', index=False)