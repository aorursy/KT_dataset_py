%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data.head()
print('survival rate = ', 100 * train_data['Survived'].mean())

train_data.groupby(['Survived']).mean().transpose()
train_data.isnull().sum()
test_data.isnull().sum()
# This list contains the two dataframes, the goal is to do not rewrite operations (we will use a for loop)

datas = [train_data, test_data]



for data in datas:

    

    data['Age'].fillna(data['Age'].median(), inplace=True)

    data['Fare'].fillna(data['Fare'].median(), inplace=True)

    data['Cabin'].fillna('NoCabin', inplace=True) 



train_data.dropna(inplace=True)
for data in datas:

    print(data.isnull().sum().sum())
list(train_data.columns)
train_data.groupby(['Pclass'])[['Survived']].describe()
for data in datas:

    data['Title'] = data['Name'].map(lambda x: x[x.index(',')+2: x.index('.')])
train_data['Title'].value_counts()
test_data['Title'].value_counts()
Title_Mapping = {

    "Mr" :        "Mr",

    "Miss" :      "Miss",

    "Mrs" :       "Mrs",

    "Mme":        "Mrs",

    "Master" :    "Master",

    "Mlle":       "Miss",

    "Ms":         "Mrs",

    "Capt":       "Officer",

    "Col":        "Officer",

    "Major":      "Officer",

    "Dr":         "Officer",

    "Rev":        "Officer",

    "Jonkheer":   "Royalty",

    "Don":        "Royalty",

    "Sir" :       "Royalty",

    "the Countess":"Royalty",

    "Dona":       "Royalty",

    "Lady" :      "Royalty"}
for data in datas:

    data['Title'] = data['Title'].map(Title_Mapping)
train_data.groupby(['Title'])['Survived'].describe()
train_data.groupby(['Title'])['Survived'].mean().plot.bar(title='Survival rate per title')
train_data.groupby(['Sex'])['Survived'].mean().plot.bar(title='Survival rate per sex')
train_data.groupby('Survived')['Age'].describe()
train_data.groupby('Survived')['Age'].plot.density(legend=True)
all_ages = train_data['Age'].append(test_data['Age'])

age_cuts = pd.cut(all_ages, 5).unique()

def get_good_age_cut(age):

    for cut in age_cuts:

        if age in cut:

            return cut

        

for data in datas:

    data['AgeBin'] = data['Age'].map(lambda x: get_good_age_cut(x))



train_data.groupby(['AgeBin'])['Survived'].mean().plot.bar(title='Survival rate per AgeBin')


for data in datas:

    data['Family'] = data['SibSp'] + data['Parch'] + 1

    

train_data.groupby(['Family'])['Survived'].describe()
def get_family_type(members_number):

    if members_number <= 1:

        return 'Small'

    if members_number <= 4:

        return 'Medium'

    return 'Big'



for data in datas:

    data['FamilyType'] = data['Family'].map(lambda x: get_family_type(x))
train_data.groupby(['FamilyType'])['Survived'].mean().plot.bar(title='Survival rate per family type')
train_data.groupby('Survived')['Fare'].plot.density(legend=True)


all_fares = train_data['Fare'].append(test_data['Fare'])

fare_cuts = pd.qcut(all_fares, 4).unique()

print(fare_cuts)

def get_good_fare_cut(fare):

    for cut in fare_cuts:

        if fare in cut:

            return cut

        

for data in datas:

    data['FareBin'] = data['Fare'].map(lambda x: get_good_fare_cut(x))



train_data.groupby(['FareBin'])['Survived'].describe()
train_data.groupby(['FareBin'])['Survived'].mean().plot.bar(title='Survival rate per FareBin')
train_data['Cabin'].unique()
def get_cabin_type(cabin_name):

    if cabin_name == 'NoCabin':

        return 'NoType'

    cabins_types = ['A', 'B', 'C', 'D', 'E', 'F']

    for cabin_type in cabins_types:

        if cabin_type in cabin_name:

            return cabin_type

    return 'NoType'



for data in datas:

    data['CabinType'] = data['Cabin'].map(lambda x: get_cabin_type(x))

train_data.groupby('CabinType')['Survived'].describe()
train_data.drop('CabinType', axis=1, inplace=True)

test_data.drop('CabinType', axis=1, inplace=True)
for data in datas:

    data['HasCabin'] = data['Cabin'].map(lambda x: x != 'NoCabin')

    

train_data.groupby(['HasCabin'])['Survived'].mean().plot.bar(legend=True)
train_data['Ticket'].head(10)
for data in datas:

    data['TicketType'] = data['Ticket'].apply(lambda x: 'Numeric' if len(x.split(' ')) == 1 else 'AplhaNumeric')
train_data.groupby('TicketType')['Survived'].describe()
print(train_data['Embarked'].value_counts())

train_data.groupby('Embarked')['Survived'].mean().plot.bar(title='Survival rate per Embarked')
list(train_data.columns)
# the first thing to do now is to capture all possible input columns and put the output in a separed variables

input_columns = ['Pclass', 'Sex', 'Embarked', 'Title', 'AgeBin', 'FamilyType',

                'FareBin', 'HasCabin', 'TicketType']



X_train, X_test = train_data[input_columns], test_data[input_columns]

y_train = train_data['Survived']
for column in input_columns:

    X_train[column] = X_train[column].astype('str')

    X_test[column] = X_test[column].astype('str')

X_train.dtypes
"""

# This code is commented because the library one_hot_encoder is not installed on kaggle kernek

from one_hot_encoder.encoder import Encoder



oh_encoder = Encoder(drop_first=True)

oh_encoder.fit(X_train)

X_train = oh_encoder.get_dummies(X_train)

X_test = oh_encoder.get_dummies(X_test)

"""

X_train = pd.get_dummies(X_train, drop_first=True)

X_test = pd.get_dummies(X_test, drop_first=True)
X_train.columns
from sklearn.ensemble import RandomForestClassifier



def find_most_important_features(X, y):

    """

    :param X: data

    :param y: labels

    :return: returns the list of columns ordered by their importance and the list of their scores

    """



    forest = RandomForestClassifier(n_estimators=250)

    forest.fit(X, y)

    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]

    columns = list(X.columns)



    columns_by_importance = []

    importance_by_column = []

    for f in range(X.shape[1]):

        columns_by_importance.append(columns[indices[f]])

        importance_by_column.append(importances[indices[f]])

    

    

    fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(15,5))

    plt.xticks(np.arange(len(columns_by_importance)), columns_by_importance, rotation=90)

    plt.bar(np.arange(len(columns_by_importance)), importance_by_column)

find_most_important_features(X_train, y_train)
# Here we will use the select kbest features algo

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.model_selection import KFold



kfold = KFold(n_splits=5, shuffle=True)



k_best_algo = SelectKBest(chi2, 10)

k_best_algo.fit(X_train, y_train)

k_best_features = X_train.columns.values[k_best_algo.get_support()]

print(k_best_features)
# here we will use the recursive features elimination alvorithm

from sklearn.feature_selection import RFECV

from sklearn.ensemble import RandomForestClassifier



len_rfe_features = float('inf')

while len_rfe_features > 7:

    rfe = RFECV(RandomForestClassifier(), step=1, cv=kfold, n_jobs=2)

    rfe.fit(X_train, y_train)

    rfe_features = X_train.columns.values[rfe.get_support()]

    len_rfe_features = len(rfe_features)

print(rfe_features)

# We will try here many machine learning algorithms to fetch those who has the best performance

def get_classififcation_algorithms():

    """returns a list of ml algorithms"""

    from sklearn import ensemble, linear_model, neighbors, svm, tree

        

    return [

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(n_estimators = 100),

    neighbors.KNeighborsClassifier(n_neighbors = 3),

    svm.SVC(probability=True),

    svm.LinearSVC(),

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier()]

from sklearn.model_selection import cross_validate



def get_classification_results(alg, X, y, cv):

    results = cross_validate(alg, X, y, cv=cv, return_train_score=True)

    

    return {

        'alg_name': alg.__class__.__name__,

        'train_score': results['train_score'].mean(),

        'test_score': results['test_score'].mean()

    }

    
ml_results = []



columns_types = [

    ('rfe', rfe_features),

    ('k_best', k_best_features),

    ('all_features', list(X_test.columns))]



for columns_type, columns in columns_types:

    for alg in get_classififcation_algorithms():

        classification_results = get_classification_results(alg, X_train[columns], y_train, kfold)

        classification_results['columns_type'] = columns_type

        ml_results.append(classification_results)



results_df = pd.DataFrame(list(ml_results))

# We build a column which represents the diff between the train and tests score

results_df['train_test_diff%'] = 100*(results_df['train_score'] - results_df['test_score'])/results_df['train_score']

results_df.sort_values(by='test_score', ascending=False).head(n=10)
results_df.groupby(['columns_type'])['train_test_diff%'].mean()
results_df.groupby(['columns_type'])['test_score'].mean()


from sklearn.ensemble import VotingClassifier

voting_estimators = [

 (est.__class__.__name__ , est)   for est in get_classififcation_algorithms()

]

from sklearn import ensemble

voting_hard = ensemble.VotingClassifier(estimators = voting_estimators , voting = 'hard')

voting_hard_cv = get_classification_results(voting_hard, X_train[rfe_features], y_train, kfold)

print('train score = ', voting_hard_cv['train_score'].mean())

print('test score = ', voting_hard_cv['test_score'].mean())



from sklearn.model_selection import GridSearchCV



rf_grid = {

        'n_estimators': [100, 200, 500],

        'max_depth': [5, 10, 20],

        'max_leaf_nodes': [None, 10, 100],

        'min_samples_split': [5, 10, 50]

}

used_columns = rfe_features



rf_alg = RandomForestClassifier()

grid_search = GridSearchCV(

        estimator=rf_alg, param_grid=rf_grid,

        n_jobs=1, cv=3, refit=True, verbose=1)

grid_search.fit(X_train[used_columns], y_train)



best_estimator = grid_search.best_estimator_

print('Random forest\'s best test score = {}'.format(grid_search.best_score_))
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_predict

import matplotlib.pyplot as plt

import numpy as np

import itertools



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure()

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()



cv_predictions = cross_val_predict(

    best_estimator, 

    X_train[used_columns], 

    y_train, cv=kfold)



plot_confusion_matrix(confusion_matrix(y_train, cv_predictions), ['Died', 'Survived'], normalize=True)
test_predictions = best_estimator.predict(X_test[used_columns])



test_data['Survived'] = test_predictions

test_data[['PassengerId', 'Survived']].to_csv('predictions.csv', index=False)
