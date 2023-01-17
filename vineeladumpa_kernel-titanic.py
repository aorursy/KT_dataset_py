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
import matplotlib.pyplot as plt

import seaborn as sns



# Machine learning

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import model_selection

from sklearn.metrics import confusion_matrix

from sklearn.metrics import jaccard_similarity_score

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
training_df = pd.read_csv("../input/train.csv")

testing_df = pd.read_csv("../input/test.csv")

combine = [training_df, testing_df]
training_df.head()
training_df.info()

print('_'*40)

testing_df.info()
training_df.describe()
training_df.describe(include=['O'])
training_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
training_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
training_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
training_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(training_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(training_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(training_df, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(training_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
column_choice_training = training_df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Name','Survived']]

column_choice_test = testing_df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Name']]

column_choice_training.head()
column_choice_training.Sex[column_choice_training.Sex == 'female'] = 0

column_choice_training.Sex[column_choice_training.Sex == 'male'] = 1

column_choice_test.Sex[column_choice_test.Sex == 'female'] = 0

column_choice_test.Sex[column_choice_test.Sex == 'male'] = 1

column_choice_training.head()
freq_port = column_choice_training.Embarked.dropna().mode()[0]

column_choice_training.Embarked = column_choice_training.Embarked.fillna(freq_port)

column_choice_test.Embarked = column_choice_test.Embarked.fillna(freq_port)

column_choice_training.Embarked[column_choice_training.Embarked == 'S'] = 0

column_choice_training.Embarked[column_choice_training.Embarked == 'C'] = 1

column_choice_training.Embarked[column_choice_training.Embarked == 'Q'] = 2

column_choice_test.Embarked[column_choice_test.Embarked == 'S'] = 0

column_choice_test.Embarked[column_choice_test.Embarked == 'C'] = 1

column_choice_test.Embarked[column_choice_test.Embarked == 'Q'] = 2

column_choice_training.head()
column_choice_training.Age = column_choice_training.Age.fillna(column_choice_training.Age.mean())

column_choice_test.Age = column_choice_test.Age.fillna(column_choice_test.Age.mean())

column_choice_training.head(6)
column_choice_training.Fare = column_choice_training.Fare.fillna(column_choice_training.Fare.mean())

column_choice_test.Fare = column_choice_test.Fare.fillna(column_choice_test.Fare.mean())

column_choice_training.head(6)
column_choice_training['Title'] = column_choice_training['Name'].str.split(', ').str[1]

column_choice_training['Title'] = column_choice_training['Title'].str.split('.').str[0]

column_choice_training = column_choice_training.drop(['Name'], axis=1)

column_choice_test['Title'] = column_choice_test['Name'].str.split(', ').str[1]

column_choice_test['Title'] = column_choice_test['Title'].str.split('.').str[0]

column_choice_test = column_choice_test.drop(['Name'], axis=1)
column_choice_training.head(6)
pd.crosstab(column_choice_training['Title'], column_choice_training['Sex'])
pd.crosstab(column_choice_test['Title'], column_choice_test['Sex'])
column_choice_training['Title'] = column_choice_training['Title'].replace(['the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Rare')

column_choice_training['Title'] = column_choice_training['Title'].replace(['Lady','Mlle', 'Ms'], 'Mrs')

column_choice_training['Title'] = column_choice_training['Title'].replace(['Mme'], 'Miss')

column_choice_test['Title'] = column_choice_test['Title'].replace(['Col', 'Dona', 'Dr', 'Rev'], 'Rare')

column_choice_test['Title'] = column_choice_test['Title'].replace(['Ms'], 'Mrs')

pd.crosstab(column_choice_training['Title'], column_choice_training['Sex'])
pd.crosstab(column_choice_test['Title'], column_choice_test['Sex'])
column_choice_training.Title[column_choice_training.Title == 'Master'] = 0

column_choice_training.Title[column_choice_training.Title == 'Miss'] = 1

column_choice_training.Title[column_choice_training.Title == 'Mr'] = 2

column_choice_training.Title[column_choice_training.Title == 'Mrs'] = 3

column_choice_training.Title[column_choice_training.Title == 'Rare'] = 4

column_choice_test.Title[column_choice_test.Title == 'Master'] = 0

column_choice_test.Title[column_choice_test.Title == 'Miss'] = 1

column_choice_test.Title[column_choice_test.Title == 'Mr'] = 2

column_choice_test.Title[column_choice_test.Title == 'Mrs'] = 3

column_choice_test.Title[column_choice_test.Title == 'Rare'] = 4

column_choice_training.head(6)
X_train = np.asarray(column_choice_training[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title']])

X_train = preprocessing.StandardScaler().fit(X_train).transform(X_train)

X_train[0:5]
X_test = np.asarray(column_choice_test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title']])

X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test)

X_test[0:5]
y_train = np.asarray(column_choice_training['Survived'])

y_train[0:5]
print ('Train set:', X_train.shape,y_train.shape)

print ('Test set:', X_test.shape)
models = [

    LogisticRegression(solver='lbfgs'), 

    RandomForestClassifier(n_estimators=100, oob_score = True, random_state = 1)

    ]



model_results = pd.DataFrame(data = {'test_score_mean': [], 'fit_time_mean': []})



# Spliting the model

cross_validation_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 )

# Performing shufflesplit cross validation, with the whole training set (the cross_validate function coupled with ShuffleSplit take care of spliting the training set) 

for model in models:

    cross_validation_results = model_selection.cross_validate(model, X_train, y_train, cv= cross_validation_split, return_train_score=True)

    # Checking the mean of test scores for each iteration of the validation

    model_results = model_results.append({'test_score_mean' : cross_validation_results['test_score'].mean(), 

                                      'fit_time_mean' : cross_validation_results['fit_time'].mean()}, ignore_index=True) 

    

model_results
#RFC = RandomForestClassifier(n_estimators=100, oob_score = True, random_state = 1)

#param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10], "min_samples_split" : [2, 4, 10, 12, 16], "n_estimators": [50, 100, 400, 700, 1000]}



#GS = GridSearchCV(estimator = RFC, param_grid = param_grid, scoring='accuracy', cv = cross_validation_split)



#GS = GS.fit(column_choice_training.iloc[:, 1:], column_choice_training.iloc[:, 0])



print(GS.best_score_)

print(GS.best_params_)

print(GS.cv_results_)
best_model = RandomForestClassifier(n_estimators=400, 

                                    oob_score = True, 

                                    criterion = 'entropy',

                                    min_samples_leaf = 1,

                                    min_samples_split = 2,

                                    random_state = 1).fit(X_train,y_train)

yhat = best_model.predict(X_train)
from sklearn.metrics import classification_report, confusion_matrix

import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



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

print(confusion_matrix(y_train, yhat, labels=[1,0]))
# Compute confusion matrix

cnf_matrix = confusion_matrix(y_train, yhat, labels=[1,0])

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
# Predicting the results of the tessting set with the model

yhat_test = best_model.predict(X_test)

# Submitting

submission = testing_df.copy()

submission['Survived'] = yhat_test

submission.to_csv('submission.csv', columns=['PassengerId', 'Survived'], index=False)



submission[['PassengerId', 'Survived']].head(15)