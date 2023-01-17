import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

import pydot

import re

!pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887

!apt update && apt install -y libsm6 libxext6

from fastai.imports import *

from fastai.structured import *



%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt



# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.wrappers.scikit_learn import KerasClassifier



import sklearn

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import export_graphviz, DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score, train_test_split
def display_confusion_matrix(sample_test, prediction, score=None):

    cm = metrics.confusion_matrix(sample_test, prediction)

    plt.figure(figsize=(9,9))

    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')

    plt.ylabel('Actual label')

    plt.xlabel('Predicted label')

    if score:

        all_sample_title = 'Accuracy Score: {0}'.format(score)

        plt.title(all_sample_title, size = 15)

    print(metrics.classification_report(sample_test, prediction))

    

        

#Load the data:

path_train = '../input/train.csv'

path_test = '../input/test.csv'

#Create a dataframe with the raw data

train_df_raw = pd.read_csv(path_train)

test_df_raw = pd.read_csv(path_test)



train_df_raw.head()
# Display some statistics

train_df_raw.describe()
train_df_raw.info()
# Display missing data

def draw_missing_data_table(df):

    total = df.isnull().sum().sort_values(ascending=False)

    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_data



draw_missing_data_table(train_df_raw)
train_df_raw['Embarked'].value_counts()
# Let's plot some histograms to have a previzualisation of some of the data ...

train_df_raw.drop(['PassengerId'], 1).hist(bins=50, figsize=(20,15))

plt.show()
def preprocess_data(df):

    

    processed_df = df

    

    #train_cats module from fastai, which changes the strings in a dataframe to a 

    #categorical values

    train_cats(processed_df)

    

    #proc_df takes a data frame df and splits off the response variable, and

    #changes the df into an entirely numeric dataframe. In this case am excluding the 

    #fields in ignore_flds as they need further processing.

    processed_df,y,nas = proc_df(processed_df,y_fld=None,ignore_flds=['Age','Name',

                                                                       'Embarked','Cabin','Parch',

                                                                       'SibSp'])

    # Introducing a new feature : the size of families (including the passenger)

    processed_df['FamilySize'] = processed_df['Parch'] + processed_df['SibSp'] + 1

    # Introducing other features based on the family size

    processed_df['Singleton'] = processed_df['FamilySize'].map(lambda s: 1 if s == 1 else 0)

    processed_df['SmallFamily'] = processed_df['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)

    processed_df['LargeFamily'] = processed_df['FamilySize'].map(lambda s: 1 if 5 <= s else 0)  

        



    # Deal with missing values

    processed_df['Embarked'].fillna('S', inplace=True)

    df_dummies = pd.get_dummies(processed_df['Embarked'], prefix='Embarked')

    processed_df = pd.concat([processed_df, df_dummies], axis=1)



    

    # Replacing missing cabins with U (for Uknown)

   # processed_df.Cabin.fillna('U', inplace=True)

    # Mapping each Cabin value with the cabin letter

   # processed_df['Cabin'] = processed_df['Cabin'].map(lambda c: c[0])

    # Dummy encoding ...

   # cabin_dummies = pd.get_dummies(processed_df['Cabin'], prefix='Cabin')    

   # processed_df = pd.concat([processed_df, cabin_dummies], axis=1)

    

    

    # Get Title and Status from Name

    processed_df['Title'] = pd.Series((name.split('.')[0].split(',')[1].strip() for name in train_df_raw['Name']), index=train_df_raw.index)

    processed_df['Title'] = processed_df['Title'].replace(['the Countess'], 'Countess')

    processed_df['Status'] = processed_df['Title'].replace(['Countess', 'Don', 'Jonkheer', 'Sir', 'Dona', 'Lady'], 'Nobility')

    processed_df['Status'] = processed_df['Status'].replace(['Major', 'Capt', 'Col'], 'Army')

    processed_df['Status'] = processed_df['Status'].replace(['Rev'], 'Church')

    processed_df['Status'] = processed_df['Status'].replace(['Master'], 'Kid')

    processed_df['Status'] = processed_df['Status'].replace(['Miss', 'Mlle'], 'Not married')

    processed_df['Status'] = processed_df['Status'].replace(['Mrs', 'Mme'], 'Married')

    processed_df['Status'] = processed_df['Status'].replace(['Mr', 'Ms'], 'Neutral')

    processed_df['Status'] = processed_df['Status'].replace(['Dr'], 'Highly Educated')

    processed_df['Specialities'] = processed_df['Title'].replace(['Countess', 'Don', 'Jonkheer', 'Sir', 'Dona', 'Lady',

                                                                  'Major', 'Capt', 'Col', 'Rev' ], 'Special')



    titles_dummies = pd.get_dummies(processed_df['Title'], prefix='Title')

    processed_df = pd.concat([processed_df, titles_dummies], axis=1)

    status_dummies = pd.get_dummies(processed_df['Status'], prefix='Status')

    processed_df = pd.concat([processed_df, status_dummies], axis=1)

    titles_dummies = pd.get_dummies(processed_df['Specialities'], prefix='Specialities')

    processed_df = pd.concat([processed_df, titles_dummies], axis=1)

    

    #summarize the Age grouped by sex, class + title

    grouped_train = processed_df.groupby(['Sex', 'Pclass', 'Title'])

    grouped_median_train = grouped_train.median()

    grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]

    def fill_age(row):

        condition = (

            (grouped_median_train['Sex'] == row['Sex']) &

            (grouped_median_train['Title'] == row['Title']) &

            (grouped_median_train['Pclass'] == row['Pclass'])

        )

        if np.isnan(grouped_median_train[condition]['Age'].values[0]):

            print('true')

            condition = (

                (grouped_median_train['Sex'] == row['Sex']) &

                (grouped_median_train['Pclass'] == row['Pclass'])

            )

        return grouped_median_train[condition]['Age'].values[0]

    # fills the missing values of the Age variable

    processed_df['Age'] = processed_df.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)

    

    processed_df = processed_df.drop(['Name', 'Ticket', 'PassengerId', 'Embarked', 'Cabin', 'Title', 'Status', 'Specialities'], 1)    

    

    return processed_df
train_df = train_df_raw.copy()

X1 = train_df.drop(['Survived'], 1)

X1 = X1.append(test_df_raw)

Y = train_df['Survived']

X1 = preprocess_data(X1)



sc = StandardScaler()

X1 = pd.DataFrame(sc.fit_transform(X1.values), index=X1.index, columns=X1.columns)



X = X1[:891].copy()

test = X1[891:].copy()



# Let's divide the train dataset in two datasets to evaluate perfomance of machine learning models used

# Split dataset for prediction

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=None)



X_train.head()
X_train.shape
# Create and train model on train data sample

dt = DecisionTreeClassifier(min_samples_split=15, min_samples_leaf=20, random_state=42)

dt.fit(X_train, Y_train)



# Predict for test data sample

dt_prediction = dt.predict(X_test)



# Compute error between predicted data and true response and display it in confusion matrix

score = metrics.accuracy_score(Y_test, dt_prediction)

display_confusion_matrix(Y_test, dt_prediction, score=score)

m = RandomForestClassifier(n_estimators=180,min_samples_leaf=3,max_features=0.5,n_jobs=-1)

m.fit(X_train,Y_train)

m.score(X_train,Y_train)

# Predicting the Test set results

m_prediction = m.predict(X_test)

from sklearn.metrics import accuracy_score

m_prediction = (m_prediction > 0.5) # convert probabilities to binary output



# Compute error between predicted data and true response and display it in confusion matrix

score = metrics.accuracy_score(Y_test, m_prediction)

display_confusion_matrix(Y_test, m_prediction, score=score)
def build_ann(optimizer='adam'):

    

    # Initializing our ANN

    ann = Sequential()

    

    # Adding the input layer and the first hidden layer of our ANN with dropout

    ann.add(Dense(units=32, kernel_initializer='glorot_normal', activation='relu', input_shape = (X_train.shape[1],)))

    # Dropout will disable some neurons (here 50% of all neurons) to avoid overfitting

    ann.add(Dropout(p=0.5))

    

    # Add other layers, it is not necessary to pass the shape because there is a layer before

    ann.add(Dense(units=64, kernel_initializer='glorot_normal', activation='relu'))

    ann.add(Dropout(p=0.5))

    ann.add(Dense(units=128, kernel_initializer='glorot_normal', activation='relu'))

    ann.add(Dropout(p=0.5))

    ann.add(Dense(units=164, kernel_initializer='glorot_normal', activation='relu'))

    ann.add(Dropout(p=0.5))

    ann.add(Dense(units=16, kernel_initializer='glorot_normal', activation='relu'))

    

    # Adding the output layer

    ann.add(Dense(units=1, kernel_initializer='glorot_normal', activation='sigmoid'))

    

    # Compilling the ANN

    ann.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    

    return ann
ann = build_ann()

# Training the ANN

ann.fit(X_train, Y_train, batch_size=10, epochs=100)
# Predicting the Test set results

ann_prediction = ann.predict(X_test)

ann_prediction = (ann_prediction > 0.5) # convert probabilities to binary output



# Compute error between predicted data and true response and display it in confusion matrix

score = metrics.accuracy_score(Y_test, ann_prediction)

display_confusion_matrix(Y_test, ann_prediction, score=score)
accuracies_tree = cross_val_score(estimator=dt, X=X_train, y=Y_train, cv=6, n_jobs=-1)

accuracies_rf = cross_val_score(estimator=m, X=X_train, y=Y_train, cv=6, n_jobs=-1)

accuracies_ann = cross_val_score(estimator=KerasClassifier(build_fn=build_ann, batch_size=10, epochs=100, verbose=0),

                                 X=X_train, y=Y_train, cv=6, n_jobs=-1)
accuracies = {'dt': accuracies_tree, 'm': accuracies_rf, 'ann': accuracies_ann}

mean = {model: acc.mean() for model, acc in accuracies.items()}

variance = {model: acc.std() for model, acc in accuracies.items()}

print('Mean accuracy:', mean, '\nVariance:', variance)
plt.figure(figsize=(20,5))

plt.plot(accuracies['dt'])

plt.plot(accuracies['m'])

plt.plot(accuracies['ann'])

plt.title('Models Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Trained fold')

plt.xticks([k for k in range(10)])

plt.legend(['tree', 'RF', 'ANN'], loc='upper left')

plt.show()
print(m.get_params())
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}
# Use the random grid to search for best hyperparameters

# First create the base model to tune

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = m, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(X_train, Y_train)
rf_random.best_params_
best_random = rf_random.best_estimator_
def evaluate(model, test_features, test_labels):

    predictions = model.predict(test_features)

    errors = abs(predictions - test_labels)

    mape = 100 * np.mean(errors / test_labels)

    accuracy = 100 - mape

    print('Model Performance')

    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))

    print('Accuracy = {:0.2f}%.'.format(accuracy))

    

    return accuracy
best_random = rf_random.best_estimator_

rf_accuracy = evaluate(best_random, X_test, Y_test)
from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 

param_grid = {

    'bootstrap': [True],

    'max_depth': [15, 17, 20, 22],

    'max_features': [2, 3, 4],

    'min_samples_leaf': [1, 2, 3],

    'min_samples_split': [8, 10, 12],

    'n_estimators': [1500, 1550, 1600, 1650, 1700]

}

# Instantiate the grid search model

grid_search = GridSearchCV(estimator = m, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)
# Fit the grid search to the data

grid_search.fit(X_train, Y_train)

grid_search.best_params_
grid_search.best_params_
best_grid = grid_search.best_estimator_

grid_accuracy = evaluate(best_grid, X_test, Y_test)
#test_df_raw = pd.read_csv(path_test)

#test = test_df_raw.copy()

#test = preprocess_data(test)

#test = test[to_keep]

test.shape

pd.DataFrame(sc.fit_transform(test.values), index=test.index, columns=test.columns)
# Create and train model on train data sample

model_test = best_grid

#X = X[to_keep]

model_test.fit(X, Y)



# Predict for test data sample

prediction = model_test.predict(test)

prediction = (prediction > 0.5)*1



ids = test_df_raw["PassengerId"]

result_df= {"PassengerId": ids,

           "Survived": prediction}

submission = pd.DataFrame(result_df)

submission.to_csv("submissionvf.csv", index=False)