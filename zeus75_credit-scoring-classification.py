# Upload Libraries

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE,ADASYN

from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import StratifiedKFold

from collections import Counter

import itertools



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Upload Dataset

dataset = pd.read_csv('../input/hmeq-data/hmeq.csv')

# Target variable

y = dataset.BAD

dataset.drop(['BAD'], axis=1, inplace=True)
# dimensions of dataset

print(dataset.shape)

# columns of dataset

dataset.columns

# list types for each attribute

dataset.dtypes

# take a peek at the first rows of the data

dataset.head(50)

# summarize attribute distributions for data frame

print(dataset.describe().T)

print(dataset.info())

def rstr(dataset): return dataset.shape, dataset.apply(lambda x: [x.unique()])

print(rstr(dataset))
# Look at the level of each feature

for column in dataset.columns:

    print(column, dataset[column].nunique())
# check missing values both to numeric features and categorical features 

feat_missing = []



for f in dataset.columns:

    missings = dataset[f].isnull().sum()

    if missings > 0:

        feat_missing.append(f)

        missings_perc = missings/dataset.shape[0]

        

        # printing summary of missing values

        print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))



# how many variables do present missing values?

print()

print('In total, there are {} variables with missing values'.format(len(feat_missing)))
# imputing missing values 

dataset = dataset.fillna(method='ffill')

dataset = dataset.fillna(method='bfill')





# summarize the class distribution

y = y.astype(object) 

count = pd.crosstab(index = y, columns="count")

percentage = pd.crosstab(index = y, columns="frequency")/pd.crosstab(index = y, columns="frequency").sum()

pd.concat([count, percentage], axis=1)
ax = sns.countplot(x=y, data=dataset).set_title("Target Variable Distribution")
# categorical features

categorical_cols = [cname for cname in dataset.columns if

                    dataset[cname].dtype in ['object']]

cat = dataset[categorical_cols]

cat.columns
# Visualizations

sns.set( rc = {'figure.figsize': (5, 5)})

fcat = ['REASON','JOB']



for col in fcat:

    plt.figure()

    sns.countplot(x=cat[col], data=cat, palette="Set3")

    plt.show()
# One-hot encode the data

HOX_dataset = pd.get_dummies(dataset)
# Numerical features

numerical_cols = [cname for cname in dataset.columns if

                 dataset[cname].dtype in ['float']]

num = dataset[numerical_cols]

num.columns
# Visualizations

sns.set( rc = {'figure.figsize': (5, 5)})

fnum = ['MORTDUE','VALUE','YOJ','DEROG','CLAGE','DEBTINC','DELINQ','NINQ','CLNO']



for col in fnum:

    plt.figure()

    x=num[col]

    sns.distplot(x, bins=10)

    plt.xticks(rotation=45)

    plt.show()
y = y.astype('int') 

smo = SMOTE(random_state=0)

X_resampled, y_resampled = smo.fit_resample(HOX_dataset, y)

print(sorted(Counter(y_resampled).items()))
# Break off train and validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X_resampled, y_resampled, train_size=0.8, test_size=0.2,

                                                                random_state=0)
# From https://www.kaggle.com/ajay1735/my-credit-scoring-model

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

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
# Test options and evaluation metric



# Spot Check Algorithms

models = []

models.append(('LogisticRegression', LogisticRegression(random_state=0)))

models.append(('Bagging', BaggingClassifier(random_state=0)))

models.append(('RandomForest', RandomForestClassifier(random_state=0)))

models.append(('AdaBoost', AdaBoostClassifier(random_state=0)))

models.append(('GBM', GradientBoostingClassifier(random_state=0)))

models.append(('XGB', XGBClassifier(random_state=0)))

results_t = []

results_v = []

names = []

score = []

skf = StratifiedKFold(n_splits=5)

for (name, model) in models:

    param_grid = {}

    my_model = GridSearchCV(model,param_grid,cv=skf)

    my_model.fit(X_train, y_train)

    predictions_t = my_model.predict(X_train) 

    predictions_v = my_model.predict(X_valid)

    accuracy_train = accuracy_score(y_train, predictions_t) 

    accuracy_valid = accuracy_score(y_valid, predictions_v) 

    results_t.append(accuracy_train)

    results_v.append(accuracy_valid)

    names.append(name)

    f_dict = {

        'model': name,

        'accuracy_train': accuracy_train,

        'accuracy_valid': accuracy_valid,

    }

    # Computing Confusion matrix for the above algorithm

    cnf_matrix = confusion_matrix(y_valid, predictions_v)

    np.set_printoptions(precision=2)



    # Plot non-normalized confusion matrix

    plt.figure()

    plot_confusion_matrix(cnf_matrix, classes=["BAD"], title="Confusion Matrix - "+str(name))

    score.append(f_dict)

plt.show()    

score = pd.DataFrame(score, columns = ['model','accuracy_train', 'accuracy_valid'])
print(score)
# Spot Check Algorithms with standardized dataset

pipelines = []

pipelines.append(('Scaled_LogisticRegression', Pipeline([('Scaler', StandardScaler()),('LogisticRegression', LogisticRegression(random_state=0))])))

pipelines.append(('Scaled_Bagging', Pipeline([('Scaler', StandardScaler()),('Bagging', BaggingClassifier(random_state=0))])))

pipelines.append(('Scaled_RandomForest', Pipeline([('Scaler', StandardScaler()),('RandomForest', RandomForestClassifier(random_state=0))])))

pipelines.append(('Scaled_AdaBoost', Pipeline([('Scaler', StandardScaler()),('AdaBoost', AdaBoostClassifier(random_state=0))])))

pipelines.append(('Scaled_GBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingClassifier(random_state=0))])))

pipelines.append(('Scaled_XGB', Pipeline([('Scaler', StandardScaler()),('XGB', XGBClassifier(random_state=0))])))

pipelines.append(('Scaled_NeuralNetwork', Pipeline([('Scaler', StandardScaler()),('NeuralNetwork', MLPClassifier(random_state=0))])))

results_t = []

results_v = []

names = []

score_sd = []

skf = StratifiedKFold(n_splits=5)

for (name, model) in pipelines:

    param_grid = {}

    my_model = GridSearchCV(model,param_grid,cv=skf)

    my_model.fit(X_train, y_train)

    predictions_t = my_model.predict(X_train) 

    predictions_v = my_model.predict(X_valid)

    accuracy_train = accuracy_score(y_train, predictions_t) 

    accuracy_valid = accuracy_score(y_valid, predictions_v) 

    results_t.append(accuracy_train)

    results_v.append(accuracy_valid)

    names.append(name)

    f_dict = {

        'model': name,

        'accuracy_train': accuracy_train,

        'accuracy_valid': accuracy_valid,

    }

    # Computing Confusion matrix for the above algorithm

    cnf_matrix = confusion_matrix(y_valid, predictions_v)

    np.set_printoptions(precision=2)



    # Plot non-normalized confusion matrix

    plt.figure()

    plot_confusion_matrix(cnf_matrix, classes=["BAD"], title="Confusion Matrix - "+str(name))

    score_sd.append(f_dict)

plt.show()   

score_sd = pd.DataFrame(score_sd, columns = ['model','accuracy_train', 'accuracy_valid'])
print(score_sd)
y = y.astype('int') 

ada = ADASYN(random_state=0)

X_resampled_, y_resampled_ = ada.fit_resample(HOX_dataset, y)

print(sorted(Counter(y_resampled_).items()))
# Break off train and validation set from training data

X_train_, X_valid_, y_train_, y_valid_ = train_test_split(X_resampled_, y_resampled_, train_size=0.8, test_size=0.2,

                                                                random_state=0)
# Test options and evaluation metric



# Spot Check Algorithms

models = []

models.append(('LogisticRegression', LogisticRegression(random_state=0)))

models.append(('Bagging', BaggingClassifier(random_state=0)))

models.append(('RandomForest', RandomForestClassifier(random_state=0)))

models.append(('AdaBoost', AdaBoostClassifier(random_state=0)))

models.append(('GBM', GradientBoostingClassifier(random_state=0)))

models.append(('XGB', XGBClassifier(random_state=0)))

results_t = []

results_v = []

names = []

score = []

skf = StratifiedKFold(n_splits=5)

for (name, model) in models:

    param_grid = {}

    my_model = GridSearchCV(model,param_grid,cv=skf)

    my_model.fit(X_train_, y_train_)

    predictions_t = my_model.predict(X_train_) 

    predictions_v = my_model.predict(X_valid_)

    accuracy_train = accuracy_score(y_train_, predictions_t) 

    accuracy_valid = accuracy_score(y_valid_, predictions_v) 

    results_t.append(accuracy_train)

    results_v.append(accuracy_valid)

    names.append(name)

    f_dict = {

        'model': name,

        'accuracy_train': accuracy_train,

        'accuracy_valid': accuracy_valid,

    }

    # Computing Confusion matrix for the above algorithm

    cnf_matrix = confusion_matrix(y_valid_, predictions_v)

    np.set_printoptions(precision=2)



    # Plot non-normalized confusion matrix

    plt.figure()

    plot_confusion_matrix(cnf_matrix, classes=["BAD"], title="Confusion Matrix - "+str(name))

    score.append(f_dict)

plt.show()    

score = pd.DataFrame(score, columns = ['model','accuracy_train', 'accuracy_valid'])
print(score)
# Spot Check Algorithms with standardized dataset

pipelines = []

pipelines.append(('Scaled_LogisticRegression', Pipeline([('Scaler', StandardScaler()),('LogisticRegression', LogisticRegression(random_state=0))])))

pipelines.append(('Scaled_Bagging', Pipeline([('Scaler', StandardScaler()),('Bagging', BaggingClassifier(random_state=0))])))

pipelines.append(('Scaled_RandomForest', Pipeline([('Scaler', StandardScaler()),('RandomForest', RandomForestClassifier(random_state=0))])))

pipelines.append(('Scaled_AdaBoost', Pipeline([('Scaler', StandardScaler()),('AdaBoost', AdaBoostClassifier(random_state=0))])))

pipelines.append(('Scaled_GBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingClassifier(random_state=0))])))

pipelines.append(('Scaled_XGB', Pipeline([('Scaler', StandardScaler()),('XGB', XGBClassifier(random_state=0))])))

pipelines.append(('Scaled_NeuralNetwork', Pipeline([('Scaler', StandardScaler()),('NeuralNetwork', MLPClassifier(random_state=0))])))

results_t = []

results_v = []

names = []

score_sd = []

skf = StratifiedKFold(n_splits=5)

for (name, model) in pipelines:

    param_grid = {}

    my_model = GridSearchCV(model,param_grid,cv=skf)

    my_model.fit(X_train_, y_train_)

    predictions_t = my_model.predict(X_train_) 

    predictions_v = my_model.predict(X_valid_)

    accuracy_train = accuracy_score(y_train_, predictions_t) 

    accuracy_valid = accuracy_score(y_valid_, predictions_v) 

    results_t.append(accuracy_train)

    results_v.append(accuracy_valid)

    names.append(name)

    f_dict = {

        'model': name,

        'accuracy_train': accuracy_train,

        'accuracy_valid': accuracy_valid,

    }

    # Computing Confusion matrix for the above algorithm

    cnf_matrix = confusion_matrix(y_valid_, predictions_v)

    np.set_printoptions(precision=2)



    # Plot non-normalized confusion matrix

    plt.figure()

    plot_confusion_matrix(cnf_matrix, classes=["BAD"], title="Confusion Matrix - "+str(name))

    score_sd.append(f_dict)

plt.show()   

score_sd = pd.DataFrame(score_sd, columns = ['model','accuracy_train', 'accuracy_valid'])
print(score_sd)