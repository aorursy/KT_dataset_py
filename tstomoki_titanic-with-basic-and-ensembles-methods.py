# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import sys
import re
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import scipy as sp

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

print("Python version: {}". format(sys.version))
print("pandas version: {}". format(pd.__version__))
print("matplotlib version: {}". format(matplotlib.__version__))
print("NumPy version: {}". format(np.__version__))
print("SciPy version: {}". format(sp.__version__)) 
import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 
import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))
print('-'*25)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

RANDOM_SEED = 19901129 # for reproducibility
np.random.seed(RANDOM_SEED)
train = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})
test  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})
full_data = [train, test]
print (train.info())
# Some features of my own that I have added in
# Gives the length of the name
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)
# Feature that tells whether a passenger had a cabin on the Titanic
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
# Feature engineering steps taken from Sina
# Create new feature FamilySize as a combination of SibSp and Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# Create new feature IsAlone from FamilySize
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# Remove all NULLS in the Embarked column
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
# Remove all NULLS in the Fare column and create a new feature CategoricalFare
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
# Create a New feature CategoricalAge
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
train['CategoricalAge'] = pd.cut(train['Age'], 5)
# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# Create a new feature Title, containing the titles of passenger names
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    
    # Age * Class
    dataset['Age*Class'] = dataset['Age'] * dataset['Pclass']
# Store our passenger ID for easy access
PassengerId = test['PassengerId']

# Feature selection
feature_columns = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Age*Class']
train = train[feature_columns + ['Survived']]
test  = test[feature_columns]
train.head()
print('enabled features'.upper())
for column_name in list(train.columns):
    print('\t' + column_name)
# visualization
g = sns.pairplot(train[list(train.columns)], hue='Survived', palette = 'seismic',height=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[])
# Some useful parameters which will come in handy later on
classifiers = []
ntrain = train.shape[0]
ntest = test.shape[0]

# Cross validate model with Kfold stratified cross val
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10)

# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = train['Survived'].ravel()
x_train = train.drop(['Survived'], axis=1).values # Creates an array of the train data
x_train_columns = train.drop(['Survived'], axis=1).columns
x_test = test.values # Creates an array of the test data

def classifier_name(clf):
    if 'ABCMeta' == clf.__class__:
        return clf.__name__
    else:
        return clf.__class__.__name__ 
from sklearn.ensemble import RandomForestClassifier
# parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'warm_start': False, 
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0,
    'random_state': RANDOM_SEED
}
rf = RandomForestClassifier(**rf_params)
classifiers.append(rf)
from sklearn.svm import SVC
svm_params = {'probability': True,
              'random_state': RANDOM_SEED}
svm = SVC(**svm_params)
classifiers.append(svm)
from sklearn.linear_model import LogisticRegression

lr_params = {
    'solver': 'liblinear',
    'random_state': RANDOM_SEED
}

lr = LogisticRegression(**lr_params)
classifiers.append(lr)
from sklearn.neighbors import KNeighborsClassifier
kn_params = {'n_neighbors': 3}
kn = KNeighborsClassifier(**kn_params)
classifiers.append(kn)
from sklearn.ensemble import GradientBoostingClassifier
gb_params = {'random_state':RANDOM_SEED}
gb = GradientBoostingClassifier(**gb_params)
classifiers.append(gb)
from sklearn.tree import DecisionTreeClassifier
dt_params = {'random_state':RANDOM_SEED}
dt = DecisionTreeClassifier(**dt_params)
classifiers.append(dt)
from sklearn.ensemble import AdaBoostClassifier
ab_params = {'random_state':RANDOM_SEED}
ab = AdaBoostClassifier(**ab_params)
classifiers.append(ab)
from sklearn.naive_bayes import GaussianNB
gnb_params = {}
gnb = GaussianNB(**gnb_params)
classifiers.append(gnb)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
ld_params = {}
ld = LinearDiscriminantAnalysis(**ld_params)
classifiers.append(ld)
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qd_params = {}
qd = QuadraticDiscriminantAnalysis(**qd_params)
classifiers.append(qd)
from xgboost import XGBClassifier
xgb_params = {'random_state':RANDOM_SEED}
xgb = XGBClassifier(**xgb_params)
classifiers.append(xgb)
import keras 
from keras.models import Sequential # intitialize the ANN
from keras.layers import Dense      # create layers
from keras.wrappers.scikit_learn import KerasClassifier

# Create function returning a compiled network
_, NUMBER_OF_FEATURES = x_train.shape

def create_network():
    # Start neural network
    network = Sequential()
    # layers
    network.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_shape=(NUMBER_OF_FEATURES,)))
    network.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
    network.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
    network.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    # Compile neural network
    network.compile(loss='binary_crossentropy', # Cross-entropy
                    optimizer='rmsprop', # Root Mean Square Propagation
                    metrics=['accuracy']) # Accuracy performance metric
    # Return compiled network
    return network

# Wrap Keras model so it can be used by scikit-learn
NN_classifier = KerasClassifier(build_fn=create_network, 
                                epochs=200, 
                                batch_size=32, 
                                verbose=1)
classifiers.append(NN_classifier)
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score
from tqdm import tqdm_notebook as tqdm

def evaluate_with_cross_validation(classifiers, test_size, split_num, X, y):
    ret_results = {}
    cv_results = []
    for classifier in tqdm(classifiers) :
        clf_name = classifier_name(classifier)
        print('cross validate with {0}'.format(clf_name))
        n_jobs = None if clf_name in ['KerasClassifier'] else -1
        cv_results.append(cross_val_score(classifier, X, y = y, scoring = "accuracy", cv = kfold, n_jobs=n_jobs))
            
    cv_means = []
    cv_std = []
    for index, cv_result in enumerate(cv_results):
        mean_accuracy = cv_result.mean()
        std = cv_result.std()
        cv_means.append(mean_accuracy)
        cv_std.append(std)
        classifier = classifiers[index]
        clf_name = classifier_name(classifier)
        ret_results[clf_name] = (mean_accuracy, classifier)
        print('{0:30s}: {1:.4f} (±{2:.4f})'.format(clf_name, mean_accuracy, std))
    cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm": list(ret_results.keys())})

    g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
    g.set_xlabel("Mean Accuracy")
    g = g.set_title("Cross validation scores")
    return ret_results

split_num = 10
test_size = 0.2
classifier_results = evaluate_with_cross_validation(classifiers, test_size, split_num, x_train, y_train)
from sklearn.model_selection import GridSearchCV
# Adaboost
DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs=-1, verbose = 1)

gsadaDTC.fit(x_train,y_train)

ada_best = gsadaDTC.best_estimator_
ada_accuracy = gsadaDTC.best_score_
classifier_results['ada_best'] = (ada_accuracy, ada_best)
ada_accuracy
#ExtraTrees
from sklearn.ensemble import ExtraTreesClassifier
ExtC = ExtraTreesClassifier()

## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs=-1, verbose = 1)

gsExtC.fit(x_train,y_train)

ext_best = gsExtC.best_estimator_

# Best score
ext_accuracy = gsExtC.best_score_
classifier_results['ext_best'] = (ext_accuracy, ext_best)
ext_accuracy
# RFC Parameters tunning 
RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs=-1, verbose = 1)

gsRFC.fit(x_train,y_train)

RFC_best = gsRFC.best_estimator_

# Best score
rfc_accuracy = gsRFC.best_score_
classifier_results['RFC_best'] = (rfc_accuracy, RFC_best)
rfc_accuracy
# Gradient boosting tunning

GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs=-1, verbose = 1)

gsGBC.fit(x_train,y_train)

GBC_best = gsGBC.best_estimator_

# Best score
gs_accuracy = gsGBC.best_score_
classifier_results['GBC_best'] = (gs_accuracy, GBC_best)
gs_accuracy
### SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs=-1, verbose = 1)

gsSVMC.fit(x_train,y_train)

SVMC_best = gsSVMC.best_estimator_

# Best score
gsSVMC_accuracy = gsSVMC.best_score_

classifier_results['SVMC_best'] = (gsSVMC_accuracy, SVMC_best)

gsSVMC_accuracy
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

g = plot_learning_curve(gsRFC.best_estimator_,"RF mearning curves",x_train,y_train,cv=kfold)
g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",x_train,y_train,cv=kfold)
g = plot_learning_curve(gsadaDTC.best_estimator_,"AdaBoost learning curves",x_train,y_train,cv=kfold)
g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",x_train,y_train,cv=kfold)
nrows = ncols = 2
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,15))

names_classifiers = [("AdaBoosting", ada_best),("ExtraTrees",ext_best), ("RandomForest",RFC_best),("GradientBoosting",GBC_best)]

nclassifier = 0
x_train_columns = train.drop(['Survived'], axis=1).columns
for row in range(nrows):
    for col in range(ncols):
        name = names_classifiers[nclassifier][0]
        classifier = names_classifiers[nclassifier][1]
        indices = np.argsort(classifier.feature_importances_)[::-1][:40]
        g = sns.barplot(y=x_train_columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h',ax=axes[row][col])
        g.set_xlabel("Relative importance",fontsize=12)
        g.set_ylabel("Features",fontsize=12)
        g.tick_params(labelsize=9)
        g.set_title(name + " feature importance")
        nclassifier += 1
# get TOP N classifiers to use in Voting Classifier
enable_classifiers_num = 5
enable_classifiers = sorted(classifier_results.items(), key=lambda kv: kv[1][0], reverse=True)[:enable_classifiers_num]
print('Top {0:,} classifiers'.format(enable_classifiers_num).upper())
for clf_name, tup in enable_classifiers:
    print('\t{0:30s}: {1:.5f}'.format(clf_name, tup[0]))
from sklearn.ensemble import VotingClassifier
estimators = [(n, tup[1]) for n, tup in enable_classifiers]
vc_params = {
    'estimators': estimators,
    'voting': 'soft',
    'n_jobs': -1,
    'flatten_transform': True
}
voting_classifier = VotingClassifier(**vc_params)
voting_classifier = voting_classifier.fit(x_train, y_train)
cv_result = cross_val_score(voting_classifier, x_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=-1)
print('{0:30s}: {1:.4f} (±{2:.4f})'.format(classifier_name(voting_classifier), cv_result.mean(), cv_result.std()))
# best_classifier_name = 'VotingClassifier'
# best_classifier = [_clf for _clf in classifiers if best_classifier_name == _clf.classifier_name()][0]
best_classifier = voting_classifier
best_classifier.fit(x_train, y_train)
predictions = best_classifier.predict(x_test)
# Generate Submission File 
submission_file = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': predictions })
from datetime import datetime
ts = datetime.now().strftime("%Y%m%d%H%M")
filename = 'submission_file_{0}.csv'.format(ts)
submission_file.to_csv(filename, index=False)
submission_file
