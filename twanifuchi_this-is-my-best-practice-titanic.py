# General Packages
import numpy as np
import pandas as pd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

# others
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

# constant
RANDOM_STATE = 10
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
train.info()
train.describe()
test.head()
test.info()
test.describe()
drop_train = train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
drop_train.head()
drop_test = test.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
drop_test.head()
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(drop_train.corr(), annot=True, ax=ax)
meanAgePclass1 = drop_train.loc[drop_train.Pclass == 1,'Age'].mean()
meanAgePclass2 = drop_train.loc[drop_train.Pclass == 2,'Age'].mean()
meanAgePclass3 = drop_train.loc[drop_train.Pclass == 3,'Age'].mean()

print('meanAgePclass1: {}'.format(meanAgePclass1))
print('meanAgePclass2: {}'.format(meanAgePclass2))
print('meanAgePclass3: {}'.format(meanAgePclass3))
modeEmbarked = drop_train.Embarked.mode().iloc[0]
print(type(modeEmbarked))
print('modeEmbarked: {}'.format(modeEmbarked))
meanFarePclass1 = drop_train.loc[drop_train.Pclass == 1,'Fare'].mean()
meanFarePclass2 = drop_train.loc[drop_train.Pclass == 2,'Fare'].mean()
meanFarePclass3 = drop_train.loc[drop_train.Pclass == 3,'Fare'].mean()

print('meanFarePclass1: {}'.format(meanFarePclass1))
print('meanFarePclass2: {}'.format(meanFarePclass2))
print('meanFarePclass3: {}'.format(meanFarePclass3))

print('row of missing Fare: \n{}'.format(drop_test.loc[drop_test.Fare.isnull()]))
imputer_train = drop_train
imputer_train.loc[(imputer_train['Pclass'] == 1) & (imputer_train['Age'].isnull()), 'Age'] = meanAgePclass1
imputer_train.loc[(imputer_train['Pclass'] == 2) & (imputer_train['Age'].isnull()), 'Age'] = meanAgePclass2
imputer_train.loc[(imputer_train['Pclass'] == 3) & (imputer_train['Age'].isnull()), 'Age'] = meanAgePclass3
imputer_train.loc[imputer_train['Embarked'].isnull(), 'Embarked'] = modeEmbarked

imputer_train.info()                                                   
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(imputer_train.corr(), annot=True, ax=ax)
imputer_test = drop_test
imputer_test.loc[(imputer_test['Pclass'] == 1) & (imputer_test['Age'].isnull()), 'Age'] = meanAgePclass1
imputer_test.loc[(imputer_test['Pclass'] == 2) & (imputer_test['Age'].isnull()), 'Age'] = meanAgePclass2
imputer_test.loc[(imputer_test['Pclass'] == 3) & (imputer_test['Age'].isnull()), 'Age'] = meanAgePclass3
imputer_test.loc[imputer_test['Fare'].isnull(), 'Fare'] = meanFarePclass3

imputer_test.info()
imputer_train['Pclass'] = imputer_train['Pclass'].astype('str')
imputer_train.info()
imputer_test['Pclass'] = imputer_test['Pclass'].astype('str')
imputer_test.info()
oneHot_train = pd.get_dummies(imputer_train)
oneHot_train.head()
oneHot_test = pd.get_dummies(imputer_test)
oneHot_test.head()
X = oneHot_train.drop(['Survived'], axis=1)
X.head()
y = oneHot_train['Survived']
y.head()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=RANDOM_STATE)

print('X_train: \n{}'.format(X_train.head()))
print()
print('X_valid: \n{}'.format(X_valid.head()))
print()
print('y_train: \n{}'.format(y_train.head()))
print()
print('y_valid: \n{}'.format(y_valid.head()))     
# MinMaxScaler
minmaxScaler = MinMaxScaler()

# X_train, X_valid
minmaxScaler.fit(X_train)
X_train_minmaxScaled = minmaxScaler.transform(X_train)
X_valid_minmaxScaled = minmaxScaler.transform(X_valid)
print('X_train_minmaxScaled:\n{}'.format(X_train_minmaxScaled[:5,:]))
print()
print('X_train_minmaxScaled:\n{}'.format(X_valid_minmaxScaled[:5,:]))
print()

# X, test
minmaxScaler.fit(X)
X_minmaxScaled = minmaxScaler.transform(X)
test_minmaxScaled = minmaxScaler.transform(oneHot_test)
print('X_minmaxScaled:\n{}'.format(X_minmaxScaled[:5,:]))
print()
print('test_minmaxScaled:\n{}'.format(test_minmaxScaled[:5,:]))
# StandardScaler

# X_train, X_valid
stdScaler = StandardScaler()
stdScaler.fit(X_train)
X_train_stdScaled = stdScaler.transform(X_train)
X_valid_stdScaled = stdScaler.transform(X_valid)
print('X_train_stdScaled:\n{}'.format(X_train_stdScaled[:5,:]))
print()
print('X_valid_stdScaled:\n{}'.format(X_valid_stdScaled[:5,:]))
print()

# X, test
stdScaler = StandardScaler()
stdScaler.fit(X)
X_stdScaled = stdScaler.transform(X)
test_stdScaled = stdScaler.transform(oneHot_test)
print('X_stdScaled:\n{}'.format(X_stdScaled[:5,:]))
print()
print('test_stdScaled:\n{}'.format(test_stdScaled[:5,:]))
cv = StratifiedKFold(shuffle=True, random_state=RANDOM_STATE)
# LogisticRegression
clf = LogisticRegression(random_state = RANDOM_STATE)

#defult NotScaled
clf.fit(X_train,y_train)
defLrTrainScore = cross_val_score(clf, X_train, y_train, cv=cv).mean()
defLrValidScore = cross_val_score(clf, X_valid, y_valid, cv=cv).mean()
print('LogisticRegression NotScaled train score: {:.3f}'.format(defLrTrainScore))
print('LogisticRegression NotScaled valid score: {:.3f}'.format(defLrValidScore))

#defult minmaxScaled
clf.fit(X_train_minmaxScaled,y_train)
defMinMaxLrTrainScore = cross_val_score(clf, X_train_minmaxScaled, y_train, cv=cv).mean()
defMinMaxLrValidScore = cross_val_score(clf, X_valid_minmaxScaled, y_valid, cv=cv).mean()
print('LogisticRegression MinMaxScaled train score: {:.3f}'.format(defMinMaxLrTrainScore))
print('LogisticRegression MinMaxScaled valid score: {:.3f}'.format(defMinMaxLrValidScore))

#defult StandardScaled
clf.fit(X_train_stdScaled,y_train)
defStdLrTrainScore = cross_val_score(clf, X_train_stdScaled, y_train, cv=cv).mean()
defStdLrValidScore = cross_val_score(clf, X_valid_stdScaled, y_valid, cv=cv).mean()
print('LogisticRegression StandardScaled train score: {:.3f}'.format(defStdLrTrainScore))
print('LogisticRegression StandardScaled valid score: {:.3f}'.format(defStdLrValidScore))
# GaussianNB
clf = GaussianNB()

#defult NotScaled
clf.fit(X_train,y_train)
defGaussianTrainScore = cross_val_score(clf, X_train, y_train, cv=cv).mean()
defGaussianValidScore = cross_val_score(clf, X_valid, y_valid, cv=cv).mean()
print('Gaussian NotScaled train score: {:.3f}'.format(defGaussianTrainScore))
print('Gaussian NotScaled valid score: {:.3f}'.format(defGaussianValidScore))

#defult minmaxScaled
clf.fit(X_train_minmaxScaled,y_train)
defMinMaxGaussianTrainScore = cross_val_score(clf, X_train_minmaxScaled, y_train, cv=cv).mean()
defMinMaxGaussianValidScore = cross_val_score(clf, X_valid_minmaxScaled, y_valid, cv=cv).mean()
print('Gaussian MinMaxScaled train score: {:.3f}'.format(defMinMaxGaussianTrainScore))
print('Gaussian MinMaxScaled valid score: {:.3f}'.format(defMinMaxGaussianValidScore))

#defult StandardScaled
clf.fit(X_train_stdScaled,y_train)
defStdGaussianTrainScore = cross_val_score(clf, X_train_stdScaled, y_train, cv=cv).mean()
defStdGaussianValidScore = cross_val_score(clf, X_valid_stdScaled, y_valid, cv=cv).mean()
print('Gaussian StandardScaled train score: {:.3f}'.format(defStdGaussianTrainScore))
print('Gaussian StandardScaled valid score: {:.3f}'.format(defStdGaussianValidScore))
# SVC
clf = SVC(random_state = RANDOM_STATE)

#defult NotScaled
clf.fit(X_train,y_train)
defSvcTrainScore = cross_val_score(clf, X_train, y_train, cv=cv).mean()
defSvcValidScore = cross_val_score(clf, X_valid, y_valid, cv=cv).mean()
print('SVC train NotScaled score: {:.3f}'.format(defSvcTrainScore))
print('SVC valid NotScaled score: {:.3f}'.format(defSvcValidScore))

#defult minmaxScaled
clf.fit(X_train_minmaxScaled,y_train)
defMinMaxSvcTrainScore = cross_val_score(clf, X_train_minmaxScaled, y_train, cv=cv).mean()
defMinMaxSvcValidScore = cross_val_score(clf, X_valid_minmaxScaled, y_valid, cv=cv).mean()
print('SVC MinMaxScaled train score: {:.3f}'.format(defMinMaxSvcTrainScore))
print('SVC MinMaxScaled valid score: {:.3f}'.format(defMinMaxSvcValidScore))

#defult StandardScaled
clf.fit(X_train_stdScaled,y_train)
defStdSvcTrainScore = cross_val_score(clf, X_train_stdScaled, y_train, cv=cv).mean()
defStdSvcValidScore = cross_val_score(clf, X_valid_stdScaled, y_valid, cv=cv).mean()
print('SVC StandardScaled train score: {:.3f}'.format(defStdSvcTrainScore))
print('SVC StandardScaled valid score: {:.3f}'.format(defStdSvcValidScore))
# LinearSVC
clf = LinearSVC(random_state = RANDOM_STATE)

#defult NotScaled
clf.fit(X_train,y_train)
defLinearSvcTrainScore = cross_val_score(clf, X_train, y_train, cv=cv).mean()
defLinearSvcValidScore = cross_val_score(clf, X_valid, y_valid, cv=cv).mean()
print('LinearSVC NotScaled train score: {:.3f}'.format(defLinearSvcTrainScore))
print('LinearSVC NotScaled valid score: {:.3f}'.format(defLinearSvcValidScore))

#defult minmaxScaled
clf.fit(X_train_minmaxScaled,y_train)
defMinMaxLinearSvcTrainScore = cross_val_score(clf, X_train_minmaxScaled, y_train, cv=cv).mean()
defMinMaxLinearSvcValidScore = cross_val_score(clf, X_valid_minmaxScaled, y_valid, cv=cv).mean()
print('LinearSVC MinMaxScaled train score: {:.3f}'.format(defMinMaxLinearSvcTrainScore))
print('LinearSVC MinMaxScaled valid score: {:.3f}'.format(defMinMaxLinearSvcValidScore))

#defult StandardScaled
clf.fit(X_train_stdScaled,y_train)
defStdLinearSvcTrainScore = cross_val_score(clf, X_train_stdScaled, y_train, cv=cv).mean()
defStdLinearSvcValidScore = cross_val_score(clf, X_valid_stdScaled, y_valid, cv=cv).mean()
print('LinearSVC StandardScaled train score: {:.3f}'.format(defStdLinearSvcTrainScore))
print('LinearSVC StandardScaled valid score: {:.3f}'.format(defStdLinearSvcValidScore))
# RandomForestClassifier
clf = RandomForestClassifier(random_state = RANDOM_STATE)

#defult NotScaled
clf.fit(X_train,y_train)
defRfTrainScore = cross_val_score(clf, X_train, y_train, cv=cv).mean()
defRfValidScore = cross_val_score(clf, X_valid, y_valid, cv=cv).mean()
print('RandomForestClassifier NotScaled train score: {:.3f}'.format(defRfTrainScore))
print('RandomForestClassifier NotScaled valid score: {:.3f}'.format(defRfValidScore))

#defult minmaxScaled
clf.fit(X_train_minmaxScaled,y_train)
defMinMaxRfTrainScore = cross_val_score(clf, X_train_minmaxScaled, y_train, cv=cv).mean()
defMinMaxRfValidScore = cross_val_score(clf, X_valid_minmaxScaled, y_valid, cv=cv).mean()
print('RandomForestClassifier MinMaxScaled train score: {:.3f}'.format(defMinMaxRfTrainScore))
print('RandomForestClassifier MinMaxScaled valid score: {:.3f}'.format(defMinMaxRfValidScore))

#defult StandardScaled
clf.fit(X_train_stdScaled,y_train)
defStdRfTrainScore = cross_val_score(clf, X_train_stdScaled, y_train, cv=cv).mean()
defStdRfValidScore = cross_val_score(clf, X_valid_stdScaled, y_valid, cv=cv).mean()
print('RandomForestClassifier StandardScaled train score: {:.3f}'.format(defStdRfTrainScore))
print('RandomForestClassifier StandardScaled valid score: {:.3f}'.format(defStdRfValidScore))
# DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state = RANDOM_STATE)

#defult NotScaled
clf.fit(X_train,y_train)
defDtTrainScore = cross_val_score(clf, X_train, y_train, cv=cv).mean()
defDtValidScore = cross_val_score(clf, X_valid, y_valid, cv=cv).mean()
print('DecisionTreeClassifier NotScaled train score: {:.3f}'.format(defDtTrainScore))
print('DecisionTreeClassifier NotScaled valid score: {:.3f}'.format(defDtValidScore))

#defult minmaxScaled
clf.fit(X_train_minmaxScaled,y_train)
defMinMaxDtTrainScore = cross_val_score(clf, X_train_minmaxScaled, y_train, cv=cv).mean()
defMinMaxDtValidScore = cross_val_score(clf, X_valid_minmaxScaled, y_valid, cv=cv).mean()
print('DecisionTreeClassifier MinMaxScaled train score: {:.3f}'.format(defMinMaxDtTrainScore))
print('DecisionTreeClassifier MinMaxScaled valid score: {:.3f}'.format(defMinMaxDtValidScore))

#defult StandardScaled
clf.fit(X_train_stdScaled,y_train)
defStdDtTrainScore = cross_val_score(clf, X_train_stdScaled, y_train, cv=cv).mean()
defStdDtValidScore = cross_val_score(clf, X_valid_stdScaled, y_valid, cv=cv).mean()
print('DecisionTreeClassifier StandardScaled train score: {:.3f}'.format(defStdDtTrainScore))
print('DecisionTreeClassifier StandardScaled valid score: {:.3f}'.format(defStdDtValidScore))
# KNeighborsClassifier
clf = KNeighborsClassifier()

#defult NotScaled
clf.fit(X_train,y_train)
defKnnTrainScore = cross_val_score(clf, X_train, y_train, cv=cv).mean()
defKnnValidScore = cross_val_score(clf, X_valid, y_valid, cv=cv).mean()
print('KNeighborsClassifier NotScaled train score: {:.3f}'.format(defKnnTrainScore))
print('KNeighborsClassifier NotScaled valid score: {:.3f}'.format(defKnnValidScore))

#defult minmaxScaled
clf.fit(X_train_minmaxScaled,y_train)
defMinMaxKnnTrainScore = cross_val_score(clf, X_train_minmaxScaled, y_train, cv=cv).mean()
defMinMaxKnnValidScore = cross_val_score(clf, X_valid_minmaxScaled, y_valid, cv=cv).mean()
print('KNeighborsClassifier MinMaxScaled train score: {:.3f}'.format(defMinMaxKnnTrainScore))
print('KNeighborsClassifier MinMaxScaled valid score: {:.3f}'.format(defMinMaxKnnValidScore))

#defult StandardScaled
clf.fit(X_train_stdScaled,y_train)
defStdKnnTrainScore = cross_val_score(clf, X_train_stdScaled, y_train, cv=cv).mean()
defStdKnnValidScore = cross_val_score(clf, X_valid_stdScaled, y_valid, cv=cv).mean()
print('KNeighborsClassifier StandardScaled train score: {:.3f}'.format(defStdKnnTrainScore))
print('KNeighborsClassifier StandardScaled valid score: {:.3f}'.format(defStdKnnValidScore))
# Perceptron
clf = Perceptron(random_state = RANDOM_STATE)

#defult NotScaled
clf.fit(X_train,y_train)
defPerceptronTrainScore = cross_val_score(clf, X_train, y_train, cv=cv).mean()
defPerceptronValidScore = cross_val_score(clf, X_valid, y_valid, cv=cv).mean()
print('Perceptron NotScaled train score: {:.3f}'.format(defPerceptronTrainScore))
print('Perceptron NotScaled valid score: {:.3f}'.format(defPerceptronValidScore))

#defult minmaxScaled
clf.fit(X_train_minmaxScaled,y_train)
defMinMaxPerceptronTrainScore = cross_val_score(clf, X_train_minmaxScaled, y_train, cv=cv).mean()
defMinMaxPerceptronValidScore = cross_val_score(clf, X_valid_minmaxScaled, y_valid, cv=cv).mean()
print('Perceptron MinMaxScaled train score: {:.3f}'.format(defMinMaxPerceptronTrainScore))
print('Perceptron MinMaxScaled valid score: {:.3f}'.format(defMinMaxPerceptronValidScore))

#defult StandardScaled
clf.fit(X_train_stdScaled,y_train)
defStdPerceptronTrainScore = cross_val_score(clf, X_train_stdScaled, y_train, cv=cv).mean()
defStdPerceptronValidScore = cross_val_score(clf, X_valid_stdScaled, y_valid, cv=cv).mean()
print('Perceptron StandardScaled train score: {:.3f}'.format(defStdPerceptronTrainScore))
print('Perceptron StandardScaled valid score: {:.3f}'.format(defStdPerceptronValidScore))
# MLPClassifier
clf = MLPClassifier(random_state = RANDOM_STATE)

#defult NotScaled
clf.fit(X_train,y_train)
defMlpTrainScore = cross_val_score(clf, X_train, y_train, cv=cv).mean()
defMlpValidScore = cross_val_score(clf, X_valid, y_valid, cv=cv).mean()
print('MLPClassifier NotScaled train score: {:.3f}'.format(defMlpTrainScore))
print('MLPClassifier NotScaled valid score: {:.3f}'.format(defMlpValidScore))

#defult minmaxScaled
clf.fit(X_train_minmaxScaled,y_train)
defMinMaxMlpTrainScore = cross_val_score(clf, X_train_minmaxScaled, y_train, cv=cv).mean()
defMinMaxMlpValidScore = cross_val_score(clf, X_valid_minmaxScaled, y_valid, cv=cv).mean()
print('MLPClassifier MinMaxScaled train score: {:.3f}'.format(defMinMaxMlpTrainScore))
print('MLPClassifier MinMaxScaled valid score: {:.3f}'.format(defMinMaxMlpValidScore))

#defult StandardScaled
clf.fit(X_train_stdScaled,y_train)
defStdMlpTrainScore = cross_val_score(clf, X_train_stdScaled, y_train, cv=cv).mean()
defStdMlpValidScore = cross_val_score(clf, X_valid_stdScaled, y_valid, cv=cv).mean()
print('MLPClassifier StandardScaler train score: {:.3f}'.format(defStdMlpTrainScore))
print('MLPClassifier StandardScaler valid score: {:.3f}'.format(defStdMlpValidScore))
# Define function plotting heatmap results from gridsearchCV
def plot_heatmap_from_grid(clf,i, name):
    # Pick up tuning Parameters
    params = [k for k in clf.cv_results_.keys() if k.startswith('param_')]
    if len(params) != 2: raise Exception('grid has to have exact 2 parameters.') 

    # Define heatmap's index, columns, values
    index = params[0]
    columns = params[1]
    values = 'mean_test_score'

    # Extract Keys from grid
    df_dict = {k: clf.cv_results_[k] for k in clf.cv_results_.keys() & {index, columns, values}}

    # Transform to pd.DataFrame and plot heatmap
    df = pd.DataFrame(df_dict)
    data = df.pivot(index=index, columns=columns, values=values)    
    sns.heatmap(data, annot=True, fmt='.3f',ax=axarr[i]).set_title(name, fontsize=18)
# LogisticRegression
lr = LogisticRegression(random_state=RANDOM_STATE)
grid_param = {'solver': ['newton-cg','lbfgs','liblinear','sag','saga'], 'C': [0.001,0.01,0.1,1,10,100]}
clf = GridSearchCV(lr, grid_param, cv=cv, scoring='accuracy')
fig, axarr = plt.subplots(1, 3, figsize=(28,8))

#GridSearched NotScaled
clf.fit(X_train, y_train)
gridLrTrainScore = clf.best_score_
gridLrValidScore = cross_val_score(clf, X_valid, y_valid, cv=cv).mean()
print('LogisticRegression GridSearchCV NotScaled best params: {}'.format(clf.best_params_))
print('LogisticRegression GridSearchCV NotScaled train score: {:.3f}'.format(gridLrTrainScore))
print('LogisticRegression GridSearchCV NotScaled valid score: {:.3f}'.format(gridLrValidScore))
plot_heatmap_from_grid(clf,0, 'LogisticRegression NotScaled')

#GridSearched minmaxScaled
clf.fit(X_train_minmaxScaled, y_train)
gridMinMaxLrTrainScore = clf.best_score_
gridMinMaxLrValidScore = cross_val_score(clf, X_valid_minmaxScaled, y_valid, cv=cv).mean()
print('LogisticRegression GridSearchCV MinMaxScaled best params: {}'.format(clf.best_params_))
print('LogisticRegression GridSearchCV MinMaxScaled train score: {:.3f}'.format(gridMinMaxLrTrainScore))
print('LogisticRegression GridSearchCV MinMaxScaled valid score: {:.3f}'.format(gridMinMaxLrValidScore))
plot_heatmap_from_grid(clf,1, 'LogisticRegression MinMaxScaled')

#GridSearched StandardScaled
clf.fit(X_train_stdScaled, y_train)
gridStdLrTrainScore = clf.best_score_
gridStdLrValidScore = cross_val_score(clf, X_valid_stdScaled, y_valid, cv=cv).mean()
print('LogisticRegression GridSearchCV StandardScaled best params: {}'.format(clf.best_params_))
print('LogisticRegression GridSearchCV StandardScaled train score: {:.3f}'.format(gridStdLrTrainScore))
print('LogisticRegression GridSearchCV StandardScaled valid score: {:.3f}'.format(gridStdLrValidScore))
plot_heatmap_from_grid(clf,2, 'LogisticRegression StandardScaled')
# GaussianNB
gridGaussianTrainScore = 0
gridMinMaxGaussianTrainScore = 0
gridStdGaussianTrainScore = 0
gridGaussianValidScore = 0
gridMinMaxGaussianValidScore = 0
gridStdGaussianValidScore = 0
# SVC
svc = SVC(random_state=RANDOM_STATE)
grid_param = {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': ['auto', 0.0001, 0.001, 0.01, 0.1, 1, 10]}
clf = GridSearchCV(svc, grid_param, cv=cv, scoring='accuracy', )
fig, axarr = plt.subplots(1, 3, figsize=(28,8))

#GridSearched NotScaled
clf.fit(X_train, y_train)
gridSvcTrainScore = clf.best_score_
gridSvcValidScore = cross_val_score(clf, X_valid, y_valid, cv=cv).mean()
print('SVC GridSearchCV NotScaled best params: {}'.format(clf.best_params_))
print('SVC GridSearchCV NotScaled train score: {:.3f}'.format(gridSvcTrainScore))
print('SVC GridSearchCV NotScaled valid score: {:.3f}'.format(gridSvcValidScore))
plot_heatmap_from_grid(clf,0,'SVC NotScaled')

#GridSearched minmaxScaled
clf.fit(X_train_minmaxScaled, y_train)
gridMinMaxSvcTrainScore = clf.best_score_
gridMinMaxSvcValidScore = cross_val_score(clf, X_valid_minmaxScaled, y_valid, cv=cv).mean()
print('SVC GridSearchCV MinMaxScaled best params: {}'.format(clf.best_params_))
print('SVC GridSearchCV MinMaxScaled train score: {:.3f}'.format(gridMinMaxSvcTrainScore))
print('SVC GridSearchCV MinMaxScaled valid score: {:.3f}'.format(gridMinMaxSvcValidScore))
plot_heatmap_from_grid(clf,1,'SVC MinMaxScaled')

#GridSearched StandardScaled
clf.fit(X_train_stdScaled, y_train)
gridStdSvcTrainScore = clf.best_score_
gridStdSvcValidScore = cross_val_score(clf, X_valid_stdScaled, y_valid, cv=cv).mean()
print('SVC GridSearchCV StandardScaled best params: {}'.format(clf.best_params_))
print('SVC GridSearchCV StandardScaled train score: {:.3f}'.format(gridStdSvcTrainScore))
print('SVC GridSearchCV StandardScaled valid score: {:.3f}'.format(gridStdSvcValidScore))
plot_heatmap_from_grid(clf,2,'SVC StandardScaled')
# LinearSVC
linear_svc = LinearSVC(random_state=RANDOM_STATE)
grid_param = { 'C':[0.001, 0.01, 0.1, 1, 3, 5], 'max_iter': [1000, 10000, 50000, 100000]}
clf = GridSearchCV(linear_svc, grid_param, cv=cv, scoring='accuracy')
fig, axarr = plt.subplots(1, 3, figsize=(28,8))

#GridSearched NotScaled
clf.fit(X_train, y_train)
gridLinearSvcTrainScore = clf.best_score_
gridLinearSvcValidScore = cross_val_score(clf, X_valid, y_valid, cv=cv).mean()
print('LinearSVC GridSearchCV NotScaled best params: {}'.format(clf.best_params_))
print('LinearSVC GridSearchCV NotScaled train score: {:.3f}'.format(gridLinearSvcTrainScore))
print('LinearSVC GridSearchCV NotScaled valid score: {:.3f}'.format(gridLinearSvcValidScore))
plot_heatmap_from_grid(clf,0,'LinearSVC NotScaled')

#GridSearched minmaxScaled
clf.fit(X_train_minmaxScaled, y_train)
gridMinMaxLinearSvcTrainScore = clf.best_score_
gridMinMaxLinearSvcValidScore = cross_val_score(clf, X_valid_minmaxScaled, y_valid, cv=cv).mean()
print('LinearSVC GridSearchCV MinMaxScaled best params: {}'.format(clf.best_params_))
print('LinearSVC GridSearchCV MinMaxScaled train score: {:.3f}'.format(gridMinMaxLinearSvcTrainScore))
print('LinearSVC GridSearchCV MinMaxScaled valid score: {:.3f}'.format(gridMinMaxLinearSvcValidScore))
plot_heatmap_from_grid(clf,1,'LinearSVC MinMaxScaled')

#GridSearched StandardScaled
clf.fit(X_train_stdScaled, y_train)
gridStdLinearSvcTrainScore = clf.best_score_
gridStdLinearSvcValidScore = cross_val_score(clf, X_valid_stdScaled, y_valid, cv=cv).mean()
print('LinearSVC GridSearchCV StandardScaled best params: {}'.format(clf.best_params_))
print('LinearSVC GridSearchCV StandardScaled train score: {:.3f}'.format(gridStdLinearSvcTrainScore))
print('LinearSVC GridSearchCV StandardScaled valid score: {:.3f}'.format(gridStdLinearSvcValidScore))
plot_heatmap_from_grid(clf,2,'LinearSVC StandardScaled')
# RandomForestClassifier
rf = RandomForestClassifier(random_state=RANDOM_STATE)
grid_param = {'n_estimators': [10, 100, 300, 500], 'max_depth': [None, 1, 3, 5, 7, 9]}
clf = GridSearchCV(rf, grid_param, cv=cv, scoring='accuracy')
fig, axarr = plt.subplots(1, 3, figsize=(28,8))

#GridSearched NotScaled
clf.fit(X_train, y_train)
gridRfTrainScore = clf.best_score_
gridRfValidScore = cross_val_score(clf, X_valid, y_valid, cv=cv).mean()
print('RandomForestClassifier GridSearchCV NotScaled best params: {}'.format(clf.best_params_))
print('RandomForestClassifier GridSearchCV NotScaled train score: {:.3f}'.format(gridRfTrainScore))
print('RandomForestClassifier GridSearchCV NotScaled valid score: {:.3f}'.format(gridRfValidScore))
plot_heatmap_from_grid(clf,0,'RandomForestClassifier NotScaled')

#GridSearched minmaxScaled
clf.fit(X_train_minmaxScaled, y_train)
gridMinMaxRfTrainScore = clf.best_score_
gridMinMaxRfValidScore = cross_val_score(clf, X_valid_minmaxScaled, y_valid, cv=cv).mean()
print('RandomForestClassifier GridSearchCV MinMaxScaled best params: {}'.format(clf.best_params_))
print('RandomForestClassifier GridSearchCV MinMaxScaled train score: {:.3f}'.format(gridMinMaxRfTrainScore))
print('RandomForestClassifier GridSearchCV MinMaxScaled valid score: {:.3f}'.format(gridMinMaxRfValidScore))
plot_heatmap_from_grid(clf,1,'RandomForestClassifier MinMaxScaled')

#GridSearched StandardScaled
clf.fit(X_train_stdScaled, y_train)
gridStdRfTrainScore = clf.best_score_
gridStdRfValidScore = cross_val_score(clf, X_valid_stdScaled, y_valid, cv=cv).mean()
print('RandomForestClassifier GridSearchCV StandardScaled best params: {}'.format(clf.best_params_))
print('RandomForestClassifier GridSearchCV StandardScaled train score: {:.3f}'.format(gridStdRfTrainScore))
print('RandomForestClassifier GridSearchCV StandardScaled valid score: {:.3f}'.format(gridStdRfValidScore))
plot_heatmap_from_grid(clf,2,'RandomForestClassifier StandardScaled')
# DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
grid_param = {'max_depth': [None, 1, 3, 5, 7, 9, 11], 'max_features': [None, 1, 3, 5, 7, 9, 11]}
clf = GridSearchCV(dt, grid_param, cv=cv, scoring='accuracy')
fig, axarr = plt.subplots(1, 3, figsize=(28,8))

#GridSearched NotScaled
clf.fit(X_train, y_train)
gridDtTrainScore = clf.best_score_
gridDtValidScore = cross_val_score(clf, X_valid, y_valid, cv=cv).mean()
print('DecisionTreeClassifier GridSearchCV NotScaled best params: {}'.format(clf.best_params_))
print('DecisionTreeClassifier GridSearchCV NotScaled train score: {:.3f}'.format(gridDtTrainScore))
print('DecisionTreeClassifier GridSearchCV NotScaled valid score: {:.3f}'.format(gridDtValidScore))
plot_heatmap_from_grid(clf,0,'DecisionTreeClassifier NotScaled')

#GridSearched minmaxScaled
clf.fit(X_train_minmaxScaled, y_train)
gridMinMaxDtTrainScore = clf.best_score_
gridMinMaxDtValidScore = cross_val_score(clf, X_valid_minmaxScaled, y_valid, cv=cv).mean()
print('DecisionTreeClassifier GridSearchCV MinMaxScaled best params: {}'.format(clf.best_params_))
print('DecisionTreeClassifier GridSearchCV MinMaxScaled train score: {:.3f}'.format(gridMinMaxDtTrainScore))
print('DecisionTreeClassifier GridSearchCV MinMaxScaled valid score: {:.3f}'.format(gridMinMaxDtValidScore))
plot_heatmap_from_grid(clf,1,'DecisionTreeClassifier MinMaxScaled')

#GridSearched StandardScaled
clf.fit(X_train_stdScaled, y_train)
gridStdDtTrainScore = clf.best_score_
gridStdDtValidScore = cross_val_score(clf, X_valid_stdScaled, y_valid, cv=cv).mean()
print('DecisionTreeClassifier GridSearchCV StandardScaled best params: {}'.format(clf.best_params_))
print('DecisionTreeClassifier GridSearchCV StandardScaled train score: {:.3f}'.format(gridStdDtTrainScore))
print('DecisionTreeClassifier GridSearchCV StandardScaled valid score: {:.3f}'.format(gridStdDtValidScore))
plot_heatmap_from_grid(clf,2,'DecisionTreeClassifier StandardScaled')
# KNeighborsClassifier
knn = KNeighborsClassifier()
grid_param = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'weights': ['uniform', 'distance']}
clf = GridSearchCV(knn, grid_param, cv=cv, scoring='accuracy')
fig, axarr = plt.subplots(1, 3, figsize=(28,8))

#GridSearched NotScaled
clf.fit(X_train, y_train)
gridKnnTrainScore = clf.best_score_
gridKnnValidScore = cross_val_score(clf, X_valid, y_valid, cv=cv).mean()
print('KNeighborsClassifier GridSearchCV NotScaled best params: {}'.format(clf.best_params_))
print('KNeighborsClassifier GridSearchCV NotScaled train score: {:.3f}'.format(gridKnnTrainScore))
print('KNeighborsClassifier GridSearchCV NotScaled valid score: {:.3f}'.format(gridKnnValidScore))
plot_heatmap_from_grid(clf,0,'KNeighborsClassifier NotScaled')

#GridSearched minmaxScaled
clf.fit(X_train_minmaxScaled, y_train)
gridMinMaxKnnTrainScore = clf.best_score_
gridMinMaxKnnValidScore = cross_val_score(clf, X_valid_minmaxScaled, y_valid, cv=cv).mean()
print('KNeighborsClassifier GridSearchCV MinMaxScaled best params: {}'.format(clf.best_params_))
print('KNeighborsClassifier GridSearchCV MinMaxScaled train score: {:.3f}'.format(gridMinMaxKnnTrainScore))
print('KNeighborsClassifier GridSearchCV MinMaxScaled valid score: {:.3f}'.format(gridMinMaxKnnValidScore))
plot_heatmap_from_grid(clf,1,'KNeighborsClassifier MinMaxScaled')

#GridSearched StandardScaled
clf.fit(X_train_stdScaled, y_train)
gridStdKnnTrainScore = clf.best_score_
gridStdKnnValidScore = cross_val_score(clf, X_valid_stdScaled, y_valid, cv=cv).mean()
print('KNeighborsClassifier GridSearchCV StandardScaled best params: {}'.format(clf.best_params_))
print('KNeighborsClassifier GridSearchCV StandardScaled train score: {:.3f}'.format(gridStdKnnTrainScore))
print('KNeighborsClassifier GridSearchCV StandardScaled valid score: {:.3f}'.format(gridStdKnnValidScore))
plot_heatmap_from_grid(clf,2,'KNeighborsClassifier StandardScaled')
# Perceptron
perceptron = Perceptron(random_state=RANDOM_STATE)
grid_param = {'penalty': [None,'l2', 'l1','elasticnet'],'max_iter': [5, 10, 50, 100, 500, 1000, 5000]}
clf = GridSearchCV(perceptron, grid_param, cv=cv, scoring='accuracy')
fig, axarr = plt.subplots(1, 3, figsize=(28,8))

#GridSearched NotScaled
clf.fit(X_train, y_train)
gridPerceptronTrainScore = clf.best_score_
gridPerceptronValidScore = cross_val_score(clf, X_valid, y_valid, cv=cv).mean()
print('Perceptron GridSearchCV NotScaled best params: {}'.format(clf.best_params_))
print('Perceptron GridSearchCV NotScaled train score: {:.3f}'.format(gridPerceptronTrainScore))
print('Perceptron GridSearchCV NotScaled valid score: {:.3f}'.format(gridPerceptronValidScore))
plot_heatmap_from_grid(clf,0,'Perceptron NotScaled')

#GridSearched minmaxScaled
clf.fit(X_train_minmaxScaled, y_train)
gridMinMaxPerceptronTrainScore = clf.best_score_
gridMinMaxPerceptronValidScore = cross_val_score(clf, X_valid_minmaxScaled, y_valid, cv=cv).mean()
print('Perceptron GridSearchCV MinMaxScaled best params: {}'.format(clf.best_params_))
print('Perceptron GridSearchCV MinMaxScaled train score: {:.3f}'.format(gridMinMaxPerceptronTrainScore))
print('Perceptron GridSearchCV MinMaxScaled valid score: {:.3f}'.format(gridMinMaxPerceptronValidScore))
plot_heatmap_from_grid(clf,1,'Perceptron MinMaxScaled')

#GridSearched StandardScaled
clf.fit(X_train_stdScaled, y_train)
gridStdPerceptronTrainScore = clf.best_score_
gridStdPerceptronValidScore = cross_val_score(clf, X_valid_stdScaled, y_valid, cv=cv).mean()
print('Perceptron GridSearchCV StandardScaled best params: {}'.format(clf.best_params_))
print('Perceptron GridSearchCV StandardScaled train score: {:.3f}'.format(gridStdPerceptronTrainScore))
print('Perceptron GridSearchCV StandardScaled valid score: {:.3f}'.format(gridStdPerceptronValidScore))
plot_heatmap_from_grid(clf,2,'Perceptron StandardScaled')
# MLPClassifier
mlp = MLPClassifier(random_state=RANDOM_STATE)
grid_param = parameters={'hidden_layer_sizes': [(10,), (100,), (100,100,)],'alpha': [0.0001, 0.001, 0.01]}
clf = GridSearchCV(mlp, grid_param, cv=cv, scoring='accuracy')
fig, axarr = plt.subplots(1, 3, figsize=(28,8))

#GridSearched NotScaled
clf.fit(X_train, y_train)
gridMlpTrainScore = clf.best_score_
gridMlpValidScore = cross_val_score(clf, X_valid, y_valid, cv=cv).mean()
print('MLPClassifier GridSearchCV NotScaled best params: {}'.format(clf.best_params_))
print('MLPClassifier GridSearchCV NotScaled train score: {:.3f}'.format(gridMlpTrainScore))
print('MLPClassifier GridSearchCV NotScaled valid score: {:.3f}'.format(gridMlpValidScore))
plot_heatmap_from_grid(clf,0,'MLPClassifier NotScaled')

#GridSearched minmaxScaled
clf.fit(X_train_minmaxScaled, y_train)
gridMinMaxMlpTrainScore = clf.best_score_
gridMinMaxMlpValidScore = cross_val_score(clf, X_valid_minmaxScaled, y_valid, cv=cv).mean()
print('MLPClassifier GridSearchCV MinMaxScaled best params: {}'.format(clf.best_params_))
print('MLPClassifier GridSearchCV MinMaxScaled train score: {:.3f}'.format(gridMinMaxMlpTrainScore))
print('MLPClassifier GridSearchCV MinMaxScaled valid score: {:.3f}'.format(gridMinMaxMlpValidScore))
plot_heatmap_from_grid(clf,1,'MLPClassifier MinMaxScaled')

#GridSearched StandardScaled
clf.fit(X_train_stdScaled, y_train)
gridStdMlpTrainScore = clf.best_score_
gridStdMlpValidScore = cross_val_score(clf, X_valid_stdScaled, y_valid, cv=cv).mean()
print('MLPClassifier GridSearchCV StandardScaled best params: {}'.format(clf.best_params_))
print('MLPClassifier GridSearchCV StandardScaled train score: {:.3f}'.format(gridStdMlpTrainScore))
print('MLPClassifier GridSearchCV StandardScaled valid score: {:.3f}'.format(gridStdMlpValidScore))
plot_heatmap_from_grid(clf,2,'MLPClassifier StandardScaled')
pd.options.display.float_format = '{:,.3f}'.format

trainScores = pd.DataFrame({
    'Model': ['LogisticRegression', 'GaussianNB', 'SVC', 'LinearSVC', 
              'RandomForestClassifier', 'DecisionTreeClassifier',
              'KNeighborsClassifier', 'Perceptron', 
              'MLPClassifier'],
    'NoScaleTrain': [defLrTrainScore, defGaussianTrainScore,
                         defSvcTrainScore, defLinearSvcTrainScore, 
                         defRfTrainScore, defDtTrainScore, 
                         defKnnTrainScore, defPerceptronTrainScore,
                        defMlpTrainScore],
    'MinMaxTrain': [defMinMaxLrTrainScore, defMinMaxGaussianTrainScore,
                         defMinMaxSvcTrainScore, defMinMaxLinearSvcTrainScore, 
                         defMinMaxRfTrainScore, defMinMaxDtTrainScore, 
                         defMinMaxKnnTrainScore, defMinMaxPerceptronTrainScore,
                        defMinMaxMlpTrainScore],
    'StdTrain': [defStdLrTrainScore, defStdGaussianTrainScore,
                         defStdSvcTrainScore, defStdLinearSvcTrainScore, 
                         defStdRfTrainScore, defStdDtTrainScore, 
                         defStdKnnTrainScore, defStdPerceptronTrainScore,
                        defStdMlpTrainScore],
    'gridNoScaleTrain': [gridLrTrainScore, gridGaussianTrainScore,
                         gridSvcTrainScore, gridLinearSvcTrainScore, 
                         gridRfTrainScore, gridDtTrainScore, 
                         gridKnnTrainScore, gridPerceptronTrainScore,
                        gridMlpTrainScore],
    'gridMinMaxTrain': [gridMinMaxLrTrainScore, gridMinMaxGaussianTrainScore,
                         gridMinMaxSvcTrainScore, gridMinMaxLinearSvcTrainScore, 
                         gridMinMaxRfTrainScore, gridMinMaxDtTrainScore, 
                         gridMinMaxKnnTrainScore, gridMinMaxPerceptronTrainScore,
                        gridMinMaxMlpTrainScore],
    'gridStdTrain': [gridStdLrTrainScore, gridStdGaussianTrainScore,
                         gridStdSvcTrainScore, gridStdLinearSvcTrainScore, 
                         gridStdRfTrainScore, gridStdDtTrainScore, 
                         gridStdKnnTrainScore, gridStdPerceptronTrainScore,
                        gridStdMlpTrainScore],
}).set_index('Model')

validScores = pd.DataFrame({
    'Model': ['LogisticRegression', 'GaussianNB', 'SVC', 'LinearSVC', 
              'RandomForestClassifier', 'DecisionTreeClassifier',
              'KNeighborsClassifier', 'Perceptron', 
              'MLPClassifier'],
    'NoScaleValid': [defLrValidScore, defGaussianValidScore,
                         defSvcValidScore, defLinearSvcValidScore, 
                         defRfValidScore, defDtValidScore, 
                         defKnnValidScore, defPerceptronValidScore,
                        defMlpValidScore],
    'MinMaxValid': [defMinMaxLrValidScore, defMinMaxGaussianValidScore,
                         defMinMaxSvcValidScore, defMinMaxLinearSvcValidScore, 
                         defMinMaxRfValidScore, defMinMaxDtValidScore, 
                         defMinMaxKnnValidScore, defMinMaxPerceptronValidScore,
                        defMinMaxMlpValidScore],
    'StdValid': [defStdLrValidScore, defStdGaussianValidScore,
                         defStdSvcValidScore, defStdLinearSvcValidScore, 
                         defStdRfValidScore, defStdDtValidScore, 
                         defStdKnnValidScore, defStdPerceptronValidScore,
                        defStdMlpValidScore],
    'gridNoScaleValid': [gridLrValidScore, gridGaussianValidScore,
                         gridSvcValidScore, gridLinearSvcValidScore, 
                         gridRfValidScore, gridDtValidScore, 
                         gridKnnValidScore, gridPerceptronValidScore,
                        gridMlpValidScore],
    'gridMinMaxValid': [gridMinMaxLrValidScore, gridMinMaxGaussianValidScore,
                         gridMinMaxSvcValidScore, gridMinMaxLinearSvcValidScore, 
                         gridMinMaxRfValidScore, gridMinMaxDtValidScore, 
                         gridMinMaxKnnValidScore, gridMinMaxPerceptronValidScore,
                        gridMinMaxMlpValidScore],
    'gridStdValid': [gridStdLrValidScore, gridStdGaussianValidScore,
                         gridStdSvcValidScore, gridStdLinearSvcValidScore, 
                         gridStdRfValidScore, gridStdDtValidScore, 
                         gridStdKnnValidScore, gridStdPerceptronValidScore,
                        gridStdMlpValidScore]
}).set_index('Model')
fig, axarr = plt.subplots(1, 2, figsize=(22, 8))
sns.heatmap(trainScores, annot=True, vmin=0.6, fmt='.3f',ax=axarr[0]).set_title("Train Scores", fontsize=18)
sns.heatmap(validScores, annot=True, vmin=0.6, fmt='.3f',ax=axarr[1]).set_title("Valid Scores", fontsize=18)
# LogisticRegression
lr = LogisticRegression(random_state=RANDOM_STATE)
grid_param = {'solver': ['newton-cg','lbfgs','liblinear','sag','saga'], 'C': [0.001,0.01,0.1,1,10,100]}
clf = GridSearchCV(lr, grid_param, cv=cv, scoring='accuracy')

#GridSearched StandardScaled
clf.fit(X_stdScaled, y)
lrStdScaledPred = clf.predict(test_stdScaled)
print('LogisticRegression StandardScaled train score: {:.3f}'.format(clf.best_score_))
# SVC
svc = SVC(random_state=RANDOM_STATE)
grid_param = {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': ['auto', 0.0001, 0.001, 0.01, 0.1, 1, 10]}
clf = GridSearchCV(svc, grid_param, cv=cv, scoring='accuracy', )

#GridSearched minmaxScaled
clf.fit(X_minmaxScaled, y)
svcMinmaxScaledPred = clf.predict(test_minmaxScaled)
print('SVC minmaxScaled train score: {:.3f}'.format(clf.best_score_))
# LinearSVC
linear_svc = LinearSVC(random_state=RANDOM_STATE)
grid_param = { 'C':[0.001, 0.01, 0.1, 1, 3, 5], 'max_iter': [1000, 10000, 50000, 100000]}
linear_svcClf = GridSearchCV(linear_svc, grid_param, cv=cv, scoring='accuracy')

#GridSearched StandardScaled
clf.fit(X_stdScaled, y)
linear_svcStdScaledPred = clf.predict(test_stdScaled)
print('LinearSVC StandardScaled train score: {:.3f}'.format(clf.best_score_))
# RandomForestClassifier
rf = RandomForestClassifier(random_state=RANDOM_STATE)
grid_param = {'n_estimators': [10, 100, 300, 500], 'max_depth': [None, 1, 3, 5, 7, 9]}
clf = GridSearchCV(rf, grid_param, cv=cv, scoring='accuracy')

# GridSearched NotScaled)
clf.fit(X, y)
rfNotScaledPred = clf.predict(oneHot_test)
print('RandomForestClassifier NotScaled train score: {:.3f}'.format(clf.best_score_))
# DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
grid_param = {'max_depth': [None, 1, 3, 5, 7, 9, 11], 'max_features': [None, 1, 3, 5, 7, 9, 11]}
clf = GridSearchCV(dt, grid_param, cv=cv, scoring='accuracy')

# GridSearched NotScaled)
clf.fit(X, y)
dtNotScaledPred = clf.predict(oneHot_test)
print('DecisionTreeClassifier NotScaled train score: {:.3f}'.format(clf.best_score_))
# KNeighborsClassifier
knn = KNeighborsClassifier()
grid_param = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'weights': ['uniform', 'distance']}
clf = GridSearchCV(knn, grid_param, cv=cv, scoring='accuracy')

#GridSearched minmaxScaled
clf.fit(X_minmaxScaled, y)
knnMinmaxScaledPred = clf.predict(test_minmaxScaled)
print('KNeighborsClassifier minmaxScaled train score: {:.3f}'.format(clf.best_score_))
# MLPClassifier
mlp = MLPClassifier(random_state=RANDOM_STATE)
grid_param = parameters={'hidden_layer_sizes': [(10,), (100,), (100,100,)],'alpha': [0.0001, 0.001, 0.01]}
clf = GridSearchCV(mlp, grid_param, cv=cv, scoring='accuracy')

#GridSearched StandardScaled
clf.fit(X_stdScaled, y)
mlpStdScaledPred = clf.predict(test_stdScaled)
print('MLPClassifier StandardScaled train score: {:.3f}'.format(clf.best_score_))
ensembleDf = pd.DataFrame({
    'lrStdScaledPred': lrStdScaledPred,
    'svcMinmaxScaledPred': svcMinmaxScaledPred,
    'linear_svcStdScaledPred': linear_svcStdScaledPred,
    'rfNotScaledPred': rfNotScaledPred,
    'dtNotScaledPred': dtNotScaledPred,
    'knnMinmaxScaledPred': knnMinmaxScaledPred,
    'mlpStdScaledPred': mlpStdScaledPred,
})

ensembleDf['mode'] = ensembleDf.mode(axis=1)

ensembleDf.head()
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': ensembleDf.loc[:,'mode']
})

submission.head()
submission.to_csv('submission.csv', index=False)