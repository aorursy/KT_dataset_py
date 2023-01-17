# regular expressions
import re 

# math and data utilities
import numpy as np
import pandas as pd
import scipy.stats as ss
import itertools

# data and statistics libraries
import sklearn.preprocessing as pre
from sklearn import model_selection
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# visualization libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Set-up default visualization parameters
mpl.rcParams['figure.figsize'] = [10,6]
viz_dict = {
    'axes.titlesize':18,
    'axes.labelsize':16,
}
sns.set_context("notebook", rc=viz_dict)
sns.set_style("whitegrid")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')
test_df = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')
# Question 1: Are we missing any data?
train_df.info()
# Look at the first few entries
train_df.head()
train_df['Title'] = train_df['Name'].str.extract(r'([A-Za-z]+)\.')
train_df.Title.value_counts()
title_dict = {
    'Mrs': 'Mrs', 'Lady': 'Lady', 'Countess': 'Lady',
    'Jonkheer': 'Lord', 'Col': 'Officer', 'Rev': 'Rev',
    'Miss': 'Miss', 'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Miss', 'Dona': 'Lady',
    'Mr': 'Mr', 'Dr': 'Dr', 'Major': 'Officer', 'Capt': 'Officer', 'Sir': 'Lord', 'Don': 'Lord', 'Master': 'Master'
}

train_df.Title = train_df.Title.map(title_dict)
sns.countplot(train_df.Title).set_title("Histogram of Categorical Data: Title")
train_df.head(1)
train_df['FamilySize'] = 1 + train_df.SibSp + train_df.Parch
sns.countplot(train_df.FamilySize)
train_df['Alone'] = train_df.FamilySize.apply(lambda x: 1 if x==1 else 0)
plt.figure(figsize=(8,5))
sns.countplot(train_df.Alone)
train_df['LName'] = train_df.Name.str.extract(r'([A-Za-z]+),')
train_df['NameLength'] = train_df.Name.apply(len)
train_df
train_df.head(1)
# nominal variables (use Cramer's V)
nom_vars = ['Survived', 'Title', 'Embarked', 'Sex', 'Alone', 'LName']

# ordinal variables (nominal-ordinal, use Rank Biserial or Kendall's Tau)
ord_vars = ['Survived', 'Pclass', 'FamilySize', 'Parch', 'SibSp', 'NameLength']

# continuous variables (use Pearson's r)
cont_vars = ['Survived', 'Fare', 'Age']
# convert all string 'object' types to numeric categories
for i in train_df.columns:
    if train_df[i].dtype == 'object':
        train_df[i], _ = pd.factorize(train_df[i])
def cramers_v_matrix(dataframe, variables):
    
    df = pd.DataFrame(index=dataframe[variables].columns, columns=dataframe[variables].columns, dtype="float64")
    
    for v1, v2 in itertools.combinations(variables, 2):
        
        # generate contingency table:
        table = pd.crosstab(dataframe[v1], dataframe[v2])
        n     = len(dataframe.index)
        r, k  = table.shape
        
        # calculate chi squared and phi
        chi2  = ss.chi2_contingency(table)[0]
        phi2  = chi2/n
        
        # bias corrections:
        r = r - ((r - 1)**2)/(n - 1)
        k = k - ((k - 1)**2)/(n - 1)
        phi2 = max(0, phi2 - (k - 1)*(r - 1)/(n - 1))
        
        # fill correlation matrix
        df.loc[v1, v2] = np.sqrt(phi2/min(k - 1, r - 1))
        df.loc[v2, v1] = np.sqrt(phi2/min(k - 1, r - 1))
        np.fill_diagonal(df.values, np.ones(len(df)))
        
    return df
fig, axes = plt.subplots(1, 3, figsize=(20,6))

# nominal variable correlation
ax1 = sns.heatmap(cramers_v_matrix(train_df, nom_vars), annot=True, ax=axes[0], vmin=0)

# ordinal variable correlation: 
ax2 = sns.heatmap(train_df[ord_vars].corr(method='kendall'), annot=True, ax=axes[1], vmin=-1)

# Pearson's correlation:
ax3 = sns.heatmap(train_df[cont_vars].corr(), annot=True, ax=axes[2], vmin=-1)

ax1.set_title("Cramer's V Correlation")
ax2.set_title("Kendall's Tau Correlation")
ax3.set_title("Pearson's R Correlation")
todrop = ['SibSp', 'Ticket', 'Cabin', 'Name']
train_df = train_df.drop(todrop, axis=1)
train_df
X = train_df.drop(['Survived'], axis = 1)
Y = train_df.loc[:, 'Survived']

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=333)
# We normalize the training and testing data separately so as to avoid data leaks.

x_train = pd.DataFrame(pre.scale(x_train), columns=x_train.columns, index=x_train.index)
x_test = pd.DataFrame(pre.scale(x_test), columns=x_test.columns, index=x_test.index)
x_train
x_train.loc[x_train.Age.isnull(), 'Age'] = x_train.loc[:, 'Age'].median()
x_test.loc[x_test.Age.isnull(), 'Age'] = x_test.loc[:, 'Age'].median()
x_train.info()
def kfold_evaluate(model, folds=5):
    eval_dict = {}
    accuracy = 0
    f1       = 0
    AUC      = 0
    
    skf = model_selection.StratifiedKFold(n_splits=folds)
    
    # perform k splits on the training data. Gather performance results.
    for train_idx, test_idx in skf.split(x_train, y_train):
        xk_train, xk_test = x_train.iloc[train_idx], x_train.iloc[test_idx]
        yk_train, yk_test = y_train.iloc[train_idx], y_train.iloc[test_idx]
    
        model.fit(xk_train, yk_train)
        y_pred = model.predict(xk_test)
        report = metrics.classification_report(yk_test, y_pred, output_dict=True)
        
        prob_array = model.predict_proba(xk_test)
    
        fpr, tpr, huh = metrics.roc_curve(yk_test, model.predict_proba(xk_test)[:,1])
        auc = metrics.auc(fpr, tpr)
        accuracy   += report['accuracy']
        f1         += report['macro avg']['f1-score']
        AUC        += auc
        
    # Average performance metrics over the k folds
    measures = np.array([accuracy, f1, AUC])
    measures = measures/folds

    # Add metric averages to dictionary and return.
    eval_dict['Accuracy']  = measures[0]
    eval_dict['F1 Score']  = measures[1]
    eval_dict['AUC']       = measures[2]  
    eval_dict['Model']     = model
    
    return eval_dict

# a function to pretty print our dictionary of dictionaries:
def pprint(web, level):
    for k,v in web.items():
        if isinstance(v, dict):
            print('\t'*level, f'{k}: ')
            level += 1
            pprint(v, level)
            level -= 1
        else:
            print('\t'*level, k, ": ", v)
evals = {}
evals['KNN'] = kfold_evaluate(KNeighborsClassifier())
evals['Logistic Regression'] = kfold_evaluate(LogisticRegression(max_iter=1000))
evals['Random Forest'] = kfold_evaluate(RandomForestClassifier())
evals['SVC'] = kfold_evaluate(SVC(probability=True))
result_df = pd.DataFrame(evals)
result_df.drop('Model', axis=0).plot(kind='bar', ylim=(0.7, 0.9)).set_title("Base Model Performance")
plt.xticks(rotation=0)
plt.show()
result_df
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

pprint(random_grid, 0)
# create RandomizedSearchCV object
searcher = model_selection.RandomizedSearchCV(estimator = RandomForestClassifier(),
                                            param_distributions = random_grid,
                                            n_iter = 10, # Number of parameter settings to sample (this could take a while)
                                            cv     = 3,  # Number of folds for k-fold validation 
                                            n_jobs = -1, # Use all processors to compute in parallel
                                            random_state=0) 
search = searcher.fit(x_train, y_train)
params = search.best_params_
params
tuning_eval = {}
tuned_rf = RandomForestClassifier(**params)
basic_rf = RandomForestClassifier()

tuning_eval['Tuned'] = kfold_evaluate(tuned_rf)
tuning_eval['Basic'] = kfold_evaluate(basic_rf)

result_df = pd.DataFrame(tuning_eval)
result_df.drop('Model', axis=0).plot(kind='bar', ylim=(0.7, 0.9)).set_title("Tuning Performance")
plt.xticks(rotation=0)
plt.show()
result_df
y_pred = tuned_rf.predict(x_test)
results = metrics.classification_report(y_test, y_pred,
                                        labels = [0, 1],
                                        target_names = ['Died', 'Survived'],
                                        output_dict = True)

pprint(results, 0)
X = pd.concat([x_train, x_test], axis=0).sort_index()
Y = pd.concat([y_train, y_test], axis=0).sort_index()
tuned_rf.fit(X, Y)
# Feature Engineering:
test_df['Title'] = test_df.Name.str.extract(r'([A-Za-z]+)\.')
test_df['LName'] = test_df.Name.str.extract(r'([A-Za-z]+),')
test_df['NameLength'] = test_df.Name.apply(len)
test_df['FamilySize'] = 1 + test_df.SibSp + test_df.Parch
test_df['Alone'] = test_df.FamilySize.apply(lambda x: 1 if x==1 else 0)
test_df.Title = test_df.Title.map(title_dict)

# Feature Selection
test_df = test_df.drop(todrop, axis=1)

# Imputation of missing age and fare data
test_df.loc[test_df.Age.isna(), 'Age'] = test_df.Age.median()
test_df.loc[test_df.Fare.isna(), 'Fare'] = test_df.Fare.median()

# encode categorical data
for i in test_df.columns:
    if test_df[i].dtype == 'object':
        test_df[i], _ = pd.factorize(test_df[i])
        
# center and scale data 
test_df = pd.DataFrame(pre.scale(test_df), columns=test_df.columns, index=test_df.index)

# ensure columns of unlabeled data are in same order as training data.
test_df = test_df[x_test.columns]
test_df
final = tuned_rf.predict(test_df)
final.sum()/len(final)
submission = pd.DataFrame({'PassengerId':test_df.index,
                           'Survived':final})
submission
submission.to_csv('submission2.csv', index=False)