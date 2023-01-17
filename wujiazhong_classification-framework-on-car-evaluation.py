# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn import model_selection
from sklearn.model_selection import train_test_split

# Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, gaussian_process, discriminant_analysis
from xgboost import XGBClassifier
import sklearn
print(sklearn.__version__)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import os

# Depress xgboost gcc eror on mac
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Any results you write to the current directory are saved as output.
# load raw data from csv file
data = pd.read_csv('../input/car.data.csv')

# See if there is any null value for each feature
print(data.isnull().any())

# Overview of the data
data.info()
data.describe()
# Overview of the dataset
fig, saxis = plt.subplots(2, 3,figsize=(16,12))
data_rating = pd.factorize(data['rating'])
data['rating'] = pd.Series(data_rating[0])
sns.barplot(x = 'buying', y = 'rating', data=data, ax = saxis[0,0])
sns.barplot(x = 'maint', y = 'rating', data=data, ax = saxis[0,1])
sns.barplot(x = 'doors', y = 'rating', data=data, ax = saxis[0,2])
sns.barplot(x = 'persons', y = 'rating', data=data, ax = saxis[1,0])
sns.barplot(x = 'lug_boot', y = 'rating', data=data, ax = saxis[1,1])
sns.barplot(x = 'safety', y = 'rating', data=data, ax = saxis[1,2])
# Factorize data label to numeric for scikit-learn pipeline
data_rating = pd.factorize(data.loc[:,'rating'])
data.loc[:,'rating'] = pd.Series(data_rating[0])

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.attribute_names]


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        for col in X.columns:
            if X.loc[:,col].dtype != 'int64':
                col_data = pd.factorize(X.loc[:,col])
                X.loc[:, col] = pd.Series(col_data[0], name=col)
        self.most_frequent_ = pd.Series([X.loc[:,c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)

# Define the data pipeline to extract, convert and transform data
cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["buying", "maint", "doors", "persons", "lug_boot", "safety"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False))
])

X_data = cat_pipeline.fit_transform(data)
col_data = pd.factorize(data.loc[:,'rating'])
y_data = pd.Series(col_data[0]).values

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=42)
# Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    # Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    # Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),

    # GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),

    # Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),

    # Nearest Neighbor
    neighbors.KNeighborsClassifier(),

    # Trees
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),

    # Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    # xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()
]

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 )
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = data.loc[:,'rating']

# a = ensemble.ExtraTreesClassifier()
# a.fit(X_train, y_train)
# print(a.score(X_test, y_test))
row_index = 0
for alg in MLA:
    fit = alg.fit(X_train, y_train)
    predicted = fit.predict(X_test)
    # fp, tp, th = roc_curve(y_test, predicted)
    MLA_name = alg.__class__.__name__
    print(alg.__class__.__name__)
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, X_data, y_data, cv=cv_split, return_train_score=True)
    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()
    # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std() * 3  # let's know the worst that can happen!

    # save MLA predictions - see section 6 for usage
    alg.fit(X_data, y_data)
    MLA_predict[MLA_name] = alg.predict(X_data)

    row_index += 1
pd.set_option('display.max_columns', None)


#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare
sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')
vote_est = [
    #Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html
    ('ada', ensemble.AdaBoostClassifier()),
    ('bc', ensemble.BaggingClassifier()),
    ('etc',ensemble.ExtraTreesClassifier()),
    ('gbc', ensemble.GradientBoostingClassifier()),
    ('rfc', ensemble.RandomForestClassifier()),

    #Gaussian Processes: http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-classification-gpc
    ('gpc', gaussian_process.GaussianProcessClassifier()),
    
    #GLM: http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ('lr', linear_model.LogisticRegressionCV()),
    
    #Navies Bayes: http://scikit-learn.org/stable/modules/naive_bayes.html
    ('bnb', naive_bayes.BernoulliNB()),
    ('gnb', naive_bayes.GaussianNB()),
    
    #Nearest Neighbor: http://scikit-learn.org/stable/modules/neighbors.html
    ('knn', neighbors.KNeighborsClassifier()),
    
    #SVM: http://scikit-learn.org/stable/modules/svm.html
    ('svc', svm.SVC(probability=True)),
    
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
   ('xgb', XGBClassifier())

]


#Hard Vote or majority rules
vote_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
vote_hard_cv = model_selection.cross_validate(vote_hard, X_train, y_train, cv  = cv_split)
vote_hard.fit(X_train, y_train)

print("Hard Voting Training w/bin score mean: {:.2f}". format(vote_hard_cv['train_score'].mean()*100)) 
print("Hard Voting Test w/bin score mean: {:.2f}". format(vote_hard_cv['test_score'].mean()*100))
print("Hard Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_hard_cv['test_score'].std()*100*3))
print('-'*10)


#Soft Vote or weighted probabilities
vote_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
vote_soft_cv = model_selection.cross_validate(vote_soft, X_data, y_data, cv  = cv_split)
vote_soft.fit(X_data, y_data)

print("Soft Voting Training w/bin score mean: {:.2f}". format(vote_soft_cv['train_score'].mean()*100)) 
print("Soft Voting Test w/bin score mean: {:.2f}". format(vote_soft_cv['test_score'].mean()*100))
print("Soft Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_soft_cv['test_score'].std()*100*3))
print('-'*10)
# This is an example how to use random search on GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint 
n_iter_search = 20
param_dist= {
    "max_depth": sp_randint(2, 17),
    "min_samples_split": sp_randint(2, 101),
    "min_samples_leaf": sp_randint(1, 101)
}

gsearch1 = RandomizedSearchCV(estimator = ensemble.GradientBoostingClassifier(learning_rate=0.1, 
                              max_features='sqrt', subsample=0.8,random_state=10),
                              param_distributions=param_dist, n_iter=n_iter_search)
gsearch1.fit(X_train, y_train)
gsearch1.best_params_, gsearch1.best_score_