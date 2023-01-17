# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

#misc libraries
import random
import time


#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
os.listdir('../input')

# Any results you write to the current directory are saved as output.

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
#from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')
sample  = pd.read_csv('../input/gender_submission.csv')
sample.head()
print(train.shape)
train.head()
print('Train columns with null values:\n', train.isnull().sum())
print("-"*10)

print('Test/Validation columns with null values:\n', test.isnull().sum())
print("-"*10)
train.info()
print("-"*10)
test.info()
data_cleaner =[train,test]
for dataset in data_cleaner:    
    #complete missing age with median
    dataset['Age'].fillna(dataset['Age'].mean(), inplace = True)

    #complete embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    #complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
    
#delete the cabin feature/column and others previously stated to exclude in train dataset
drop_column = ['PassengerId','Cabin', 'Ticket']
#data_val.drop(drop_column, axis=1, inplace = True)
#data1.drop(drop_column, axis=1, inplace = True)

print(train.isnull().sum())
print("-"*10)
print(test.isnull().sum())
###CREATE: Feature Engineering for train and test/validation dataset
for dataset in data_cleaner:    
    #Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
    dataset['Fare_Bin'] = pd.qcut(dataset['Fare'], 5)

    #Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
    dataset['Age_Bin'] = pd.cut(dataset['Age'].astype(int), 5)
train['Age_Bin']
#CONVERT: convert objects to category using Label Encoder for train and test/validation dataset
from sklearn import preprocessing
#code categorical data
label = preprocessing.LabelEncoder()
for dataset in data_cleaner:    
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Pclass_Code'] = label.fit_transform(dataset['Pclass'])
    dataset['Age_Bin_Code'] = label.fit_transform(dataset['Age_Bin'])
    #dataset['Fare_Bin_Code'] = label.fit_transform(dataset['Fare_Bin'])
# columns=['Sex_Code','Embarked_Code','Title_Code','AgeBin_Code','FareBin_Code']
print(train.info())
print(train.isnull().sum())
Target = ['Survived']
columns =['Sex_Code','Pclass_Code','Age_Bin_Code']

train_data = train[columns]
test_data = test[columns]
y_true = train[Target]
sub = test['PassengerId']
print('Train columns with null values: \n', train_data.isnull().sum())
print("-"*20)
print (train_data.info())
print('y_true shape :',y_true.shape)
print("*_*"*10)

print('Test/Validation columns with null values: \n', test_data.isnull().sum())
print("-"*10)
print (test_data.info())
print("-"*10)
train_data.head()
from sklearn import ensemble,naive_bayes,neighbors,gaussian_process,linear_model,svm,tree,discriminant_analysis
#Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    naive_bayes.MultinomialNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),
   
    ]
y = y_true[Target]
from sklearn.model_selection import train_test_split
x1,x2,y1,y2 = train_test_split(train_data,y)
print(x1.shape)
print(y1.shape)
from sklearn.metrics import accuracy_score
for alg in MLA:
    Name = alg.__class__.__name__
    alg.fit(x1,y1)
    y_pred=alg.predict(x2)
    print(Name,' Accuracy => ',accuracy_score(y2,y_pred)*100)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(train_data,y)
y_pred = model.predict(test_data)
sample1 = sample
sample1.Survived=y_pred
print(sample1.head())
sample.head()

sample1.to_csv('submission.csv', index=False)
