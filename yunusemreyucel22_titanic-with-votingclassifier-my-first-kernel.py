# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import string

import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
data_val = pd.read_csv('/kaggle/input/titanic/test.csv')

data1 = pd.read_csv('/kaggle/input/titanic/train.csv')



data_all = [data1, data_val]



data1.info()

data_val.info()
print(data1.isin([0]).sum(),"--------"*9,data_val.isin([0]).sum())

for dataset in data_all:

    

    #Title Feature

    dataset['Title'] = dataset['Name'].str.split(", ",expand = True)[1].str.split(".",expand = True)[0]

    title_names = (dataset["Title"].value_counts() < 10)

    dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

    

    #FamilySize Feature

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    

    #IsAlone Feature

    dataset['IsAlone'] = 1

    dataset['IsAlone'].loc[dataset['FamilySize']> 1] = 0

    dataset['IsAlone'].loc[dataset['FamilySize']> 1] = 0 

    

    #Cabin Feature

    dataset.Cabin= [1 if each.__class__ == str else 0 for each in dataset.Cabin]

    
print(data1.groupby('Title')['Age'].median())



#Master     3.5   

#Misc      44.5

#Miss      21.0

#Mr        30.0

#Mrs       35.0



for dataset in data_all:

    m1 = dataset.Title == "Master"

    m2 = dataset.Title == "Misc"

    m3 = dataset.Title == "Miss"

    m4 = dataset.Title == "Mr"

    m5 = dataset.Title == "Mrs";





    dataset.loc[m1,'Age'] = dataset.loc[m1,'Age'].fillna(3.5)

    dataset.loc[m2,'Age'] = dataset.loc[m2,'Age'].fillna(44.5)

    dataset.loc[m3,'Age'] = dataset.loc[m3,'Age'].fillna(21)

    dataset.loc[m4,'Age'] = dataset.loc[m4,'Age'].fillna(30)

    dataset.loc[m5,'Age'] = dataset.loc[m5,'Age'].fillna(35)
# Dealing With Embarked



print(data1.info())



for dataset in data_all:

    dataset['Embarked'].fillna(data1['Embarked'].mode()[0], 

            inplace = True) # using train dataset's mode value to fill data_val again.
# Dealing with Fare and Age



sns.distplot(data1.Age)

sns.distplot(data1.Fare)   

       
# Dealing with fare 1/4



""" There are some fare values with " 0 " and 1 data_val fare value is nan. Before using fare to make feature engineering,

    I should deal with those. data_val will be easy, but what to do with 0 values?

    Most likely those tickets were " free " for them, Maybe a prize for something maybe nepotism, who knows?

    But giving fare = 0 value to people who are in first class? It's certainly bad for the model."""

def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

correlation_heatmap(data1)



#Acording to correlation map, to fill those values I can make prediction with cabin and Pclass features. 
# Dealing with fare 2/4

# since there is only 1 missing value for data_val, I will do better on that.





temp = data1[(data1['Embarked']=='S') & (data1['Cabin']==0) & (data1['IsAlone']==1) & (data1['Pclass']==3)]

print("median: ",temp.Fare.median())

print("mode: ",temp.Fare.mode())

print("mean: ",temp.Fare.mean())



# I feel like its best to fill it with 8.05, it is most frequent value plus between median and mean values.



data_val.Fare.fillna(8.05,inplace = True)
# Dealing with fare 3/4



# now time to deal with fare = 0 cases.



temp1 = data1[(data1['Pclass']==1 ) & (data1['Cabin']==0) ].Fare.median()

temp2 = data1[(data1['Pclass']==1 ) & (data1['Cabin']==1) ].Fare.median()



temp3 = data1[(data1['Pclass']==2 ) & (data1['Cabin']==0) ].Fare.median()

temp4 = data1[(data1['Pclass']==2 ) & (data1['Cabin']==1) ].Fare.median()



temp5 = data1[(data1['Pclass']==3 ) & (data1['Cabin']==0) ].Fare.median()

temp6 = data1[(data1['Pclass']==3 ) & (data1['Cabin']==1) ].Fare.median()



print("temp1: ",temp1)

print("temp2: ",temp2)

print("temp3: ",temp3)

print("temp4: ",temp4)

print("temp5: ",temp5)

print("temp6: ",temp6)



# now I will change zero values 



for dataset in data_all:

    

    dataset['Fare'].loc[(dataset.Pclass == 1) & (dataset.Cabin == 0 ) & (dataset.Fare == 0)] = temp1

    dataset['Fare'].loc[(dataset.Pclass == 1) & (dataset.Cabin == 1 ) & (dataset.Fare == 0)] = temp2

    dataset['Fare'].loc[(dataset.Pclass == 2) & (dataset.Cabin == 0 ) & (dataset.Fare == 0)] = temp3

    dataset['Fare'].loc[(dataset.Pclass == 2) & (dataset.Cabin == 1 ) & (dataset.Fare == 0)] = temp4

    dataset['Fare'].loc[(dataset.Pclass == 3) & (dataset.Cabin == 0 ) & (dataset.Fare == 0)] = temp5
#temp2:  nan

#temp4:  nan

#temp6:  nan

    

# It is not a mistake. it indicates that there are not that kind of data there,

# I didn'T check it one by one, coded every possibilty just in case.



#check here: no null values now.



print(data1.info(),data_val.info())
for dataset in data_all:

    dataset['FarePerPerson'] = dataset.Fare/dataset.FamilySize



# instead of using Fare feature, FarePerPerson makes more sense.  

    
# Transforming Categorical Data Into Numerical Data part1

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

    

for dataset in data_all:

    

    #Title Feature

    dataset['Title'] = dataset['Name'].str.split(", ",expand = True)[1].str.split(".",expand = True)[0]

    title_names = (dataset["Title"].value_counts() < 10)

    dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)



    #SexCode Feature

    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])

    

    #Transforming Age and Fare features into discrete features.



    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 8) # I tried different numbers for those 2 values, 8 - 5 seems okey. 

    dataset['FareBin'] = pd.qcut(dataset['FarePerPerson'], 5) # 

    

    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])

    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])
# Transforming Categorical Data Into Numerical Data part2, using dummies.

    

dummies1 = pd.get_dummies(data1['Embarked'])

dummies2 = pd.get_dummies(data1['Pclass'])

dummies3 = pd.get_dummies(data1['Title'])

data1  = pd.concat([data1,dummies1,dummies2,dummies3],axis = 1)



dummies1 = pd.get_dummies(data_val['Embarked'])

dummies2 = pd.get_dummies(data_val['Pclass'])

dummies3 = pd.get_dummies(data_val['Title'])

data_val  = pd.concat([data_val,dummies1,dummies2,dummies3],axis = 1)

# 2 new Features: baby, old



data1['baby'] = [1 if each <= 3 else 0 for each in data1.Age]

data1['Old'] = [1 if 65 < each else 0 for each in data1.Age]



data_val['baby'] = [1 if each <= 3 else 0 for each in data_val.Age]

data_val['Old'] = [1 if 65 < each else 0 for each in data_val.Age]

    
# Dropping unnecassary columns + creating train and target dataframes



data_train = data1.drop(['Embarked','Pclass','Title','Fare','AgeBin','FareBin',

                         'FarePerPerson','Survived','Age','Name','PassengerId','Ticket','Sex'],axis = 1)

data_target = pd.DataFrame(data1['Survived'])

# Lets check our training data now:



data_train.sample(10)
correlation_heatmap(data_train)
import scipy as sp 

import time

import warnings

warnings.filterwarnings('ignore')



#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier #bu yok bende 1. hata



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics
# I will test how those models with default parameters work with the data



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



    

    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html

    XGBClassifier()    

    ]
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 

               'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']



MLA_compare = pd.DataFrame(columns = MLA_columns)

MLA_predict = data_target.copy(deep = True) 

MLA_Score = pd.DataFrame()

#note: this is an alternative to train_test_split

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%



row_index = 0



for alg in MLA:

    

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index,'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    

    cv_results = model_selection.cross_validate(alg, data_train, data_target, return_train_score = True, cv  = cv_split)

    

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()

    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()

    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()

    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!



    #save MLA predictions - see section 6 for usage

    alg.fit(data_train, data_target)

    MLA_predict[MLA_name] = alg.predict(data_train)

    MLA_Score[MLA_name] = MLA_compare['MLA Test Accuracy Mean']

    

    row_index += 1



MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)



sns.barplot(y = MLA_compare['MLA Name'], x = MLA_compare['MLA Test Accuracy Mean'],color = "c", order = MLA_compare['MLA Name'] )

plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('Accuracy Score (%)')

plt.ylabel('Algorithm')
MLA_predict.drop(['SGDClassifier','Perceptron','GaussianNB','PassiveAggressiveClassifier', 'QuadraticDiscriminantAnalysis','BernoulliNB','ExtraTreeClassifier',

                  'ExtraTreesClassifier','DecisionTreeClassifier','GaussianProcessClassifier','KNeighborsClassifier'],axis = 1, inplace = True)
#%%  Building up a GridSearch model.





vote_est = [

    #Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html

#    ('ada', ensemble.AdaBoostClassifier()),111

#    ('bc', ensemble.BaggingClassifier()),111

#    #('etc',ensemble.ExtraTreesClassifier()),

#    ('gbc', ensemble.GradientBoostingClassifier()),

     ('rfc', ensemble.RandomForestClassifier()),

#    

#    #GLM: http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

    ('lr', linear_model.LogisticRegressionCV()),

    

    ('gpc', gaussian_process.GaussianProcessClassifier()),

    

    #Navies Bayes: http://scikit-learn.org/stable/modules/naive_bayes.html

    #('bnb', naive_bayes.BernoulliNB()), discarded

#    ('gnb', naive_bayes.GaussianNB()), 

#    

#    #Nearest Neighbor: http://scikit-learn.org/stable/modules/neighbors.html

#    ('knn', neighbors.KNeighborsClassifier(algorithm= 'brute', n_neighbors= 7, weights= 'uniform')),

    

    #SVM: http://scikit-learn.org/stable/modules/svm.html

    ('svc', svm.SVC(probability=True)),

#    

#    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html

    ('xgb', XGBClassifier()),

    

#    ('ridge', linear_model.RidgeClassifierCV()),

#    ('lsvc',  svm.LinearSVC()),

    ('lda', discriminant_analysis.LinearDiscriminantAnalysis())



]



# Remember, cv_results = model_selection.cross_validate(alg, data_train, data_target, return_train_score = True, cv  = cv_split) was coded before. 
grid_n_estimator = [10, 50, 100, 300,350,400]

grid_ratio = [.1, .25, .5, .75, 1.0]

grid_learn = [.01, .03, .05, .1, .25]

grid_max_depth = [2, 4, 6, 8, 10, None]

grid_min_samples = [5, 10, .03, .05, .10]

grid_criterion = ['gini', 'entropy']

grid_bool = [True, False]

grid_seed = [0]





grid_param = [

#            [{

#            #AdaBoostClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

#            'n_estimators': grid_n_estimator, #default=50

#            'learning_rate':[ 0.05], #default=1

#            'algorithm': ['SAMME.R'], #default=’SAMME.R

#            'random_state': grid_seed

#            }],

#       

#    

#            [{

#            #BaggingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier

#            'n_estimators': grid_n_estimator, #default=10

#            'max_samples': [0.1], #default=1.0

#            'random_state': grid_seed

#             }],



    

#            [{

#            #ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier

#            'n_estimators': grid_n_estimator, #default=10

#            'criterion': grid_criterion, #default=”gini”

#            'max_depth': grid_max_depth, #default=None

#            'random_state': grid_seed

#             }],





#            [{

#            #GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier

#            'loss': ['deviance'], #default=’deviance’

#            'learning_rate': [0.05], 

#            'n_estimators': [300], 

#            'criterion':  ['mse'], #default=”friedman_mse”

#            'max_depth': [2], #default=3   

#            'random_state': grid_seed

#             }],

    

    

            [{

            #RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier

            'n_estimators': grid_n_estimator, #default=10

            'criterion': ['gini'], #default=”gini”

            'max_depth': [6], #default=None

            'oob_score': grid_bool,

            'random_state': grid_seed

             }],



            [{

            #LogisticRegressionCV - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV

            'fit_intercept': [False], #default: True

            #'penalty': [None, 'l2'],

            'solver': ['saga'], #default: lbfgs

            'random_state': grid_seed

             }],



#            [{

#            #BernoulliNB - http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB

#            'alpha': grid_ratio, #default: 1.0

#             }],

    

    

            #GaussianNB - 

            [{}],

    

#            [{

#            #KNeighborsClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier

#            'n_neighbors': [1,2,3,4,5,6,7], #default: 5

#            'weights': ['uniform', 'distance'], #default = ‘uniform’

#            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']

#            }],

             

            [{

            #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

            #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r

            'kernel': ['rbf'],

            'C': [2], #default=1.0

            'gamma': [0.1], #edfault: auto

            'decision_function_shape': ['ovo'], #default:ovr

            'probability': [True],

            'random_state': grid_seed

             }],



    

            [{

            #XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html

            'learning_rate': [0.03], #default: .3

            'max_depth': [2], #default 2

            'n_estimators': grid_n_estimator, 

            'seed': grid_seed  

             }],   

    

#             [{

#             #RidgeClassifier ,no predict_proba can't be used on soft voting

#            'fit_intercept':grid_bool,

#            'normalize':grid_bool,

#            }],

#            

#            [{

##             #LinearSVC ,no predict_proba can't be used on soft voting

#            'dual': grid_bool,

#            'fit_intercept': grid_bool,

#            'random_state': grid_seed



#            }],

    

            [{

            #LinearDiscriminantAnalysis

            'store_covariance': grid_bool



            }],

        ]    



start_total = time.perf_counter() #https://docs.python.org/3/library/time.html#time.perf_counter

for clf, param in zip (vote_est, grid_param): 



    #print(clf[1]) #vote_est is a list of tuples, index 0 is the name and index 1 is the algorithm

    #print(param)

    

    

    start = time.perf_counter()        

    best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv = cv_split, scoring = 'roc_auc')

    best_search.fit(data_train, data_target)

    run = time.perf_counter() - start



    best_param = best_search.best_params_

    print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__, best_param, run))

    clf[1].set_params(**best_param) 





run_total = time.perf_counter() - start_total

print('Total optimization time was {:.2f} minutes.'.format(run_total/60))



print('-'*10)
# I checked Parch and SibSp, as I suspected, including them with their high rate of

# correlation decreases accuracy. 



data_train = data_train.drop(['Parch'],axis =1) 

data_train = data_train.drop(['SibSp'],axis =1) 
data_train.head()
#%% Hard Voting 





grid_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')

grid_hard_cv = model_selection.cross_validate(grid_hard, data_train, data_target, cv  = cv_split,return_train_score = True)

grid_hard.fit(data_train, data_target)



print("Hard Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_hard_cv['train_score'].mean()*100)) 

print("Hard Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_hard_cv['test_score'].mean()*100))

print("Hard Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_hard_cv['test_score'].std()*100*3))

print('-'*10)



#Soft Vote or weighted probabilities w/Tuned Hyperparameters









grid_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')

grid_soft_cv = model_selection.cross_validate(grid_soft, data_train, data_target, cv  = cv_split, return_train_score = True)

grid_soft.fit(data_train, data_target)



print("Soft Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_soft_cv['train_score'].mean()*100)) 

print("Soft Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_soft_cv['test_score'].mean()*100))

print("Soft Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_soft_cv['test_score'].std()*100*3))

print('-'*10)

# Those lines of codes for submitting result.

data_val = data_val.drop(['Embarked','Pclass','Title','Fare','AgeBin','FareBin','FarePerPerson','Age','Name','PassengerId','Ticket','Sex','Parch','SibSp'],axis = 1)

data_val.head()
data_val_raw = pd.read_csv('/kaggle/input/titanic/test.csv')



predictions = grid_soft.predict(data_val)



output = pd.DataFrame({'PassengerId': data_val_raw.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")