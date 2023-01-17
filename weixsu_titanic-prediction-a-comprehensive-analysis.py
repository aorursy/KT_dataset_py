# This block is from https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy
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

import seaborn as sns #collection of functions for data visualization
print("seaborn version: {}". format(sns.__version__))

from sklearn.preprocessing import OneHotEncoder #OneHot Encoder


#misc libraries
import random
import time


#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)
#this is from https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
#from pandas.tools.plotting import scatter_matrix

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
%matplotlib inline
#mpl.style.use('ggplot')
#sns.set_style('white')
#pylab.rcParams['figure.figsize'] = 12,8
#load data
train_raw = pd.read_csv('../input/titanic/train.csv')
test_raw = pd.read_csv('../input/titanic/test.csv')
train_raw.head()
train_raw.info()
train_raw.describe()
#let split the data for more targeted handling
txt_cols = [cname for cname in train_raw.columns if train_raw[cname].dtype == "object"]

# Select numerical columns
num_cols = [cname for cname in train_raw.columns if train_raw[cname].dtype in ['int64', 'float64']]

txt_data = train_raw[txt_cols].copy()
num_data = train_raw[num_cols].copy()
# to make sure we didn't accidentally drop any cols
txt_data.shape[1] + num_data.shape[1] == train_raw.shape[1]
#now let us look at the numaric cols
num_data.info()
age_missing_per = num_data.Age.isnull().sum()/len(num_data.Age)
print("{:.2%}".format(age_missing_per))
# to find corrolating features
sns.heatmap(train_raw.corr())
#we see that Pclass, Sibsp, Parch are highly corrolated with Age
Pclass_Sibsp_Parch = train_raw['Pclass'].apply(str)+'-'+train_raw['SibSp'].apply(str)+'-'+train_raw['Parch'].apply(str)
Pclass_Sibsp_Parch.value_counts()
frame = { 'Pclass_Sibsp_Parch': Pclass_Sibsp_Parch, 'Age': train_raw.Age } 
age_psp = pd.DataFrame(frame)
age_classes = ['3-0-0', '1-0-0', '2-0-0']
#now we have the data in the right state to be converted to the desired lookup table
age_psp['Pclass_Sibsp_Parch'] = age_psp['Pclass_Sibsp_Parch'].apply(lambda x: x if x in age_classes else 'other')
age_psp['Pclass_Sibsp_Parch'].value_counts()
#creating the lookup table with Pclass_Sibsp_Parch as index
age_lookup= age_psp.groupby("Pclass_Sibsp_Parch", as_index=False).mean()
age_lookup = age_lookup.set_index('Pclass_Sibsp_Parch')
#create temp columns in the df to impute missing Age values
num_data['temp_psp'] = age_psp['Pclass_Sibsp_Parch']
num_data['temp_age_cat_mean'] = num_data['temp_psp']
num_data['temp_age_cat_mean'] = num_data['temp_age_cat_mean'].apply(lambda x: age_lookup['Age'][x])
num_data['Age'] = num_data['Age'].fillna(num_data['temp_age_cat_mean'])
#check to see if there is still null values in the data
num_data['Age'].isnull().sum()
#clean up after the imputation
num_data = num_data.drop('temp_psp', axis=1)
num_data = num_data.drop('temp_age_cat_mean', axis=1)
num_data.isnull().sum()
num_data['PassengerId'].hist()
num_data = num_data.drop('PassengerId', axis=1)
txt_data.isnull().sum()
txt_data = txt_data.drop('Cabin', axis=1)
txt_data['Embarked'] = txt_data['Embarked'].fillna(txt_data['Embarked'].value_counts().index[0])
txt_data.isnull().sum()
def data_clean (df):
    #create lookup table for Age
    psp = df['Pclass'].apply(str)+'-'+df['SibSp'].apply(str)+'-'+df['Parch'].apply(str)
    frame = { 'Pclass_Sibsp_Parch': psp, 'Age': df.Age } 
    psp_age = pd.DataFrame(frame)
    #define age classes, other not yet included
    age_classes = ['3-0-0', '1-0-0', '2-0-0']
    #convert excluding items to 'other'
    psp_age['Pclass_Sibsp_Parch'] = psp_age['Pclass_Sibsp_Parch'].apply(lambda x: x if x in age_classes else 'other')
    #transform to a lookup table
    am_lookup= psp_age.groupby('Pclass_Sibsp_Parch').mean()
    #using the lookup table
    df['temp_psp'] = psp_age['Pclass_Sibsp_Parch'] #setup a temp col with psp lables
    df['temp_age_cat_mean'] = df['temp_psp'] #create a col for age means conversions
    df['temp_age_cat_mean'] = df['temp_age_cat_mean'].apply(lambda x: am_lookup['Age'][x]) #convert values in this col to age means according to the psp lable
    df['Age'] = df['Age'].fillna(df['temp_age_cat_mean']) #fill na according to the tempt mean col
    #drop the temp cols
    df = df.drop('temp_psp', axis=1)
    df = df.drop('temp_age_cat_mean', axis=1)
    
    #found that there are missing valuse in Fare in the test dataset
    df['Fare'] = df['Fare'].fillna(method='ffill')
    
    #for reasons stated above we don't want PassengerId, Cabin cols
    df = df.drop('PassengerId', axis=1)
    df = df.drop('Cabin', axis=1)
    
    #fill na for Embarked, only two 
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].value_counts().index[0])
    

    
    return df
    
num_data.describe()
num_data['FareBin'] = pd.qcut(num_data['Fare'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
num_data['AgeBin'] = pd.cut(num_data['Age'].astype(int), 5, labels=[1, 2, 3, 4, 5]).astype(int)
num_data
#take a look at if the new features are actually useful
num_data.corr()
#now we drop the original features
num_data = num_data.drop(['Fare', 'Age'], axis=1)
num_data
FamilySize = num_data['SibSp']+num_data['Parch']+1
IsAlone = FamilySize>1
IsAlone = IsAlone.apply(int)
num_data['IsAlone'] = IsAlone
num_data['FamilySize'] = FamilySize
num_data.corr()
txt_data 
#borrowed from https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy notebook
#quick and dirty code split title from name: http://www.pythonforbeginners.com/dictionary/python-split
txt_data['Title'] = txt_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

txt_data.Title.value_counts()
stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
title_names = (txt_data['Title'].value_counts() < stat_min) #this will create a true false series with title name as index

#apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
txt_data['Title'] = txt_data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
print(txt_data['Title'].value_counts())
Ticket_len = txt_data.Ticket.apply(len)
Ticket_len.value_counts()
stat_min_ti = 30
Ticket_len_ls = (Ticket_len.value_counts() < stat_min_ti)
Ticket_len = Ticket_len.apply(lambda x: '30' if Ticket_len_ls.loc[x] == True else x)
Ticket_len.value_counts()
txt_data['Ticket_len'] = Ticket_len
txt_data = txt_data.drop(['Name', 'Ticket'], axis=1)
txt_data
#imputer only works on str or numbers
txt_data['Ticket_len'] = txt_data['Ticket_len'].astype(str)
OH_en = OneHotEncoder(handle_unknown='ignore', sparse=False)
imp_txt_cols = OH_en.fit_transform(txt_data[['Sex','Embarked', 'Title', 'Ticket_len']])
imp_txt_cols = pd.DataFrame(imp_txt_cols)
#now we align the index and col names
imp_txt_cols.index = txt_data[['Sex','Embarked', 'Title', 'Ticket_len']].index
imp_txt_cols.columns = OH_en.get_feature_names(['Sex','Embarked', 'Title', 'Ticket_len'])
imp_txt_cols
txt_data
#now we complete the txt df with the imputed cols
txt_data = txt_data.drop(['Sex','Embarked', 'Title', 'Ticket_len'], axis=1).join(imp_txt_cols)
txt_data.corr()
def f_eng (clean_data):
    #let's start with num cols
    #create value bins for continuouse values
    clean_data['FareBin'] = pd.qcut(clean_data['Fare'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    clean_data['AgeBin'] = pd.cut(clean_data['Age'].astype(int), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    #now we drop the original features
    clean_data = clean_data.drop(['Fare', 'Age'], axis=1)
    
    #create new features
    FamilySize = clean_data['SibSp']+clean_data['Parch']+1
    IsAlone = FamilySize>1
    IsAlone = IsAlone.apply(int)
    clean_data['IsAlone'] = IsAlone
    clean_data['FamilySize'] = FamilySize
    
    #next we work on text data
    #extrat title from Name
    clean_data['Title'] = clean_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    
    stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
    title_names = (clean_data['Title'].value_counts() < stat_min) #this will create a true false series with title name as index
    #apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
    clean_data['Title'] = clean_data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
    
    #cat tickets by length
    Ticket_len = clean_data.Ticket.apply(len)
    stat_min_ti = 30
    Ticket_len_ls = (Ticket_len.value_counts() < stat_min_ti)
    Ticket_len = Ticket_len.apply(lambda x: '30' if Ticket_len_ls.loc[x] == True else x)
    Ticket_len.value_counts()
    clean_data['Ticket_len'] = Ticket_len
    #imputer only works on str or numbers
    clean_data['Ticket_len'] = clean_data['Ticket_len'].astype(str)
    clean_data = clean_data.drop(['Name', 'Ticket'], axis=1)
    
    #do imputation on txt data
    OH_en = OneHotEncoder(handle_unknown='ignore', sparse=False)
    imp_cols = OH_en.fit_transform(clean_data[['Sex','Embarked', 'Title', 'Ticket_len']])
    imp_cols = pd.DataFrame(imp_cols)
    #now we align the index and col names
    imp_cols.index = clean_data[['Sex','Embarked', 'Title', 'Ticket_len']].index
    imp_cols.columns = OH_en.get_feature_names(['Sex','Embarked', 'Title', 'Ticket_len'])
    clean_data = clean_data.drop(['Sex','Embarked', 'Title', 'Ticket_len'], axis=1).join(imp_cols)

    return clean_data
test_data = train_raw.copy()
test_clean = data_clean(test_data)
final = f_eng(test_clean)
final.corr()
def pre_p (train, test):
    train_c = data_clean(train)
    train_f = f_eng(train_c)
    y_train = train_f.Survived
    X_train = train_f.drop('Survived', axis=1)
    
    test_c = data_clean(test)
    test_f = f_eng(test_c)
    X_test = test_f
    
    return X_train, y_train, X_test
X_train, y_train, X_test = pre_p (train_raw, test_raw)
def base_model (X_tr, y_tr):
    #this is from https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy
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
    
    #split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
    #note: this is an alternative to train_test_split
    #cv_split = model_selection.ShuffleSplit(test_size = .2, train_size = .8, random_state = 0 ) # run model 10x with 80/20 split intentionally leaving out 10%

    #create table to compare MLA metrics
    MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
    MLA_compare = pd.DataFrame(columns = MLA_columns)

    #create table to compare MLA predictions
    MLA_predict = y_tr

    #index through MLA and save performance to table
    row_index = 0
    for alg in MLA:

        #set name and parameters
        MLA_name = alg.__class__.__name__
        #print(MLA_name)
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
        #print(y_tr.shape)

        #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
        cv_results = model_selection.cross_validate(alg, X_tr, y_tr, cv = 5, scoring='accuracy', return_train_score=True)
        

        MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
        MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
        MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
        #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
        MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!


        #save MLA predictions - see section 6 for usage
        #alg.fit(X_tr, y_tr)
        #MLA_predict[MLA_name] = alg.predict(X_tr)

        row_index+=1


    #print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
    MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
    return MLA_compare
    #MLA_predict
base_alg = base_model(X_train, y_train)
base_alg
#barplot using https://seaborn.pydata.org/generated/seaborn.barplot.html
sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = base_alg, color = 'm')

#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')
#this is borrowed from the Kaggle notebook https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy
#the original idea was to take the top ten from our list above, but since the alg list below covers a large portion of our top ten, we will use it directly
def hp_tune(X_tr, y_tr, base_alg):
    
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
    
    #Hyperparameter Tune with GridSearchCV: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    grid_n_estimator = [10, 50, 100, 300, 500]
    grid_ratio = [.1, .25, .5, .75, 1.0]
    grid_learn = [.01, .03, .05, .1, .25]
    grid_max_depth = [2, 4, 6, 8, 10, None]
    grid_min_samples = [5, 10, .03, .05, .10]
    grid_criterion = ['gini', 'entropy']
    grid_bool = [True, False]
    grid_seed = [0]


    grid_param = [
                [{
                #AdaBoostClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
                'n_estimators': grid_n_estimator, #default=50
                'learning_rate': grid_learn, #default=1
                #'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R
                'random_state': grid_seed
                }],


                [{
                #BaggingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
                'n_estimators': grid_n_estimator, #default=10
                'max_samples': grid_ratio, #default=1.0
                'random_state': grid_seed
                 }],


                [{
                #ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
                'n_estimators': grid_n_estimator, #default=10
                'criterion': grid_criterion, #default=”gini”
                'max_depth': grid_max_depth, #default=None
                'random_state': grid_seed
                 }],


                [{
                #GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
                #'loss': ['deviance', 'exponential'], #default=’deviance’
                'learning_rate': [.05], #default=0.1 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
                'n_estimators': [300], #default=100 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
                #'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”
                'max_depth': grid_max_depth, #default=3   
                'random_state': grid_seed
                 }],


                [{
                #RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
                'n_estimators': grid_n_estimator, #default=10
                'criterion': grid_criterion, #default=”gini”
                'max_depth': grid_max_depth, #default=None
                'oob_score': [True], #default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.
                'random_state': grid_seed
                 }],

                [{    
                #GaussianProcessClassifier
                'max_iter_predict': grid_n_estimator, #default: 100
                'random_state': grid_seed
                }],


                [{
                #LogisticRegressionCV - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
                'fit_intercept': grid_bool, #default: True
                #'penalty': ['l1','l2'],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], #default: lbfgs
                'random_state': grid_seed
                 }],


                [{
                #BernoulliNB - http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
                'alpha': grid_ratio, #default: 1.0
                 }],


                #GaussianNB - 
                [{}],

                [{
                #KNeighborsClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
                'n_neighbors': [1,2,3,4,5,6,7], #default: 5
                'weights': ['uniform', 'distance'], #default = ‘uniform’
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                }],


                [{
                #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
                #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
                #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'C': [1,2,3,4,5], #default=1.0
                'gamma': grid_ratio, #edfault: auto
                'decision_function_shape': ['ovo', 'ovr'], #default:ovr
                'probability': [True],
                'random_state': grid_seed
                 }],


                [{
                #XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html
                'learning_rate': grid_learn, #default: .3
                'max_depth': [1,2,4,6,8,10], #default 2
                'n_estimators': grid_n_estimator, 
                'seed': grid_seed  
                 }]   
            ]

    
    #create a table to display key metrics
    hp_columns = ['Alg Name', 'Best Score', 'Score Before Tuning', 'Best Parameters']
    hp_compare = pd.DataFrame(columns = hp_columns)

    #index through MLA and save performance to table
    row_index = 0

    for clf, param in zip (vote_est, grid_param): #https://docs.python.org/3/library/functions.html#zip

     
        best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv = 5, scoring = 'accuracy', n_jobs = -1)
        best_search.fit(X_tr, y_tr)

        best_param = best_search.best_params_
        best_score = best_search.best_score_
        alg_name = clf[1].__class__.__name__
        #print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__, best_param, run))
        clf[1].set_params(**best_param) 
        
        hp_compare.loc[row_index, 'Alg Name'] = alg_name
        hp_compare.loc[row_index, 'Best Parameters'] = str(best_param)
        hp_compare.loc[row_index, 'Best Score'] = best_score
        hp_compare.loc[row_index, 'Score Before Tuning'] = base_alg.loc[base_alg['MLA Name'] == alg_name]['MLA Test Accuracy Mean'].tolist()[0]
        row_index+=1


    print('Done')
    print('-'*10)
    
    return vote_est, hp_compare

tuned_algs, hp_compare = hp_tune(X_train, y_train,base_alg)
hp_compare.sort_values(by='Best Score', ascending=False)
def en_alg (algs, X_tr, y_tr):
    
    grid_hard = ensemble.VotingClassifier(estimators = algs , voting = 'hard')
    grid_hard_cv = model_selection.cross_validate(grid_hard, X_tr, y_tr, cv  = 5, return_train_score=True)
    grid_hard.fit(X_tr, y_tr)

    print("Hard Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_hard_cv['train_score'].mean()*100)) 
    print("Hard Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_hard_cv['test_score'].mean()*100))
    print("Hard Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_hard_cv['test_score'].std()*100*3))
    print('-'*10)

    #Soft Vote or weighted probabilities w/Tuned Hyperparameters
    grid_soft = ensemble.VotingClassifier(estimators = algs , voting = 'soft')
    grid_soft_cv = model_selection.cross_validate(grid_soft, X_tr, y_tr, cv  = 5, return_train_score=True)
    grid_soft.fit(X_tr, y_tr)

    print("Soft Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_soft_cv['train_score'].mean()*100)) 
    print("Soft Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_soft_cv['test_score'].mean()*100))
    print("Soft Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_soft_cv['test_score'].std()*100*3))
    print('-'*10)
    
    return grid_hard, grid_soft
vc_hard_all, vc_soft_all = en_alg(tuned_algs, X_train, y_train)
hp_compare.sort_values(by='Best Score', ascending=False)['Alg Name']
def top_algs (sorted_score_df, algs, num_algs):
    top_alg_ls = []
    
    for alg_name in sorted_score_df:
        for alg in algs:
            if alg[1].__class__.__name__ == alg_name:
                top_alg_ls.append(alg)
        
    return top_alg_ls[:num_algs]
sorted_alg_names = hp_compare.sort_values(by='Best Score', ascending=False)['Alg Name']
top_five_algs = top_algs(sorted_alg_names, tuned_algs, 5)
vc_hard_tfive, vc_soft_tfive = en_alg(top_five_algs, X_train, y_train)
def gen_submissions(top_five, b_vc_hard, b_vc_soft, X_tr, y_tr, X_test, tag):
    
    #create submissions for each alg in the top_five list
    for alg in top_five:
        alg[1].fit(X_tr, y_tr)
        y_pred = alg[1].predict(X_test).astype(int)
        final_data = {'PassengerId': test_raw.PassengerId, 'Survived': y_pred}
        submission = pd.DataFrame(data=final_data)
        submission.to_csv('submissions/submission_'+tag+'_'+alg[0]+'.csv', index =False)
        
    #create submissions best hard and soft vc
    #they are already fitted in the evaluation steps
    y_pred_b_vc_hard = b_vc_hard.predict(X_test).astype(int)
    y_pred_b_vc_soft = b_vc_soft.predict(X_test) .astype(int)
    vc_hard_sub_data = {'PassengerId': test_raw.PassengerId, 'Survived': y_pred_b_vc_hard}
    vc_soft_sub_data = {'PassengerId': test_raw.PassengerId, 'Survived': y_pred_b_vc_soft}
    vc_hard_sub = pd.DataFrame(data=vc_hard_sub_data)
    vc_soft_sub = pd.DataFrame(data=vc_soft_sub_data)
    vc_hard_sub.to_csv('submissions/submission_'+tag+'_vc_hard.csv', index =False) 
    vc_soft_sub.to_csv('submissions/submission_'+tag+'vc_soft.csv', index =False) 
    
    print('Done')
    
#after some debugging, we found that Ticket_len_10 is missing from the X_test, as it was auto generated in X_train, need to improve the process in the next project
X_test.insert(17, "Ticket_len_10", 0, allow_duplicates = False)
gen_submissions(top_five_algs, vc_hard_all, vc_soft_all, X_train, y_train, X_test)
#creating a base model, the best result we got from single model approach was XGB, we will use it as our base model in this experiment
xgb = XGBClassifier(n_estimators=500)  
#prepare data for baseline evaluation 
train_i2 = train_raw.copy()
train_i2 = data_clean(train_i2)
def base_f_eng (clean_data):
    
    #next we work on text data
    #extrat title from Name
    clean_data['Title'] = clean_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    
    stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
    title_names = (clean_data['Title'].value_counts() < stat_min) #this will create a true false series with title name as index
    #apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
    clean_data['Title'] = clean_data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
    
    
    #do imputation on txt data
    OH_en = OneHotEncoder(handle_unknown='ignore', sparse=False)
    imp_cols = OH_en.fit_transform(clean_data[['Sex','Embarked', 'Title']])
    imp_cols = pd.DataFrame(imp_cols)
    #now we align the index and col names
    imp_cols.index = clean_data[['Sex','Embarked', 'Title']].index
    imp_cols.columns = OH_en.get_feature_names(['Sex','Embarked', 'Title'])
    clean_data = clean_data.drop(['Sex','Embarked', 'Title'], axis=1).join(imp_cols)
    
    clean_data = clean_data.drop('Ticket', axis=1)
    clean_data = clean_data.drop('Name', axis=1)

    return clean_data
def score (X, y):
    cv_agebin_results = model_selection.cross_validate(xgb, X, y, cv = 5, scoring='accuracy', return_train_score=True)
    return cv_agebin_results['test_score'].mean()
def xysplit (df):
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    return X, y
train_i2_base = base_f_eng (train_i2)
X_base = train_i2_base.drop('Survived', axis=1)
y_base = train_i2_base['Survived']
score (X_base, y_base)
def age_bin (clean_data):
    AgeBin = pd.cut(clean_data['Age'].astype(int), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    AgeBin.name = 'AgeBin'
    train_i2_agebin = train_i2_base.join(AgeBin)
    train_i2_agebin = train_i2_agebin.drop('Age', axis =1)
    
    X, y = xysplit(train_i2_agebin)
    
    return X, y
X_agebin, y_agebin = age_bin (train_i2)
score (X_agebin, y_agebin)
def fare_bin (clean_data):
    FareBin = pd.cut(clean_data['Fare'].astype(int), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    FareBin.name = 'FareBin'
    train_i2_agebin = train_i2_base.join(FareBin)
    train_i2_agebin = train_i2_agebin.drop('Fare', axis =1)
    
    X, y = xysplit(train_i2_agebin)
    
    return X, y
X_farebin, y_farebin = fare_bin (train_i2)
score (X_farebin, y_farebin)
#have all Fare values plus one to remove zeros
fare_po = train_i2['Fare']+1
from scipy import stats
boxcox_fare = stats.boxcox(fare_po)
train_i2['Fare'].hist()
boxcox_fare = pd.Series(boxcox_fare[0], index=train_i2['Fare'].index, name='boxcox_fare')
boxcox_fare.hist(bins=50)
sqrt_fare = np.sqrt(train_i2['Fare'])
sqrt_fare.name = 'sqrt_fare'
log_fare = np.log(train_i2['Fare'])
log_fare.name = 'log_fare'
sqrt_fare.hist(bins=50)
log_fare.hist(range=(0, 8), bins=50)
def fare_val (base_df, nor_fare):
    
    train_i2_nor_fare = base_df.join(nor_fare)
    train_i2_nor_fare_final = train_i2_nor_fare.drop('Fare', axis=1)
    X, y = xysplit(train_i2_nor_fare_final)
    return score (X,y)
score_log = fare_val(train_i2_base, log_fare)
print(score_log)
score_sqrt = fare_val(train_i2_base, sqrt_fare)
print(score_sqrt)
score_bc = fare_val(train_i2_base, boxcox_fare)
print(score_bc)
#what if we include both Farebin and nor_fare in the data
def fare_w_fbin_val (base_df, nor_fare):
    
    train_i2_nor_fare = base_df.join(nor_fare)
    train_i2_nor_fare_final = train_i2_nor_fare.drop('Fare', axis=1)
    train_i2_nor_fare_final['FareBin'] = X_farebin['FareBin']
    X, y = xysplit(train_i2_nor_fare_final)
    return score (X,y)
score_wf = fare_w_fbin_val(train_i2_base, log_fare)
print(score_wf)
train_i2_base.head()
#FamilySize evaluation
train_i2_fz = train_i2_base.copy()
train_i2_fz['FamilySize'] = train_i2_fz['SibSp']+train_i2_fz['Parch']+1
train_i2_fz.head()
X_fz, y_fz = xysplit(train_i2_fz)
score (X_fz, y_fz)
train_i2_fz_noraw = train_i2_fz.drop(['SibSp', 'Parch'], axis=1)
X_fz_nr, y_fz_nr = xysplit(train_i2_fz_noraw)
score (X_fz_nr, y_fz_nr)
train_i2_fz_isAlone = train_i2_fz_noraw.copy()
train_i2_fz_isAlone['IsAlone'] = train_i2_fz_isAlone['FamilySize'] == 1
train_i2_fz_isAlone['IsAlone'] = train_i2_fz_isAlone['IsAlone'].astype(int)
X_fz_ia, y_fz_ia = xysplit(train_i2_fz_isAlone)
score (X_fz_ia, y_fz_ia)
def ticket_len (df):
    #making sure we are not massing with the raw data
    clean_data = df.copy()
    #cat tickets by length
    Ticket_len = clean_data.Ticket.apply(len)
    stat_min_ti = 30
    Ticket_len_ls = (Ticket_len.value_counts() < stat_min_ti)
    Ticket_len = Ticket_len.apply(lambda x: '30' if Ticket_len_ls.loc[x] == True else x)
    Ticket_len = Ticket_len.astype(str)
    
    
    #do imputation on txt data
    OH_en = OneHotEncoder(handle_unknown='ignore', sparse=False)
    imp_cols = OH_en.fit_transform(pd.DataFrame(Ticket_len))
    imp_cols = pd.DataFrame(imp_cols)
    #now we align the index and col names
    imp_cols.index = Ticket_len.index
    imp_cols.columns = OH_en.get_feature_names(['Ticket_len'])
        

    return imp_cols

Ticket_len = ticket_len (train_i2)
train_i2_tl = train_i2_base.copy()
train_i2_tl = train_i2_tl.join(Ticket_len)
X_fz_tl, y_fz_tl = xysplit(train_i2_tl)
score (X_fz_tl, y_fz_tl)
train_i2.head()
def interactions (df):
    
    PSE = df.Pclass.astype(str) + df.Sex + df.Embarked
    
    #do imputation on txt data
    OH_en = OneHotEncoder(handle_unknown='ignore', sparse=False)
    imp_cols = OH_en.fit_transform(pd.DataFrame(PSE))
    imp_cols = pd.DataFrame(imp_cols)
    #now we align the index and col names
    imp_cols.index = PSE.index
    imp_cols.columns = OH_en.get_feature_names(['Pclass_Sex_Embarked'])
    
    return imp_cols
pse = interactions(train_i2)
train_i2_pse = train_i2_base.copy()
train_i2_pse = train_i2_pse.join(pse)
X_pse, y_pse = xysplit(train_i2_pse)
score (X_pse, y_pse)
train_i2_pse_noraw = train_i2_pse.drop(['Pclass', 'Sex_male','Sex_female','Embarked_C','Embarked_Q','Embarked_S'], axis=1)
X_pse_nr, y_pse_nr = xysplit(train_i2_pse_noraw)
score (X_pse_nr, y_pse_nr)
def f_eng_i2 (clean_data):
    #let's start with num cols
    #create value bins for continuouse values
    clean_data['FareBin'] = pd.qcut(clean_data['Fare'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    clean_data['AgeBin'] = pd.cut(clean_data['Age'].astype(int), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    #now we drop the original features
    clean_data = clean_data.drop(['Fare', 'Age'], axis=1)
    
    #create new features
    FamilySize = clean_data['SibSp']+clean_data['Parch']+1
    clean_data['FamilySize'] = FamilySize
    
    #next we work on text data
    #extrat title from Name
    clean_data['Title'] = clean_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    
    stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
    title_names = (clean_data['Title'].value_counts() < stat_min) #this will create a true false series with title name as index
    #apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
    clean_data['Title'] = clean_data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
    
    #cat tickets by length
    Ticket_len = clean_data.Ticket.apply(len)
    stat_min_ti = 30
    Ticket_len_ls = (Ticket_len.value_counts() < stat_min_ti)
    Ticket_len = Ticket_len.apply(lambda x: '30' if Ticket_len_ls.loc[x] == True else x)
    Ticket_len.value_counts()
    clean_data['Ticket_len'] = Ticket_len
    #imputer only works on str or numbers
    clean_data['Ticket_len'] = clean_data['Ticket_len'].astype(str)
    clean_data = clean_data.drop(['Name', 'Ticket'], axis=1)
    
    #adding Pclass_Sex_Embarked
    PSE = clean_data.Pclass.astype(str) + clean_data.Sex + clean_data.Embarked
    clean_data['Pclass_Sex_Embarked'] = PSE
    
    
    #do imputation on txt data
    OH_en = OneHotEncoder(handle_unknown='ignore', sparse=False)
    imp_cols = OH_en.fit_transform(clean_data[['Title', 'Ticket_len', 'Pclass_Sex_Embarked']])
    imp_cols = pd.DataFrame(imp_cols)
    #now we align the index and col names
    imp_cols.index = clean_data[['Title', 'Ticket_len', 'Pclass_Sex_Embarked']].index
    imp_cols.columns = OH_en.get_feature_names(['Title', 'Ticket_len', 'Pclass_Sex_Embarked'])
    clean_data = clean_data.drop(['Sex','Embarked', 'Title', 'Ticket_len', 'SibSp', 'Parch', 'Pclass', 'Pclass_Sex_Embarked'], axis=1).join(imp_cols)

    return clean_data
train_i2_af = f_eng_i2(train_i2)
X_af, y_af = xysplit(train_i2_af)
score (X_af, y_af)
#create a feature engineering function that retains all features
def f_eng_all (clean_data):
    #let's start with num cols
    #create value bins for continuouse values
    clean_data['FareBin'] = pd.qcut(clean_data['Fare'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    clean_data['AgeBin'] = pd.cut(clean_data['Age'].astype(int), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    
    #create new features
    FamilySize = clean_data['SibSp']+clean_data['Parch']+1
    clean_data['FamilySize'] = FamilySize
    
    #next we work on text data
    #extrat title from Name
    clean_data['Title'] = clean_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    
    stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
    title_names = (clean_data['Title'].value_counts() < stat_min) #this will create a true false series with title name as index
    #apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
    clean_data['Title'] = clean_data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
    
    #cat tickets by length
    Ticket_len = clean_data.Ticket.apply(len)
    stat_min_ti = 30
    Ticket_len_ls = (Ticket_len.value_counts() < stat_min_ti)
    Ticket_len = Ticket_len.apply(lambda x: '30' if Ticket_len_ls.loc[x] == True else x)
    Ticket_len.value_counts()
    clean_data['Ticket_len'] = Ticket_len
    #imputer only works on str or numbers
    clean_data['Ticket_len'] = clean_data['Ticket_len'].astype(str)
    clean_data = clean_data.drop(['Name', 'Ticket'], axis=1)
    
    #adding Pclass_Sex_Embarked
    PSE = clean_data.Pclass.astype(str) + clean_data.Sex + clean_data.Embarked
    clean_data['Pclass_Sex_Embarked'] = PSE
    
    
    #do imputation on txt data
    OH_en = OneHotEncoder(handle_unknown='ignore', sparse=False)
    imp_cols = OH_en.fit_transform(clean_data[['Sex','Embarked','Title', 'Ticket_len', 'Pclass_Sex_Embarked']])
    imp_cols = pd.DataFrame(imp_cols)
    #now we align the index and col names
    imp_cols.index = clean_data[['Sex','Embarked','Title', 'Ticket_len', 'Pclass_Sex_Embarked']].index
    imp_cols.columns = OH_en.get_feature_names(['Sex','Embarked','Title', 'Ticket_len', 'Pclass_Sex_Embarked'])
    clean_data = clean_data.drop(['Sex','Embarked', 'Title', 'Ticket_len', 'Pclass_Sex_Embarked'], axis=1).join(imp_cols)

    return clean_data
train_i2_all = f_eng_all(train_i2)
#creating a base score with all features included
X_all, y_all = xysplit(train_i2_all)
score (X_all, y_all)
def k_class (X, y, kc):
    
    #borrowed from https://www.kaggle.com/matleonard/feature-selection
    # Keep 5 features
    selector = feature_selection.SelectKBest(feature_selection.f_classif, k=kc)

    X_new = selector.fit_transform(X, y)

    # Get back the features we've kept, zero out all other features
    selected_features = pd.DataFrame(selector.inverse_transform(X_new), 
                                     index=X.index, 
                                     columns=X.columns)

    # Dropped columns have values of all 0s, so var is 0, drop them
    selected_columns = selected_features.columns[selected_features.var() != 0]
    selected_columns = selected_columns.insert(0, 'Survived')

    return selected_columns
    
def run_kc (X, y, kc_ls):
    for kc in kc_ls:
        k_cols = k_class(X, y, kc)
        X_kc, y_kc = xysplit(train_i2_all[k_cols])
        auc_score = score (X_kc, y_kc)
        print('k value {0} score {1}'.format(kc, auc_score))
kc_ls = [5, 10, 15, 20]
run_kc (X_all, y_all, kc_ls)
def L_one (X, y):

    # Set the regularization parameter C=1
    logistic = linear_model.LogisticRegression(C=1, penalty="l1", solver='liblinear', random_state=7).fit(X, y)
    model = feature_selection.SelectFromModel(logistic, prefit=True)
    X_new = model.transform(X)
    # Get back the kept features as a DataFrame with dropped columns as all 0s
    selected_features = pd.DataFrame(model.inverse_transform(X_new), 
                                 index=X.index,
                                 columns=X.columns)

    # Dropped columns have values of all 0s, keep other columns 
    selected_columns = selected_features.columns[selected_features.var() != 0]
    selected_columns = selected_columns.insert(0, 'Survived')
    return selected_columns
L_cols = L_one(X_all, y_all)
X_lc, y_lc = xysplit(train_i2_all[L_cols])
score (X_lc, y_lc)
base_L1_alg = base_model(X_lc, y_lc)
base_L1_alg
tuned_L1_algs, hp_compare = hp_tune(X_lc, y_lc, base_L1_alg)
hp_compare.sort_values(by='Best Score', ascending=False)
vc_hard_L1, vc_soft_L1 = en_alg(tuned_L1_algs, X_lc, y_lc)
sorted_L1_algs = hp_compare.sort_values(by='Best Score', ascending=False)['Alg Name']
top_five_L1_algs = top_algs(sorted_L1_algs, tuned_L1_algs, 5)
X_test_i2_clean = data_clean(test_raw)
X_test_i2_fe = f_eng_all(X_test_i2_clean)
#aligning columns
X_test_i2_fe.insert(14, "Ticket_len_10", 0, allow_duplicates = False)
X_test_i2_lc = X_test_i2_fe[X_lc.columns]
gen_submissions(top_five_L1_algs, vc_hard_L1, vc_soft_L1, X_lc, y_lc, X_test_i2_lc)
ps_tuned_L1_algs = top_five_L1_algs
best_alg = ps_tuned_L1_algs[0][1].fit(X_lc, y_lc)
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(best_alg, random_state=1).fit(X_lc, y_lc)
eli5.show_weights(perm, feature_names = X_lc.columns.tolist(), top = 30)
import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(best_alg)

# calculate shap values. This is what we will plot.
# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
shap_values = explainer.shap_values(X_lc)

# Make plot. Index of [1] is explained in text below.
shap.summary_plot(shap_values, X_lc)
