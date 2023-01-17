from sklearn.preprocessing import StandardScaler

from sklearn import ensemble, gaussian_process, linear_model, naive_bayes, neighbors, svm, tree, discriminant_analysis, model_selection

from xgboost import XGBClassifier 

import pandas as pd

import numpy as np

import kaggle





# Custom imputer that handles numerical and categorical values

from sklearn.base import TransformerMixin

class CustomImputer(TransformerMixin):



    def __init__(self):

        """Impute missing values.



        Columns of dtype object are imputed with the most frequent value 

        in column.



        Columns of other types are imputed with median of column.



        """

    def fit(self, X, y=None):



        self.fill = pd.Series([X[c].value_counts().index[0]

            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],

            index=X.columns)



        return self



    def transform(self, X, y=None):

        return X.fillna(self.fill)



# Gather data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_copy = test_data.copy()



# Create family feature

train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']

y = train_data['Survived']

test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']

train_data['IsAlone'] = train_data['FamilySize'] == 0

test_data['IsAlone'] = test_data['FamilySize'] == 0



# Extract title

def extract_title(train_data):

    name_list = train_data['Name']

    title_list = [name.split(", ")[1].split(".")[0] for name in name_list]

    title_dict = {

        "Mr": "Mr",

        "Mrs": "Mrs",

        "Miss": "Miss",

        "Master": "Master",

        "Don": "Noble",

        "Dona": "Noble",

        "Rev": "Rev",

        "Dr": "Dr",

        "Mme": "Noble",

        "Ms": "Miss",

        "Major": "Military",

        "Lady": "Noble",

        "Sir": "Noble",

        "Mlle": "Noble",

        "Col": "Military",

        "Capt": "Military",

        "the Countess": "Noble",

        "Jonkheer": "Noble"

    }

    title_feature = [title_dict[key] for key in title_list]

    return title_feature

train_data['Title'] = extract_title(train_data)

test_data['Title'] = extract_title(test_data)





# Remove unwanted features

unwanted_features = ['PassengerId', "Cabin", 'Ticket','Name']

train_data.drop(columns=unwanted_features, inplace=True)

train_data.drop(columns="Survived", inplace=True)

test_data.drop(columns=unwanted_features, inplace=True)



# Impute median missing values for both numerical and categorical features

CI = CustomImputer()

train_data = CI.fit_transform(train_data)

test_data = CI.transform(test_data)



# Define categorical variables

cat_vars = ['Sex', 'Pclass', 'Embarked', 'Title']

num_vars = [var for var in train_data.columns if var not in cat_vars]



# Transform to dummy variables

train_data = pd.get_dummies(train_data, columns=cat_vars, drop_first=True) 

test_data = pd.get_dummies(test_data, columns=cat_vars, drop_first=True)



# Rescale the data

scaler = StandardScaler()

train_data[train_data.columns] = scaler.fit_transform(train_data)

test_data[test_data.columns] = scaler.transform(test_data)



#MLA_predict



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



grid_n_estimator = [10, 50, 100, 300]

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

            'oob_score': [False], #default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.

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

            'solver': ['liblinear'], #default: lbfgs

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

import time

from sklearn.model_selection import GridSearchCV, cross_val_score

start_total = time.perf_counter() #https://docs.python.org/3/library/time.html#time.perf_counter

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6)

for clf, param in zip (vote_est, grid_param): #https://docs.python.org/3/library/functions.html#zip



    #print(clf[1]) #vote_est is a list of tuples, index 0 is the name and index 1 is the algorithm

    #print(param)

    

    

    start = time.perf_counter()        

    best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv = cv_split, scoring = 'roc_auc')

    best_search.fit(train_data, y)

    run = time.perf_counter() - start



    best_param = best_search.best_params_

    print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__, best_param, run))

    clf[1].set_params(**best_param) 





run_total = time.perf_counter() - start_total

print('Total optimization time was {:.2f} minutes.'.format(run_total/60))



print('-'*10)

from sklearn.ensemble import VotingClassifier

VCensemble = VotingClassifier(estimators=vote_est, voting="hard")

scores = cross_val_score(VCensemble, train_data, y, cv=10, scoring='accuracy')

print(scores.mean(), scores.std())

    

# Make submission

VCensemble.fit(train_data, y)

predictions = VCensemble.predict(test_data)

output = pd.DataFrame({'PassengerId': test_copy.PassengerId, 'Survived': predictions})

output.to_csv('title_soft_submission.csv', index=False)

print("Your submission was successfully saved!")