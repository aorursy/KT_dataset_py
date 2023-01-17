#Imports and Pandas display settings

import numpy as np

import pandas as pd

import os

import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.float_format',lambda x: "%.2f"%x if x!=np.nan else np.nan)

pd.set_option('display.max_columns',1000)

pd.set_option('display.max_rows',1000)
#Load data

train_df = pd.read_csv('/kaggle/input/titanic/train.csv',index_col=False)

test_df = pd.read_csv('/kaggle/input/titanic/test.csv',index_col=False)



train_copy = train_df.copy(deep=True)

test_copy = test_df.copy(deep=True)
best_features = []

def featurize(dataset,isTest=False):

    global best_features

    

    dataset.Embarked.fillna(dataset.Embarked.mode()[0],inplace=True)

    

    dataset['Median_Age'] = dataset.groupby(['Sex','Pclass']).Age.transform('median')

    dataset.Age.fillna(dataset.Median_Age,inplace=True)

    dataset.drop(columns=['Median_Age'],inplace=True)

    

    dataset['Median_Fare'] = dataset.groupby(['Sex','Pclass']).Fare.transform('median')

    dataset.Fare.fillna(dataset.Median_Fare,inplace=True)

    dataset.drop(columns=['Median_Fare'],inplace=True)

    

    title_to_id = {

    'Mr': 1,

    'Mrs': 4,

    'Miss': 3,

    'Master': 3,

    'Don': 0,

    'Dona': 0,

    'Rev':0,

    'Dr': 2,

    'Mme': 5,

    'Ms': 5,

    'Major': 2,

    'Lady': 5,

    'Sir': 5,

    'Mlle': 5,

    'Col': 2,

    'Capt': 0,

    'the Countess': 5,

    'Jonkheer': 0

    }

    dataset['Title'] = dataset.Name.str.split(',').str[1].str.split('.').str[0].str.strip()

    dataset['Title'] = dataset.Title.map(title_to_id)

    

    cabin_to_id = {

    'D':3,

    'E':3, 

    'B':3,

    'F':2,

    'C':2,

    'G':2,

    'A':2,

    'Z':1,

    'T':1

    }

    dataset['Code'] = dataset.Cabin.str.split(' ').str[0].str[0]

    dataset['Code'].fillna('Z',inplace=True)

    dataset['Code'] = dataset.Code.map(cabin_to_id)

    

    dataset['Fare_Bin'] = pd.cut(dataset.Fare,bins=4).cat.codes

    dataset['Age_Bin'] = pd.cut(dataset.Age,bins=5).cat.codes

    dataset['Sex'] = dataset.Sex.map({'male':0,'female':1})

    

    def family_size_mapper(value):

        if value>=5:

            return 3

        elif value==1:

            return 2

        elif value>1 and value<=4:

            return 1

        return value



    dataset['Family_Size'] = dataset.Parch + dataset.SibSp + 1

    dataset.Family_Size = dataset.Family_Size.apply(family_size_mapper)



    dataset = pd.concat([dataset,pd.get_dummies(dataset.Family_Size,prefix='Family_Size')],axis=1)

    dataset = pd.concat([dataset,pd.get_dummies(dataset.Age_Bin,prefix='Age_Bin')],axis=1)

    dataset = pd.concat([dataset,pd.get_dummies(dataset.Fare_Bin,prefix='Fare_Bin')],axis=1)

    dataset = pd.concat([dataset,pd.get_dummies(dataset.Embarked,prefix='Embarked')],axis=1)

    dataset = pd.concat([dataset,pd.get_dummies(dataset.Title,prefix='Title')],axis=1)

    dataset = pd.concat([dataset,pd.get_dummies(dataset.Pclass,prefix='Pclass')],axis=1)

    dataset = pd.concat([dataset,pd.get_dummies(dataset.Code,prefix='Code')],axis=1)

    dataset.drop(columns=['PassengerId','Family_Size','Name','Age','SibSp','Parch','Ticket','Fare','Embarked','Title','Fare_Bin','Age_Bin','Pclass','Cabin','Code'],inplace=True)

    

    return dataset
id_array = test_df.PassengerId.tolist()

train_df = featurize(train_df)

test_df = featurize(test_df,isTest=True)
#Judge feature importance 

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

def feature_selector(N,dataset):

    X = dataset.iloc[:,1:]

    y = dataset.iloc[:,0]

    selector_model = SelectKBest(score_func=chi2, k=len(dataset.columns)-1)

    fit = selector_model.fit(X,y)

    featureScores = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(fit.scores_)],axis=1)

    featureScores.columns = ['Features','Score']

    return featureScores.sort_values(by='Score',ascending=False).reset_index(drop=True).head(N).Features.tolist()
#Models

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



MLA =  {

    #Ensemble Methods

    'adaboost':ensemble.AdaBoostClassifier(),

    'bagging':ensemble.BaggingClassifier(),

    'extra_trees':ensemble.ExtraTreesClassifier(),

    'gradient_boost':ensemble.GradientBoostingClassifier(),

    'rf':ensemble.RandomForestClassifier(),



    #Gaussian Processes

    'gaussian':gaussian_process.GaussianProcessClassifier(),

    

    #GLM

    'logit':linear_model.LogisticRegressionCV(),

    'pac':linear_model.PassiveAggressiveClassifier(),

    'ridge':linear_model.RidgeClassifierCV(),

    'sgd':linear_model.SGDClassifier(),

    'perceptron':linear_model.Perceptron(),

    

    #Navies Bayes

    'bernoulli':naive_bayes.BernoulliNB(),

    'gaussian_nb':naive_bayes.GaussianNB(),

    

    #Nearest Neighbor

    'knn':neighbors.KNeighborsClassifier(),

    

    #SVM

    'svc':svm.SVC(probability=True),

    'nu_svc':svm.NuSVC(probability=True),

    'lin_svc':svm.LinearSVC(),

    

    #Trees    

    'decision_tree':tree.DecisionTreeClassifier(),

    'ex_tree':tree.ExtraTreeClassifier(),

    

    #Discriminant Analysis

    'lin_disc':discriminant_analysis.LinearDiscriminantAnalysis(),

    'quad_disc':discriminant_analysis.QuadraticDiscriminantAnalysis(),



    

    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html

    'xgboost':XGBClassifier()    

   }
#Fix this code, make it pretty

pd.set_option('display.float_format',lambda x: "%.4f"%x if x!=np.nan else np.nan)

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%



MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']

MLA_compare = pd.DataFrame(columns = MLA_columns)

Target = 'Survived'

MLA_predict = {}

row_index = 0

for k in MLA:

    alg = MLA[k]

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    cv_results = model_selection.cross_validate(alg, train_df.iloc[:,1:], train_df.iloc[:,0], cv  = cv_split)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()

    #MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()

    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   

    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3 

    alg.fit(train_df.iloc[:,1:], train_df.iloc[:,0])

    MLA_predict[MLA_name] = alg.predict(train_df[train_df.columns[1:]])

    print(MLA_name+" completed!")

    row_index+=1



MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)

MLA_compare[['MLA Name','MLA Test Accuracy Mean']]
# prediction_array = MLA['lin_disc'].predict(test_df)

# my_submission = pd.DataFrame({'PassengerId': id_array, 'Survived': prediction_array})

# my_submission.to_csv('miracle.csv', index=False)
def voting_prediction(dataset,MLA_keys):

    pred_dict = {}

    for k in MLA_keys:

        pred_dict[k] = MLA[k].predict(dataset)

    results_df = pd.DataFrame(pred_dict)

    return results_df.mode(axis=1)[0]
prediction_array = MLA['lin_svc'].predict(test_df)

my_submission = pd.DataFrame({'PassengerId': id_array, 'Survived': prediction_array})

my_submission.to_csv('using_lin_svc.csv', index=False)