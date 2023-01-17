import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

# Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.tree import export_graphviz

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8

import math
import copy

from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')
dfTrain = pd.read_csv("/kaggle/input/titanic/train.csv")
dfTest = pd.read_csv("/kaggle/input/titanic/test.csv")
dfTestIndex = dfTest["PassengerId"]
datas = [dfTrain, dfTest]
for df in datas:
    df.info()

dict_titles = {'Mr.': 1,
               'Mrs.': 2, 
               'Miss.': 3, 
               'Ms.': 4, 
               'Rev.': 5, 
               'Dr.': 6, 
               'Master.': 7, 
               'Don.': 8,
               'Major.': 9,
               'Mme.': 10,
               'Mlle.': 11,
               'Col.': 12,
               'Capt.': 13, 
               'Jonkheer.': 14, 
               'Countess.': 15, 
               'Sir.': 16, 
               'Lady.': 17,
               'Dona.': 18 # just one person in Test data
              }

for df in datas:
    for i in range(len(df)):
        for title in dict_titles:
            if title in df.loc[i,'Name']:
                df.loc[i,'TitleIndex']=int(dict_titles[title])
    df["TitleIndex"]=df["TitleIndex"].astype(int)

print(dfTrain["TitleIndex"].value_counts())
print(dfTest["TitleIndex"].value_counts())

print(dfTrain.Embarked.value_counts())
print("Total: {}".format(dfTrain.Embarked.value_counts().sum()))
print("Number of NaN values: {}".format(len(dfTrain[dfTrain.Embarked.isna()]['PassengerId'])))
for df in datas:
    df["Embarked"].fillna("Q", inplace=True)
    if set(df.Embarked.unique())=={"S","C","Q"}: # check if already this replacement was done
        df["Embarked"] = df["Embarked"].map({"S": 2,"C": 1,"Q": 0}).astype(int)
print("After transformation")
for df in datas:
    print(df.Embarked.value_counts())
for df in datas:
    df["FamilySize"] = df["SibSp"]+df["Parch"]+1
    df.drop(["SibSp","Parch"],axis=1, inplace=True)

    df["Age"].fillna(df["Age"].dropna().median(),inplace=True)
    df['AgeRange'] = pd.cut(df['Age'], [0,5,18,50,100], labels=[1,2,3,4])
    df.drop(["Age"],axis=1, inplace=True)
    df["AgeRange"]=df["AgeRange"].astype(int)    
for df in datas:
    df["Sex"] = df["Sex"].map({"female":1,"male":0}).astype(int)
for df in datas:
    df.drop(["PassengerId","Name","Ticket","Fare","Cabin"],axis=1,inplace=True)
for df in datas:
    print(df.info())
#for vColVal in set([i for i in dfTrain.columns])-set(['Survived']):
#    dfTrain[vColVal] = (dfTrain[vColVal] - dfTrain[vColVal].mean()) / (dfTrain[vColVal].std())    
#print(dfTrain.mean(),dfTrain.var())

#for vColVal in set([i for i in dfTest.columns])-set(['Survived']):
#    dfTest[vColVal] = (dfTest[vColVal] - dfTest[vColVal].mean()) / (dfTest[vColVal].std())
#print(dfTest.mean(),dfTest.var())
CLF = [
    # navies_bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    # linear_model
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),

    # neighbors (K Nearest Neighbors)
    neighbors.KNeighborsClassifier(),

    # tree (Decision Tree)
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),    
    
    # ensemble
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    # gaussian_processes
    gaussian_process.GaussianProcessClassifier(),
        
    # svm (Support Vector Machine)
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    # discriminant_analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),
    
    # xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()    
    ]
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) 

CLF_cols = ['Name', 'Parameters','Train Accuracy mean()', 'Test Accuracy mean()', 'Test Accuracy in 3sigma' ,'Time']
CLF_compare = pd.DataFrame(columns = CLF_cols)

labels = dfTrain['Survived']
df = pd.DataFrame()
df = dfTrain.drop('Survived',axis=1)

row_index = 0
for alg in CLF:
    CLF_compare.loc[row_index, 'Name'] = alg.__class__.__name__
    CLF_compare.loc[row_index, 'Parameters'] = str(alg.get_params())
    
    cv_results = model_selection.cross_validate(alg, df, labels, cv=cv_split, return_train_score=True)

    CLF_compare.loc[row_index, 'Time'] = cv_results['fit_time'].mean()
    CLF_compare.loc[row_index, 'Train Accuracy mean()'] = cv_results['train_score'].mean()
    CLF_compare.loc[row_index, 'Test Accuracy mean()'] = cv_results['test_score'].mean()   
    CLF_compare.loc[row_index, 'Test Accuracy in 3sigma'] = cv_results['test_score'].std()*3

    row_index+=1
    
CLF_compare.sort_values(by=['Test Accuracy mean()'], ascending=False, inplace=True)
CLF_compare
if False: # if you want to continue investigation - set this parameter to False
    data_train = dfTrain
    labels_train = data_train["Survived"]
    data_train = data_train.drop("Survived",axis=1)    

    xgb_clf = XGBClassifier()          
    xgb_clf = xgb_clf.fit(data_train, labels_train)    
    cv_results = model_selection.cross_validate(xgb_clf, data_train, labels_train, cv=5, return_train_score=True)
    for p in ['fit_time', 'train_score', 'test_score']:
        print("{:<15}:{:>15.3f}".format(p,cv_results[p].mean()))    
    
    best_res = xgb_clf.predict(dfTest)   
    dfFinal=pd.DataFrame({
        "PassengerId":dfTestIndex,
        "Survived": best_res })
    dfFinal.to_csv("submission.csv",index=False)
data_train = dfTrain
labels_train = data_train["Survived"]
data_train = data_train.drop("Survived",axis=1)

xgb_clf = XGBClassifier()
xgb_clf.fit(data_train, labels_train)
cv_results = model_selection.cross_validate(xgb_clf, data_train, labels_train, cv=5, return_train_score=True)
print("Before tuning", xgb_clf.score(data_train, labels_train))
for p in ['fit_time', 'train_score', 'test_score']:
    print("{:<15}:{:>15.3f}".format(p, cv_results[p].mean()))       
    
# tune model
# parameters description can be found at http://xgboost.readthedocs.io/en/latest/parameter.html
grid_n_estimator = [10, 50, 100, 300]
grid_learn = [.01, .03, .05, .1, .25]
grid_seed = [0]
params = {
            'learning_rate': grid_learn, #default: .3
            'max_depth': [1,2,4,6,8,10], #default 2
            'n_estimators': grid_n_estimator, 
            'seed': grid_seed  
}

# choose best model parameters with GridSearchCV:
# cv: cross-validation ration
# n_jobs: how many parallel tasks will be started (-1 means "all possible")
# verbose: verbosity of logging
xgb_grid = model_selection.GridSearchCV(estimator=xgb_clf, param_grid=params, scoring='roc_auc', cv=10, n_jobs=-1, verbose=False)
xgb_grid.fit(data_train, labels_train)
cv_results = model_selection.cross_validate(xgb_grid, data_train, labels_train, cv=5, return_train_score=True)

print("After tuning", xgb_grid.score(data_train, labels_train))
for p in ['fit_time', 'train_score', 'test_score']:
    print("{:<15}:{:>15.3f}".format(p, cv_results[p].mean()))    
print('Raw AUC score:', xgb_grid.best_score_)
for param_name in sorted(xgb_grid.best_params_.keys()):
    print("%s: %r" % (param_name, xgb_grid.best_params_[param_name]))
xgb_grid.best_estimator_.learning_rate    
tuned_xgb_clf = XGBClassifier(learning_rate=xgb_grid.best_estimator_.learning_rate, 
                              max_depth=xgb_grid.best_estimator_.max_depth,
                              n_estimators=xgb_grid.best_estimator_.n_estimators,
                              seed=xgb_grid.best_estimator_.seed)
tuned_xgb_clf.fit(data_train, labels_train)
cv_results = model_selection.cross_validate(tuned_xgb_clf, data_train, labels_train, cv=5, return_train_score=True)
print("After tuning", tuned_xgb_clf.score(data_train, labels_train))
for p in ['fit_time', 'train_score', 'test_score']:
    print("{:<15}:{:>15.3f}".format(p, cv_results[p].mean()))       

if False:
    best_res = xgb_grid.predict(dfTest)
    
    dfFinal=pd.DataFrame({
        "PassengerId": dfTestIndex,
        "Survived": best_res })
    dfFinal.to_csv("submission.csv", index=False)
data = dfTrain
labels = data["Survived"]
data = data.drop("Survived",axis=1)

logreg = linear_model.LogisticRegression()
logreg.fit(data,labels)
cv_results = model_selection.cross_validate(logreg, data, labels, cv=5, return_train_score=True)
print("Before tuning", logreg.score(data, labels))
for p in ['fit_time', 'train_score', 'test_score']:
    print("{:<15}:{:>15.3f}".format(p, cv_results[p].mean()))  

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression    
penalty = ['l1', 'l2'] # penalization norm NB only 'saga' supports 'elasticnet', penalty 'none' is not supported by 'liblinear'
#dual = [True, False] # Dual or primal formulation
tol = [0.01, 0.001] # tolerance for stopping criteria
C = np.logspace(0,4,10) # Inverse of regularization strength; must be a positive float.
max_iter = np.arange(10,210,10) # default=100 Maximum number of iterations taken for the solvers to converge.
#fit_intercept = [True, False]
#solver = ['liblinear', 'sag', 'saga'] # default=’lbfgs’, NB 'newton-cg', 'lbfgs'only for l2 

parameters_dict = dict(penalty=penalty, tol=tol, C=C, max_iter=max_iter)

xgb_grid = model_selection.GridSearchCV(estimator=logreg, param_grid=parameters_dict, cv=5, verbose=1)
model = xgb_grid.fit(data, labels)

cv_results = model_selection.cross_validate(model, data, labels, cv=5, return_train_score=True)
print("After tuning", model.score(data, labels))
for p in ['fit_time', 'train_score', 'test_score']:
    print("{:<15}:{:>15.3f}".format(p, cv_results[p].mean()))  

#logreg_rep = metrics.classification_report(y_test, logreg_res)
#print (logreg_rep)

df = pd.DataFrame(logreg.get_params(), index=['Before tune'])
df.loc['After tune'] = model.best_estimator_.get_params()
df
res1 = model.predict(dfTest)
res2 = logreg.predict(dfTest)

for i in range(len(res1)):
    if res1[i]==res2[i]:
        print("!", end="")
    else:
        print("0", end="")
data = dfTrain
labels = data["Survived"]
data = data.drop("Survived",axis=1)

#X_train, X_test, y_train, y_test = model_selection.train_test_split(data.values, labels, test_size=0.3, random_state=1)
#knn = neighbors.KNeighborsClassifier(n_neighbors=4)
#knn.fit(X_train, y_train)  
#knn_res = knn.predict(X_test)

param_grid = {'n_neighbors': np.arange(1,8), 'p': [1,2]}
cv = 3
estimator_kNN = neighbors.KNeighborsClassifier()
optimazer_kNN = model_selection.GridSearchCV(estimator_kNN, param_grid, cv = cv)
optimazer_kNN.fit(data, labels)
print(optimazer_kNN.best_score_)
print(optimazer_kNN.best_params_)



#print ("knn.score={}".format(knn.score(X_test, y_test)))  
#knn_rep = metrics.classification_report(y_test, knn_res)
#print (knn_rep)
data = dfTrain
labels = data["Survived"]
data = data.drop("Survived",axis=1)

param_grid = {'n_estimators': np.arange(20,101,10), 'min_samples_split': np.arange(4,11, 1)}
cv = 3
estimator_rf = ensemble.RandomForestClassifier()
optimazer_rf = model_selection.GridSearchCV(estimator_rf, param_grid, cv=cv)
optimazer_rf.fit(data, labels)
print(optimazer_rf.best_score_)
print(optimazer_rf.best_params_)

data_train = dfTrain
labels_train = data_train["Survived"]
data_train = data_train.drop("Survived",axis=1)

data_test = dfTest

X_train, X_test, y_train, y_test = model_selection.train_test_split(data_train.values, labels_train, test_size=0.3, random_state=17)

tree_clf = tree.DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)
cv_results = model_selection.cross_validate(tree_clf, data_train.values, labels_train, cv  = 5)

#feature selection
tree_clf_rfe = feature_selection.RFECV(tree_clf, step = 1, scoring = 'accuracy', cv = 5)
tree_clf_rfe.fit(X_train, y_train)
X_rfe = data_train.columns.values[tree_clf_rfe.get_support()]
cv_results = model_selection.cross_validate(tree_clf_rfe, data_train[X_rfe], labels_train, cv  = 5)

#tune model
param_grid = {
              'criterion': ['gini', 'entropy'],
              'max_depth': range(1,11),
              'min_samples_split': [2,5,10,.03,.05],
              'min_samples_leaf': [1,5,10,.03,.05],
              'max_features': range(1,6),
              'random_state': [0]
             }
#choose best model with grid_search:
tree_clf = model_selection.GridSearchCV(tree_clf, param_grid=param_grid, scoring = 'roc_auc', cv=5, n_jobs=-1, verbose=False)
tree_clf.fit(X_train, y_train)
cv_results = model_selection.cross_validate(tree_clf, data_train[X_rfe], labels_train, cv  = 5)

tree_rep = cv_results
print('Raw AUC score:', tree_clf.best_score_)
for param_name in sorted(tree_clf.best_params_.keys()):
    print("%s: %r" % (param_name, tree_clf.best_params_[param_name]))
if False:
    best_res = tree_clf.predict(dfTest)
    
    dfFinal=pd.DataFrame({
        "PassengerId": dfTestIndex,
        "Survived": best_res })
    dfFinal.to_csv("submission.csv", index=False)
data = dfTrain 
labels = data["Survived"]
data = data.drop("Survived",axis=1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(data.values, labels, test_size=0.3, random_state=1)
#gbc = XGBClassifier()
#gbc.fit(X_train, y_train)
#gbc_res=gbc.predict(X_test)

gbc = ensemble.GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_res = gbc.predict(X_test)
print("Before tuning, gbc.score={}".format(gbc.score(X_test, y_test)))
gbc_rep = metrics.classification_report(y_test, gbc_res)
print (gbc_rep)

#tune hyper-parameters: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
grid_n_estimator = [10, 50, 100, 300]
grid_ratio = [.1, .25, .5, .75, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, .03, .05, .10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]
param_grid = {'learning_rate': [.05], #default=0.1
            'n_estimators': grid_n_estimator, #default=100
            'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”
            'max_depth': grid_max_depth, #default=3   
            'random_state': grid_seed
             }
     
tune_model = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = 3)
tune_model.fit(X_train, y_train)
gbc_res=tune_model.predict(X_test)
print("After tuning, gbc.score={}".format(tune_model.score(X_test, y_test)))
gbc_rep = metrics.classification_report(y_test, gbc_res)
print (gbc_rep)
print('Raw AUC score:', tune_model.best_score_)
for param_name in sorted(tune_model.best_params_.keys()):
    print("%s: %r" % (param_name, tune_model.best_params_[param_name]))
if True:
    best_res = tune_model.predict(dfTest)
    
    dfFinal=pd.DataFrame({
        "PassengerId": dfTestIndex,
        "Survived": best_res })
    dfFinal.to_csv("submission.csv", index=False)
data = dfTrain 
labels = data["Survived"]
data = data.drop("Survived", axis=1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(data.values, labels, test_size=0.3, random_state=1)

svc_clf = svm.SVC()
svc_clf.fit(X_train, y_train)
svc_res = svc_clf.predict(X_test)

print("Before tuning, svc_clf.score={}".format(svc_clf.score(X_test, y_test)))
svc_rep = metrics.classification_report(y_test, svc_res)
print (svc_rep)

param_grid = {
            'C': [1,2,3,4,5], #default=1.0
            'gamma': [.1, .25, .5, .75, 1.0], #default: auto
            'decision_function_shape': ['ovo', 'ovr'], #default:ovr
            'probability': [True],
            'random_state': [0]
}
     
tune_model = model_selection.GridSearchCV(svm.SVC(), param_grid=param_grid, scoring = 'roc_auc', cv = 3)
tune_model.fit(df, labels)
svc_res = tune_model.predict(X_test)
print("After tuning, svc.score={}".format(tune_model.score(data.values, labels)))
svc_rep = metrics.classification_report(y_test, svc_res)
print (svc_rep)    

print('Raw AUC score:', tune_model.best_score_)
for param_name in sorted(tune_model.best_params_.keys()):
    print("%s: %r" % (param_name, tune_model.best_params_[param_name]))
print("Logistic regression results:")
print(logreg_rep)
print("K nearest neighbours results:")
print(knn_rep)
print("Decision tree results:")
print(tree_rep)
print("Gradient boosting results:")
print(gbc_rep)
print("SVC:")
print(svc_rep)
print(X_rfe)
data_for_predictions = dfTest
data_train = dfTrain
labels_train = data_train["Survived"]
data_train = data_train.drop("Survived",axis=1)
tuned_xgb_clf.fit(data_train, labels_train)
best_res = tuned_xgb_clf.predict(data_for_predictions)
dfFinal=pd.DataFrame({
    "PassengerId":dfTestIndex,
    "Survived": best_res })
dfFinal.to_csv("submission.csv",index=False)