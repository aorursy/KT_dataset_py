# Importing Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re, os

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from scipy import stats

import warnings



from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import learning_curve

from sklearn import metrics

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix



warnings.filterwarnings('ignore')

%matplotlib inline

print(os.listdir("../input"))
#Bring data

train = pd.read_csv('../input/nassCDS1.csv')

train = train.drop(['Unnamed: 0','Unnamed: 0.1','caseid'], axis=1)

train = train[pd.notnull(train['injSeverity'])]

train = train[pd.notnull(train['yearVeh'])]

train.head()
# Transforming Dead

le = LabelEncoder()

train.dead=le.fit_transform(train.dead)

train.head()
ax = sns.barplot(x="frontal", y="dead", hue='sex', data=train)
ax = sns.barplot(x="seatbelt", y="dead", hue='occRole', data=train)
ax = sns.barplot(x="airbag", y="dead", hue='occRole', data=train)
ax = sns.barplot(x="abcat", y="dead", hue='occRole', data=train)
ax = sns.barplot(x="dvcat", y="weight", hue='dead', data=train)
data = pd.concat([train['weight'], train['ageOFocc']], axis=1)

data.plot.scatter(x='ageOFocc', y='weight');
# Transforming DVCAT

le = LabelEncoder()

le.fit(['1-9km/h','10-24','25-39','40-54', '55+'])

train.dvcat=le.transform(train.dvcat)

train.head()
g = sns.FacetGrid(train, col='dead')

g.map(plt.hist, 'dvcat', bins=20)
grid2 = sns.FacetGrid(train, row='seatbelt', col='dead', size=2.2, aspect=1.6)

grid2.map(sns.barplot, 'sex', 'dvcat', alpha=.5, ci=None)

grid2.add_legend()
ax = sns.barplot(x="injSeverity", y="dvcat", hue='dead', data=train)
# Transforming Airbag

airbag  = pd.get_dummies( train.airbag , prefix='has'  )

train= pd.concat([train, airbag], axis=1)  

# we should drop one of the columns

train = train.drop(['has_none','airbag'], axis=1)

train.head()
# Transforming seatbelt

seatbelt  = pd.get_dummies( train.seatbelt , prefix='is'  )

train= pd.concat([train, seatbelt], axis=1)  

# we should drop one of the columns

train = train.drop(['is_none','seatbelt'], axis=1)

train.head()
# Transforming Abcat

abcat  = pd.get_dummies( train.abcat , prefix='abcat'  )

train= pd.concat([train, abcat], axis=1)  

# we should drop one of the columns

train = train.drop(['abcat_unavail','abcat'], axis=1)

train.head()
# Transforming occRole

occRole  = pd.get_dummies( train.occRole , prefix='is'  )

train= pd.concat([train, occRole], axis=1)  

# we should drop one of the columns

train = train.drop(['is_pass','occRole'], axis=1)

train.head()
# Transforming Sex

le = LabelEncoder()

train.sex=le.fit_transform(train.sex)

train.head()
train.head(10)
corr=train.corr()

fig = plt.figure(figsize=(10,10))

r = sns.heatmap(corr, cmap='Purples')

r.set_title("Correlation ")
#price range correlation

corr.sort_values(by=["dead"],ascending=False).iloc[0].sort_values(ascending=False)
train = train.drop(['abcat_deploy','deploy','is_driver','yearacc','abcat_nodeploy','yearVeh','has_airbag','frontal','weight','is_belted'], axis=1)

train.head()
y = train.dead

X_data=train.drop(["dead"],axis=1)

#x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data,y,test_size = 0.2,random_state=1)
#DesicionTree

from sklearn.tree import DecisionTreeClassifier

model= DecisionTreeClassifier(random_state=1234)

#Hyper Parameters Set

params = {'max_features': [None,'auto', 'sqrt', 'log2'],

          'max_depth' : [None,1,3,5,7,9,11],

          'min_samples_split': [2,3,4,5,6,7,8,9,10], 

          'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11],

          'random_state':[123]}

#Making models with hyper parameters sets

model1 = GridSearchCV(model, param_grid=params, n_jobs=-1)

#Learning

model1.fit(X_train,y_train)

#The best hyper parameters set

print("Best Hyper Parameters:",model1.best_params_)

#Prediction

prediction=model1.predict(X_test)

#evaluation(Accuracy)

print("Accuracy:",metrics.accuracy_score(prediction,y_test))

print("F1-Score:",f1_score(y_test, prediction, average="macro"))

print("Precision:",precision_score(y_test, prediction, average="macro"))

print("Recall:",recall_score(y_test, prediction, average="macro"))  
#DesicionTree

# creating list

max_depth_list = [1,3,5,7,9,11,13,15]

# empty list that will hold cv scores

cv_scores = []

# perform 10-fold cross validation

for d in max_depth_list:

    dtc= DecisionTreeClassifier(max_depth=d, min_samples_split=2,min_samples_leaf=9, random_state=123)

    scores = cross_val_score(dtc, X_train, y_train, cv=10, scoring='accuracy')

    cv_scores.append(scores.mean())

    

# changing to misclassification error

errors = [1 - x for x in cv_scores]



# determining best max_depth

optimal_max_depth = max_depth_list[errors.index(min(errors))]

print ("The optimal max depth is %d" % optimal_max_depth)



# plot misclassification error vs k

plt.plot(max_depth_list, errors)

plt.xlabel('Max Depth')

plt.ylabel('Validation Error')

plt.xticks(max_depth_list)

plt.show()
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=None, train_sizes=np.linspace(.2, .4, 2.0, 5)):



    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Train Size")

    plt.ylabel("Accuracy")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)*100

    train_scores_std = np.std(train_scores, axis=1)*100

    test_scores_mean = np.mean(test_scores, axis=1)*100

    test_scores_std = np.std(test_scores, axis=1)*100

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





title = r"Decision Tree Classifier Learning Curves"

cv = 10

estimator = DecisionTreeClassifier(max_depth=3, min_samples_split=2,min_samples_leaf=9, random_state=123)

plot_learning_curve(estimator, title, X_train, y_train, (80, 100), cv=cv, n_jobs=4, train_sizes=[1,1000,2000,3000,4000,4450])

plt.grid()

plt.show()
#SVM

from sklearn import svm

#making the instance

model=svm.SVC()

#Hyper Parameters Set

params = {'C': [.001,.003,.01,.03,.1,.3,1,3], 

          'kernel': ['linear','rbf'],

          'gamma':['auto',0,.01,.1]}

#Making models with hyper parameters sets

model1 = GridSearchCV(model, param_grid=params, n_jobs=-1)

#Learning

model1.fit(X_train,y_train)

#The best hyper parameters set

print("Best Hyper Parameters:\n",model1.best_params_)

#Prediction

prediction=model1.predict(X_test)

#evaluation(Accuracy)

print("Accuracy:",metrics.accuracy_score(prediction,y_test))

print("F1-Score:",f1_score(y_test, prediction, average="macro"))

print("Precision:",precision_score(y_test, prediction, average="macro"))

print("Recall:",recall_score(y_test, prediction, average="macro"))  
#SVM

# creating list

c_list = [.001,.003,.01,.03,.1,.3,1,3]

# empty list that will hold cv scores

cv_scores = []

# perform 10-fold cross validation

for c in c_list:

    svc= svm.SVC(kernel='linear', C=c)

    scores = cross_val_score(svc, X_train, y_train, cv=10, scoring='accuracy')

    cv_scores.append(scores.mean())

    

# changing to misclassification error

errors = [1 - x for x in cv_scores]

# determining best max_depth

optimal_c = c_list[errors.index(min(errors))]

print ("The C depth is %f" % optimal_c)



# plot misclassification error vs k

plt.plot(c_list, errors)

plt.xlabel('C')

plt.ylabel('Validation Error')

plt.xticks(c_list)

plt.xscale(value='log')

plt.show()
#SVM

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=None, train_sizes=np.linspace(.01, 1.0, 5)):



    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Train Size")

    plt.ylabel("Accuracy")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)*100

    train_scores_std = np.std(train_scores, axis=1)*100

    test_scores_mean = np.mean(test_scores, axis=1)*100

    test_scores_std = np.std(test_scores, axis=1)*100

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



title = r"SVM Classifier Learning Curves"

cv = 10

estimator = svm.SVC(kernel='linear', C=.003)

plot_learning_curve(estimator, title, X_train, y_train, (80, 100), cv=cv, n_jobs=-1)

plt.grid()

plt.show()
#kNearestNeighbors

#importing modules

from sklearn.neighbors import KNeighborsClassifier

#making the instance

model = KNeighborsClassifier(n_jobs=-1)

#Hyper Parameters Set

params = {'n_neighbors':[1,3,5,7,9],

          'leaf_size':[1,2,3,5],

          'weights':['uniform', 'distance'],

          'algorithm':['auto', 'ball_tree','kd_tree','brute'],

          'n_jobs':[-1]}

#Making models with hyper parameters sets

model1 = GridSearchCV(model, param_grid=params, n_jobs=1)

#Learning

model1.fit(X_train,y_train)

#The best hyper parameters set

print("Best Hyper Parameters:\n",model1.best_params_)

#Prediction

prediction=model1.predict(X_test)

#evaluation(Accuracy)

print("Accuracy:",metrics.accuracy_score(prediction,y_test))

print("F1-Score:",f1_score(y_test, prediction, average="macro"))

print("Precision:",precision_score(y_test, prediction, average="macro"))

print("Recall:",recall_score(y_test, prediction, average="macro"))  
#KNN

# creating list

n_neighbors_list = [1,3,5,7,9,11]

# empty list that will hold cv scores

cv_scores = []

# perform 10-fold cross validation

for n in n_neighbors_list:

    knn= KNeighborsClassifier(algorithm='brute', n_neighbors=n, n_jobs= -1, weights= 'distance')

    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')

    cv_scores.append(scores.mean())

    

# changing to misclassification error

errors = [1 - x for x in cv_scores]

#print(errors)

# determining best max_depth

optimal_n = n_neighbors_list[errors.index(min(errors))]

print ("The optimal N neighbor is %f" % optimal_n)



# plot misclassification error vs k

plt.plot(n_neighbors_list, errors)

plt.xlabel('N neighbor')

plt.ylabel('Validation Error')

plt.xticks(n_neighbors_list)

#plt.xscale(value='log')

plt.show()
#KNN

title = r"KNN Classifier Learning Curves"

cv = 10

estimator = KNeighborsClassifier(algorithm='brute', n_neighbors=3, n_jobs= -1, weights= 'distance')

plot_learning_curve(estimator, title, X_train, y_train, (80, 100), cv=cv, n_jobs=-1)

plt.grid()

plt.show()
#Randomforest

#importing modules

from sklearn.ensemble import RandomForestClassifier

#making the instance

model=RandomForestClassifier()

#hyper parameters set

params = {'criterion':['gini','entropy'],

          'n_estimators':[10,15,20,25,30],

          'min_samples_leaf':[1,2,3],

          'min_samples_split':[3,4,5,6,7], 

          'random_state':[123],

          'n_jobs':[-1]}

#Making models with hyper parameters sets

model1 = GridSearchCV(model, param_grid=params, n_jobs=-1)

#learning

model1.fit(X_train,y_train)

#The best hyper parameters set

print("Best Hyper Parameters:\n",model1.best_params_)

#Prediction

prediction=model1.predict(X_test)

#evaluation(Accuracy)

print("Accuracy:",metrics.accuracy_score(prediction,y_test))

print("F1-Score:",f1_score(y_test, prediction, average="macro"))

print("Precision:",precision_score(y_test, prediction, average="macro"))

print("Recall:",recall_score(y_test, prediction, average="macro"))  
#RandomForest

# creating list

estimators_list = [10,15,20,25,30]

# empty list that will hold cv scores

cv_scores = []

# perform 10-fold cross validation

for n in estimators_list:

    dfc= RandomForestClassifier(criterion= 'gini', n_estimators= n, n_jobs= -1, random_state= 123)

    scores = cross_val_score(dfc, X_train, y_train, cv=10, scoring='accuracy')

    cv_scores.append(scores.mean())

    

# changing to misclassification error

errors = [1 - x for x in cv_scores]

#print(errors)

# determining best max_depth

optimal_n = estimators_list[errors.index(min(errors))]

print ("The optimal N estimators is %f" % optimal_n)



# plot misclassification error vs k

plt.plot(estimators_list, errors)

plt.xlabel('N Estimator')

plt.ylabel('Validation Error')

plt.xticks(estimators_list)

#plt.xscale(value='log')

plt.show()
#RandomForest

title = r"Random Forest Classifier Learning Curves"

cv = 10

estimator = RandomForestClassifier(criterion= 'gini', n_estimators= 20, n_jobs= -1, random_state= 123)

plot_learning_curve(estimator, title, X_train, y_train, (90, 100), cv=cv, n_jobs=-1)

plt.grid()

plt.show()
import numpy as np

from scipy import interp

import matplotlib.pyplot as plt



from sklearn import svm, datasets

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import StratifiedKFold



X = X_train

y = y_train

n_samples, n_features = X.shape



# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves

cv = StratifiedKFold(n_splits=6)

classifier = svm.SVC(kernel='linear', probability=True,random_state=1)



tprs = []

aucs = []

mean_fpr = np.linspace(0, 1, 100)



i = 0

for train, test in cv.split(X, y):

    #print("TRAIN:", train, "TEST:", test)

    probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])

    # Compute ROC curve and area the curve

    fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])

    tprs.append(interp(mean_fpr, fpr, tpr))

    tprs[-1][0] = 0.0

    roc_auc = auc(fpr, tpr)

    aucs.append(roc_auc)

    plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))



    i += 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)



mean_tpr = np.mean(tprs, axis=0)

mean_tpr[-1] = 1.0

mean_auc = auc(mean_fpr, mean_tpr)

std_auc = np.std(aucs)

plt.plot(mean_fpr, mean_tpr, color='b',

         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),

         lw=2, alpha=.8)



std_tpr = np.std(tprs, axis=0)

tprs_upper = np.minimum(mean_tpr + std_tpr, 1)

tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,

                 label=r'$\pm$ 1 std. dev.')



plt.xlim([-0.05, 1.05])

plt.ylim([-0.05, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
from sklearn.tree import DecisionTreeClassifier

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

dtc = DecisionTreeClassifier(max_depth=3, min_samples_split=2,min_samples_leaf=9, random_state=123)

svc = svm.SVC(kernel='linear', C=.003)

knn = KNeighborsClassifier(algorithm='brute', n_neighbors=3, n_jobs= -1, weights= 'distance')

rfc = RandomForestClassifier(criterion= 'gini', n_estimators= 20, n_jobs= -1, random_state= 123)

MLA = [

    dtc,

    svc,

    knn,

    rfc

]
MLA_columns = []

MLA_compare = pd.DataFrame(columns = MLA_columns)



row_index = 0

for alg in MLA:

    

    predicted = alg.fit(X_train, y_train).predict(X_test)

    fp, tp, th = roc_curve(y_test, predicted)

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index,'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(alg.score(X_train, y_train), 4)

    MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round(alg.score(X_test, y_test), 4)

    MLA_compare.loc[row_index, 'MLA Precission'] = precision_score(y_test, predicted)

    MLA_compare.loc[row_index, 'MLA Recall'] = recall_score(y_test, predicted)

    MLA_compare.loc[row_index, 'MLA F1-score'] = f1_score(y_test, predicted)

    MLA_compare.loc[row_index, 'MLA AUC'] = auc(fp, tp)



    row_index+=1

    

MLA_compare.sort_values(by = ['MLA Test Accuracy'], ascending = False, inplace = True)    

MLA_compare
plt.subplots(figsize=(15,6))

sns.barplot(x="MLA Name", y="MLA F1-score",data=MLA_compare,palette='hot',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('MLA F1-score Comparison')

plt.show()
plt.subplots(figsize=(15,6))

sns.barplot(x="MLA Name", y="MLA Recall",data=MLA_compare,palette='hot',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('MLA Recall Comparison')

plt.show()
index = 1

for alg in MLA:

    predicted = alg.fit(X_train, y_train).predict(X_test)

    fp, tp, th = roc_curve(y_test, predicted)

    roc_auc_mla = auc(fp, tp)

    MLA_name = alg.__class__.__name__

    plt.plot(fp, tp, lw=2, alpha=0.3, label='ROC %s (AUC = %0.2f)'  % (MLA_name, roc_auc_mla))

   

    index+=1



plt.title('ROC Curve comparison')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.plot([0,1],[0,1],'r--')

plt.xlim([0,1])

plt.ylim([0,1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')    

plt.show()
index = 1

for alg in MLA:

    predicted = alg.fit(X_train, y_train).predict(X_test)

    fp, tp, th = roc_curve(y_test, predicted)

    roc_auc_mla = auc(fp, tp)

    MLA_name = alg.__class__.__name__

    plt.plot(fp, tp, lw=2, alpha=0.3, label='ROC %s (AUC = %0.2f)'  % (MLA_name, roc_auc_mla))

   

    index+=1



plt.title('ROC Curve comparison')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.plot([0,1],[0,1],'r--')

plt.xlim([0,0.15])

plt.ylim([0.75,1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')    

plt.show()