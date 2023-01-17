import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
import time

# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
X = df.drop('Class', axis=1)
y = pd.DataFrame(df['Class'])
df.info()
print(y.describe().round(decimals=2))

sns.countplot('Class', data=df)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
plt.show()
print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
X.describe().round(3)
fig = plt.figure(figsize=(12,20))
plt.title('Distribution of Feature Attributes')
for i in range(len(X.columns)):
    fig.add_subplot(8,4,i+1)
    sns.distplot(X.iloc[:,i].dropna())
    plt.xlabel(X.columns[i])

plt.tight_layout()
plt.show()
fig = plt.figure(figsize=(12,20))
plt.title('Numerical Feature (before dropping identified outliers)')
for i in range(len(X.columns)):
    fig.add_subplot(8,4,i+1)
    sns.scatterplot(X.iloc[:,i], y.iloc[:,0])
    plt.xlabel(X.columns[i])

plt.tight_layout()
plt.show()
correlation = X.corr()

f, ax = plt.subplots(figsize=(14,12))
plt.title('Correlation of numerical attributes', size=30)
sns.heatmap(correlation)
plt.show()

y_corr = pd.DataFrame(X.corrwith(y.Class),columns=["Correlation with target variable"])
y_corr_sorted= y_corr.sort_values(by=['Correlation with target variable'],ascending=False)
y_corr_sorted
fig = plt.figure(figsize=(6,10))
plt.title('Correlation with target variable')
a=sns.barplot(y_corr_sorted.index,y_corr_sorted.iloc[:,0],data=y_corr)
a.set_xticklabels(labels=y_corr_sorted.index,rotation=90)
plt.tight_layout()
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
from imblearn.over_sampling import SMOTE
resampling_method = SMOTE()

X_train_resampled, y_train_resampled = resampling_method.fit_resample(X_train, y_train)
sns.countplot('Class', data=y_train_resampled)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
plt.show()
print('No Frauds:', round(y_train_resampled['Class'].value_counts()[0]), 'data points')

print('Frauds Data points Before SMOTE:', round(df['Class'].value_counts()[1]))
print('Frauds Data points After SMOTE:', round(y_train_resampled['Class'].value_counts()[1]), 'data points')

Fraud_obs_added = round((y_train_resampled['Class'].value_counts()[1])-df['Class'].value_counts()[1])
print('Frauds Data points Added:',Fraud_obs_added , 'data points')


print('No Frauds:', round(y_train_resampled['Class'].value_counts()[0]/len(y_train_resampled) * 100,2), '% of the dataset')
print('Frauds:', round(y_train_resampled['Class'].value_counts()[1]/len(y_train_resampled) * 100,2), '% of the dataset')

y_corr = pd.DataFrame(X_train_resampled.corrwith(y_train_resampled.Class),columns=["Correlation with target variable"])
y_corr_sorted= y_corr.sort_values(by=['Correlation with target variable'],ascending=False)
y_corr_sorted
fig = plt.figure(figsize=(6,10))
plt.title('Correlation with target variable')
a=sns.barplot(y_corr_sorted.index,y_corr_sorted.iloc[:,0],data=y_corr)
a.set_xticklabels(labels=y_corr_sorted.index,rotation=90)
plt.tight_layout()
plt.show()
from imblearn.pipeline import Pipeline 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer,roc_curve

resampling = SMOTE(random_state=4)
roc_auc = make_scorer(roc_auc_score)
training_cv_list={}
training_cv_best_auc={}

test_best_auc = {}
test_best_classification_report={}
test_confusion_matrix={}
from sklearn.linear_model import LogisticRegression

model_name = "Logistics Regression (Ridge)"
model=LogisticRegression(max_iter=4000, penalty="l2")

param_grid = [{model_name+'__C':[0.01,0.1,1,10,100]}]
#LR
pipeline = Pipeline([('SMOTE', resampling), (model_name, model)])


clf=GridSearchCV(pipeline,param_grid,cv=5, scoring=roc_auc, n_jobs=-1)
clf.fit(X_train,y_train.to_numpy())


#Record the best grid search paramters into the list.
training_cv_list[model_name]=clf
training_cv_best_auc[model_name]=clf.best_score_
#print out the best param and best score 
print('best training param:',clf.best_params_)
print('best training score', clf.best_score_)
print('\n')

#make prediction on X_test
pred_prob_y = clf.predict_proba(X_test)
pred_y = clf.predict(X_test)

#compute auc, classification report,confusion matrix 
aucroc = roc_auc_score(y_test,pred_prob_y[:,1])
confusionmatrix = confusion_matrix(y_test,pred_y)
classificationreport = classification_report(y_test,pred_y)
fpr, tpr, thresholds = roc_curve(y_test, pred_prob_y[:,1])

#store results
test_best_auc[model_name]=aucroc
test_best_classification_report[model_name]=confusionmatrix
test_confusion_matrix[model_name]=classificationreport

#print results
print('test auc roc:',aucroc)
print('test confusion matrix: \n',confusionmatrix)
print('test classification report \n', classificationreport)
plt.plot(fpr, tpr, marker='.')
plt.title('ROC Plot: '+model_name)
model_name = "Logistics Regression (Ridge)"
model=LogisticRegression(max_iter=4000, penalty="l1",solver='liblinear')

param_grid = [{model_name+'__C':[0.01,0.1,1,10,100]}]
pipeline = Pipeline([('SMOTE', resampling), (model_name, model)])


clf=GridSearchCV(pipeline,param_grid,cv=5, scoring=roc_auc, n_jobs=-1)
clf.fit(X_train,y_train.to_numpy())


#Record the best grid search paramters into the list.
training_cv_list[model_name]=clf
training_cv_best_auc[model_name]=clf.best_score_
#print out the best param and best score 
print('best training param:',clf.best_params_)
print('best training score', clf.best_score_)
print('\n')

#make prediction on X_test
pred_prob_y = clf.predict_proba(X_test)
pred_y = clf.predict(X_test)

#compute auc, classification report,confusion matrix 
aucroc = roc_auc_score(y_test,pred_prob_y[:,1])
confusionmatrix = confusion_matrix(y_test,pred_y)
classificationreport = classification_report(y_test,pred_y)
fpr, tpr, thresholds = roc_curve(y_test, pred_prob_y[:,1])

#store results
test_best_auc[model_name]=aucroc
test_best_classification_report[model_name]=confusionmatrix
test_confusion_matrix[model_name]=classificationreport

#print results
print('test auc roc:',aucroc)
print('test confusion matrix: \n',confusionmatrix)
print('test classification report \n', classificationreport)
plt.plot(fpr, tpr, marker='.')
plt.title('ROC Plot: '+model_name)
from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier()
model_name = 'DecisionTreeClassifier'
param_grid = [{model_name+'__'+'splitter':['best','random'],model_name+'__'+'max_depth':[1,2,3,5,10,20]}]
pipeline = Pipeline([('SMOTE', resampling), (model_name, model)])

clf=GridSearchCV(pipeline,param_grid,cv=5, scoring=roc_auc, n_jobs=-1)
clf.fit(X_train,y_train.to_numpy())


#Record the best grid search paramters into the list.
training_cv_list[model_name]=clf
training_cv_best_auc[model_name]=clf.best_score_
#print out the best param and best score 
print('best training param:',clf.best_params_)
print('best training score', clf.best_score_)
print('\n')

#make prediction on X_test
pred_prob_y = clf.predict_proba(X_test)
pred_y = clf.predict(X_test)

#compute auc, classification report,confusion matrix 
aucroc = roc_auc_score(y_test,pred_prob_y[:,1])
confusionmatrix = confusion_matrix(y_test,pred_y)
classificationreport = classification_report(y_test,pred_y)
fpr, tpr, thresholds = roc_curve(y_test, pred_prob_y[:,1])

#store results
test_best_auc[model_name]=aucroc
test_best_classification_report[model_name]=confusionmatrix
test_confusion_matrix[model_name]=classificationreport

#print results
print('test auc roc:',aucroc)
print('test confusion matrix: \n',confusionmatrix)
print('test classification report \n', classificationreport)
plt.plot(fpr, tpr, marker='.')
plt.title('ROC Plot: '+model_name)
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

model_name = 'BaggingDecisionTreeClassifier'
model=BaggingClassifier(DecisionTreeClassifier())

param_grid = [{model_name+'__'+'base_estimator__splitter': ['best','random'],
              model_name+'__'+'base_estimator__max_depth':np.arange(1,5)
              }]
pipeline = Pipeline([('SMOTE', resampling), (model_name, model)])


clf=GridSearchCV(pipeline,param_grid,cv=5, scoring=roc_auc, n_jobs=-1)
clf.fit(X_train,y_train.to_numpy())


#Record the best grid search paramters into the list.
training_cv_list[model_name]=clf
training_cv_best_auc[model_name]=clf.best_score_
#print out the best param and best score 
print('best training param:',clf.best_params_)
print('best training score', clf.best_score_)
print('\n')

#make prediction on X_test
pred_prob_y = clf.predict_proba(X_test)
pred_y = clf.predict(X_test)

#compute auc, classification report,confusion matrix 
aucroc = roc_auc_score(y_test,pred_prob_y[:,1])
confusionmatrix = confusion_matrix(y_test,pred_y)
classificationreport = classification_report(y_test,pred_y)
fpr, tpr, thresholds = roc_curve(y_test, pred_prob_y[:,1])

#store results
test_best_auc[model_name]=aucroc
test_best_classification_report[model_name]=confusionmatrix
test_confusion_matrix[model_name]=classificationreport

#print results
print('test auc roc:',aucroc)
print('test confusion matrix: \n',confusionmatrix)
print('test classification report \n', classificationreport)
plt.plot(fpr, tpr, marker='.')
plt.title('ROC Plot: '+model_name)
from sklearn.ensemble import RandomForestClassifier
param_grid = {
    model_name+'__'+'max_depth':[1,2,3,5,10,20]}

model=RandomForestClassifier()
model_name = 'RandomForestClassifier'

pipeline = Pipeline([('SMOTE', resampling), (model_name, model)])


clf=GridSearchCV(pipeline,param_grid,cv=5, scoring=roc_auc, n_jobs=-1)
clf.fit(X_train,y_train.to_numpy())


#Record the best grid search paramters into the list.
training_cv_list[model_name]=clf
training_cv_best_auc[model_name]=clf.best_score_
#print out the best param and best score 
print('best training param:',clf.best_params_)
print('best training score', clf.best_score_)
print('\n')

#make prediction on X_test
pred_prob_y = clf.predict_proba(X_test)
pred_y = clf.predict(X_test)

#compute auc, classification report,confusion matrix 
aucroc = roc_auc_score(y_test,pred_prob_y[:,1])
confusionmatrix = confusion_matrix(y_test,pred_y)
classificationreport = classification_report(y_test,pred_y)
fpr, tpr, thresholds = roc_curve(y_test, pred_prob_y[:,1])

#store results
test_best_auc[model_name]=aucroc
test_best_classification_report[model_name]=confusionmatrix
test_confusion_matrix[model_name]=classificationreport

#print results
print('test auc roc:',aucroc)
print('test confusion matrix: \n',confusionmatrix)
print('test classification report \n', classificationreport)
plt.plot(fpr, tpr, marker='.')
plt.title('ROC Plot: '+model_name)
# from sklearn.svm import SVC

# model=SVC()
# model_name = 'SVC'

# param_grid = [
#   {model_name+'__'+'C': [2**-5,2**-3,2**-1,2**1,2**3,2**5,2**7,2**9,2**11,2**13,2**15], model_name+'__'+'kernel': ['linear','poly','rbf','sigmoid'],
#    model_name+'__'+'gamma':['scale','auto',2**-5,2**-3,2**-1,2**1,2**3,2**5,2**7,2**9,2**11,2**13,2**15]
#   }]


# %%time
# #SVM
# pipeline = Pipeline([('SMOTE', resampling), (model_name, model)])


# clf=GridSearchCV(pipeline,param_grid,cv=5, scoring=roc_auc, n_jobs=-1)
# clf.fit(X_train,y_train.to_numpy())


# #Record the best grid search paramters into the list.
# training_cv_list[model_name]=clf
# training_cv_best_auc[model_name]=clf.best_score_
# #print out the best param and best score 
# print('best training param:',clf.best_params_)
# print('best training score', clf.best_score_)
# print('\n')

# #make prediction on X_test
# pred_prob_y = clf.predict_proba(X_test)
# pred_y = clf.predict(X_test)

# #compute auc, classification report,confusion matrix 
# aucroc = roc_auc_score(y_test,pred_prob_y[:,1])
# confusionmatrix = confusion_matrix(y_test,pred_y)
# classificationreport = classification_report(y_test,pred_y)
# fpr, tpr, thresholds = roc_curve(y_test, pred_prob_y[:,1])

# #store results
# test_best_auc[model_name]=aucroc
# test_best_classification_report[model_name]=confusionmatrix
# test_confusion_matrix[model_name]=classificationreport

# #print results
# print('test auc roc:',aucroc)
# print('test confusion matrix: \n',confusionmatrix)
# print('test classification report \n', classificationreport)
# plt.plot(fpr, tpr, marker='.')
# plt.title('ROC Plot: '+model_name)
best_auc_score = pd.DataFrame.from_dict(test_best_auc,orient='index')
best_auc_score

