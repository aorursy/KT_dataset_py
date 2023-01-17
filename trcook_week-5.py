#Add packages
%matplotlib inline 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn import tree 
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as ss
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.grid_search import GridSearchCV
import time
from operator import itemgetter
import os
os.getcwd()
#import data
df = pd.read_csv("../input/week-2-data/Churn_Calls.csv", sep=',')
df.head(10)
# See each collum name
print(df.columns)
df.shape
# designate target variable name
targetName = 'churn'
# move target variable into first column
targetSeries = df[targetName]
del df[targetName]
df.insert(0, targetName, targetSeries)
expected=targetName
df.head(10)
gb = df.groupby(targetName)
targetEDA=gb[targetName].aggregate(len)
plt.figure()
targetEDA.plot(kind='bar', grid=False)
plt.axhline(0, color='k')
from sklearn import preprocessing
le_dep = preprocessing.LabelEncoder()
#to convert into numbers
df['churn'] = le_dep.fit_transform(df['churn'])
# perform data transformation
for col in df.columns[1:]:
	attName = col
	dType = df[col].dtype
	missing = pd.isnull(df[col]).any()
	uniqueCount = len(df[attName].value_counts(normalize=False))
	# discretize (create dummies)
	if dType == object:
		df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
		del df[attName]
# split dataset into testing and training
features_train, features_test, target_train, target_test = train_test_split(
    df.ix[:,1:].values, df.ix[:,0].values, test_size=0.40, random_state=0)
print(features_test.shape)
print(features_train.shape)
print(target_test.shape)
print(target_train.shape)
print("Percent of Target that is Yes", target_test.mean())
#data.groupby(['col1', 'col2'])
#Decision Tree train model
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, target_train)
#DT test model
target_predicted_dt = clf.predict(features_test)
print("DT Accuracy Score", accuracy_score(target_test, target_predicted_dt))
# print classification report
target_names = ["Fail = no", "Fail = yes"]
print(classification_report(target_test, target_predicted_dt, target_names=target_names))
#verify DT with Cross Validation
scores = cross_val_score(clf, features_train, target_train, cv=10)
print("Cross Validation Score for each K",scores)
scores.mean()                             
# display confusion matrix
cm = confusion_matrix(target_test, target_predicted_dt)
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
print(cm)
# train random forest model
#paralleized to 4 cores 
rf = RandomForestClassifier(n_estimators= 500, n_jobs=-1,oob_score=True)
rf.fit(features_train, target_train)
# test random forest model
target_predicted_rf = rf.predict(features_test)
print(accuracy_score(target_test, target_predicted_rf))
target_names = ["Churn = no", "Churn = yes"]
print(classification_report(target_test, target_predicted_rf, target_names=target_names))
print(confusion_matrix(target_test, target_predicted_rf))

#verify RF with cross validation
scores_rf = cross_val_score(rf, features_train, target_train, cv=10, n_jobs=-1)
print("Cross Validation Score for each K",scores_rf)
scores_rf.mean()
# display confusion matrix
cm = confusion_matrix(target_test, target_predicted_rf)
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
print(cm)
from sklearn.model_selection import GridSearchCV
# use a full grid over all parameters
param_grid = {"max_features": [2, 3, 4, 5]}
start_time = time.clock()




# run grid search
grid_search = GridSearchCV(rf, param_grid=param_grid,n_jobs=-1)

grid_search.fit(features_train, target_train)

print(time.clock() - start_time, "seconds")
print(grid_search.cv_results_)

#Show importance of each feature in Random Forest
zip(df.columns[1:20], rf.feature_importances_)
# Determine the false positive and true positive rates
fpr, tpr, _ = roc_curve(target_test, rf.predict_proba(features_test)[:,1]) 
    
# Calculate the AUC
roc_auc = auc(fpr, tpr)
print('ROC AUC: %0.3f' % roc_auc)
 
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
from sklearn.neural_network import MLPClassifier
nn = MLPClassifier(activation='relu',hidden_layer_sizes=(5, 2), solver='adam',learning_rate_init=0.001, max_iter=500,validation_fraction=0.1, verbose=False,
       warm_start=False)
nn.fit(features_train, target_train)

y_valid = nn.predict(features_test)

score = nn.score(features_test, target_test)
print(score)
from sklearn.neural_network import MLPClassifier
clf_NN1 =MLPClassifier(activation='relu', hidden_layer_sizes=(15,), learning_rate_init=0.001, max_iter=200, random_state=1, 
       solver='adam', validation_fraction=0.1, verbose=False,
       )
clf_NN1.fit(features_train, target_train)
# test random forest model
target_predicted_NN1 = clf_NN1.predict(features_test)
print("Accuracy", accuracy_score(target_test, target_predicted_NN1))
target_names = ["Churn = no", "Churn = yes"]
print(classification_report(target_test, target_predicted_NN1, target_names=target_names))
print(confusion_matrix(target_test, target_predicted_NN1))
