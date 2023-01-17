#structures
import pandas as pd
import numpy as np

#visualization
import matplotlib.pyplot as plt
%matplotlib inline
import math
import seaborn as sns
sns.set()
from mpl_toolkits.mplot3d import Axes3D

#get model duration
import time
from datetime import date

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#load train dataset
loan_data = '../input/loan-application/loan_application_dataset.csv'
df = pd.read_csv(loan_data)
df.shape
df.columns #finding name of columns of the dataset
df.head(5)
#checking datatypes of features
df.dtypes
#for full summary with every column & row
df.describe(include = "all")
df.info()
df.isnull().sum()
#output missing data
missing_data = df.isnull()
missing_data.sample(10)
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")    
sns.heatmap(df.isnull(), yticklabels=False, cmap="viridis")
df.head()
df = df[['Unnamed: 0', 'Unnamed: 0.1', 'Principal', 'terms', 'effective_date', 'due_date', 'age', 'education', 'Gender', 'loan_status']]
df.head()
df.dtypes
df['effective_date'] = pd.to_datetime(df['effective_date'])
df['due_date'] = pd.to_datetime(df['due_date'])
df.education.unique()
df.Gender.unique()
df.loan_status.unique()
df_1 = df.copy()
df_1.head()
encoded_features = {"education": {"High School or Below": 0, "Bechalor": 1, "college": 2, "Master or Above": 3},
                    "Gender": {"male": 0, "female": 1},
                    "loan_status": {"PAIDOFF": 0, "COLLECTION": 1}}
df_1.replace(encoded_features, inplace=True)
df_1.head()
df_2 = df.copy()
df_2.head()
from sklearn.preprocessing import LabelEncoder
lbEncoder = LabelEncoder()
df_2["education"] = lbEncoder.fit_transform(df_2["education"])
df_2["Gender"] = lbEncoder.fit_transform(df_2["Gender"])
df_2["loan_status"] = lbEncoder.fit_transform(df_2["loan_status"])
df_2.head()
df_1.head()
df_1_toscale = df_1.drop(['effective_date', 'due_date', 'education', 'Gender', 'loan_status'], axis=1)
df_1_toscale.head()
df_1_toscale.shape
from sklearn.preprocessing import StandardScaler
#stadardize data
df_1_scaledvalues = StandardScaler().fit_transform(df_1_toscale.values)
df_1_scaledvalues[:3,:] #lost the indices
df_1_scaled = pd.DataFrame(df_1_scaledvalues, index=df_1_toscale.index, columns=df_1_toscale.columns)
df_1_scaled.head()
df_1_scaled.shape
df_1_tocombine = df_1.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Principal', 'terms', 'age'], axis=1)
df_1_tocombine.head()
df_1_scaled = pd.concat([df_1_scaled, df_1_tocombine],axis=1)
df_1_scaled.head()
df_1_scaled.shape
df_1_scaled.drop(columns=['effective_date', 'due_date'], inplace=True)
df_1_scaled.head()
df_1_scaled.shape
#Correlation Heatmap
corr_mat = df_1_scaled.corr()

plt.figure(figsize = (13,5))
sns_plot = sns.heatmap(data = corr_mat, annot = True, cmap='GnBu')
plt.show()
#create a dataframe with all training data except the target column
X = df_1_scaled.drop(columns=['loan_status'])
#check that the target variable has been removed
X.head()
#separate target values
y = df_1_scaled['loan_status'].values
#view target values
y[0:5]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)
X_train.head()
import math
math.sqrt(len(X_test))
from sklearn.neighbors import KNeighborsClassifier
knnmodel = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean') #p is 2 cuz we are looking for 'PAIDOFF' or 'COLLECTION': 2 results
#Fit Model
knnmodel.fit(X_train, y_train)
#predict the test set results
yhat_knn = knnmodel.predict(X_test)
print(yhat_knn)
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, knnmodel.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat_knn))
Ks = 15
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    knnmodel_2 = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat_knn2=knnmodel_2.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat_knn2)

    
    std_acc[n-1]=np.std(yhat_knn2==y_test)/np.sqrt(yhat_knn2.shape[0])

mean_acc
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()
knnmodel = KNeighborsClassifier(n_neighbors=3, p=2, metric='euclidean') #p is 2 cuz we are looking for 'PAIDOFF' or 'COLLECTION': 2 results
#Fit Model
knnmodel.fit(X_train, y_train)
#predict the test set results
yhat_knn = knnmodel.predict(X_test)
print(yhat_knn)
print("Train set Accuracy: ", metrics.accuracy_score(y_train, knnmodel.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat_knn))
from sklearn.metrics import jaccard_score
jaccard_score(y_test, yhat_knn)
from sklearn.metrics import f1_score
f1_score(y_test, yhat_knn)
from sklearn.metrics import log_loss
yhat_knnprob = knnmodel.predict_proba(X_test)
log_loss(y_test, yhat_knnprob)
from sklearn.tree import DecisionTreeClassifier
loanTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
loanTree.fit(X_train, y_train)
yhat_tree = loanTree.predict(X_test)
print (yhat_tree [0:5])
print (y_test [0:5])
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, yhat_tree))
jaccard_score(y_test, yhat_tree)
f1_score(y_test, yhat_tree)
yhat_treeprob = loanTree.predict_proba(X_test)
log_loss(y_test, yhat_treeprob)
from sklearn import svm
loanSVM = svm.SVC(kernel='rbf')
loanSVM.fit(X_train, y_train)
yhat_svm = loanSVM.predict(X_test)
print (yhat_svm [0:5])
print (y_test [0:5])
print("SVM's Accuracy: ", metrics.accuracy_score(y_test, yhat_svm))
jaccard_score(y_test, yhat_svm)
f1_score(y_test, yhat_svm)
from sklearn.linear_model import LogisticRegression
lregressionmodel = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
lregressionmodel
yhat_lregression = lregressionmodel.predict(X_test)
print (yhat_lregression [0:5])
print (y_test [0:5])
print("Logistics Regression's Accuracy: ", metrics.accuracy_score(y_test, yhat_lregression))
jaccard_score(y_test, yhat_lregression)
f1_score(y_test, yhat_lregression)
yhat_lregressionprob = lregressionmodel.predict_proba(X_test)
log_loss(y_test, yhat_lregressionprob)