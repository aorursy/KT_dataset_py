# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import models 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# metrics 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve

# graphs 
import matplotlib.pyplot as plt 

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/column_2C_weka.csv')
print ( 'distinct class values : ' , data['class'].unique())

data['class'] = [1 if each == 'Normal' else 0 for each in data['class'] ]
x = data.loc[:,data.columns != 'class'] # veya data.drop(['class'], axis = 1) 
y = data.loc[:,'class']

x.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 1)

print ('x_train shape: {} '.format(x_train.shape))
print ('y_train shape: {} '.format(y_train.shape))
print ('x_test shape: {} '.format(x_test.shape))
print ('y_test shape: {} '.format(y_test.shape))

# find  best k value 
knn_accuracy_list =[]
for k in  range (1,25):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    knn_accuracy_list.append(knn.score(x_test, y_test))
    
print ('best k is :{} , best acccuracy is :{} '.format(  knn_accuracy_list.index (np.max(knn_accuracy_list))+1, np.max(knn_accuracy_list)))

#knn classifier 
knn = KNeighborsClassifier(n_neighbors = knn_accuracy_list.index (np.max(knn_accuracy_list))+1 )
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
print ('KNeighborsClassifier test accuracy ' , knn.score(x_test, y_test) )

rfc_accuracy_list =[]
for r in  range (1,25):
    rfc = RandomForestClassifier(random_state = r)
    rfc.fit(x_train, y_train)
    rfc_accuracy_list.append(rfc.score(x_test, y_test))
    
print ('best r is :{} , best acccuracy is :{} '.format(  rfc_accuracy_list.index (np.max(rfc_accuracy_list))+1, np.max(rfc_accuracy_list)))
    
rfc = RandomForestClassifier(random_state = rfc_accuracy_list.index (np.max(rfc_accuracy_list))+1)
rfc.fit(x_train,y_train)
y_pred_rfc = rfc.predict(x_test)
print ('RandomForestClassifier test accuracy ' , rfc.score(x_test, y_test) )
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred_logreg = logreg.predict(x_test)
print ('LogisticRegression test accuracy ' , logreg.score(x_test, y_test) )
cm = confusion_matrix(y_test,y_pred_knn)
print('KNN Confusion matrix: \n',cm)
print('KNN Classification report: \n',classification_report(y_test,y_pred_knn))

cm = confusion_matrix(y_test,y_pred_rfc)
print('RandomForestClassifier Confusion matrix: \n',cm)
print('RandomForestClassifier Classification report: \n',classification_report(y_test,y_pred_rfc))

cm = confusion_matrix(y_test,y_pred_logreg)
print('LogisticRegression Confusion matrix: \n',cm)
print('LogisticRegression Classification report: \n',classification_report(y_test,y_pred_logreg))

y_pred_knn_prob = knn.predict_proba(x_test)[:,1]
y_pred_rfc_prob = rfc.predict_proba(x_test)[:,1]
y_pred_lr_prob = logreg.predict_proba(x_test)[:,1]

fpr_knn, tpr_knn, thresholds = roc_curve(y_test, y_pred_knn_prob) 
fpr_rfc, tpr_rfc, thresholds = roc_curve(y_test, y_pred_rfc_prob) 
fpr_lr, tpr_lr, thresholds = roc_curve(y_test, y_pred_lr_prob) 

plt.figure (figsize=[13 ,8])
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_knn, tpr_knn, label='KNN')
plt.plot(fpr_rfc, tpr_rfc, label='Random Forest')
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC')
plt.show()