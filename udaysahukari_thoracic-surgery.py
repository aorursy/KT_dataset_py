# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/thoracicsurgery/Thoracic-Surgery.csv', index_col = 'id')
df.head()
df.info()
df[['PRE7', 'PRE8', 'PRE9', 'PRE10', 'PRE11', 'PRE17', 'PRE19', 'PRE25', 'PRE30', 'PRE32', 'Risk1Yr']] = (df[['PRE7', 'PRE8', 'PRE9', 'PRE10', 'PRE11', 'PRE17', 'PRE19', 'PRE25', 'PRE30', 'PRE32', 'Risk1Yr']] == 'T').astype(int)
df.head()
df.shape
df['DGN'] = df['DGN'].str[-1:].astype(int)

df['PRE6'] = df['PRE6'].str[-1:].astype(int)

df['PRE14'] = df['PRE14'].str[-1:].astype(int)
df.columns
col = ['Daignosis','Forced_Capacity','Forced_Expiration','Zubrod_scale','Pain',' Haemoptysis','Dyspnoea',

       'Cough','Weakness','Size_of_tumor','diabetes','MI_6months','PAD','Smoker','Asthmatic','Age','Risk_1y']

df.columns = col
df.head()
# Train test split 

from sklearn.model_selection import train_test_split 

X = df.drop('Risk_1y', axis=1) 

y = df.Risk_1y 

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 3) 
X_train.head() 
y_test[:10] 
from sklearn.linear_model import LogisticRegression 

model_lr = LogisticRegression() 

model_lr.fit(X_train, y_train) 

model_lr.score(x_test, y_test) 
y_pred = model_lr.predict(x_test)  
from sklearn.metrics import accuracy_score 

print(accuracy_score(y_test, y_pred)) 
# from sklearn.metrics import precision_score

# print(precision_score(y_true, y_pred))



# The precision is the ratio tp / (tp + fp)

# y_true = [0, 1, 2, 0, 1, 2]

# y_pred = [0, 2, 1, 0, 0, 1]
from sklearn.metrics import confusion_matrix 

c_matrix = confusion_matrix(y_test, y_pred) 

c_matrix
import seaborn as sns

sns.heatmap(c_matrix, annot=True, cmap = 'Blues')

plt.show()
from sklearn.neighbors import KNeighborsClassifier 

model_knc = KNeighborsClassifier(n_neighbors=5)  

model_knc.fit(X_train, y_train)  

y_knn_pred = model_knc.predict(x_test)
accuracy_score(y_test, y_knn_pred) 
model_knc.predict_proba(x_test) 
model_knc.score(x_test, y_test) 
c_matrix = confusion_matrix(y_test, y_knn_pred) 

print(c_matrix)



sns.heatmap(c_matrix, annot=True, cmap = 'Blues') 

plt.show() 
# For KNN = 1 

model_knc_1 = KNeighborsClassifier(n_neighbors = 1) 

model_knc_1.fit(X_train, y_train) 

y_knn_pred_1 = model_knc_1.predict(x_test) 

accuracy_score(y_test, y_knn_pred_1) 
c_matrix = confusion_matrix(y_test, y_knn_pred_1) 

print(c_matrix)



sns.heatmap(c_matrix, annot=True, cmap = 'Blues') 

plt.show() 
# for KNN = 3 

model_knc_3 = KNeighborsClassifier(n_neighbors = 3) 

model_knc_3.fit(X_train, y_train) 

y_knn_pred_3 = model_knc_3.predict(x_test) 

accuracy_score(y_test, y_knn_pred_3)  
c_matrix = confusion_matrix(y_test, y_knn_pred_3) 

print(c_matrix)



sns.heatmap(c_matrix, annot=True, cmap = 'Blues') 

plt.show() 
# for KNN = 5 

model_knc_5 = KNeighborsClassifier(n_neighbors = 5) 

model_knc_5.fit(X_train, y_train) 

y_knn_pred_5 = model_knc_5.predict(x_test) 

accuracy_score(y_test, y_knn_pred_5)  
c_matrix = confusion_matrix(y_test, y_knn_pred_5) 

print(c_matrix)



sns.heatmap(c_matrix, annot=True, cmap = 'Blues') 

plt.show() 
# for KNN = 7 

model_knc_7 = KNeighborsClassifier(n_neighbors = 7) 

model_knc_7.fit(X_train, y_train) 

y_knn_pred_7 = model_knc_7.predict(x_test) 

accuracy_score(y_test, y_knn_pred_7)  
c_matrix = confusion_matrix(y_test, y_knn_pred_7) 

print(c_matrix)



sns.heatmap(c_matrix, annot=True, cmap = 'Blues') 

plt.show() 
# for KNN = 9 

model_knc_9 = KNeighborsClassifier(n_neighbors = 9) 

model_knc_9.fit(X_train, y_train) 

y_knn_pred_9 = model_knc_9.predict(x_test) 

accuracy_score(y_test, y_knn_pred_9) 
c_matrix = confusion_matrix(y_test, y_knn_pred_9) 

print(c_matrix)



sns.heatmap(c_matrix, annot=True, cmap = 'Blues') 

plt.show() 
# Evaluation of KNN models on bases of Accuracy. 

dict_ms = {'N_Neighbors':[1,3,5,7,9], 'Accuracy':[accuracy_score(y_test, y_knn_pred_1) * 100, 

                                                  accuracy_score(y_test, y_knn_pred_3) * 100, 

                                                  accuracy_score(y_test, y_knn_pred) * 100,

                                                  accuracy_score(y_test, y_knn_pred_7) * 100,

                                                  accuracy_score(y_test, y_knn_pred_9) * 100]} 

model_selection = pd.DataFrame(dict_ms)  

model_selection 
x = df.drop('Risk_1y', axis=1) 

x_norm = (x-np.min(x))/(np.max(x)-np.min(x))   

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3) 
# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge

logreg=LogisticRegression()

logreg_cv=GridSearchCV(logreg,grid,cv=10)

logreg_cv.fit(x_train,y_train)



print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)

print("accuracy :",logreg_cv.best_score_) 
logreg2=LogisticRegression(C=0.001,penalty="l2")

logreg2.fit(x_train,y_train)

print("score",logreg2.score(x_test,y_test)) 
c_matrix = confusion_matrix(y_test, y_pred) 

print(c_matrix)



sns.heatmap(c_matrix, annot=True, cmap = 'Blues') 

plt.show() 
df.info()  
category_col = ['Daignosis','Zubrod_scale','Pain', ' Haemoptysis', 'Dyspnoea', 'Cough','Weakness',

               'Size_of_tumor', 'diabetes', 'MI_6months', 'PAD', 'Smoker', 'Asthmatic', 'Risk_1y'] 

for col in category_col:

     df[col]=df[col].astype('category')
df.info() 
X_dtrees = df.drop('Risk_1y', axis=1) 

y_dtrees = df['Risk_1y'] 
from sklearn.model_selection import train_test_split  

X_train, x_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size = 0.3, 

                                                    random_state = 3) 

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

clf = DecisionTreeClassifier()



# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)



#Predict the response for test dataset

y_pred = clf.predict(x_test)  
from sklearn.metrics import accuracy_score 

accuracy_score(y_test, y_pred) 
c_matrix = confusion_matrix(y_test, y_pred) 

print(c_matrix)



sns.heatmap(c_matrix, annot=True, cmap = 'Blues') 

plt.show() 
#Import Random Forest Model

from sklearn.ensemble import RandomForestClassifier



#Create a Gaussian Classifier

clf=RandomForestClassifier(n_estimators=100)



#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X_train,y_train)



y_pred=clf.predict(x_test)  



print("Accuracy:", accuracy_score(y_test, y_pred)) 
c_matrix = confusion_matrix(y_test, y_pred) 

print(c_matrix)



sns.heatmap(c_matrix, annot=True, cmap = 'Blues') 

plt.show() 
from sklearn.feature_selection import RFE

logReg = LogisticRegression()

rfe_selector = RFE(logReg, 10) 

rfe_selector.fit(X_train, y_train)
rfe_selector.support_
rfe_selector.ranking_
X_train.columns
top_features = ['Daignosis', 'Forced_Expiration', 'Zubrod_scale','Pain', 'Dyspnoea', 'Cough', 'Size_of_tumor', 

                'diabetes', 'MI_6months', 'Smoker']

X_selected = X_train[top_features] 

x_test_selected = x_test[top_features]
logreg3 = LogisticRegression()

logreg3.fit(X_selected,y_train)

print("score",logreg3.score(x_test_selected, y_test)) 