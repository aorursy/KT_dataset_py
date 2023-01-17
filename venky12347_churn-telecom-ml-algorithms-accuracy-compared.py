# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import datetime

from sklearn.preprocessing import LabelEncoder

from math import sqrt



pd.set_option('display.max_columns', 0)

pd.set_option('display.max_rows', 500)



from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score





from sklearn import tree

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import DecisionTreeRegressor

churn_df=pd.read_csv("../input/churn.csv")
churn_df.shape
churn_df.isna().sum()
churn_df.dtypes.value_counts()
churn_df=churn_df.rename(columns={'Churn?':'churn'})

churn_df=churn_df.rename(columns={"Int'l Plan":"Intl Plan"})

churn_df.columns
churn_df['churn'].value_counts()
churn_df['churn']=churn_df['churn'].apply(lambda x:1 if x=="True." else 0 )

churn_df['Intl Plan']=churn_df['Intl Plan'].apply(lambda x:1 if x=="yes" else 0 )

churn_df['VMail Plan']=churn_df['VMail Plan'].apply(lambda x:1 if x=="yes" else 0 )

churn_df['VMail Plan'].value_counts()

churn_df.head(10)
churn_df_dropped=churn_df.drop(['State','Area Code','Phone'],axis=1)
churn_df_dropped.columns
X=churn_df_dropped.drop(columns=['churn'])

y=churn_df_dropped[['churn']]
X_train, X_test, y_train, y_test = train_test_split(

X, y, test_size = 0.2, random_state = 100)



#y_train = y_train.ravel()

#y_test = y_test.ravel()
# using logistic regression

from sklearn import metrics

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

train_Pred = logreg.predict(X_train)

test_pred = logreg.predict(X_test)

print("Accuracy of Logistic regression for train",metrics.accuracy_score(y_train,train_Pred))

print("Accuracy of Logistic regression for test",metrics.accuracy_score(y_test,test_pred))
y_train=np.ravel(y_train)

y_test=np.ravel(y_test)

accuracy_train_dict={}

accuracy_test_dict={}

df_len=round(sqrt(len(churn_df_dropped)))

for k in range(3,df_len):

    K_value = k+1

    neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')

    neigh.fit(X_train, y_train) 

    y_pred_train = neigh.predict(X_train)

    y_pred_test = neigh.predict(X_test)    

    train_accuracy=accuracy_score(y_train,y_pred_train)*100

    test_accuracy=accuracy_score(y_test,y_pred_test)*100

    accuracy_train_dict.update(({k:train_accuracy}))

    accuracy_test_dict.update(({k:test_accuracy}))

    print ("Accuracy for train :",train_accuracy ," and test :",test_accuracy,"% for K-Value:",K_value)
# using Naive bayes

from sklearn.naive_bayes import GaussianNB

NB=GaussianNB()

NB.fit(X_train, y_train)

train_pred=NB.predict(X_train)

test_pred=NB.predict(X_test)

print("Accuracy of Naive bayes train set",accuracy_score(train_pred,y_train))

print("Accuracy of Naive bayes test set",accuracy_score(test_pred,y_test))
## Boosting

import matplotlib.pyplot as plt

# Adaboost Classifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

dtree = DecisionTreeClassifier(criterion='gini',max_depth=1)



adabst_fit = AdaBoostClassifier(base_estimator= dtree,

        n_estimators=5000,learning_rate=0.05,random_state=42)



adabst_fit.fit(X_train, y_train)



#print ("\nAdaBoost - Train Confusion Matrix\n\n",pd.crosstab(y_train,adabst_fit.predict(X_train),rownames = ["Actuall"],colnames = ["Predicted"]))      

print ("\nAdaBoost  - Train accuracy",round(accuracy_score(y_train,adabst_fit.predict(X_train)),3))

#print ("\nAdaBoost  - Train Classification Report\n",classification_report(y_train,adabst_fit.predict(X_train)))



#print ("\n\nAdaBoost  - Test Confusion Matrix\n\n",pd.crosstab(y_test,adabst_fit.predict(X_test),rownames = ["Actuall"],colnames = ["Predicted"]))      

print ("\nAdaBoost  - Test accuracy",round(accuracy_score(y_test,adabst_fit.predict(X_test)),3))

#print ("\nAdaBoost - Test Classification Report\n",classification_report(y_test,adabst_fit.predict(X_test)))
# Gradientboost Classifier

from sklearn.ensemble import GradientBoostingClassifier



gbc_fit = GradientBoostingClassifier(loss='deviance',learning_rate=0.05,n_estimators=5000,

                                     min_samples_split=2,min_samples_leaf=1,max_depth=1,random_state=42 )

gbc_fit.fit(X_train,y_train)



#print ("\nGradient Boost - Train Confusion Matrix\n\n",pd.crosstab(y_train,gbc_fit.predict(X_train),rownames = ["Actuall"],colnames = ["Predicted"]))      

print ("\nGradient Boost - Train accuracy",round(accuracy_score(y_train,gbc_fit.predict(X_train)),3))

#print ("\nGradient Boost  - Train Classification Report\n",classification_report(y_train,gbc_fit.predict(X_train)))



#print ("\n\nGradient Boost - Test Confusion Matrix\n\n",pd.crosstab(y_test,gbc_fit.predict(X_test),rownames = ["Actuall"],colnames = ["Predicted"]))      

print ("\nGradient Boost - Test accuracy",round(accuracy_score(y_test,gbc_fit.predict(X_test)),3))

#print ("\nGradient Boost - Test Classification Report\n",classification_report(y_test,gbc_fit.predict(X_test)))

# Xgboost Classifier

import xgboost as xgb



xgb_fit = xgb.XGBClassifier(max_depth=2, n_estimators=5000, learning_rate=0.05)

xgb_fit.fit(X_train, y_train)



#print ("\nXGBoost - Train Confusion Matrix\n\n",pd.crosstab(y_train,xgb_fit.predict(X_train),rownames = ["Actuall"],colnames = ["Predicted"]))      

print ("\nXGBoost - Train accuracy",round(accuracy_score(y_train,xgb_fit.predict(X_train)),3))

#print ("\nXGBoost  - Train Classification Report\n",classification_report(y_train,xgb_fit.predict(X_train)))



#print ("\n\nXGBoost - Test Confusion Matrix\n\n",pd.crosstab(y_test,xgb_fit.predict(X_test),rownames = ["Actuall"],colnames = ["Predicted"]))      

print ("\nXGBoost - Test accuracy",round(accuracy_score(y_test,xgb_fit.predict(X_test)),3))

#print ("\nXGBoost - Test Classification Report\n",classification_report(y_test,xgb_fit.predict(X_test)))
from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV



# Choose the type of classifier. 

clf = DecisionTreeClassifier()





# Choose some parameter combinations to try

parameters = {'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [i for i in range(1,20)], 

              'min_samples_split': [i for i in range(2,10)],

              'min_samples_leaf': [i for i in range(2,10)]

             }



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



# Run the grid search

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

print(clf.fit(X_train, y_train))
#Predict target value and find accuracy score

y_pred_train = clf.predict(X_train)

print("Accuracy score of train is ",accuracy_score(y_train, y_pred_train))

y_pred = clf.predict(X_test)

print("Accuracy score of test is ",accuracy_score(y_test, y_pred))