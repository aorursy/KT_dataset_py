import pandas as pd

pd_df=pd.read_csv('/kaggle/input/detection-of-parkinson-disease/parkinsons.csv')

pd_df
#get a feature and label

features=pd_df.loc[:,pd_df.columns!='status'].values[:,1:]

labels=pd_df.loc[:,'status'].values



print(labels[labels==1].shape[0], labels[labels==0].shape[0])
#Initialize a MinMaxScaler and scale the features to between -1 and 1 to normalize them.

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler((-1,1))

x=scaler.fit_transform(features)

y=labels

#Split the dataset

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.25, random_state=7)

x_train
from sklearn.linear_model import LogisticRegression

log=LogisticRegression()

log.fit(x_train,y_train)

predicted=log.predict(x_test)

predicted
from sklearn.metrics import average_precision_score,confusion_matrix,f1_score,recall_score,roc_auc_score,precision_score

confusion_matrix(y_test,predicted)
average_precision_score(y_test,predicted)
f1_score(y_test,predicted)
recall_score(y_test,predicted)
roc_auc_score(y_test,predicted)
precision_score(y_test,predicted)
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve



y_pred_proba=log.predict_proba(x_test)[::,1]

fpr,tpr,threshold=roc_curve(y_test,y_pred_proba)

auc=roc_auc_score(y_test,y_pred_proba)



plt.plot(fpr,tpr,label="auc="+str(auc))

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('AUC-ROC.Curve')



plt.legend (loc=8)

plt.show()

tpr
# decision tree classifier

from sklearn.tree import DecisionTreeClassifier

decision=DecisionTreeClassifier()

decision.fit(x_train,y_train)

predicted=decision.predict(x_test)

predicted
from sklearn.metrics import average_precision_score,confusion_matrix,f1_score,recall_score,roc_auc_score,precision_score

confusion_matrix(y_test,predicted)

confusion_matrix(y_test,predicted)
average_precision_score(y_test,predicted)
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve



y_pred_proba=decision.predict_proba(x_test)[::,1]

fpr,tpr,threshold=roc_curve(y_test,y_pred_proba)

auc=roc_auc_score(y_test,y_pred_proba)



plt.plot(fpr,tpr,label="auc="+str(auc))

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('AUC-ROC.Curve')



plt.legend (loc=8)

plt.show()
from sklearn.naive_bayes import GaussianNB

gb=GaussianNB()

gb.fit(x_train,y_train)

predicted=gb.predict(x_test)

predicted
from sklearn.metrics import accuracy_score

accuracy_score(y_test,predicted)
import sys

!{sys.executable}  -m pip install xgboost

from xgboost import XGBClassifier

model=XGBClassifier()

model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print(accuracy_score(y_test, y_pred)*100)
from sklearn.svm import SVC

svm = SVC(random_state = 42)

svm.fit(x_train,y_train)

print("Accuracy of SVM: ",svm.score(x_test,y_test))



from sklearn.metrics import accuracy_score

accuracy_score(y_test,predicted)