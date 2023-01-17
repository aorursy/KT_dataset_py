# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")
df.head()
df.describe()
df.isnull()
df.fillna(0,inplace=True)
df['DEATH_EVENT'].value_counts()
X=df.values
X[0:2]
xt=X[:,0:12]
xt[0:1]
y=X[:,12]
y[0:5]
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
%matplotlib inline
xt=preprocessing.StandardScaler().fit(xt).transform(xt)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(xt,y,test_size=0.1,random_state=4)
from sklearn.metrics import classification_report,confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greys):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
from sklearn.neighbors import KNeighborsClassifier
kmax=10
mean_acc=np.zeros((kmax-1))

for i in range(1,kmax):
    kn=KNeighborsClassifier(n_neighbors=i).fit(xtrain,ytrain)
    yhat=kn.predict(xtest)
    mean_acc[i-1]=metrics.accuracy_score(ytest,yhat)
    
plt.plot(range(1,kmax),mean_acc,'r')
plt.ylabel('Accuracy')
plt.xlabel('Number of neighbors')
plt.tight_layout()
plt.show()
    
    

print("The best accuracy of KNN was", mean_acc.max(),"with k=",mean_acc.argmax()+1)
kn=KNeighborsClassifier(n_neighbors=7).fit(xtrain,ytrain)
yhat=kn.predict(xtest)
cnf_matrix=confusion_matrix(ytest,yhat,labels=[1,0])
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix,classes=['DeathEvent=1','DeathEvent=0'],normalize='False',title='LOGISTIC REGRESSION LIBLINEAR',cmap=plt.cm.Blues)

print(classification_report(ytest,yhat))

from sklearn.linear_model import LogisticRegression

LR1=LogisticRegression(C=0.01,solver="liblinear").fit(xtrain,ytrain)
yhat1=LR1.predict(xtest)

cnf_matrix=confusion_matrix(ytest,yhat1,labels=[1,0])
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix,classes=['DeathEvent=1','DeathEvent=0'],normalize='False',title='LOGISTIC REGRESSION LIBLINEAR',cmap=plt.cm.Reds)

print(classification_report(ytest,yhat1))

LR2=LogisticRegression(C=0.01,solver="newton-cg").fit(xtrain,ytrain)
yhat2=LR2.predict(xtest)

cnf_matrix=confusion_matrix(ytest,yhat2,labels=[1,0])
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix,classes=['DeathEvent=1','DeathEvent=0'],normalize='False',title='LOGISTIC LINEAR NEWTON-CG')

print(classification_report(ytest,yhat2))
from sklearn import svm
svmM1=svm.SVC(C=0.7,kernel='rbf')
svmM1.fit(xtrain,ytrain)
yhatsvm1=svmM1.predict(xtest)

cnf_matrix=confusion_matrix(ytest,yhatsvm1,labels=[1,0])
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix,classes=['DeathEvent=1','DeathEvent=0'],normalize='False',title='SVM rbf',cmap=plt.cm.Greens)

print(classification_report(ytest,yhatsvm1))
svmM2=svm.SVC(kernel='linear')
svmM2.fit(xtrain,ytrain)
yhatsvm2=svmM2.predict(xtest)

cnf_matrix=confusion_matrix(ytest,yhatsvm2,labels=[1,0])
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix,classes=['DeathEvent=1','DeathEvent=0'],normalize='False',title='SVM linear',cmap=plt.cm.Oranges)

print(classification_report(ytest,yhatsvm2))
svmM3=svm.SVC(kernel='poly')
svmM3.fit(xtrain,ytrain)
yhatsvm3=svmM3.predict(xtest)

cnf_matrix=confusion_matrix(ytest,yhatsvm3,labels=[1,0])
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix,classes=['DeathEvent=1','DeathEvent=0'],normalize='False',title='SVM poly',cmap=plt.cm.Blues)

print(classification_report(ytest,yhatsvm3))
from sklearn import metrics
print("KNN Accuracy                            : ",metrics.accuracy_score(ytest,yhat))
print("Logistic Regression(liblinear) Accuracy : ",metrics.accuracy_score(ytest,yhat1))
print("Logistic Regression(newton-cg) Accuracy : ",metrics.accuracy_score(ytest,yhat2))
print("SVM(rbf Kernel) Accuracy                : ",metrics.accuracy_score(ytest,yhatsvm1))
print("SVM(linear Kernel) Accuracy             : ",metrics.accuracy_score(ytest,yhatsvm2))
print("SVM(polynomial Kernel)                  : ",metrics.accuracy_score(ytest,yhatsvm3))