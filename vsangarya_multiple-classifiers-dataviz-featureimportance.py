import numpy as np
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
import seaborn as sns
import plotly.express as px
from mpl_toolkits import mplot3d 



df=pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
df.head()

df["good quality"] = 0
df.loc[df["quality"]>=7,"good quality"] = 1
df=df.drop(['quality'],axis=1)
df.head()
df.describe()
df.isna().any()
X=df.values
x=X[:,0:11]
y=X[:,11]
print(x.shape, y.shape)
fig = px.box(df, x='good quality', y='alcohol', points="all",color_discrete_sequence =['red']*len(df))
fig.update_layout(title_text="Quality = 1 Quality =0")
fig.show()
fig = px.box(df, x='good quality', y='sulphates', points="all",color_discrete_sequence =['blue']*len(df))
fig.update_layout(title_text="Quality = 1 Quality =0")
fig.show()
fig = px.box(df, x='good quality', y='volatile acidity', points="all",color_discrete_sequence =['green']*len(df))
fig.update_layout(title_text="Quality = 1 Quality =0")
fig.show()

x=preprocessing.StandardScaler().fit(x).transform(x)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=4)
print(xtrain.shape,ytrain.shape)
print(xtest.shape, ytest.shape)
from sklearn.neighbors import KNeighborsClassifier

kmax=7
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
kn=KNeighborsClassifier(n_neighbors=1).fit(xtrain,ytrain)
yhat=kn.predict(xtest)
plt.figure()
plot_confusion_matrix(kn,xtest, ytest,normalize='true',cmap=plt.cm.Blues)
a1=metrics.accuracy_score(ytest,yhat)
print(classification_report(ytest,yhat))
from sklearn.linear_model import LogisticRegression
LR1=LogisticRegression(C=0.05,solver="liblinear").fit(xtrain,ytrain)
yhat1=LR1.predict(xtest)
a2=metrics.accuracy_score(ytest,yhat1)
plt.figure()
plot_confusion_matrix(LR1,xtest, ytest,normalize='true',cmap=plt.cm.Reds)
print(classification_report(ytest,yhat1))
print("Logistic Regression(liblinear) Accuracy : ",metrics.accuracy_score(ytest,yhat1))

fi = abs(LR1.coef_[0])
for i,v in enumerate(fi):
    print('Feature: %0d, Score: %.5f' % (i,v))
plt.bar([x for x in range(len(fi))], fi)
plt.show()
from sklearn import svm
svmM3=svm.SVC(kernel='rbf')
svmM3.fit(xtrain,ytrain)
yhatsvm3=svmM3.predict(xtest)
a3=metrics.accuracy_score(ytest,yhatsvm3)
plot_confusion_matrix(svmM3,xtest, ytest,normalize='true',cmap=plt.cm.Reds)

print(classification_report(ytest,yhatsvm3))
print("SVM : ",metrics.accuracy_score(ytest,yhatsvm3))

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(xtrain, ytrain)
y_pred=rf.predict(xtest)
print(classification_report(ytest,y_pred))
plot_confusion_matrix(svmM3,xtest, ytest,normalize='true',cmap=plt.cm.Reds)
a4=metrics.accuracy_score(ytest,y_pred)
print("Random Forest Classifier accuracy: ",metrics.accuracy_score(ytest,yhatsvm3))

fi3 = rf.feature_importances_
for i,v in enumerate(fi3):
    print('Feature: %0d, Score: %.5f' % (i,v))
plt.bar([x for x in range(len(fi3))], fi3)
plt.show()
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
classifiers = ['KNN', 'Log.Regression', 'SVM','Random Forest']
accuracies = [a1,a2,a3,a4]
ax.bar(classifiers,accuracies,align='center', width=0.4,color='orange')
plt.ylim([0.85, 0.93])
plt.show()