import numpy as np 

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import seaborn as sns

import plotly.express as px

from mpl_toolkits import mplot3d 

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn import metrics

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import classification_report,plot_confusion_matrix



df=pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
df.head()
df.describe()
df.isna().any()
df.dtypes
hm = df.corr()

sns.heatmap(hm, xticklabels=hm.columns, yticklabels=hm.columns , cmap='Blues')
fig = px.box(df, x='sex', y='age', points="all",color_discrete_sequence =['green']*len(df))

fig.update_layout(title_text="Male = 1 Female =0")

fig.show()
fig = px.box(df, x='target', y='age', color_discrete_sequence =['red']*len(df))

fig.show()
fig = plt.figure(figsize =(14, 9)) 

ax = plt.axes(projection ='3d') 



ax.scatter(df['age'], df['sex'], df['chol'], c='orange', marker='o') 

ax.set_xlabel('Age')

ax.set_ylabel('Sex')

ax.set_zlabel('Cholestrol')  

plt.show() 
X=df.values

x=X[:,0:13]

y=X[:,13]

x=preprocessing.StandardScaler().fit(x).transform(x)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=4)
from sklearn.neighbors import KNeighborsClassifier

kmax=22

mean_acc=np.zeros((kmax-1))



for i in range(1,kmax):

    kn=KNeighborsClassifier(n_neighbors=i).fit(xtrain,ytrain)

    yhat=kn.predict(xtest)

    n1=1000

    mean_acc[i-1]=metrics.accuracy_score(ytest,yhat)

    

plt.plot(range(1,kmax),mean_acc,'r')

plt.ylabel('Accuracy')

plt.xlabel('Number of neighbors')

plt.tight_layout()

plt.show()
print("The best accuracy of KNN was", mean_acc.max(),"with k=",mean_acc.argmax()+1)

kn=KNeighborsClassifier(n_neighbors=19).fit(xtrain,ytrain)

yhat=kn.predict(xtest)



plot_confusion_matrix(kn,xtest,ytest,cmap=plt.cm.Blues)

plt.show()

print(classification_report(ytest,yhat))

a1=metrics.accuracy_score(ytest,yhat)



from sklearn.linear_model import LogisticRegression

LR1=LogisticRegression(C=0.01,solver="liblinear").fit(xtrain,ytrain)

yhat1=LR1.predict(xtest)



plot_confusion_matrix(LR1,xtest,ytest,cmap=plt.cm.Reds)

plt.show()



print(classification_report(ytest,yhat1))

a2=metrics.accuracy_score(ytest,yhat1)

LR2=LogisticRegression(C=0.01,solver="newton-cg").fit(xtrain,ytrain)

yhat2=LR2.predict(xtest)



plot_confusion_matrix(LR2,xtest,ytest,cmap=plt.cm.Greys)

plt.show()



print(classification_report(ytest,yhat2))

a3=metrics.accuracy_score(ytest,yhat2)

LR3=LogisticRegression(C=0.01,solver="sag").fit(xtrain,ytrain)

yhat3=LR3.predict(xtest)



plot_confusion_matrix(LR3,xtest,ytest,cmap=plt.cm.YlOrBr)

plt.show()



print(classification_report(ytest,yhat3))

a4=metrics.accuracy_score(ytest,yhat3)

from sklearn import svm

svmM1=svm.SVC(C=0.2,kernel='rbf')

svmM1.fit(xtrain,ytrain)

yhatsvm1=svmM1.predict(xtest)



plot_confusion_matrix(svmM1,xtest,ytest,cmap=plt.cm.Greens)

plt.show()



print(classification_report(ytest,yhatsvm1))

a5=metrics.accuracy_score(ytest,yhatsvm1)

svmM2=svm.SVC(C=0.1,kernel='linear')

svmM2.fit(xtrain,ytrain)

yhatsvm2=svmM2.predict(xtest)



plot_confusion_matrix(svmM2,xtest,ytest,cmap=plt.cm.Blues)

plt.show()



print(classification_report(ytest,yhatsvm2))

a6=metrics.accuracy_score(ytest,yhatsvm2)

svmM3=svm.SVC(C=1.5,kernel='poly')

svmM3.fit(xtrain,ytrain)

yhatsvm3=svmM3.predict(xtest)



plot_confusion_matrix(svmM3,xtest,ytest,cmap=plt.cm.Oranges)

plt.show()



print(classification_report(ytest,yhatsvm3))

a7=metrics.accuracy_score(ytest,yhatsvm3)

from sklearn.tree import DecisionTreeClassifier



dt=DecisionTreeClassifier(criterion="entropy")

dt.fit(xtrain,ytrain)

yhatdt=dt.predict(xtest)



plot_confusion_matrix(dt,xtest,ytest,cmap=plt.cm.Reds)

plt.show()



print(classification_report(ytest,yhatdt))

a8=metrics.accuracy_score(ytest,yhatdt)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=n1)

rf = rf.fit(xtrain, ytrain)

yhatrf=rf.predict(xtest)



plot_confusion_matrix(rf,xtest,ytest,cmap=plt.cm.Greys)

plt.show()



print(classification_report(ytest,yhatrf))

a9=metrics.accuracy_score(ytest,yhatrf)

from sklearn.ensemble import AdaBoostClassifier

ab = AdaBoostClassifier(n_estimators=17)

ab = ab.fit(xtrain, ytrain)

yhatab=ab.predict(xtest)



plot_confusion_matrix(ab,xtest,ytest,cmap=plt.cm.Greens)

plt.show()



print(classification_report(ytest,yhatab))

a10=metrics.accuracy_score(ytest,yhatab)

print("KNN Accuracy                            : %.5f"%metrics.accuracy_score(ytest,yhat))

print("Logistic Regression(liblinear) Accuracy : %.5f"%metrics.accuracy_score(ytest,yhat1))

print("Logistic Regression(newton-cg) Accuracy : %.5f"%metrics.accuracy_score(ytest,yhat2))

print("Logistic Regression(sag) Accuracy       : %.5f"%metrics.accuracy_score(ytest,yhat2))

print("SVM(rbf Kernel) Accuracy                : %.5f"%metrics.accuracy_score(ytest,yhatsvm1))

print("SVM(linear Kernel) Accuracy             : %.5f"%metrics.accuracy_score(ytest,yhatsvm2))

print("SVM(polynomial Kernel)                  : %.5f"%metrics.accuracy_score(ytest,yhatsvm3))

print("Decision Tree Accuracy                  : %.5f"%metrics.accuracy_score(ytest,yhatdt))

print("Random Forest Accuracy                  : %.5f"%metrics.accuracy_score(ytest,yhatrf))

print("Ada Boost Accuracy                      : %.5f"%metrics.accuracy_score(ytest,yhatab))







fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

classifiers = ['KNN', 'LR1', 'LR2', 'LR3', 'SVM1','SVM2','SVM3','DT','RF','AB']

accuracies = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]

ax.bar(classifiers,accuracies,align='center', width=0.4)

plt.ylim([0.7, 0.95])

plt.show()