%matplotlib inline

import pandas as pd
#creditcard=r'J:\creditcard\creditcard.csv'
#creditcard=r'f:\bank\creditcard.csv'

creditcard='../input/creditcard.csv'
cdt=pd.read_csv(creditcard)

cdt.head()
cdt.describe()
172792/3600/24
cdt['Hours']=cdt.Time/3600/24
import seaborn as sns, numpy as np
ax = sns.distplot(cdt.Hours)

ax = sns.distplot(cdt[cdt.Class==0].Hours)

ax = sns.distplot(cdt[cdt.Class==1].Hours)

cdt['Hours']=cdt.Time/3600%24
ax = sns.distplot(cdt.Hours)

ax = sns.distplot(cdt[cdt.Class==0].Hours)

ax = sns.distplot(cdt[cdt.Class==1].Hours)

import matplotlib.pyplot as plt

# is there any difference between Amount？
g = sns.FacetGrid(cdt, col='Class')
#2.画每组的什么标签？（年龄连续变量）
g.map(plt.hist, 'Amount',bins=50)
# too bad!
#make them in one plt
ax1=sns.kdeplot(cdt.Amount[cdt.Class==1],color='b')
ax2=sns.kdeplot(cdt.Amount[cdt.Class==0],color='r')
# not any help
sns.kdeplot(cdt.Amount[cdt.Class==0],color='b')
sns.kdeplot(cdt.Amount[cdt.Class==1],color='r')
cdt[cdt.Class==0].describe()
cdt[cdt.Class==1].describe()
cdt.columns
cdt.info()
# there is no missing value!
f, ax = plt.subplots(figsize=(25,25))
g = sns.heatmap(cdt[[ 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
       'Class']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm",)

import numpy as np
a=[0,0,1,1]
b=[1,0,0,1]
c=[0,1,0,1]
lists=[a,b,c]
testexample=pd.DataFrame(np.array(pd.DataFrame(lists)).T)
testexample.columns=['x1','x2','y']
testexample
g = sns.heatmap(testexample[['x1','x2','y']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm",)
# try  a random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

X=cdt.drop(['Time','Class'],axis=1).values
y=cdt.Class.values


for train_index, test_index in sss.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = RandomForestClassifier(max_depth=10, random_state=0,n_jobs=-1)
    clf.fit(X_train, y_train)
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print(acc)

train_index
#check predict on the entire datasets.
from sklearn import metrics
import itertools

cnf_matrix = metrics.confusion_matrix(y, clf.predict(X))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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

#混淆绘制
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['0','1'],
                      title='Confusion matrix')

# which means there are 187 fatal error classify.
# try class weight

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

X=cdt.drop(['Time','Class'],axis=1).values
y=cdt.Class.values


for train_index, test_index in sss.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = RandomForestClassifier(max_depth=10, random_state=0,class_weight={0:0.1,1:1},n_jobs=-1)
    clf.fit(X_train, y_train)
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print(acc)
from sklearn import metrics
import itertools

cnf_matrix = metrics.confusion_matrix(y, clf.predict(X))

#混淆绘制
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['0','1'],
                      title='Confusion matrix')
#as u can see it get much better!
#then we try sth more weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

X=cdt.drop(['Time','Class'],axis=1).values
y=cdt.Class.values


for train_index, test_index in sss.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = RandomForestClassifier(max_depth=10, random_state=0,class_weight={0:0.001,1:1},n_jobs=-1)
    clf.fit(X_train, y_train)
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print(acc)

cnf_matrix = metrics.confusion_matrix(y, clf.predict(X))
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['0','1'],
                      title='Confusion matrix')
# thought the predict get better a new problem coming! we get more wrong classiction
# now let's try generate sth random for class 1
from sklearn.metrics import precision_recall_fscore_support
ypredict=clf.predict(X)
print(precision_recall_fscore_support(y, ypredict))

cnf_matrix = metrics.confusion_matrix(y, ypredict)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['0','1'],
                      title='Confusion matrix')
len(cdt[cdt.Class==0])/len(cdt[cdt.Class==1])
# lets try add some class==1 sample
#we make the copy!
cdtc1=cdt[cdt.Class==1]
templist=[cdtc1]*500
newcdt=pd.concat(templist+[cdt],axis=0)
newcdt.info()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

X=newcdt.drop(['Time','Class'],axis=1).values
y=newcdt.Class.values


for train_index, test_index in sss.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = RandomForestClassifier(max_depth=15, random_state=0,class_weight={0:1,1:1},n_jobs=-1)
    clf.fit(X_train, y_train)
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print(acc)
from sklearn.metrics import precision_recall_fscore_support
X=cdt.drop(['Time','Class'],axis=1).values
y=cdt.Class.values
ypredict=clf.predict(X)
print(precision_recall_fscore_support(y, ypredict))

cnf_matrix = metrics.confusion_matrix(y, ypredict)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['0','1'],
                      title='Confusion matrix')

# cdtc1
print(len(cdtc1))
cdtc1.columns.values

cdtc1.columns.values[1]
# plot all the data
for i in range(1,29):
    sns.kdeplot(cdt[cdt.Class==0][cdtc1.columns.values[i]],color='b')
    plt.show()
    sns.kdeplot(cdt[cdt.Class==1][cdtc1.columns.values[i]],color='r')
    plt.show()
addedexamplenumber=499
random_index=np.random.randint(492,size=492*addedexamplenumber)
cdtc1r=cdt[cdt.Class==1].iloc[random_index,:]
cdtc1r.info()
#addedexamplenumber=499
adjustcol=3
keylist=[0]*(28-adjustcol*2)+[1]*adjustcol+[-1]*adjustcol
keylist=np.array(keylist)
keylist[0]
npkey=np.array([keylist[i] for i in np.random.randint(len(keylist),size=(492*addedexamplenumber,28))])
print(npkey.shape)
npkey*0.05+1
adjst=np.multiply(npkey*0.05+1,cdtc1r.iloc[:,list(range(1,29))].values)
cdtc1r.iloc[:,list(range(1,29))]=adjst
cdtc1r.info()
newcdt2=pd.concat([cdt,cdtc1r],axis=0)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

X=newcdt2.drop(['Time','Class'],axis=1).values
y=newcdt2.Class.values


for train_index, test_index in sss.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = RandomForestClassifier(max_depth=10, random_state=0,class_weight={0:1,1:1},n_jobs=-1)
    clf.fit(X_train, y_train)
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print(acc)
X=cdt.drop(['Time','Class'],axis=1).values
y=cdt.Class.values
ypredict=clf.predict(X)
print(precision_recall_fscore_support(y, ypredict))

cnf_matrix = metrics.confusion_matrix(y, ypredict)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['0','1'],
                      title='Confusion matrix')
X=newcdt.drop(['Time','Class'],axis=1).values
y=newcdt.Class.values
Xo=cdt.drop(['Time','Class'],axis=1).values
yo=cdt.Class.values
t=[]
for train_index, test_index in sss.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    for depth in range(3,17):
        clf = RandomForestClassifier(max_depth=depth, random_state=0,class_weight={0:1,1:1},n_jobs=-1)
        clf.fit(X_train, y_train)
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        print(acc)
        ypredict=clf.predict(Xo)
        prfs=precision_recall_fscore_support(yo, ypredict)
        cnf_matrix = metrics.confusion_matrix(yo, ypredict)
        tr=[depth]+np.concatenate([np.array(prfs).reshape(1,-1),np.array(cnf_matrix).reshape(1,-1)],axis=1)[0].tolist()
        t.append(tr)
r=pd.DataFrame(t)
r.columns=['depth','precision0','precision1','recall0','recall1','fscore0','fscore1','support0','support1','TN','FP','FN','TP']
r
X=cdt.drop(['Time','Class'],axis=1).values
y=cdt.Class.values
Xo=cdt.drop(['Time','Class'],axis=1).values
yo=cdt.Class.values
t=[]
for train_index, test_index in sss.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    for depth in range(3,17):
        clf = RandomForestClassifier(max_depth=depth, random_state=0,class_weight={0:1,1:1},n_jobs=-1)
        clf.fit(X_train, y_train)
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        print(acc)
        
        
        
        
        ypredict=clf.predict(Xo)
        prfs=precision_recall_fscore_support(yo, ypredict)
        cnf_matrix = metrics.confusion_matrix(yo, ypredict)
        tr=[depth]+np.concatenate([np.array(prfs).reshape(1,-1),np.array(cnf_matrix).reshape(1,-1)],axis=1)[0].tolist()
        t.append(tr)
r=pd.DataFrame(t)
r
X=newcdt.drop(['Time','Class'],axis=1).values
y=newcdt.Class.values
Xo=cdt.drop(['Time','Class'],axis=1).values
yo=cdt.Class.values
t=[]
for train_index, test_index in sss.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    depth=13
    clf = RandomForestClassifier(max_depth=depth, random_state=0,class_weight={0:1,1:1},n_jobs=-1)
    clf.fit(X_train,y_train)
    yo_predict=clf.predict(Xo)
    
print(precision_recall_fscore_support(yo, yo_predict))

cnf_matrix = metrics.confusion_matrix(yo, yo_predict)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['0','1'],
                      title='Confusion matrix')


    

# the model runs too long!!!!!!!!!!!!!!!!
#from sklearn.svm import SVC
#clf2 = SVC(C=1.0,kernel='rbf')
#clf2.fit(X_train,y_train)
#train_predictions = clf.predict(X_test)
#acc = accuracy_score(y_test, train_predictions)
#print(acc)
#ypredict=clf.predict(Xo)
#prfs=precision_recall_fscore_support(yo, ypredict)
#cnf_matrix = metrics.confusion_matrix(yo, ypredict)
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=['0','1'],
#                      title='Confusion matrix')
