import pandas as pd

df1=pd.read_csv('../input/dataset_train_woed.csv')
df1.info()
#df2=pd.read_csv('..dataset_test_woed.csv')
#df2.info()
df1.head()
#df2.head()
df3=pd.read_csv('../input/UCI_Credit_Card.csv')
df3.info()
df1.describe()
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

X=df1.drop(['Unnamed: 0','ID','target'],axis=1).values
y=df1.target.values


for train_index, test_index in sss.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = LogisticRegressionCV()
    #clf = RandomForestClassifier(max_depth=10, random_state=0,class_weight={0:1,1:1},n_jobs=-1)
    clf.fit(X_train, y_train)
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print(acc)
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import itertools

from sklearn.metrics import precision_recall_fscore_support
ypredict=clf.predict(X)
print(precision_recall_fscore_support(y, ypredict))

cnf_matrix = metrics.confusion_matrix(y, ypredict)

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
df1c1=df1[df1.target==1]
templist=[df1c1]*4
newdf1=pd.concat(templist+[df1],axis=0)
newdf1.info()
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

X=newdf1.drop(['Unnamed: 0','ID','target'],axis=1).values
y=newdf1.target.values


for train_index, test_index in sss.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = LogisticRegressionCV(n_jobs=-1)
    #clf = RandomForestClassifier(max_depth=10, random_state=0,class_weight={0:1,1:1},n_jobs=-1)
    clf.fit(X_train, y_train)
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print(acc)
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import itertools
X=df1.drop(['Unnamed: 0','ID','target'],axis=1).values
y=df1.target.values


from sklearn.metrics import precision_recall_fscore_support
ypredict=clf.predict(X)
print(precision_recall_fscore_support(y, ypredict))

cnf_matrix = metrics.confusion_matrix(y, ypredict)

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
from sklearn.metrics import roc_curve, auc  

probas_=clf.predict_proba(X)
fpr, tpr, thresholds = roc_curve(y, probas_[:, 1]) 
auc(fpr, tpr)  
import matplotlib.pyplot as plt  
plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (0,auc(fpr, tpr) ))  
