import pandas as pd

import numpy as np

from matplotlib import pyplot

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dataset = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

dataset.shape
dataset.head()
from matplotlib import pyplot as plt

import seaborn as sns

corr = dataset.corr()

fig, ax = plt.subplots(figsize=(30, 18))

colormap = sns.diverging_palette(220, 10, as_cmap=True)

dropSelf = np.zeros_like(corr)

dropSelf[np.triu_indices_from(dropSelf)] = True

colormap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=colormap, linewidths=.5, annot=True, fmt=".2f", mask=dropSelf)

plt.title('Fraud - Features Correlations')

plt.show()
sns.set(style="whitegrid")

num = [f for f in dataset.columns if ((dataset.dtypes[f] != 'object')& (dataset.dtypes[f]!='int64'))]

nd = pd.melt(dataset, value_vars = num)

n1 = sns.FacetGrid (nd, col='variable', col_wrap=4, sharex=False, sharey = False)

n1 = n1.map(sns.distplot, 'value')

n1
num = [f for f in dataset.columns if ((dataset.dtypes[f] != 'object')& (dataset.dtypes[f]!='int64'))]

nd = pd.melt(dataset, value_vars = num)

n1 = sns.FacetGrid (nd, col='variable', col_wrap=4, sharex=False, sharey = False)

n1 = n1.map(sns.boxplot, 'value')

n1
target = dataset['Class']

train=dataset.drop('Class',axis=1)
#applying SMOTE

from imblearn.combine  import SMOTETomek

smk=SMOTETomek(random_state=42)

train_new,target_new=smk.fit_sample(train,target)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_new, target_new, test_size = 0.30, random_state = 0)
from xgboost import XGBClassifier

classifier = XGBClassifier()
from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)
from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



accuracy=accuracy_score(y_test,y_pred) 

precision=precision_score(y_test,y_pred,average='weighted')

recall=recall_score(y_test,y_pred,average='weighted')

f1=f1_score(y_test,y_pred,average='weighted')



print('Accuracy - {}'.format(accuracy))

print('Precision - {}'.format(precision))

print('Recall - {}'.format(recall))

print('F1 - {}'.format(f1))
from sklearn.metrics import average_precision_score

average_precision = average_precision_score(y_test, y_pred)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import plot_precision_recall_curve

import matplotlib.pyplot as plt



disp = plot_precision_recall_curve(classifier, X_test, y_test)

disp.ax_.set_title('2-class Precision-Recall curve: '

                   'AP={0:0.2f}'.format(average_precision))
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)

print(cm)
import itertools



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        #print("Normalized confusion matrix")

    else:

        1#print('Confusion matrix, without normalization')



    #print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
cnf_matrix = confusion_matrix(y_test,y_pred)

np.set_printoptions(precision=2)

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()
from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier(criterion='gini', splitter='best',

                             max_depth=16, min_samples_split=2,

                             min_samples_leaf=1, min_weight_fraction_leaf=0.0,

                             max_features=None, random_state=None,

                             max_leaf_nodes=None, min_impurity_decrease=0.0, 

                             min_impurity_split=None, class_weight=None, 

                             presort='deprecated', ccp_alpha=0.0)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)
from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



accuracy=accuracy_score(y_test,y_pred) 

precision=precision_score(y_test,y_pred,average='weighted')

recall=recall_score(y_test,y_pred,average='weighted')

f1=f1_score(y_test,y_pred,average='weighted')



print('Accuracy - {}'.format(accuracy))

print('Precision - {}'.format(precision))

print('Recall - {}'.format(recall))

print('F1 - {}'.format(f1))
from sklearn.metrics import average_precision_score

average_precision = average_precision_score(y_test, y_pred)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import plot_precision_recall_curve

import matplotlib.pyplot as plt



disp = plot_precision_recall_curve(classifier, X_test, y_test)

disp.ax_.set_title('2-class Precision-Recall curve: '

                   'AP={0:0.2f}'.format(average_precision))
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_pred,y_test)

print(cm)
import itertools



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        #print("Normalized confusion matrix")

    else:

        1#print('Confusion matrix, without normalization')



    #print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
cnf_matrix = confusion_matrix(y_test,y_pred)

np.set_printoptions(precision=2)

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()
from sklearn.naive_bayes import GaussianNB

nb_model=GaussianNB()

nb_model.fit(X_train,y_train)

y_pred=nb_model.predict(X_test)
from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



accuracy=accuracy_score(y_test,y_pred) 

precision=precision_score(y_test,y_pred,average='weighted')

recall=recall_score(y_test,y_pred,average='weighted')

f1=f1_score(y_test,y_pred,average='weighted')



print('Accuracy - {}'.format(accuracy))

print('Precision - {}'.format(precision))

print('Recall - {}'.format(recall))

print('F1 - {}'.format(f1))
from sklearn.metrics import average_precision_score

average_precision = average_precision_score(y_test, y_pred)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import plot_precision_recall_curve

import matplotlib.pyplot as plt



disp = plot_precision_recall_curve(classifier, X_test, y_test)

disp.ax_.set_title('2-class Precision-Recall curve: '

                   'AP={0:0.2f}'.format(average_precision))
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)

print(cm)
import itertools



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        #print("Normalized confusion matrix")

    else:

        1#print('Confusion matrix, without normalization')



    #print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
cnf_matrix = confusion_matrix(y_test,y_pred)

np.set_printoptions(precision=2)

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()