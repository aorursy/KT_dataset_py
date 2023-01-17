import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import seaborn as sns

from matplotlib import pyplot as plt
dataset = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
dataset.isnull().sum()
sns.set(style="whitegrid")
sns.distplot(dataset.salary)
dataset['salary']=dataset['salary'].fillna(0)
sns.barplot(x = dataset['gender'],y = dataset['salary'])
sns.barplot(x = dataset['gender'],y = dataset['ssc_p'])
sns.barplot(x = dataset['gender'],y = dataset['hsc_p'])
sns.barplot(x = dataset['gender'],y = dataset['degree_p'])
sns.barplot(x = dataset['gender'],y = dataset['mba_p'])
sns.boxplot(x = dataset['status'], y = dataset['mba_p'])
sns.boxplot(x = dataset['status'], y = dataset['degree_p'])
sns.boxplot(x = dataset['status'], y = dataset['hsc_p'])
sns.boxplot(x = dataset['status'], y = dataset['ssc_p'])
#dataset=dataset[dataset.salary<600000]
sns.jointplot(x = dataset['ssc_p'], y = dataset['salary'], kind='hex')
sns.jointplot(x = dataset['hsc_p'], y = dataset['salary'], kind='hex')
sns.jointplot(x = dataset['degree_p'], y = dataset['salary'], kind='hex')
sns.jointplot(x = dataset['mba_p'], y = dataset['salary'], kind='hex')
sns.jointplot(x = dataset['etest_p'], y = dataset['salary'], kind='hex')
numeric_data = dataset.select_dtypes(include=[np.number])

cat_data = dataset.select_dtypes(exclude=[np.number])

print ("There are {} numeric and {} categorical columns in train data".format(numeric_data.shape[1],cat_data.shape[1]))
dataset.dtypes
# Label Encoding

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

def encoder(df,col_name):

    df[col_name] = le.fit_transform(dataset[col_name])

    

encoder(dataset,'gender')

encoder(dataset,'ssc_b')

encoder(dataset,'hsc_b')

encoder(dataset,'hsc_s')

encoder(dataset,'degree_t')

encoder(dataset,'workex')

encoder(dataset,'specialisation')

encoder(dataset,'status')
# splitting dataset

# removing salary feature as it is a dependent feature

target = dataset['status']

drop = ['sl_no','status','salary']

train = dataset.drop(drop,axis=1)
# Using SMOTE to balance the categories



from imblearn.combine import SMOTETomek

smk = SMOTETomek(random_state = 42)

train, target = smk.fit_sample(train,target)
#Now we will split the dataset in the ratio of 75:25 for train and test



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size = 0.25, random_state = 0)
from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier(criterion='gini', splitter='best',

                             max_depth=5, min_samples_split=2,

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



disp = plot_precision_recall_curve(model, X_test, y_test)

disp.ax_.set_title('2-class Precision-Recall curve: '

                   'AP={0:0.2f}'.format(average_precision))
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

        



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
from sklearn.metrics import confusion_matrix

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



disp = plot_precision_recall_curve(model, X_test, y_test)

disp.ax_.set_title('2-class Precision-Recall curve: '

                   'AP={0:0.2f}'.format(average_precision))
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

        



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
from sklearn.metrics import confusion_matrix

cnf_matrix = confusion_matrix(y_test,y_pred)

np.set_printoptions(precision=2)

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()