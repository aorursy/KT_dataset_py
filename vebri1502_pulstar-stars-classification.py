import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
df = pd.read_csv(r"../input/predicting-a-pulsar-star/pulsar_stars.csv")
df.sample(10)
df['target_class'].value_counts()
df.columns
df.hist(column = ' Standard deviation of the integrated profile', bins = 50)
x = df[[' Mean of the integrated profile',

       ' Standard deviation of the integrated profile',

       ' Excess kurtosis of the integrated profile',

       ' Skewness of the integrated profile', ' Mean of the DM-SNR curve',

       ' Standard deviation of the DM-SNR curve',

       ' Excess kurtosis of the DM-SNR curve', ' Skewness of the DM-SNR curve']].values



x
y = df['target_class'].values

y
x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))

x
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 4)

print('Train set: ', x_train.shape, y_train.shape)

print('Test set: ', x_test.shape, y_test.shape)
Ks = 30

mean_acc = np.zeros((Ks-1))

std_acc = np.zeros((Ks-1))

ConfusionMx = []

for n in range(1, Ks):

    neigh = KNeighborsClassifier(n_neighbors= n).fit(x_train, y_train)

    yhat = neigh.predict(x_test)

    mean_acc[n-1] = metrics.accuracy_score(y_test,yhat)

    std_acc[n-1] = np.std(yhat == y_test)/np.sqrt(yhat.shape[0])

    

mean_acc
plt.plot(range(1,Ks),mean_acc,'g')

plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)

plt.legend(('Accuracy ', '+/- 3xstd'))

plt.ylabel('Accuracy ')

plt.xlabel('Number of Nabors (K)')

plt.tight_layout()

plt.show()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
!apt-get -qq install -y graphviz && pip install -q pydotplus

from sklearn.tree import DecisionTreeClassifier

from sklearn.externals.six import StringIO

from sklearn import tree

import pydotplus

import matplotlib.image as mpimg

%matplotlib inline
pulsar_stars = DecisionTreeClassifier(criterion = 'gini', max_depth = 10)

pulsar_stars.fit(x_train, y_train)
predTree = pulsar_stars.predict(x_test)

print("DecisionTree's Accuracy: ", metrics.accuracy_score(y_test, predTree))
dot_data = StringIO()

filename = "pulsar_stars.png"

featuresNames = df.columns[0:8]

targetNames = ['0', '1']

out = tree.export_graphviz(pulsar_stars, 

                           feature_names = featuresNames, 

                           out_file= dot_data, 

                           class_names = targetNames, 

                           filled = True, 

                           special_characters = True,

                           rotate = False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_png(filename)

img = mpimg.imread(filename)

plt.figure(figsize=(60, 60))

plt.imshow(img, interpolation = 'nearest')
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import jaccard_similarity_score

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import log_loss

import itertools
LR = LogisticRegression(C=0.05, solver='liblinear').fit(x_train, y_train)

LR
yhat = LR.predict(x_test)

yhat
yhat_prob = LR.predict_proba(x_test)

yhat_prob
jaccard_similarity_score(y_test, yhat)
def plot_confusion_matrix(cm,

                         classes,

                         normalize = False, #normalize can be True

                         title='Confusion Matrix',

                         cmap = plt.cm.Blues):

    if normalize:

        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]

        print ("Normalize confusion matrix")

    else:

        print('Confusion matrix, without normalization')

    

    print (cm)

    

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation = 45)

    plt.yticks(tick_marks, classes)

    

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max()/2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                horizontalalignment = 'center',

                color = 'white' if cm[i, j] > thresh else "black")

        

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

print(confusion_matrix(y_test, yhat, labels = [1, 0]))
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])

np.set_printoptions(precision = 2)



plt.figure()

plot_confusion_matrix(cnf_matrix, classes=['target_class = 1', 'target_class = 0'], normalize = False, title = 'Confusion matrix')
print(classification_report(y_test, yhat))
log_loss(y_test, yhat_prob)
from sklearn import svm

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import jaccard_similarity_score

import itertools
clf = svm.SVC(kernel = 'rbf', gamma = 'auto')

clf.fit(x_train, y_train)
yhat = clf.predict(x_test)

yhat[0:5]

yhat
cnf_matrix = confusion_matrix(y_test, yhat)

np.set_printoptions(precision = 2)



print(classification_report(y_test, yhat))



plt.figure()

plot_confusion_matrix(cnf_matrix, classes=['Benign(2)', 'Malignant(4)'], normalize = False, title = 'Confusion Matrix')

f1_score(y_test, yhat, average = 'weighted')
jaccard_similarity_score(y_test, yhat)