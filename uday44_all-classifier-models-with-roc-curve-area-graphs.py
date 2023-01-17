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
import seaborn as sb
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import sklearn.metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from scipy import interp
from itertools import cycle
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import DiscriminationThreshold
from yellowbrick.classifier import PrecisionRecallCurve
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
data = pd.read_csv('/kaggle/input/fetal-health-classification/fetal_health.csv')
data.head()
data.info()
#rename columns
data.columns = ['FHR', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV',
               'ALTV', 'MLTV', 'Width', 'Min', 'Max', 'NMax', 'Nzeros', 
                'Mode', 'Mean', 'Median', 'Variance', 'Tendency', 'Class']
data.columns
#convert target class to Int

data.Class = data.Class.astype('int')
label_encoder = preprocessing.LabelEncoder()

y_en = label_encoder.fit_transform(data.Class)
data = data.drop(['Class'], axis = 1)
data.head()
#0-Normal
#1-Suspect
#2-Pathologic

y_en
#AC - # of accelerations per second
#FM - # of fetal movements per second
#UC - # of uterine contractions per second
#DL - # of light decelerations per second
#DS - # of severe decelerations per second
#DP - # of prolongued decelerations per second

#All the variables are measured per second and has very low values, this cause scaling issues in our models. 
#So we convert them into per min, as FHR is also  measured per min.

clms = ['AC', 'FM', 'UC', 'DL', 'DS', 'DP']

for column in clms:
    data[column] = data[column]*60
    
data.head(10)
plt.figure(figsize=[15,10])
x=data.corr()
sb.heatmap(x,annot=True)
plt.figure(figsize = [10, 8])
sb.distplot(data['FHR'])
plt.figure(figsize=[15,8])
sb.violinplot(x=y_en, y=data.FHR, palette="deep")
sb.pairplot(data[['AC', 'UC', 'DL', 'FM', 'DS', 'DP']])
def zero_table(df):
    for column in df.columns:
        zero_count = (df[column] == 0).sum()
        if zero_count != 0:
            zero_percentage = 100*zero_count/len(df[column])
            if zero_percentage > 60:
                print("%s has %s Zeros" % (column, zero_count))
                print("Percentage of Zeros %0.1f%%" % (zero_percentage))
                print("-"*25)
zero_table(data)
sb.pairplot(data[['AC', 'UC', 'DL']])
plt.figure(figsize=[15,8])
sb.scatterplot(x = data['ASTV'], y = data['MSTV'],
              hue = y_en, palette="deep")
plt.figure(figsize=[15,8])
sb.scatterplot(x = data['ALTV'], y = data['MLTV'],
              hue = y_en, palette="deep")
plt.figure(figsize = [10, 8])
sb.distplot(data['Variance'])
plt.figure(figsize=[15,8])
sb.violinplot(x=y_en, y=data.Variance, palette="deep")
plt.figure(figsize = [10, 8])
sb.distplot(data['Width'])
plt.figure(figsize=[15,8])
sb.violinplot(x=y_en, y=data.Width, palette="deep")
plt.figure(figsize = [10, 8])
sb.distplot(data['Max'])
plt.figure(figsize=[15,8])
sb.violinplot(x=y_en, y=data.Max, palette="deep")
plt.figure(figsize=[15,8])
sb.violinplot(x=y_en, y=data.NMax, palette="deep")
plt.figure(figsize=[15,8])
sb.scatterplot(x=data.FHR, y=data.Variance, hue=y_en, palette="deep")
def classifier_results(x, y):
    
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.3)

    classifiers = {
        'L1 logistic': LogisticRegression(penalty = 'l2', solver = 'saga', 
                            multi_class = 'multinomial', max_iter = 10000),
        'L2 logistic (Multinomial)': LogisticRegression(penalty = 'l1', solver = 'saga', 
                             multi_class = 'multinomial', max_iter = 10000),
        'L2 logistic (OvR)': LogisticRegression(penalty='l2', solver='saga',
                       multi_class='ovr', max_iter=10000),
        'Linear SVC': SVC(kernel='linear', probability=True),

    }
    
    class_names = ['Normal', 'Suspect', 'Pathologic']

    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (test) for %s: %0.1f%% " % (name, accuracy * 100))
        print('-'*40)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        visualizer = ClassificationReport(classifier, classes=class_names, support=True, ax=ax)
        visualizer.fit(x_train, y_train)       
        visualizer.score(x_test, y_test)       
        visualizer.show()
        
        fig, ax = plt.subplots(figsize=(12, 7))
        cm = ConfusionMatrix(classifier, classes = class_names, ax=ax)
        cm.fit(x_train, y_train)
        cm.score(x_test, y_test)
        cm.show()
        
        y_lb = label_binarize(y_en, classes=[0, 1, 2])
        n_classes = y_lb.shape[1]
        
        x2_train,x2_test,y2_train,y2_test = train_test_split(data, y_lb, test_size=0.3)

        estimator = OneVsRestClassifier(classifier)
        y2_dist = estimator.fit(x2_train, y2_train).decision_function(x2_test)
        y2_pred = estimator.predict(x2_test)
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # Compute ROC curve and ROC area for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y2_test[:, i], y2_dist[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y2_test.ravel(), y2_dist.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure(figsize=[15,7])
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i+1, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()
classifier_results(data, y_en)
def RF_AdB_GNB_classifier_results(x, y):
    
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.3)

    classifiers = {
        'RandomForest': RandomForestClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'GaussianNB': GaussianNB()

    }
    
    class_names = ['Normal', 'Suspect', 'Pathologic']

    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (test) for %s: %0.1f%% " % (name, accuracy * 100))
        print('-'*40)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        visualizer = ClassificationReport(classifier, classes=class_names, support=True, ax=ax)
        visualizer.fit(x_train, y_train)        # Fit the visualizer and the model
        visualizer.score(x_test, y_test)        # Evaluate the model on the test data
        visualizer.show()
        
        fig, ax = plt.subplots(figsize=(12, 7))
        cm = ConfusionMatrix(classifier, classes = class_names, ax=ax)
        cm.fit(x_train, y_train)
        cm.score(x_test, y_test)
        cm.show()

        fig, ax = plt.subplots(figsize=(12, 7))
        roc = ROCAUC(classifier, classes=class_names, ax=ax)
        roc.fit(x_train, y_train)        # Fit the training data to the visualizer
        roc.score(x_test, y_test)        # Evaluate the model on the test data
        roc.show()
        
        fig, ax = plt.subplots(figsize=(12, 7))
        prc = PrecisionRecallCurve(classifier,
                                   classes=class_names,
                                   colors=["purple", "cyan", "blue"],
                                   iso_f1_curves=True,
                                   per_class=True,
                                   micro=False, ax=ax)
        prc.fit(x_train, y_train)
        prc.score(x_test, y_test)
        prc.show()
RF_AdB_GNB_classifier_results(data, y_en)
