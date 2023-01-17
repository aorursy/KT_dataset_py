import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.datasets import make_classification, make_blobs

from matplotlib.colors import ListedColormap

%matplotlib notebook
df=pd.read_csv('../input/ionosphere/ionosphere.txt',header=None)
df.head()
X=df.iloc[:,0:33]

Y=df.iloc[:,34]
X.head()
from sklearn.svm import LinearSVC

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0)



clf = LinearSVC().fit(X_train, y_train)

print('Ionosphere dataset')

print('Accuracy of Linear SVC classifier on training set: {:.2f}'

     .format(clf.score(X_train, y_train)))

print('Accuracy of Linear SVC classifier on test set: {:.2f}'

     .format(clf.score(X_test, y_test)))
clf1 = LinearSVC(C=5, random_state = 67).fit(X_train, y_train)

print('Coefficients:\n', clf1.coef_)

print('Intercepts:\n', clf1.intercept_)
from sklearn.svm import SVC



svm = SVC(kernel='rbf', C=10).fit(X_train, y_train)

svm.score(X_test, y_test)
svm_poly=SVC(kernel='poly', C=100,probability=True).fit(X_train, y_train)

svm_poly.score(X_test,y_test)

svm_sigmoid=SVC(kernel='sigmoid', C=100,probability=True).fit(X_train, y_train)

svm_sigmoid.score(X_test,y_test)
svm_decf = SVC(kernel='rbf', C=1,decision_function_shape='ovr').fit(X_train, y_train)

svm_decf.score(X_test, y_test)

# svm_decf.decision_function(X.iloc[:,6:8])
X_train
clf = SVC(kernel='rbf', C=1)

clf.fit(X_train[[6,14]], y_train)

plt.figure()

plt.scatter(X.iloc[:,6:7],X.iloc[:,14:15],zorder=10,

            c=['r','b'],cmap=plt.cm.Paired,edgecolor='k', s=10)

plt.scatter(X_test.iloc[:,6:7],X_test.iloc[:,14:15],s=50,

            facecolors='none',zorder=10,edgecolor='k')

plt.axis('tight')



x_min = X.iloc[:,6].min()

x_max = X.iloc[:,6].max()

y_min = X.iloc[:,14].min()

y_max = X.iloc[:,14].max()







print(x_min)

print(x_max)

print(y_min)

print(y_max)



XX, YY = np.mgrid[ x_min : x_max :200j,  y_min : y_max :200j]

Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])



    # Put the result into a color plot

Z = Z.reshape(XX.shape)

plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)

plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],

                linestyles=['--', '-', '--'], levels=[-.5, 0, .5])





plt.show()

clf = SVC(kernel='rbf', C=1)

clf.fit(X_train[[15,28]], y_train)

plt.figure()

plt.scatter(X.iloc[:,15],X.iloc[:,28],zorder=10,

            c=['r','b'],cmap=plt.cm.Paired,edgecolor='k', s=10)

plt.scatter(X_test.iloc[:,6:7],X_test.iloc[:,14:15],s=50,

            facecolors='none',zorder=10,edgecolor='k')

plt.axis('tight')



x_min = X.iloc[:,15].min()

x_max = X.iloc[:,15].max()

y_min = X.iloc[:,28].min()

y_max = X.iloc[:,28].max()







print(x_min)

print(x_max)

print(y_min)

print(y_max)



XX, YY = np.mgrid[ x_min : x_max :200j,  y_min : y_max :200j]

Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])



    # Put the result into a color plot

Z = Z.reshape(XX.shape)

plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)

plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],

                linestyles=['--', '-', '--'], levels=[-.5, 0, .5])





plt.show()

df_with_dummies= pd.get_dummies(Y, prefix='Category_', columns=['Category'])
df2=df

df2.rename(columns={34:"class"},inplace=True)

df2.head()
import seaborn as sns

p1 = sns.heatmap(df.iloc[:,2:32])

corr_matrix=df.corr()



# plot it

# sns.heatmap(corr_matrix, cmap='PuOr')

#sns.plt.show()





# library



np.random.seed(0)

 

# Create a dataset (fake)

plt.figure()

 

# Can be great to plot only a half matrix

mask = np.zeros_like(corr_matrix)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    p2 = sns.heatmap(corr_matrix, mask=mask, square=True)

plt.show()
df2['info']=[0 if i=='b' else 1 for i in df['class']]

df2.head()
plt.figure()

import matplotlib.dates as mdates





plt.scatter(df2.iloc[:,6:7],df2['info'],c='red',label='6')

plt.scatter(df2.iloc[:,14:15],df2['info'],c='green',label='14')

plt.scatter(df2.iloc[:,8:9],df2['info'],c='b',label='8')

plt.scatter(df2.iloc[:,4:5],df2['info'],c='c',label='4')

plt.scatter(df2.iloc[:,5:6],df2['info'],c='m',label='5')

plt.scatter(df2.iloc[:,7:8],df2['info'],c='y',label='7')

plt.scatter(df2.iloc[:,9:10],df2['info'],c='k',label='9')

plt.scatter(df2.iloc[:,10:11],df2['info'],c='black',label='10')

plt.scatter(df2.iloc[:,12:13],df2['info'],c='red',label='12')

plt.scatter(df2.iloc[:,16:17],df2['info'],c='lavender',label='16')

plt.scatter(df2.iloc[:,18:19],df2['info'],c='salmon',label='18')

plt.scatter(df2.iloc[:,19:20],df2['info'],c='coral',label='19')

plt.xlim(-1.2,1.2)

plt.xticks(np.arange(-1.2, 1.2, 0.1))

plt.legend()

plt.show()
dfilter=df2[[4,5,6,7,8,9,10,12,14,16,18,19,'info']]

dfilter.head()
sns_plot = sns.pairplot(dfilter,hue='info')
plt.figure()

plt.scatter(df.iloc[:,6:7],df.iloc[:,14:15],c=['r','b'])

plt.show()

plt.figure()

plt.scatter(df.iloc[:,10:11],df.iloc[:,18:19],c=['r','b'])

plt.show()

from sklearn.datasets import load_iris

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split



clf = DecisionTreeClassifier().fit(X_train, y_train)



print('Accuracy of Decision Tree classifier on training set: {:.2f}'

     .format(clf.score(X_train, y_train)))

print('Accuracy of Decision Tree classifier on test set: {:.2f}'

     .format(clf.score(X_test, y_test)))



clf2 = DecisionTreeClassifier(max_depth = 3).fit(X_train, y_train)



print('Accuracy of Decision Tree classifier on training set: {:.2f}'

     .format(clf2.score(X_train, y_train)))

print('Accuracy of Decision Tree classifier on test set: {:.2f}'

     .format(clf2.score(X_test, y_test)))
print('Feature importances: {}'.format(clf.feature_importances_))
feat=[]

for i in range(0,len(clf.feature_importances_)):

    feat.append((list(clf.feature_importances_)[i],i))

feat.sort(reverse=True)





        
## important feature

f1=[4,26,2,3,27]
import graphviz 

from sklearn import tree

dot_data = tree.export_graphviz(clf, out_file=None) 

graph = graphviz.Source(dot_data) 

graph.render("ionosphere") 



feature=[str(i) for i in X.columns]

dot_data = tree.export_graphviz(clf, out_file=None, 

                         feature_names=feature,  

                         class_names=['g','r'],  

                         filled=True, rounded=True,  

                         special_characters=True) 

graph=graphviz.Source(dot_data)

graph
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0)



clf = GradientBoostingClassifier().fit(X_train, y_train)

clf.score(X_test,y_test)



from sklearn.ensemble import RandomForestClassifier



X_train, X_test, y_train, y_test = train_test_split(X,Y, random_state = 0)



clf = RandomForestClassifier(max_features = 12, random_state = 0)

clf.fit(X_train, y_train)



print('Accuracy of RF classifier on training set: {:.2f}'

     .format(clf.score(X_train, y_train)))

print('Accuracy of RF classifier on test set: {:.2f}'

     .format(clf.score(X_test, y_test)))
feature_list_random=list(clf.feature_importances_)

# feature_list_random

l=[]



for i in range(0,len(feature_list_random)):

    l.append((feature_list_random[i],i))

l.sort(reverse=True)

print(l)
X1_limited=df[f1]





X_train, X_test, y_train, y_test = train_test_split(X1_limited,Y, random_state = 0)



clf = RandomForestClassifier(max_features=3,random_state = 0)

clf.fit(X_train, y_train)



print('Accuracy of RF classifier on training set: {:.2f}'

     .format(clf.score(X_train, y_train)))

print('Accuracy of RF classifier on test set: {:.2f}'

     .format(clf.score(X_test, y_test)))
RANDOM_STATE=0

from collections import OrderedDict

ensemble_clfs = [

    ("RandomForestClassifier, max_features='sqrt'",

        RandomForestClassifier(warm_start=True, oob_score=True,

                               max_features="sqrt",

                               random_state=RANDOM_STATE)),

    ("RandomForestClassifier, max_features='log2'",

        RandomForestClassifier(warm_start=True, max_features='log2',

                               oob_score=True,

                               random_state=RANDOM_STATE)),

    ("RandomForestClassifier, max_features=None",

        RandomForestClassifier(warm_start=True, max_features=None,

                               oob_score=True,

                               random_state=RANDOM_STATE))

]



# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.

error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)



# Range of `n_estimators` values to explore.

min_estimators = 15

max_estimators = 175

plt.figure()

for label, clf in ensemble_clfs:

    for i in range(min_estimators, max_estimators + 1):

        clf.set_params(n_estimators=i)

        clf.fit(X, Y)



        # Record the OOB error for each `n_estimators=i` setting.

        oob_error = 1 - clf.oob_score_

        error_rate[label].append((i, oob_error))



# Generate the "OOB error rate" vs. "n_estimators" plot.

for label, clf_err in error_rate.items():

    xs, ys = zip(*clf_err)

    plt.plot(xs, ys, label=label)



plt.xlim(min_estimators, max_estimators)

plt.xlabel("n_estimators")

plt.ylabel("OOB error rate")

plt.legend(loc="upper right")

plt.show()
from sklearn.linear_model import LogisticRegression



X_train, X_test, y_train, y_test = train_test_split(X,Y, random_state = 0)



clf = LogisticRegression().fit(X_train, y_train)



print('Accuracy of Logistic regression classifier on training set: {:.2f}'

     .format(clf.score(X_train, y_train)))

print('Accuracy of Logistic regression classifier on test set: {:.2f}'

     .format(clf.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix



y=df2['info']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0)



clf = RandomForestClassifier(max_features = 12, random_state = 0)

clf.fit(X_train, y_train)

predicted=clf.predict(X_test)

confusion = confusion_matrix(y_test,predicted)



print ('Most frequent class \n', confusion)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Accuracy = TP + TN / (TP + TN + FP + FN)

# Precision = TP / (TP + FP)

# Recall = TP / (TP + FN)  Also known as sensitivity, or True Positive Rate

# F1 = 2 * Precision * Recall / (Precision + Recall) 

print('Accuracy: {:.2f}'.format(accuracy_score(y_test, predicted)))

print('Precision: {:.2f}'.format(precision_score(y_test, predicted)))

print('Recall: {:.2f}'.format(recall_score(y_test, predicted)))

print('F1: {:.2f}'.format(f1_score(y_test, predicted)))
from sklearn.metrics import precision_recall_curve



precision, recall, thresholds = precision_recall_curve(y_test, predicted)

closest_zero = np.argmin(np.abs(thresholds))

closest_zero_p = precision[closest_zero]

closest_zero_r = recall[closest_zero]



plt.figure()

plt.xlim([0.0, 1.01])

plt.ylim([0.0, 1.01])

plt.plot(precision, recall, label='Precision-Recall Curve')

plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)

plt.xlabel('Precision', fontsize=16)

plt.ylabel('Recall', fontsize=16)

plt.axes().set_aspect('equal')

plt.show()
from sklearn.metrics import roc_curve, auc





fpr_lr, tpr_lr, _ = roc_curve(y_test, predicted)

roc_auc_lr = auc(fpr_lr, tpr_lr)



plt.figure()

plt.xlim([-0.01, 1.00])

plt.ylim([-0.01, 1.01])

plt.plot(fpr_lr, tpr_lr, lw=3, label='ROC curve (area = {:0.2f})'.format(roc_auc_lr))

plt.xlabel('False Positive Rate', fontsize=16)

plt.ylabel('True Positive Rate', fontsize=16)

plt.title('ROC curve ', fontsize=16)

plt.legend(loc='lower right', fontsize=13)

plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')

plt.axes().set_aspect('equal')

plt.show()
from sklearn.metrics import average_precision_score

average_precision = average_precision_score(y_test, predicted)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))

precision, recall, _ = precision_recall_curve(y_test, predicted)



plt.figure()

plt.step(recall, precision, color='b', alpha=0.2,

         where='post')

plt.fill_between(recall, precision, step='post', alpha=0.2,

                 color='b')



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(

          average_precision))