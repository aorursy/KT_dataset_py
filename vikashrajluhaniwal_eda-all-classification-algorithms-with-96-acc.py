import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("/kaggle/input/human-activity-recognition-with-smartphones/train.csv")

test = pd.read_csv("/kaggle/input/human-activity-recognition-with-smartphones/test.csv")
print('Number of duplicates in train : ',sum(train.duplicated()))

print('Number of duplicates in test : ', sum(test.duplicated()))
print('Total number of missing values in train : ', train.isna().values.sum())

print('Total number of missing values in train : ', test.isna().values.sum())
plt.figure(figsize=(10,8))

plt.title('Barplot of Activity')

sns.countplot(train.Activity)

plt.xticks(rotation=90)
facetgrid = sns.FacetGrid(train, hue='Activity', height=5,aspect=3)

facetgrid.map(sns.distplot,'tBodyAccMag-mean()', hist=False).add_legend()

plt.annotate("Static Activities", xy=(-.996,21), xytext=(-0.9, 23),arrowprops={'arrowstyle': '-', 'ls': 'dashed'})

plt.annotate("Static Activities", xy=(-.999,26), xytext=(-0.9, 23),arrowprops={'arrowstyle': '-', 'ls': 'dashed'})

plt.annotate("Static Activities", xy=(-0.985,12), xytext=(-0.9, 23),arrowprops={'arrowstyle': '-', 'ls': 'dashed'})

plt.annotate("Dynamic Activities", xy=(-0.2,3.25), xytext=(0.1, 9),arrowprops={'arrowstyle': '-', 'ls': 'dashed'})

plt.annotate("Dynamic Activities", xy=(0.1,2.18), xytext=(0.1, 9),arrowprops={'arrowstyle': '-', 'ls': 'dashed'})

plt.annotate("Dynamic Activities", xy=(-0.01,2.15), xytext=(0.1, 9),arrowprops={'arrowstyle': '-', 'ls': 'dashed'})
plt.figure(figsize=(12,8))

plt.subplot(1,2,1)

plt.title("Static Activities(closer view)")

sns.distplot(train[train["Activity"]=="SITTING"]['tBodyAccMag-mean()'],hist = False, label = 'Sitting')

sns.distplot(train[train["Activity"]=="STANDING"]['tBodyAccMag-mean()'],hist = False,label = 'Standing')

sns.distplot(train[train["Activity"]=="LAYING"]['tBodyAccMag-mean()'],hist = False, label = 'Laying')

plt.axis([-1.02, -0.5, 0, 35])

plt.subplot(1,2,2)

plt.title("Dynamic Activities(closer view)")

sns.distplot(train[train["Activity"]=="WALKING"]['tBodyAccMag-mean()'],hist = False, label = 'Sitting')

sns.distplot(train[train["Activity"]=="WALKING_DOWNSTAIRS"]['tBodyAccMag-mean()'],hist = False,label = 'Standing')

sns.distplot(train[train["Activity"]=="WALKING_UPSTAIRS"]['tBodyAccMag-mean()'],hist = False, label = 'Laying')
plt.figure(figsize=(10,7))

sns.boxplot(x='Activity', y='tBodyAccMag-mean()',data=train, showfliers=False)

plt.ylabel('Body Acceleration Magnitude mean')

plt.title("Boxplot of tBodyAccMag-mean() column across various activities")

plt.axhline(y=-0.7, xmin=0.05,dashes=(3,3))

plt.axhline(y=0.020, xmin=0.35, dashes=(3,3))

plt.xticks(rotation=90)
plt.figure(figsize=(10,7))

sns.boxplot(x='Activity', y='angle(X,gravityMean)', data=train, showfliers=False)

plt.axhline(y=0.08, xmin=0.1, xmax=0.9,dashes=(3,3))

plt.ylabel("Angle between X-axis and gravityMean")

plt.title('Box plot of angle(X,gravityMean) column across various activities')

plt.xticks(rotation = 90)
plt.figure(figsize=(10,7))

sns.boxplot(x='Activity', y='angle(Y,gravityMean)', data = train, showfliers=False)

plt.ylabel("Angle between Y-axis and gravityMean")

plt.title('Box plot of angle(Y,gravityMean) column across various activities')

plt.xticks(rotation = 90)

plt.axhline(y=-0.35, xmin=0.01, dashes=(3,3))
from sklearn.manifold import TSNE
X_for_tsne = train.drop(['subject', 'Activity'], axis=1)
%time

tsne = TSNE(random_state = 42, n_components=2, verbose=1, perplexity=50, n_iter=1000).fit_transform(X_for_tsne)
plt.figure(figsize=(12,8))

sns.scatterplot(x =tsne[:, 0], y = tsne[:, 1], hue = train["Activity"],palette="bright")
y_train = train.Activity

X_train = train.drop(['subject', 'Activity'], axis=1)

y_test = test.Activity

X_test = test.drop(['subject', 'Activity'], axis=1)

print('Training data size : ', X_train.shape)

print('Test data size : ', X_test.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report



import warnings

warnings.filterwarnings("ignore")
parameters = {'C':np.arange(10,61,10), 'penalty':['l2','l1']}

lr_classifier = LogisticRegression()

lr_classifier_rs = RandomizedSearchCV(lr_classifier, param_distributions=parameters, cv=5,random_state = 42)

lr_classifier_rs.fit(X_train, y_train)

y_pred = lr_classifier_rs.predict(X_test)
lr_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy using Logistic Regression : ", lr_accuracy)
# function to plot confusion matrix

def plot_confusion_matrix(cm,lables):

    fig, ax = plt.subplots(figsize=(12,8)) # for plotting confusion matrix as image

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),

    yticks=np.arange(cm.shape[0]),

    xticklabels=lables, yticklabels=lables,

    ylabel='True label',

    xlabel='Predicted label')

    plt.xticks(rotation = 90)

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, int(cm[i, j]),ha="center", va="center",color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
cm = confusion_matrix(y_test.values,y_pred)

plot_confusion_matrix(cm, np.unique(y_pred))  # plotting confusion matrix
#function to get best random search attributes

def get_best_randomsearch_results(model):

    print("Best estimator : ", model.best_estimator_)

    print("Best set of parameters : ", model.best_params_)

    print("Best score : ", model.best_score_)
# getting best random search attributes

get_best_randomsearch_results(lr_classifier_rs)
from sklearn.svm import LinearSVC
parameters = {'C':np.arange(1,12,2)}

lr_svm = LinearSVC(tol=0.00005)

lr_svm_rs = RandomizedSearchCV(lr_svm, param_distributions=parameters,random_state = 42)

lr_svm_rs.fit(X_train, y_train)

y_pred = lr_svm_rs.predict(X_test)
lr_svm_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy using linear SVM : ",lr_svm_accuracy)
cm = confusion_matrix(y_test.values,y_pred)

plot_confusion_matrix(cm, np.unique(y_pred)) # plotting confusion matrix
# getting best random search attributes

get_best_randomsearch_results(lr_svm_rs)
from sklearn.svm import SVC
np.linspace(2,22,6)
parameters = {'C':[2,4,8,16],'gamma': [0.125, 0.250, 0.5, 1]}

kernel_svm = SVC(kernel='rbf')

kernel_svm_rs = RandomizedSearchCV(kernel_svm,param_distributions=parameters,random_state = 42)

kernel_svm_rs.fit(X_train, y_train)

y_pred = kernel_svm_rs.predict(X_test)
kernel_svm_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy using Kernel SVM : ", kernel_svm_accuracy)
cm = confusion_matrix(y_test.values,y_pred)

plot_confusion_matrix(cm, np.unique(y_pred)) # plotting confusion matrix
# getting best random search attributes

get_best_randomsearch_results(kernel_svm_rs)
from sklearn.tree import DecisionTreeClassifier

parameters = {'max_depth':np.arange(2,10,2)}

dt_classifier = DecisionTreeClassifier()

dt_classifier_rs = RandomizedSearchCV(dt_classifier,param_distributions=parameters,random_state = 42)

dt_classifier_rs.fit(X_train, y_train)

y_pred = dt_classifier_rs.predict(X_test)
dt_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy using Decision tree : ", dt_accuracy)
cm = confusion_matrix(y_test.values,y_pred)

plot_confusion_matrix(cm, np.unique(y_pred)) # plotting confusion matrix
# getting best random search attributes

get_best_randomsearch_results(dt_classifier_rs)
from sklearn.ensemble import RandomForestClassifier

params = {'n_estimators': np.arange(20,101,10), 'max_depth':np.arange(2,16,2)}

rf_classifier = RandomForestClassifier()

rf_classifier_rs = RandomizedSearchCV(rf_classifier, param_distributions=params,random_state = 42)

rf_classifier_rs.fit(X_train, y_train)

y_pred = rf_classifier_rs.predict(X_test)
rf_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy using Random forest : ", rf_accuracy)
cm = confusion_matrix(y_test.values,y_pred)

plot_confusion_matrix(cm, np.unique(y_pred)) # plotting confusion matrix
# getting best random search attributes

get_best_randomsearch_results(rf_classifier_rs)