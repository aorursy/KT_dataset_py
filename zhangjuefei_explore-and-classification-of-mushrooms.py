# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/mushrooms.csv")

data.describe()
feature_columns = data.columns[1:]

for i, f in zip(np.arange(1, len(feature_columns) + 1), feature_columns):

    print('feature {:d}:\t{}'.format(i, f))
import matplotlib.pyplot as plt

plt.style.use('ggplot')



fig, axes = plt.subplots(nrows=11, ncols=2, figsize=(8, 60))

data['id'] = np.arange(1, data.shape[0] + 1)



for f, ax in zip(feature_columns, axes.ravel()):

    data.groupby(['class', f])['id'].count().unstack(f).plot(kind='bar', ax=ax, legend=False, grid=True, title=f)
from sklearn.feature_selection import chi2, SelectKBest

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

numeric_data = pd.DataFrame()

for f in feature_columns:

    numeric_data[f] = le.fit_transform(data[f])

    

chi_statics, p_values = chi2(numeric_data, data['class'])



chi2_result = pd.DataFrame({'features': feature_columns, 'chi2_statics': chi_statics, 'p_values': p_values})

chi2_result.dropna(axis=0, how='any', inplace=True)



print(chi2_result.sort_values(by='chi2_statics', ascending=False)[['features', 'chi2_statics', 'p_values']].reset_index().drop('index', axis=1))



_ = chi2_result.sort_values(by='chi2_statics', ascending=True).set_index('features')['chi2_statics'].plot(kind='barh', logx=True, rot=-2)
use_features = chi2_result.sort_values(by='chi2_statics', ascending=False)['features'].head(5).values



print('top 5 most useful features are:')

for f in use_features:

    print(f)
data_reduced = pd.DataFrame()

for f in use_features:

    dummies = data[f].str.get_dummies()

    dummies.columns = ['{}_{}'.format(f, v) for v in dummies.columns]

    data_reduced = pd.concat([data_reduced, dummies], axis=1)



data_reduced['class'] = data['class']
from sklearn.neural_network import MLPClassifier

from sklearn.cross_validation import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

import seaborn as sns



# A two-hidden layer network.

nw = MLPClassifier(hidden_layer_sizes = (30, 30), activation='logistic', alpha=0.001, solver='lbfgs', learning_rate='constant')



# Prepare the training set & testing set.

train_features, test_features, train_labels, test_labels = train_test_split(data_reduced[data_reduced.columns[: -1]], data_reduced['class'], train_size=0.8)



nw.fit(train_features, train_labels)



# Check metrics on training set.

train_predict_labels = nw.predict(train_features)

print('\nTraining Classification Report:')

print(classification_report(train_labels, train_predict_labels))



# Check metrics on testing set.

test_predict_labels = nw.predict(test_features)

print('\nTesting Classification Report:')

print(classification_report(test_labels, test_predict_labels))



# Confision Matrix.

cm = confusion_matrix(test_labels, test_predict_labels)



# print('\nConfusion Matrix:')

_ = sns.heatmap(cm, square = True, xticklabels = ['e', 'p'], annot = True, annot_kws = {'fontsize': 12}, yticklabels = ['e', 'p'], cbar = True, cbar_kws = {"orientation": "horizontal"}, cmap = "Blues").set(xlabel = "predicted", ylabel = "true", title = 'Confusion Matrix')
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve



test_labels = (test_labels == 'p').astype(np.int)

test_predict_labels = (test_predict_labels == 'p').astype(np.int)



# merics.

test_predict_proba = nw.predict_proba(test_features)

fpr, rc, th = roc_curve(test_labels, test_predict_proba[:, 1])

precision, recall, threshold = precision_recall_curve(test_labels, test_predict_proba[:, 1])

roc_auc = auc(fpr, rc)



print('\nMetrics: Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, AUC: {:.3f}'.format(accuracy_score(test_labels, test_predict_labels), precision_score(test_labels, test_predict_labels), recall_score(test_labels, test_predict_labels), roc_auc))



# draw some charts.

fig = plt.figure(figsize=(16, 4))

ax = fig.add_subplot(131)

ax.set_xlabel('False Positive Rate')

ax.set_ylabel('Recall')

ax.set_title('ROC Curve')

ax.plot(fpr, rc, 'b')

ax.plot([0.0, 1.0], [0.0, 1.0], 'r--')

ax.text(0.80, 0.05, 'auc: {:.2f}'.format(roc_auc))



ax = fig.add_subplot(132)

ax.set_xlabel('Threshold')

ax.set_ylabel('Precision & Recall')

ax.set_title('Precsion & Recall')

ax.set_xlim([threshold.min(), threshold.max()])

ax.set_ylim([0.0, 1.0])

ax.plot(threshold, precision[:-1], 'b', label='Precision')

ax.plot(threshold, recall[:-1], 'r', label='Recall')

_ = ax.legend(loc='best')



ts = np.arange(0, 1.02, 0.02)

accuracy = []

for t in ts:

    predict_label = (test_predict_proba[:, 1] >= t).astype(np.int)

    accuracy.append(accuracy_score(test_labels, test_predict_labels))



ax = fig.add_subplot(133)

ax.set_xlabel("Threshold")

ax.set_ylabel("Accuracy")

ax.set_ylim([0.0, 1.0])

ax.set_title('Accuracy')

ax.plot([0.0, 1.0], [0.5, 0.5], 'r--')

ax.plot(ts, accuracy, 'b')



plt.show()