#import libraries and data and split into Training and Testing

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



df = pd.read_csv('../input/mushroom-classification/mushrooms.csv')

X = df.iloc[:, 1:].values

y = df.iloc[:, 0].values
TEST_SIZE = 0.3

RANDOM_STATE = 0
# Encoding categorical data

# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder

n_rows_X, n_cols_X = X.shape

for cols in range(0,n_cols_X):

    labelencoder_X = LabelEncoder()

    X[:, cols] = labelencoder_X.fit_transform(X[:, cols])



labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)
#Train/test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
# '#create CART classifier'

from sklearn import tree

clf = tree.DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=None, \

                                     min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., \

                                     max_features=None, random_state=0, max_leaf_nodes=None, min_impurity_decrease=0., \

                                     min_impurity_split=None, class_weight=None, presort=False)

#clf = tree.DecisionTreeClassifier()

clf.fit(X_train, y_train)
# Predicting the Test set results

y_pred = clf.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm
import pydot

from IPython.display import Image

from sklearn.tree import export_graphviz



tree.export_graphviz(clf, out_file='tree_1.dot',feature_names=df.columns[:-1])  

(graph,) = pydot.graph_from_dot_file('tree_1.dot')

graph.write_png('tree_1.png')

Image("tree_1.png", width=700, height=700)
from IPython.display import Image

Image(filename='../input/images-decision-trees-created/tree_1.png')
#Appears get_dummies does the job of onehotenoder and removes dummy variables we do not need 

#AND automatically gives nice column names ... win :)

df_2 = pd.get_dummies(df,drop_first=True)
X_new = df_2.iloc[:, 1:].values

y = df_2.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

clf = tree.DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=None, \

                                     min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., \

                                     max_features=None, random_state=0, max_leaf_nodes=None, min_impurity_decrease=0., \

                                     min_impurity_split=None, class_weight=None, presort=False)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

cm
tree.export_graphviz(clf, out_file='tree_2.dot',feature_names=df_2.columns[:-1])

(graph,) = pydot.graph_from_dot_file('tree_2.dot')

graph.write_png('tree_2.png')

Image("tree_2.png", width=700, height=700)
Image(filename='../input/images-decision-trees-created/tree_2.png')
Image(filename='../input/images-decision-trees-created/tree_weka.png')
#Remove odor and continue as normal

df_1 = df.drop('odor',1)
df_2 = pd.get_dummies(df_1,drop_first=True)



X_new = df_2.iloc[:, 1:].values

y = df_2.iloc[:, 0].values



X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

clf = tree.DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=None, \

                                     min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., \

                                     max_features=None, random_state=0, max_leaf_nodes=None, min_impurity_decrease=0., \

                                     min_impurity_split=None, class_weight=None, presort=False)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

cm



tree.export_graphviz(clf, out_file='tree_3.dot',feature_names=df_2.columns[:-1])

(graph,) = pydot.graph_from_dot_file('tree_3.dot')

graph.write_png('tree_3.png')

Image("tree_3.png", width=700, height=700)
Image(filename='../input/images-decision-trees-created/tree_3.png')