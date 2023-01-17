import pandas as pd
data = pd.read_csv('../input/data.csv')
classes = data['diagnosis'].unique()
labels = data['diagnosis'].map(lambda x: list(classes).index(x))
features = data.drop(['diagnosis', 'id', 'Unnamed: 32'], axis=1)
from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.1)
train_features.describe()
test_features.describe()
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(train_features, train_labels)
clf.score(test_features, test_labels)
from DecisionTreeConstraints import SizeConstraintPruning

MAX_SIZE=6
SizeConstraintPruning(MAX_SIZE).pruneToSizeK(clf)
clf.score(test_features, test_labels)
import graphviz
dot_data = tree.export_graphviz(clf, out_file=None, 
                      feature_names=features.columns,
                      class_names=classes,
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)
graph
