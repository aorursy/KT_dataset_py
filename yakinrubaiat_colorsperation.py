!pip install pydotplus
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pydotplus #pip install pydotplus
from sklearn.tree import export_graphviz
%matplotlib inline
df = pd.read_csv(os.path.join('..','input','Color.csv'))
df.head()
X = df[['B','G','R','Time']]
y = df[['Label']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, 
                                  random_state=17)
clf_tree.fit(X_train, y_train)
predicted = clf_tree.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(predicted, y_test)
def tree_graph_to_png(tree, feature_names, png_file_to_save):
    tree_str = export_graphviz(tree, feature_names=feature_names, 
                                     filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)  
    graph.write_png(png_file_to_save)
tree_graph_to_png(tree=clf_tree, feature_names=['B','G','R','Time'], 
                  png_file_to_save='topic3_tree1.png')
from joblib import dump, load
dump(clf_tree, 'filename.joblib') 
clf = load('filename.joblib') 
