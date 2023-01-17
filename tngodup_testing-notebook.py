import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('../input/data.csv')
df.head()
df.shape
df_nona = df.dropna()
df_nona['shot_made_flag'].head()
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(df_nona[['team_id','minutes_remaining']], df_nona[['shot_made_flag']])
tree.export_graphviz(clf,out_file='tree.dot') 

from IPython.display import Image  
from sklearn.externals.six import StringIO  
import pydotplus

dot_data = StringIO()  
df_feature = list(df_nona[['team_id','minutes_remaining']].columns.values)
tree.export_graphviz(clf, out_file=dot_data,feature_names=df_feature)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png()) 
!pip install pydotplus
