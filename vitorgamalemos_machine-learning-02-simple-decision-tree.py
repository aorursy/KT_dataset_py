import pandas as pd 

  

data = [['Day1', 'Sun', 'Warm', 'High', 'Weak', 'No'], 

        ['Day2', 'Sun', 'Warm', 'High', 'Strong', 'No'],

        ['Day3', 'Cloud', 'Warm', 'High', 'Weak', 'Yes'],

        ['Day4', 'Rain', 'Soft', 'High', 'Weak', 'Yes'], 

        ['Day5', 'Rain', 'Fresh', 'Normal', 'Weak', 'Yes'],

        ['Day6', 'Rain', 'Fresh', 'Normal', 'Strong', 'No'],

        ['Day7', 'Cloud', 'Fresh', 'Normal', 'Weak', 'Yes'], 

        ['Day8', 'Sun', 'Soft', 'High', 'Weak', 'No'],

        ['Day9', 'Sun', 'Fresh', 'Normal', 'Weak', 'Yes'],

        ['Day10', 'Rain', 'Soft', 'Normal', 'Strong', 'Yes'],

        ['Day11', 'Sun', 'Soft', 'Normal', 'Strong', 'Yes'], 

        ['Day12', 'Cloud', 'Soft', 'High', 'Strong', 'Yes'],

        ['Day13', 'Cloud', 'Warm', 'Normal', 'Weak', 'Yes'],

        ['Day14', 'Rain', 'Soft', 'High', 'Strong', 'No']]





df = pd.DataFrame(data, columns = ['Day', 'Aspect', 'Temperature', 'Humidity','Wind','Output']) 

df.head(12)
df_type_1 = df.select_dtypes(include=['int64']).copy()

df_type_2 = df.select_dtypes(include=['float64']).copy()

df_type_3 = pd.concat([df_type_2, df_type_1], axis=1, join_axes=[df_type_1.index])
from sklearn import preprocessing 

from sklearn.preprocessing import LabelEncoder



categorization = preprocessing.LabelEncoder()

categorization.fit(df["Aspect"].astype(str))

list(categorization.classes_)



df_object = df.astype(str).apply(categorization.fit_transform)

df_formated = pd.concat([df_type_3, df_object], axis=1, join_axes=[df_type_3.index])

df_formated.head(12)
import matplotlib.pyplot as plt

df_formated.hist(alpha=0.5, figsize=(15, 15), color='blue')

plt.show()
import pandas

from pandas.plotting import scatter_matrix



scatter_matrix(df_formated, alpha=0.5, figsize=(20, 20))

plt.show()
from sklearn.datasets import load_iris

from sklearn.tree import DecisionTreeClassifier



X = df_formated.ix[:,'Day':'Wind':] 

y = df_formated.ix[:, 'Output':]
tree_clf = DecisionTreeClassifier(max_depth=6)

tree_clf.fit(X, y)
from sklearn.tree import export_graphviz

from IPython.display import Image

from subprocess import call

from graphviz import Digraph



t = export_graphviz(tree_clf,out_file='tree_decision.dot',

                    feature_names=['Day','Aspect','Temperature','Humidity','Wind'],

                    class_names=['Yes', 'No'],

                    rounded=True,filled=True)



call(['dot', '-Tpng', 'tree_decision.dot', '-o', 'tree.png', '-Gdpi=800'])

Image(filename = 'tree.png')