import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

# apply ignore

import warnings

warnings.filterwarnings('ignore')
#load train data

train_data = pd.read_csv('../input/learn-together/train.csv')

train_data.head()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier



y = train_data.Cover_Type

from sklearn import tree

import graphviz



selected_features = ['Elevation', 'Horizontal_Distance_To_Roadways', 'Wilderness_Area4']



X1 = train_data[selected_features]

train_X, val_X, train_y, val_y = train_test_split(X1, y, random_state=1)

tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)



# shrink names

selected_features = ['Elv', 'HDRoad', 'WildA4']



tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=selected_features)

graphviz.Source(tree_graph)
from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots



# 'Id' is a false feature, although adding it makes smaller mse error

feature_names = [cname for cname in train_data.columns if cname not in ['Id','Cover_Type']]

X = train_data[feature_names]



# Build Random Forest model

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

rf_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)



feature_to_plot = 'Elevation'

pdp_dist = pdp.pdp_isolate(model=rf_model, dataset=val_X, model_features=feature_names, feature=feature_to_plot)



pdp.pdp_plot(pdp_dist, feature_to_plot)

plt.show()
# use pdp_interact and pdp_interact_plot 

# 'Elevation', 'Horizontal_Distance_To_Roadways', 'Wilderness_Area4'

features_to_plot = ['Horizontal_Distance_To_Roadways', 'Wilderness_Area4']

inter1  =  pdp.pdp_interact(model=rf_model, dataset=val_X, model_features=feature_names, features=features_to_plot)



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour', plot_pdp=True)

plt.show()