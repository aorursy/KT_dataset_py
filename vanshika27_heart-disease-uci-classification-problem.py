import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold

from collections import Counter

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn import tree, metrics

import graphviz 

import os
# Load the dataset

heart_data = pd.read_csv("../input/heart.csv")
heart_data.head()
heart_data.shape
heart_data.info()
# Correlation between the features

heart_data.corr()
plt.hist(heart_data['thalach'], bins = 10, color = 'blue')

plt.xlabel('thalach')

plt.ylabel('Frequency')

plt.title('Histogram of Age')

plt.show()
for index,value in zip(heart_data['sex'].value_counts().index, heart_data['sex'].value_counts()):

    print('Class {} =>  {} ({:.2f}%)'.format(index, value, (value/heart_data.shape[0]*100)))
# Convert continous data into categorial data

def label_data(feature_data):

    mean = feature_data.mean()

    first = feature_data.min()

    second = (first + mean)/2

    third = (feature_data.max() + mean)/2

    return feature_data.apply(label, args = (first, second, third))

    

def label(val, *boundries):

    if val < boundries[0]:

        val = 1

    elif val < boundries[1]:

        val = 2

    elif val < boundries[2]:

        val = 3

    else:

        val = 4

    return val
heart_data['thalach'] = label_data(heart_data['thalach'])

heart_data['trestbps'] = label_data(heart_data['trestbps'])

heart_data['chol'] = label_data(heart_data['chol'])

heart_data['oldpeak'] = label_data(heart_data['oldpeak'])

heart_data['age'] = label_data(heart_data['age'])
heart_data.head(10)
# Segregate data and target

data = heart_data.iloc[:,0:13]

target = heart_data.iloc[:,13]

data.shape, target.shape
# Split the data into train and test data

x_train, x_test, y_train, y_test = train_test_split(data, target)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
# Get list of all the features in the dataset

features = x_train.columns
plt.scatter(x_train['thalach'], y_train)

plt.xlabel('thalach');

plt.ylabel('target')

plt.show()
# Calculate Gini Index for all the features

def gini_impurity(data, total):

    gini_split = 0

    for cat in data:

        gini_split += (data[cat]/total)**2

    gini_index = 1 - gini_split

    return gini_index
for feature in features:

    impurity = gini_impurity(Counter(x_train[feature]), x_train.shape[0])

    print('Impurity at {} node is: {}'.format(feature, impurity))
# Find appropriate max_depth for classification

for depth in range(1,20):

    fold_accuracy = []

    tree_model = tree.DecisionTreeClassifier(max_depth=depth)

    model = tree_model.fit(X = x_train, y = y_train)   

    valid_acc = tree_model.score(X = x_test, y = y_test)

    print('Accuracy for depth {} is {}'.format(depth, valid_acc))
# Train Model

tree_model = tree.DecisionTreeClassifier(max_depth = 6)

tree_model.fit(x_train, y_train)
y_pred = tree_model.predict(x_test)
print('Accuracy is: ',metrics.accuracy_score(y_test,y_pred))
os.environ["PATH"] += os.pathsep + 'C:/Users/pc/Anaconda3/pkgs/graphviz-2.38.0-4/Library/bin/graphviz'
dot_data = export_graphviz(tree_model, out_file=None, 

                         feature_names=features, 

                         filled=True, rounded=True,  

                         special_characters=True)  

graph = graphviz.Source(dot_data)  

graph