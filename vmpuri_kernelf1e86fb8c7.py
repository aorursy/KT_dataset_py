# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import matplotlib.pyplot as plt

import numpy as np
import csv

rows = []

def class_switch(class_char):
    if(class_char == 'e'):
        return 0
    # else class_char == 'p' since only 2 classes
    else:
        return 1
    
# Convert ASCII characters to their integer equivalent to create the feature vector
def generate_vector(row):
    return [
        class_switch(row['class'])
        ,ord(row['cap-shape'])
        ,ord(row['cap-surface'])
        ,ord(row['cap-color'])
        ,ord(row['bruises'])
        ,ord(row['odor'])
        ,ord(row['gill-attachment'])
        ,ord(row['gill-spacing'])
        ,ord(row['gill-size'])
        ,ord(row['gill-color'])
        ,ord(row['stalk-shape'])
        ,ord(row['stalk-root'])
        ,ord(row['stalk-surface-above-ring'])
        ,ord(row['stalk-surface-below-ring'])
        ,ord(row['stalk-color-above-ring'])
        ,ord(row['stalk-color-below-ring'])
        ,ord(row['veil-type'])
        ,ord(row['veil-color'])
        ,ord(row['ring-number'])
        ,ord(row['ring-type'])
        ,ord(row['spore-print-color'])
        ,ord(row['population'])
        ,ord(row['habitat'])
    ]

# load data into memory from file
with open('/kaggle/input/mushroom-classification/mushrooms.csv') as infile:
    reader = csv.DictReader(infile)
    
    for row in reader:
        rows.append(generate_vector(row))
    print(len(rows), rows[0])
    
    vectors = []
    classes = []
    for row in rows:
        vectors.append(row[1:])
        classes.append(row[0])

    # Initialize the decision tree classifier
    numtrain = 6031       # Number of records to train on 
                          # 6031 = 75% of training set
    print("Training on the first", numtrain, "records.")
    
    # Having fun with different classifiers
    # Decision Tree
    dtree = tree.DecisionTreeClassifier(criterion="entropy")
    dtree.fit(vectors[:numtrain], classes[:numtrain])
    
    # K Nearest Neighbors Classifier
    neigh = KNeighborsClassifier(n_neighbors=8)
    neigh.fit(vectors[:numtrain], classes[:numtrain])
    
    # Support Vector Machines
    sv = svm.SVC()
    sv.fit(vectors[:numtrain], classes[:numtrain])
    
    # Check how well the decision tree works against the original dataset
    treepred = 0
    knnpred = 0
    svmpred = 0
    for i in range(0, len(rows)):
        if(dtree.predict([vectors[i]])[0] == classes[i]):
            treepred += 1
        if(neigh.predict([vectors[i]])[0] == classes[i]):
            knnpred += 1
        if(sv.predict([vectors[i]])[0] == classes[i]):
            svmpred += 1
      
    #print the number of poisonous mushrooms identified (actual = 3916)
    print("Decision Tree Accuracy\t",treepred/8124)
    print("KNN Accuracy\t",knnpred/8124)
    print("SVM Accuracy\t",svmpred/8124)
plt.figure(figsize=(20, 20))
tree.plot_tree(dtree, rounded=True, fontsize=15)
plt.show()

