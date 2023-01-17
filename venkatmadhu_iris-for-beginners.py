# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Load the data :

data = pd.read_csv('/kaggle/input/iris/Iris.csv')
print('Shape of the data:',data.shape)
# Tells the dtypes of each feature :
data.info()
#Just have a feel of the data :
data.head(1)
#Features provided :

print('Nof features provided:',list(data.columns.unique()))
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#We can have a look how the features is varuting with the target variable :

sns.set_style('whitegrid')
sns.boxplot(x = 'SepalLengthCm',y = 'Species',data = data)
plt.title('Species VS SepalLengthCm')
# A violin plot combines the benefits of the previous two plots and simplifies them
# Denser regions of the data are fatter, and sparser thiner in a violin plot
sns.violinplot(x="Species", y="PetalLengthCm", data=data, size=6)
# Another useful seaborn plot is the pairplot, which shows the bivariate relation
# between each pair of features
# 
# From the pairplot, we'll see that the Iris-setosa species is separataed from the other
# two across all feature combinations
sns.pairplot(data.drop("Id", axis=1), hue="Species", size=3)


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=data)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=data)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=data)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=data)


#Importing all the libraries :

from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.model_selection import train_test_split #to split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
#Drop the Id feature as itis irrelevant :
data.drop('Id',axis=1,inplace=True)
plt.figure(figsize=(7,4)) 
sns.heatmap(data.corr(),annot=True,cmap='cubehelix_r') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())
plt.show()
train, test = train_test_split(data, test_size = 0.3)# in this our main data is split into train and test
# the attribute test_size=0.3 splits the data into 70% and 30% ratio. train=70% and test=30%
print(train.shape)
print(test.shape)

X_train = train.drop('Species',axis=1)
y_train = train[['Species']]

X_test = test.drop('Species',axis=1)
y_test = test[['Species']]
model = LogisticRegression()
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print('The accuracy of the Logistic Regression is',round(metrics.accuracy_score(prediction,y_test),2))


model=DecisionTreeClassifier()
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print('The accuracy of the Decision Tree is',round(metrics.accuracy_score(prediction,y_test),2))

# Here in the Decision Tree we have an advantage we can see how our model is working internally by plotting the tree structure 
from IPython.display import Image  
from sklearn import tree
import pydotplus

#VIsualsing the Tree :
feature_names = list(X_train.columns.unique())
print(feature_names)

## Create DOT data
dot_data = tree.export_graphviz(model, out_file=None, 
                                feature_names=feature_names,  
                                class_names=data.Species,
                               )

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)  

# Show graph
Image(graph.create_png())
tree.plot_tree(model) 
