# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/heart.csv")

data.info()

df = pd.DataFrame(data)

df.columns

df.index
data.describe()
import seaborn as sns # used for plot interactive graph.

import matplotlib.pyplot as plt



plt.figure(figsize=(12,8))

sns.countplot(data['age'],label ="count")

#correlation graph to find the correlation

corr = data.corr()

plt.figure(figsize = (14,14))

sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},

           cmap= 'coolwarm')
#conclusion, none of the features are highlt correltaed, so we could use all 
selected_features = ['age', 'cp', 'chol', 'fbs', 'restecg', 'thalach',

       'exang', 'oldpeak', 'slope', 'ca', 'thal']





plt.figure(figsize = (14,14))

color_function = {0: "blue", 1: "red"} # Here Red color will be 1 which means M and blue foo 0 means B

colors = data["target"].map(lambda x: color_function.get(x))# mapping the color fuction with diagnosis column

pd.scatter_matrix(data[selected_features], c=colors, alpha = 0.5, figsize = (15, 15)); # plotting scatter plot matrix
#First get an general overview

data.hist(figsize=(15,20))

plt.figure()
#cp: chest pain type

data["cp"].value_counts()

plt.xlabel('pain type')

plt.ylabel('count')

plt.title('Four type chest pain')

sns.countplot(data['cp'])
import plotly

import plotly.graph_objs as go



x_data = data['age']

y_data = data['thalach']

colors = np.random.rand(2938)

sz = np.random.rand(2000)*30



fig = go.Figure()

fig.add_scatter(x = x_data,

                y = y_data,

                mode = 'markers',

                marker = {'size': sz,

                         'color': colors,

                         'opacity': 0.6,

                         'colorscale': 'Portland'

                       })

plotly.offline.iplot(fig)
#adding filter to data

data[(data['thalach']>190)]
#adding filter to data



data[(data['age']>40) & (data['sex']==0)]

   
#using swarmplot to learn about age, gender ad orobability of having heart disearse.

#first append a new row about age measuremnet based on probability

threshold = sum(data.age)/len(data.age)

threshold_chol = sum(data.chol)/len(data.chol)

print("threshold of age:" , threshold)

print("threshold of chol: ", threshold_chol)



data['probability'] = ['high' if i> threshold else 'low' for i in data.age]

data['probability']



sns.swarmplot(x = 'sex', y = 'age', hue = "probability", data = data)
data_original = pd.read_csv("../input/heart.csv")

data['sex'] = data_original.sex

data['sex']=['female' if i == 0 else 'male' for i in data.sex]

plt.figure(figsize=(14,8))

sns.swarmplot(x = 'age', y = 'chol', hue = "sex", data = data)
data.head(5)
#aplit the dataset ready for training and testing, and apply machine leaning ALG

from sklearn.model_selection import train_test_split

x,y = data.loc[:,data.columns != 'target'], data.loc[:,'target']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)

data_orginal = pd.read_csv("../input/heart.csv")

data['sex']=data_orginal.sex

data.dtypes.sample(10)

#data.select_dtypes(exclude=['object'])



#Machine learning by KNN

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors = 3)

#x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']

x_train_one_hot_encoded = pd.get_dummies(x_train)

x_test_one_hot_encoded = pd.get_dummies(x_test)

knn.fit(x_train_one_hot_encoded,y_train)

prediction = knn.predict(x_test_one_hot_encoded)

acc = accuracy_score(y_test, prediction)

k = knn.score(x_test_one_hot_encoded,y_test)

print("acc score with one hot encoded: ", acc)

print("knn score: ", k)

print("just only drop the non-numrical feature--->")

x_train_2 = x_train.select_dtypes(exclude=['object'])

x_test_2 = x_test.select_dtypes(exclude=['object'])

knn.fit(x_train_2,y_train)

prediction = knn.predict(x_test_2)

acc = accuracy_score(y_test, prediction)

print("acc score without one hot encoded: ", acc)
# the training score is relativly low. I will apply Grid Search to find the best parameters of KNN
x_train = x_train.select_dtypes(exclude=['object'])

x_test = x_test.select_dtypes(exclude=['object'])
# grid search cross validation with 1 hyperparameter

from sklearn.model_selection import GridSearchCV

grid = {'n_neighbors': np.arange(1,100)}

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv=3) # GridSearchCV

knn_cv.fit(x_train,y_train)# Fit



# Print hyperparameter

print("Tuned hyperparameter k: {}".format(knn_cv.best_params_)) 

print("Best score: {}".format(knn_cv.best_score_))
#After apply GridSearch the score is not much impoved
#apply sklearn decision tree classifier on the dataset

#reference https://www.kaggle.com/drgilermo/playing-with-the-knobs-of-sklearn-decision-tree

import time

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import Imputer

from sklearn.metrics import confusion_matrix

from subprocess import check_output

from sklearn import tree

from IPython.display import Image as PImage

from subprocess import check_call

from PIL import Image, ImageDraw, ImageFont

import re



classifier = DecisionTreeClassifier(max_depth = 3)

classifier.fit(x_train, y_train)



print("Decision tree score : {}".format(classifier.score(x_test, y_test))) 



clf = DecisionTreeClassifier(max_depth = 3, criterion = "entropy")

clf.fit(x_train,y_train)

print("Decision tree score : {}".format(clf.score(x_test, y_test))) 
#apply best split and random split

t = time.time()

clf_random = DecisionTreeClassifier(max_depth = 3, splitter = 'random')

clf_random.fit(x_train,y_train)

print('random Split accuracy...',clf_random.score(x_test,y_test))

clf_best = DecisionTreeClassifier(max_depth = 3, splitter = 'best')

clf_best.fit(x_train,y_train)

print('best Split accuracy...',clf_best.score(x_test,y_test))

# conclusion: random split is not nessesary worse than best split
with open("tree1.dot", 'w') as f:

     f = tree.export_graphviz(clf_best,

                              out_file=f,

                              max_depth = 5,

                              impurity = False,# true will show thw im-purity of each node~ gini value

                              feature_names = x_test.columns.values,

                             class_names = ['No', 'Yes'],

                            #  class_names = True,

                              rounded = True,

                              filled= True )#False no color indication

        

#Convert .dot to .png to allow display in web notebook

check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])



# Annotating chart with PIL

img = Image.open("tree1.png")

draw = ImageDraw.Draw(img)

img.save('sample-out.png')

PImage("sample-out.png")

# we have generared rhe random decision tree classifier, see below.
