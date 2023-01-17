# data analysis

import numpy as np

import pandas as pd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

%matplotlib inline



# machine learning

from sklearn.datasets import make_moons

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# plot the decision function

# use at the end of this notebook



def plot_decision_regions(X, y, classifier=None, resolution=0.02,ax=None):



    if ax is None:

        ax = plt.gca()

    # setup marker generator and color map

    markers = ('o', 'v', 's', '^', 'v')

    colors = ('navy', 'orangered', 'lightgreen', 'red', 'blue')

    cmap = ListedColormap(colors[:len(np.unique(y))])



    # plot the decision surface

    if classifier is not None:

        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1

        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),

                               np.arange(x2_min, x2_max, resolution))

        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

        Z = Z.reshape(xx1.shape)

        ax.contourf(xx1, xx2, Z, alpha=0.6, cmap=cmap)

        plt.xlim(xx1.min(), xx1.max())

        plt.ylim(xx2.min(), xx2.max())



    # scatter plot

    for idx, cl in enumerate(np.unique(y)):

        ax.scatter(x=X[y == cl, 0], 

                    y=X[y == cl, 1],

                    alpha=1, 

                    c=colors[idx],

                    marker=markers[idx], 

                    label=cl, 

                    edgecolor='black')
# parameters for the generated data

NOISE = 0.3        # Data generation

SAMPLES = 300       # Data generation

TEST_SIZE = 0.30    # Split Train/test



# Generate the data

X, y = make_moons(n_samples=SAMPLES, noise=NOISE, random_state=42)

plot_decision_regions(X,y)



# Split the data between train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE,random_state=42)



# parameters of RandomForest (max_depth,n_estimators)

parameters=np.array([[2,1],[2,10],[2,100],[5,1],[5,10],[5,100],[15,1],[15,10],[15,100]])



# init

models=[]

train_scores=[]

test_scores=[]
# Train the models for each instance of parameters

for params in parameters:    

    # train

    forest = RandomForestClassifier(max_depth = params[0], n_estimators = params[1], random_state = 42)    

    model = forest.fit(X_train, y_train)

    # save the model and the scores

    train_score = accuracy_score(y_train,model.predict(X_train))

    test_score = accuracy_score(y_test,model.predict(X_test))

    models.append(model)

    train_scores.append(train_score)

    test_scores.append(test_score)

    # print scores

    print("max_depth={} \t n_estimators={} \t train_score={:.4f} \t test_score={:.4f}".format(params[0],params[1],train_score,test_score))
max_depth = np.unique(parameters[:,0])

n_estimators  = np.unique(parameters[:,1])

train_scores_2d = np.around(np.array(train_scores).reshape(3,3),decimals=4)

test_scores_2d =  np.around(np.array(test_scores).reshape(3,3),decimals=4)



def plot_heap_map(X,x_params,y_params,title,ax):

    #fig, ax = plt.subplots()

    im = ax.imshow(X)

    ax.set_xticks(np.arange(len(x_params)))

    ax.set_yticks(np.arange(len(y_params)))

    ax.set_xticklabels(x_params)

    ax.set_yticklabels(y_params)

    for i in range(len(x_params)):

        for j in range(len(y_params)):

            text = ax.text(j, i, X[i, j],

                           ha="center", va="center", color="r")

    ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(15, 15))

plot_heap_map(train_scores_2d,n_estimators,max_depth,'train_scores',axes[0])

plot_heap_map(test_scores_2d,n_estimators,max_depth,'test_scores',axes[1])

plt.show()
# Plot the result    

def plot_all_decision_regions(X,y):    

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    for i, a in zip(range(9), axes.ravel()):

        a.set_title("max_depth={} n_estimators={}".format(parameters[i][0],parameters[i][1]))

        plot_decision_regions(X, y,classifier=models[i],resolution=0.02,ax=a)

    plt.show()

plot_all_decision_regions(X_train,y_train)
plot_all_decision_regions(X_test,y_test)