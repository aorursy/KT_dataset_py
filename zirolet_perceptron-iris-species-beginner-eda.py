#Used for interactivity

!jupyter nbextension enable --py widgetsnbextension

from ipywidgets import interact, interactive, fixed, interact_manual

import ipywidgets as  widgets 
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from scipy.special import comb

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/iris/Iris.csv")

df

df = df.drop(labels = "Id", axis = 1)

df
sLength = df[['SepalLengthCm','Species']]

sLength
sLength_bySpecies = {}

for i, id in enumerate(sLength['Species'].unique()):

    sLength_bySpecies[id] = sLength[sLength['Species'] == id]

for spec in sLength_bySpecies:

    print("Describe for " + spec + ":")

    print(sLength_bySpecies[spec].describe())
sWidth = df[['SepalWidthCm','Species']]

sWidth_bySpecies = {}

for i, id in enumerate(sWidth['Species'].unique()):

    sWidth_bySpecies[id] = sWidth[sWidth['Species'] == id]

for spec in sWidth_bySpecies:

    print("Describe for " + spec + ":")

    print(sWidth_bySpecies[spec].describe())
pLength = df[['PetalLengthCm','Species']]

pLength_bySpecies = {}

for i, id in enumerate(pLength['Species'].unique()):

    pLength_bySpecies[id] = pLength[pLength['Species'] == id]

for spec in pLength_bySpecies:

    print("Describe for " + spec + ":")

    print(pLength_bySpecies[spec].describe())
pWidth = df[['PetalWidthCm','Species']]

pWidth_bySpecies = {}

for i, id in enumerate(pWidth['Species'].unique()):

    pWidth_bySpecies[id] = pWidth[pWidth['Species'] == id]

for spec in pWidth_bySpecies:

    print("Describe for " + spec + ":")

    print(pWidth_bySpecies[spec].describe())
#Everything must be done by COPY

X , y = df.iloc[:,[0,1,2,3]].copy() , df['Species'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.5, random_state = 1)

print(y_train.value_counts())

X_train = X_train.values; X_test = X_test.values; y_train = y_train.values; y_test = y_test.values
#Returns 2-D Array of combinations in form: [[x1,x2],[x3,x4]...[xi,xj]]

def combinations(x):

    count = 0

    mesh = np.array(np.meshgrid(x,x))

    combination = mesh.T.reshape(-1,2)  #Gives me same pairs like [0,0], and converse/commutative like [0,1] [1,0] i need to get rid of these

    arr = [tuple(row) for row in np.sort(combination)]; uniq1 = np.unique(arr, axis = 0) #This will get rid of converse/commutative ones

    for i,row in enumerate(uniq1): #This will get rid of same pairs like [0,0]

        if row[0] == row[1]:

            uniq1 = np.delete(uniq1, count, axis = 0)

            count = count - 1

        count = count + 1

    return uniq1
#Will graph grapha a scatterplot coloring the unique classes

def scatterClass(axe, feature, xval, yval):

    for i,idx in enumerate(df[feature].unique()):

        x = np.array(y_train[:] == idx)

        axe.scatter(X_train[x,xval], X_train[x,yval], label = idx) #It's somehow automatically coloring using distinct colors per species

        axe.set_xlabel(X.columns[xval])

        axe.set_ylabel(X.columns[yval])

        axe.legend()
combs = combinations(np.array([0,1,2,3]))

iterations = np.size(combs, axis=0)

plt.figure(figsize=(30,10))

for j in range(iterations):

    ax = plt.subplot(2,4,j+1)

    tup = combs[j,:]#grab tupple of combinations in form (x,y) and I should have num = iterations of combinations

    scatterClass(ax, 'Species', tup[0], tup[1])

plt.tight_layout()

plt.show()
plt.figure(figsize=(10,10))

ax = plt.subplot()

scatterClass(ax, 'Species', 2, 3)

plt.show()
#Using countourf

#Lets first make a normal line y = x + 0 which means w0 = 0, w1 = 1, w2 = -1

w = np.array([0,1,-1])

delta = 0.025

def weightsContourf(w0,w1,w2):

    return contourfLine(np.array([w0,w1,w2]))

def contourfLine(w):

    x = np.arange(df['PetalLengthCm'].values.min() - 0.25, df['PetalLengthCm'].values.max() + 0.25,delta) #We are using PetalLengthCm

    y = np.arange(df['PetalWidthCm'].values.min() - 0.25,df['PetalWidthCm'].values.max() + 0.25,delta)

    x1,x2 = np.meshgrid(x,y)

    #Now I want to be able to make tuples to span through the entire graph from 0 to 3

    x_vector = np.array([x1.ravel(),x2.ravel()]).T #We have coordinates [x1,x2] = [0,0],[0,1],[0,2]...[3,3]

    Z = x_vector.dot(w[1:]) + w[0] #This is z = w0 + w1*x1 ... + wn*xn

    Z = Z.reshape(x1.shape)

    y_hat = np.where(Z >= 0, 1, -1) #This is where the prediction is made according to the value of Z

    fig, ax = plt.subplots()

    ax.contourf(x1, x2, y_hat, alpha = 0.8)

    scatterClass(ax,'Species',2,3)

    plt.title("Contourf")

    plt.xlabel('PetalLengthCm')

    plt.ylabel('PetalWidthCm')

    plt.show()

dis1 = interactive(weightsContourf, w0=widgets.FloatSlider(min=-3,max=3,step=0.1,value=2), w1=widgets.FloatSlider(min=-3,max=3,step=0.1,value=-0.5), w2=widgets.FloatSlider(min=-3,max=3,step=0.1,value=-1))

display(dis1)
#Using line plot

def weightsLine(w0,w1,w2):

    return decisionLine(np.array([w0,w1,w2]))

def decisionLine(w):

    x = np.arange(df['PetalLengthCm'].values.min() - 0.25, df['PetalLengthCm'].values.max() + 0.25,delta)

    fig, ax = plt.subplots()

    if(w[2] == 0 and (w[0] == 0 or w[1] == 0)):

        ax.plot()

    else:

        y = -(w[1]/w[2])*x - (w[0]/w[2])

        ax.plot(x,y)

    plt.title("Line Plot")

    scatterClass(ax,'Species',2,3)

    plt.xlabel('PetalLengthCm')

    plt.ylabel('PetalWidthCm')

    plt.xlim(df['PetalLengthCm'].values.min() - 0.25, df['PetalLengthCm'].values.max())

    plt.ylim(df['PetalWidthCm'].values.min() - 0.25,df['PetalWidthCm'].values.max())

    plt.show()

    print()

disp2 = interactive(weightsLine, w0=widgets.FloatSlider(min=-3,max=3,step=0.1,value=2), w1=widgets.FloatSlider(min=-3,max=3,step=0.1,value=-0.5), w2=widgets.FloatSlider(min=-3,max=3,step=0.1,value=-1))

display(disp2)
#This is used to make sure we don't have Iris-virginica in our data to keep it an easy binary classification

#Note, y_test[2] didn't work because the index is not correct

X_train_novirg = X_train.copy()

y_train_novirg = y_train.copy()

counter = 0

for i in range(len(y_train)):

    if y_train[i] == 'Iris-virginica':

        X_train_novirg = np.delete(X_train_novirg,counter, axis=0)

        y_train_novirg = np.delete(y_train_novirg,counter)

        counter = counter - 1

    counter = counter + 1

X_train_novirg = X_train_novirg[:,[2,3]] #This will give us [PetalLengthCm,PetalWidthCm]

y_train_novirg = np.where(y_train_novirg == 'Iris-setosa',1,-1) #Turns class labels into 1 and -1



X_test_novirg = X_test.copy()

y_test_novirg = y_test.copy()

counter = 0

for i in range(len(y_test)):

    if y_test[i] == 'Iris-virginica':

        X_test_novirg = np.delete(X_test_novirg,counter,axis=0)

        y_test_novirg = np.delete(y_test_novirg,counter,axis=0)

        counter = counter - 1

    counter = counter + 1

X_test_novirg = X_test_novirg[:,[2,3]]

y_test_novirg = np.where(y_test_novirg == 'Iris-setosa',1,-1)
from sklearn.linear_model import Perceptron

ppt = Perceptron(penalty = None, alpha = 0.01, max_iter = 10, shuffle = True, eta0 = 1, random_state = 1) #tol is the stopping criteria, which is a parameter as well

ppt.fit(X_train_novirg,y_train_novirg)

weightsContourf(ppt.intercept_,ppt.coef_[0,0],ppt.coef_[0,1])
print("Score for training set: " + str(ppt.score(X_train_novirg,y_train_novirg)*100) + "%")

print("Score for test set: " + str(ppt.score(X_test_novirg,y_test_novirg)*100) + "%")

print("Weights:" ,ppt.coef_, "Intercept:", ppt.intercept_)
X_train_noset = X_train[:,[2,3]].copy()

y_train_noset = y_train.copy()

X_test_noset = X_test[:,[2,3]].copy()

y_test_noset = y_test.copy()

counter = 0

for i in range(len(y_train)):

    if y_train[i] == 'Iris-setosa':

        X_train_noset = np.delete(X_train_noset,counter,axis=0)

        y_train_noset = np.delete(y_train_noset,counter)

        counter = counter - 1

    counter = counter + 1

y_train_noset = np.where(y_train_noset == 'Iris-versicolor', 1 ,-1)

counter = 0

for i in range(len(y_test)):

    if y_test[i] == 'Iris-setosa':

        X_test_noset = np.delete(X_test_noset,counter,axis=0)

        y_test_noset = np.delete(y_test_noset,counter)

        counter = counter - 1

    counter = counter + 1

y_test_noset = np.where(y_test_noset == 'Iris-versicolor', 1 ,-1)
ppt2 = Perceptron(max_iter = 1000000, shuffle = True, random_state = 1, eta0 = 0.000001, tol = None) #Note, setting tol to None will allow it to go past 1e-3 which is very important

ppt2.fit(X_train_noset,y_train_noset)

weightsContourf(ppt2.intercept_,ppt2.coef_[0,0],ppt2.coef_[0,1])
print("Score for training set: ", str(ppt2.score(X_train_noset,y_train_noset)*100)+"%")

print("# of wrong classifications:" , str((1. - ppt2.score(X_train_noset,y_train_noset))*len(X_train_noset)))

print()

print("Score for testing set: ", str(ppt2.score(X_test_noset,y_test_noset)*100)+"%")

print("# of wrong classifications:" , str((1. - ppt2.score(X_test_noset,y_test_noset))*len(X_test_noset)))
#First we will use SepalWidthCm

X_3d = df[['SepalWidthCm','PetalLengthCm','PetalWidthCm']].copy().values

y_3d = df['Species'].values

#Now delete Iris-setosa

X_vv = X_3d.copy()

y_vv = y_3d.copy()

counter = 0

for i in range(len(y)):

    if y[i] == 'Iris-setosa':

        X_vv = np.delete(X_vv,counter, axis = 0)

        y_vv = np.delete(y_vv,counter)

        counter = counter - 1

    counter = counter + 1

#split the training and testing data set

X_vv_train, X_vv_test, y_vv_train, y_vv_test = train_test_split(X_vv, y_vv, stratify = y_vv, test_size = 0.50, random_state = 1)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

for i in np.unique(y_vv):

    index = np.array(y_vv[:] == i)

    ax.scatter(X_vv[index,0],X_vv[index,1],X_vv[index,2], label = i)

plt.show()
ppt3 = Perceptron(max_iter = 1000000, shuffle = True, random_state = 1, eta0 = 0.000001, tol = None)

ppt3.fit(X_vv_train, y_vv_train)

print("Score for training set: ", str(ppt3.score(X_vv_train,y_vv_train)*100)+"%")

print("# of wrong classifications:" , str((1. - ppt3.score(X_vv_train,y_vv_train))*len(X_vv_train)))

print()

print("Score for testing set: ", str(ppt3.score(X_vv_test,y_vv_test)*100)+"%")

print("# of wrong classifications:" , str((1. - ppt3.score(X_vv_test,y_vv_test))*len(X_vv_test)))
ppt3 = Perceptron(max_iter = 10000, shuffle = True, random_state = 1, eta0 = 0.00000001, tol = None)

ppt3.fit(X_vv_train,y_vv_train)

print("Training accuracy: ", str(ppt3.score(X_vv_train,y_vv_train)*100)+"%")

print("Testing accuracy: ", str(ppt3.score(X_vv_test,y_vv_test)*100)+"%")
X_all = df.iloc[:,[0,1,2,3]].values

y_all = df.iloc[:,-1].values

y_all = np.reshape(y_all,(-1,1))

#Now delete Iris-setosa

X_all_vv = X_all.copy()

y_all_vv = y_all.copy()

counter = 0

for i in range(len(y_all)):

    if y_all[i] == 'Iris-setosa':

        X_all_vv = np.delete(X_all_vv,counter, axis = 0)

        y_all_vv = np.delete(y_all_vv,counter)

        counter = counter - 1

    counter = counter + 1

#split the training and testing data set

X_all_vv_train, X_all_vv_test, y_all_vv_train, y_all_vv_test = train_test_split(X_all_vv, y_all_vv, stratify = y_vv, test_size = 0.50, random_state = 1)
ppt4 = Perceptron(max_iter = 1000, eta0 = 0.001, shuffle = True, random_state = 1, tol = None)

ppt4.fit(X_all_vv_train, y_all_vv_train)

print("Training accuracy: ", str(ppt4.score(X_all_vv_train,y_all_vv_train)*100)+"%")

print("Testing accuracy: ", str(ppt4.score(X_all_vv_test,y_all_vv_test)*100)+"%")
ppt4 = Perceptron(max_iter = 3000000, eta0 = 0.000001, shuffle = True, random_state = 1, tol = None)

ppt4.fit(X_all_vv_train, y_all_vv_train)

print("Training accuracy: ", str(ppt4.score(X_all_vv_train,y_all_vv_train)*100)+"%")

print("Testing accuracy: ", str(ppt4.score(X_all_vv_test,y_all_vv_test)*100)+"%")